use index_vec::IndexVec;

use itertools::Itertools;
use log::info;
use std::{
    collections::{BTreeMap, BTreeSet, HashMap, hash_map::Entry},
    time::Instant,
};

use crate::{
    aph::BidiagonalAph,
    formats::ft::{
        ctmc_aut::{CTMCAut, Transition},
        modules::get_modules,
        node::{FTNode, Status},
        nodetree::{NodeTree, Restriction, Trigger},
        parser::{GateType, NodeId, ParsedTree, parse_tree_w_fdep},
        state::{State, StateCode},
        utils::{ConstructionMethod, RoundMode},
    },
    linalg::{
        Vector,
        fields::{PseudoField, Round},
    },
    operations::{convolution_bidiagonal, maximum_bidiagonal, minimum_bidiagonal},
    representation::Representation,
    voting::build_vot_bidi,
};

mod ctmc_aut;
mod modules;
mod node;
mod nodetree;
mod parser;
mod state;
pub mod utils;

/// Computes all possible rates from the state [st] in the tree [ft].
// If we want to change the semantics so that only the last failure is blocked, then, we need to rework this part.
fn rates_from<F: PseudoField>(
    st: &State,
    ft: &NodeTree<F>,
    _restrictions: &[Restriction],
) -> Vec<(NodeId, F)> {
    // TODO: What should happen if a dependency is fail safe?
    // Because Im describing the ONLY state of the components, a priori, I do not know if a rate would actually create new behaviour or not. Which makes the state space exploration less efficient.
    if let State::Transient(status) = st {
        let mut rates = vec![];
        for (nid, status) in status.iter() {
            let node = &ft.nodes[nid.raw() as usize];
            match (node, status) {
                (FTNode::Component(aph, dorm), Status::Active(phase))
                    if !st.cuc_spares().contains(&nid) && ft.warm_spares().contains(nid) =>
                {
                    // If dormancy factor must be used (is not a CuC and is a warm spare) and the dorm is not zero.
                    if !dorm.is_zero() {
                        // Use the dormancy factor if is a node that can be dormant (i.e. child of a WSP) and is not currently use
                        let phase_u: usize = (*phase).into();
                        let mut value = aph.repr().get(phase_u, phase_u);
                        if !value.is_zero() {
                            value.mul_assign(dorm);
                            rates.push((*nid, value));
                        }
                    }
                }
                (FTNode::Component(aph, _), Status::Active(phase)) => {
                    // Otherwise, any other active component contribute a rate to exit the state normally.
                    let phase_u: usize = (*phase).into();
                    let value = aph.repr().get(phase_u, phase_u);
                    if !value.is_zero() {
                        rates.push((*nid, value));
                    }
                }
                (FTNode::Gate(_, _), _)
                | (_, Status::Passive(_))
                | (_, Status::FailSafe)
                | (_, Status::Failed)
                | (_, Status::CuC(_)) => {
                    // Only Component failures add new rates.
                }
            }
        }
        rates
    } else {
        vec![]
    }
}

/// Sanitize the state. First, triggers the functional dependencies, if any.
/// Then, it check the status of the FT under the state.
/// Finally, it activates the nodes that were sequentialized, if any.
fn sanitize<F: PseudoField>(
    state: State,
    ft: &NodeTree<F>,
    rest: &[Restriction],
    deps: &[Trigger],
) -> State {
    let state = if ft.has_dependencies() {
        deps.iter()
            .fold(state, |st, trg| trg.propagate_dependencies(&st, ft))
    } else {
        state
    };

    let state = ft.under_state(&state);

    if ft.has_sequences() {
        rest.iter()
            .fold(state, |st, r| r.activate_sequentialized(&st, ft))
    } else {
        state
    }
}

fn explore_state_space<F: PseudoField>(ft: &NodeTree<F>) -> CTMCAut<F> {
    let time_start = Instant::now();
    let mut queue = Vec::new(); // States queue.
    let (seen_components, restrictions, dependencies) = ft.initialize_state_space_exploration();
    let mut transitions: Vec<Transition<F>> = vec![]; // Transitions collection

    // All seen "raw" states. If a "raw" state was already seen, its sanitized version is already in the set of seen states,
    // together with its initial probability. Use of HashMap because insertions are cheaper than in a BTreeMap.
    let mut seen_states_map: HashMap<StateCode, (StateCode, F)> = HashMap::new();
    let mut already_seen_states: BTreeSet<StateCode> = BTreeSet::new();

    let st = State::Transient(seen_components.into_iter().collect());
    seen_states_map.insert(st.to_code(), (st.to_code(), st.compute_initial(ft)));
    queue.push(st);

    while let Some(st) = queue.pop() {
        already_seen_states.insert(st.to_code());
        let rates = rates_from(&st, ft, &restrictions);

        for (nid, rate) in rates.into_iter() {
            let neighbor_raw_st =
                st.take_transition_from(&nid, ft.nodes[nid].component_size().try_into().unwrap());

            //TODO: We could have a parallel state, that contains the NodeId of the gates, on which we can mark them as failed directly, and the we use that to sanitize
            let neighbor_raw_code = neighbor_raw_st.to_code();
            let neighbor_code = match seen_states_map.entry(neighbor_raw_code.clone()) {
                Entry::Occupied(entry) => Some(entry.get().clone().0),
                Entry::Vacant(entry) => {
                    let neighbor_st =
                        sanitize(neighbor_raw_st.clone(), ft, &restrictions, &dependencies);
                    // Restrictions are applied when taking the new transitions. We can skip validation, although is not a bad idea to double check.
                    let neighbor_code = neighbor_st.to_code();
                    entry.insert((neighbor_code.clone(), neighbor_st.compute_initial(ft)));
                    // If the state is the ABS, or has been seen, or will be seen, we dont add it to the queue.
                    if !neighbor_st.is_absorbing()
                        && !already_seen_states.contains(&neighbor_code)
                        && !queue.contains(&neighbor_st)
                    {
                        queue.push(neighbor_st)
                    };
                    Some(neighbor_code)
                }
            };
            if let Some(to_code) = neighbor_code {
                transitions.push(Transition(st.to_code(), to_code, rate));
            }
        }
    }

    drop(already_seen_states);

    // TODO: Improve all of this... The initials and the retain of non-abs should be more clear.
    let mut seen_states: Vec<(StateCode, F)> = seen_states_map.into_values().collect();
    seen_states.retain(|(x, _)| !x.is_abs());
    seen_states = seen_states.into_iter().unique().collect_vec();
    seen_states.sort_by(|a, b| a.0.cmp(&b.0));

    let (seen_states, initial): (Vec<StateCode>, Vec<F>) = seen_states.into_iter().unzip();

    info!(
        "Elapsed `exploration`: {:?}. #states: {:?}. #transitions: {:?}",
        time_start.elapsed(),
        seen_states.len(),
        transitions.len()
    );

    // Merge all transitions. first group them by the source and target, then we add the rates up.
    let tt: BTreeMap<StateCode, HashMap<StateCode, F>> = transitions
        .into_iter()
        .into_group_map_by(|x| (x.0.clone(), x.1.clone()))
        .into_iter()
        // add all rates from p to q.
        .map(|(key, values)| {
            (
                (key.0, key.1),
                values
                    .into_iter()
                    .fold(F::zero(), |acc, Transition(_, _, rate)| acc + rate),
            )
        })
        .into_group_map_by(|((from, _), _)| from.clone())
        .into_iter()
        // Separate into shape: from : {to : rate}
        .map(|(key, grouped)| {
            (
                key.clone(),
                // Set up a hashmap for each (`target`, `rate`) from  state `key`.
                grouped
                    .into_iter()
                    .map(|((_, to), rate)| (to, rate))
                    .collect::<HashMap<StateCode, F>>(),
            )
        })
        .collect();
    CTMCAut::new(Vector::from(initial), seen_states, tt)
}

fn create_node_w_exploration<F: PseudoField>(
    ft: &NodeTree<F>,
    reduce: bool,
    method: ConstructionMethod,
    round: Round,
) -> Option<FTNode<F>> {
    let ftaut: CTMCAut<F> = explore_state_space(ft);
    let mut aph = ftaut.generate_aph(method, ft.root_id, round);
    if reduce {
        aph.reduce();
    }

    Some(FTNode::Component(aph, F::one()))
}

fn create_node<F: PseudoField>(
    ft: &NodeTree<F>,
    reduce: bool,
    method: ConstructionMethod,
    round: Round,
    mode: RoundMode,
) -> Option<FTNode<F>> {
    let time_start = Instant::now();
    let aph = if ft.is_static() && ft.is_independent() {
        info!("Creating node ({:?}) with operators.", ft.root_id);
        if matches!(mode, RoundMode::DepthTLE) || matches!(mode, RoundMode::MixTLE) {
            if ft.root_id == 0 {
                create_node_w_operators(ft.root_id, ft, reduce, round, mode)
            } else {
                create_node_w_operators(ft.root_id, ft, reduce, Round::Nearest, mode)
            }
        } else {
            create_node_w_operators(ft.root_id, ft, reduce, round, mode)
        }
    } else {
        info!("Creating node ({:?}) with exploration.", ft.root_id);
        if matches!(mode, RoundMode::DepthTLE) || matches!(mode, RoundMode::MixTLE) {
            if ft.root_id == 0 {
                create_node_w_exploration(ft, reduce, method, round)
            } else {
                create_node_w_exploration(ft, reduce, method, Round::Nearest)
            }
        } else {
            create_node_w_exploration(ft, reduce, method, round)
        }
    };
    info!("Elapsed `Construction`: {:?}.", time_start.elapsed());

    aph
}

fn create_node_w_operators<F: PseudoField>(
    self_id: NodeId,
    ft: &NodeTree<F>,
    reduce: bool,
    round: Round,
    mode: RoundMode,
) -> Option<FTNode<F>> {
    let node = ft.nodes[self_id].clone();
    match &node {
        FTNode::Component(_, _) => Some(node),
        // Gatetypes are only And - Or - Vot - Seq
        FTNode::Gate(gate_type, children) => {
            let child_components = children
                .iter()
                .map(|child_id| {
                    create_node_w_operators::<F>(*child_id, ft, reduce, round, mode).unwrap()
                })
                .collect::<Vec<FTNode<F>>>();
            let mut aph: BidiagonalAph<F> = match gate_type {
                GateType::And => maximum_bidiagonal(
                    child_components.iter().map(|n| n.get_component().unwrap()),
                    round,
                    mode,
                ),
                GateType::Or => minimum_bidiagonal(
                    child_components.iter().map(|n| n.get_component().unwrap()),
                    round,
                    mode,
                ),
                GateType::Seq => convolution_bidiagonal(
                    child_components.iter().map(|n| n.get_component().unwrap()),
                    round,
                ),
                GateType::KofN(k) => {
                    let generators = child_components
                        .iter()
                        .map(|n| n.get_component().unwrap())
                        .collect_vec();
                    build_vot_bidi(&generators, *k)
                }
                _ => unreachable!(
                    "This should not happen. The FT Metadata has been collected wrong."
                ),
            };

            if reduce {
                aph.reduce();
            }
            Some(FTNode::Component(aph, F::one()))
        }
    }
}

fn convert_ft<F: PseudoField>(
    tree: ParsedTree,
    module_ids: Vec<NodeId>,
    reduce: bool,
    method: ConstructionMethod,
    round: Round,
    mode: RoundMode,
) -> Option<FTNode<F>> {
    let root_id = tree.rootid;
    let mut nodes = tree
        .nodes
        .into_iter()
        .map(|e| FTNode::<F>::from_node_kind(&e, &tree.lookup_table))
        .collect::<IndexVec<NodeId, FTNode<F>>>();

    module_ids.iter().for_each(|&m_id| {
        let ft = NodeTree::new(m_id, nodes.clone());
        let comp_node = create_node::<F>(&ft, reduce, method, round, mode).unwrap();
        nodes[m_id] = comp_node;
    });
    let ft = NodeTree::new(root_id, nodes);
    create_node::<F>(&ft, reduce, method, round, mode)
}

pub fn aph_from_ft<F: PseudoField>(
    source: &str,
    reduce: bool,
    modularise: bool,
    method: ConstructionMethod,
    round: Round,
    mode: RoundMode,
) -> BidiagonalAph<F> {
    let time_start = Instant::now();
    let (mut tree, triggered) = parse_tree_w_fdep(source).unwrap();
    if log::log_enabled!(log::Level::Info) {
        let _lookup = tree
            .lookup_table
            .clone()
            .into_iter()
            .map(|(k, v)| (v, k))
            .collect::<BTreeMap<NodeId, String>>();
    }
    info!("Elapsed `Parsing DFT`: {:?}.", time_start.elapsed());

    let module_ids = if modularise {
        // Transform the triggered components to OR of itslef and the trigger.
        let time_start = Instant::now();
        for (t_node, trigger) in triggered.iter().rev() {
            tree.make_triggered(t_node, trigger);
        }
        let mut module_ids = get_modules(&tree);
        module_ids.reverse();
        // Revert transformed events and remove OR gates.
        for (t_node, _) in triggered.into_iter() {
            tree.revert_triggered_be(format!("{t_node}_t"));
        }
        info!(
            "Elapsed `Modularisation`: {:?}. Found {} modules.",
            time_start.elapsed(),
            module_ids.len()
        );
        module_ids
    } else {
        vec![]
    };
    match convert_ft::<F>(tree, module_ids, reduce, method, round, mode) {
        Some(FTNode::Component(aph, _)) => aph,
        _ => todo!(),
    }
}
