use std::collections::HashMap;

use index_vec::IndexVec;
use itertools::{Itertools, all};

use crate::{
    formats::ft::{
        State,
        node::{FTNode, Status},
        parser::{GateType, NodeId, TriggerID},
    },
    linalg::fields::PseudoField,
};

/// Restrictions are constraint to the behavior of the Fault Tree.
/// For example, a Sequence enforcer gate, a PAND gate, etc.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub(crate) enum Restriction {
    /// In order to let the first element to fail, all other elements from the vec must be failed.
    /// Used by priority And and Sequence enforcers.
    Sequentialized(NodeId, Vec<NodeId>),
}

impl Restriction {
    /// Return the [NodeId] of the node that is restricted.
    pub(crate) fn restricted_id(&self) -> &NodeId {
        match self {
            Restriction::Sequentialized(nid, _) => nid,
        }
    }

    pub(crate) fn activate_sequentialized<F: PseudoField>(
        &self,
        st: &State,
        ft: &NodeTree<F>,
    ) -> State {
        match (&self, &st) {
            (_, State::Absorbing) => State::Absorbing,
            (_, State::Locked) => State::Locked,
            // Spares gate activation depends on the gate in question. Is done in the [`under_state()`] method
            (
                Restriction::Sequentialized(restriced_id, required),
                State::Transient(transient_state_map),
            ) => {
                let mut map_cl = transient_state_map.clone();
                // The restriction is lifted, if there is all required nodes that are failed.
                if all(required, |nid| ft.nodes[*nid].is_failed(nid, st, ft)) {
                    ft.get_child_from_node(restriced_id)
                        .iter()
                        .for_each(|&c_id| {
                            map_cl.entry(c_id).and_modify(|e| e.set_active(0));
                        });
                }
                State::Transient(map_cl)
            }
        }
    }
}

#[non_exhaustive]
#[derive(Debug, Clone)]
/// If the first element fails, all other elements from the vec fail.
/// Used by functional dependency gates.
pub(crate) struct Trigger(NodeId, Vec<NodeId>);

impl Trigger {
    pub(crate) fn propagate_dependencies<F: PseudoField>(
        &self,
        st: &State,
        ft: &NodeTree<F>,
    ) -> State {
        match (&self, &st) {
            (Trigger(_, _), State::Absorbing) => State::Absorbing,
            (Trigger(_, _), State::Locked) => State::Locked,
            (Trigger(trigger_id, children), State::Transient(transient_state_map)) => {
                let mut new_map = transient_state_map.clone();
                let is_trigger_failed = ft.nodes[*trigger_id].is_failed(trigger_id, st, ft);
                if is_trigger_failed {
                    children.iter().for_each(|c_id| {
                        new_map.insert(*c_id, Status::Failed);
                    });
                }
                State::Transient(new_map)
            }
        }
    }
}

#[derive(Clone, Debug, Default)]
pub(crate) struct Metadata {
    pub(crate) has_sequences: bool,
    pub(crate) has_spares: bool,
    pub(crate) has_dependencies: bool,
    pub(crate) min_max_vot_seq: bool,
    pub(crate) is_independent: bool,
    pub(crate) warm_spares: Vec<NodeId>,
}

// TODO: Only a ref to the nodes?
#[derive(Debug)]
pub(crate) struct NodeTree<F: PseudoField> {
    pub root_id: NodeId,
    pub nodes: IndexVec<NodeId, FTNode<F>>,
    pub metadata: Metadata,
}

#[allow(unused)]
impl<F: PseudoField> NodeTree<F> {
    pub fn is_static(&self) -> bool {
        self.metadata.min_max_vot_seq
    }

    pub fn is_independent(&self) -> bool {
        self.metadata.is_independent
    }

    pub fn has_dependencies(&self) -> bool {
        self.metadata.has_dependencies
    }

    pub fn has_sequences(&self) -> bool {
        self.metadata.has_sequences
    }

    pub fn has_spares(&self) -> bool {
        self.metadata.has_spares
    }

    pub fn warm_spares(&self) -> &Vec<NodeId> {
        &self.metadata.warm_spares
    }

    pub fn new(root_id: NodeId, nodes: IndexVec<NodeId, FTNode<F>>) -> Self {
        let mut ft = Self {
            root_id,
            nodes,
            metadata: Metadata::default(),
        };
        ft.metadata = ft.fill_metadata();
        ft
    }

    /// Returns the Metadata of [self]. Does a visiting dfs to check which gates are visited.
    fn fill_metadata(&self) -> Metadata {
        // visited once, more than once.
        type Visits = (bool, bool);
        fn visit_dfs<F: PseudoField>(
            nodes_visits: &mut IndexVec<NodeId, Visits>,
            nodes: &IndexVec<NodeId, FTNode<F>>,
            curr_idx: NodeId,
        ) {
            // Take current node using the idx
            let curr_node_visits = &mut nodes_visits[curr_idx];
            let curr_child: &FTNode<F> = &nodes[curr_idx];
            // Second time? Only mark
            if curr_node_visits.0 && !curr_node_visits.1 {
                curr_node_visits.1 = true;
            }
            // First time? Mark and propagate.
            if !curr_node_visits.0 {
                curr_node_visits.0 = true;
                if let Some(node_children) = curr_child.children() {
                    for child_nid in node_children
                        .into_iter()
                        .filter(|c_id| !nodes_visits[*c_id].1)
                        .collect::<Vec<NodeId>>()
                    {
                        visit_dfs(nodes_visits, nodes, child_nid);
                    }
                }
            }
        }
        let mut vis_nodes: IndexVec<NodeId, Visits> =
            IndexVec::from_vec(vec![(false, false); self.nodes.len()]);
        visit_dfs(&mut vis_nodes, &self.nodes, self.root_id);

        let (_, _comp_from_triggers, triggered_ids) = self.all_triggers();
        let mut metadata = Metadata::default();
        let mut seen_twice = false;
        let mut warm_child = vec![];
        let mut requires_expl = false;

        vis_nodes
            .iter_enumerated()
            .zip(&self.nodes)
            .for_each(|((idx, (fst, snd)), node)| {
                if *fst {
                    if !seen_twice && *snd {
                        seen_twice = true;
                    }
                    if matches!(node, FTNode::Gate(GateType::Pand(_), _)) {
                        todo!("Priority AND not yet supported.");
                    }
                    if matches!(node, FTNode::Gate(GateType::Wsp(_), _)) {
                        warm_child.extend(node.children().unwrap());
                        requires_expl = true;
                        metadata.has_spares = true;
                    }
                    if matches!(node, FTNode::Gate(GateType::Csp(_), _))
                        || matches!(node, FTNode::Gate(GateType::Hsp(_), _))
                    {
                        requires_expl = true;
                        metadata.has_spares = true;
                    }
                    if matches!(node, FTNode::Gate(GateType::Seq, _)) {
                        metadata.has_sequences = true;
                    }
                    // For Functional dependencies, we actually want to keep them only if they do play a role in the FT.
                    if triggered_ids.contains(&idx) {
                        metadata.has_dependencies = true;
                        requires_expl = true;
                    }
                }
            });

        metadata.min_max_vot_seq = !requires_expl;
        metadata.is_independent = !seen_twice;
        metadata.warm_spares = warm_child;

        metadata
    }

    /// List all reachable [NodeId] from the [root_id].
    /// This can considerable help the state space, given that it can occur that components are
    /// in the .dft file, but never used.
    pub fn initialize_state_space_exploration(
        &self,
    ) -> (HashMap<NodeId, Status>, Vec<Restriction>, Vec<Trigger>) {
        let (triggers, comp_from_triggers, _) = self.all_triggers();
        let mut queue = Vec::new();
        let mut initial_transient_map = HashMap::new();
        let mut restrictions = vec![];
        let mut spare_nids: Vec<NodeId> = vec![];

        // Does a BFS on the (possible modular) FT.
        queue.push(&self.root_id);
        while let Some(nid) = queue.pop() {
            match &self.nodes[*nid] {
                FTNode::Component(_, _) => {
                    initial_transient_map.insert(*nid, Status::default());
                }
                FTNode::Gate(gt, children) => {
                    queue.extend(children);
                    // Additional behaviour for gates, i.e. Dependencies, restrictions, triggers, etc.
                    match gt {
                        GateType::Csp(TriggerID::ID(cuc_id)) => {
                            queue.push(cuc_id);
                            initial_transient_map.insert(*nid, Status::new_spare(*cuc_id));
                            for c in children.iter() {
                                spare_nids.extend(self.get_child_from_node(c));
                            }
                        }
                        GateType::Wsp(TriggerID::ID(cuc_id))
                        | GateType::Hsp(TriggerID::ID(cuc_id)) => {
                            // Warm and Hot spares behave the same for the state.
                            queue.push(cuc_id);
                            initial_transient_map.insert(*nid, Status::new_spare(*cuc_id));
                        }
                        GateType::Pand(TriggerID::ID(trigger)) => restrictions.push(
                            children
                                .iter()
                                .map(|c| Restriction::Sequentialized(*c, vec![*trigger]))
                                .collect(),
                        ),
                        GateType::Seq => {
                            let mut requirements = vec![];
                            let mut history = Vec::new();
                            for &c in children.iter() {
                                if !history.is_empty() {
                                    requirements
                                        .push(Restriction::Sequentialized(c, history.clone()));
                                }
                                history.push(c);
                            }
                            restrictions.push(requirements)
                        }
                        // FDep are never inserted into the queue.
                        // GateType::FDep(TriggerID::ID(t_id)) => {
                        //     dependencies.push(Dependency::Trigger(*t_id, children.clone()))
                        // }
                        _ => {}
                    }
                }
            }
        }

        // Check restrictions:
        // Gates propagate activity as follows:
        // A direct child of a spare gate is active if the spare gate is active and using that child.
        // The first direct child of an FDEP gate is always active.
        // Sequence enforcers and inspection modules do not affect activity of their children.
        // The direct children of a gate of any other type are active if the gate is active.
        // If none of the above specify that a BE should be active, it is dormant.

        // First we put to sleep all spared elements
        if self.has_spares() {
            initial_transient_map.iter_mut().for_each(|t| {
                if spare_nids.contains(t.0) {
                    t.1.set_passive(0);
                }
            });
        }

        // We activate or insert to the map all component ids from the triggers of fdep.
        if self.has_dependencies() {
            comp_from_triggers.into_iter().for_each(|c_id| {
                initial_transient_map
                    .entry(c_id)
                    .or_insert_with(Status::default);
            });
        }

        // We put to sleep all the restricted nodes from sequence gates.
        let restrictions = if self.has_sequences() {
            let mut restrictions: Vec<Restriction> = restrictions.into_iter().flatten().collect();

            // TODO: We must consider all the restrictions from above.
            let res_ids = restrictions
                .iter()
                .flat_map(|r| self.get_child_from_node(r.restricted_id()))
                .collect::<Vec<NodeId>>();

            initial_transient_map.iter_mut().for_each(|t| {
                if res_ids.contains(t.0) {
                    t.1.set_passive(0);
                }
            });
            restrictions
        } else {
            vec![]
        };

        (initial_transient_map, restrictions, triggers)
    }

    pub(crate) fn get_child_from_node(&self, node_id: &NodeId) -> Vec<NodeId> {
        match &self.nodes[*node_id] {
            FTNode::Component(_, _) => vec![*node_id],
            FTNode::Gate(GateType::Hsp(TriggerID::ID(tid)), children)
            | FTNode::Gate(GateType::Pand(TriggerID::ID(tid)), children)
            | FTNode::Gate(GateType::Csp(TriggerID::ID(tid)), children) => {
                let mut childs = self.get_child_from_node(tid);
                childs.extend(children.iter().flat_map(|c| self.get_child_from_node(c)));
                childs
            }
            FTNode::Gate(_, children) => children
                .iter()
                .flat_map(|c| self.get_child_from_node(c))
                .collect(),
        }
    }

    fn can_fail_safe(&self) -> bool {
        self.nodes
            .iter()
            .any(|n| matches!(n, FTNode::Gate(GateType::Pand(_), _)))
    }

    fn is_failed(&self, st: &State) -> bool {
        self.nodes[self.root_id].is_failed(&self.root_id, st, self)
    }

    fn is_fail_safe(&self, st: &State) -> bool {
        self.nodes[self.root_id].is_fail_safe(st, self)
    }

    /// Return all triggers present in the Tree, and all the basic components that appear in the trigger.
    fn all_triggers(&self) -> (Vec<Trigger>, Vec<NodeId>, Vec<NodeId>) {
        let mut components_under_trigger = vec![];
        let mut triggered_ids = vec![];
        let t1 = self
            .nodes
            .iter()
            .enumerate()
            .filter_map(|(id, node)| match &node {
                FTNode::Gate(GateType::FDep(TriggerID::ID(t_id)), deps) => {
                    components_under_trigger.extend(self.get_child_from_node(t_id));
                    triggered_ids.extend(deps);
                    Some(Trigger(*t_id, deps.clone()))
                }
                FTNode::Gate(GateType::FDep(TriggerID::Name(_)), _) => {
                    panic!("This should not happen. Unproccesed node {node:?}.")
                }
                _ => None,
            })
            .collect();

        (t1, components_under_trigger, triggered_ids)
    }

    /// Checks if the state `st` makes the FT failed.
    /// If the FT is not failed in `st`, and there are spare nodes in the FT, it finds
    /// new available childs for the spare gates.
    /// It returns the new state.
    pub fn under_state(&self, st: &State) -> State {
        let s = if self.is_failed(st) {
            State::Absorbing
        } else if self.can_fail_safe() && self.is_fail_safe(st) {
            State::Locked
        } else {
            match st {
                State::Absorbing => State::Absorbing,
                State::Locked => State::Locked,
                State::Transient(map) if self.has_spares() => {
                    // This case only falls here if we have restrictions like PAND, SEQ, or Spares.
                    let mut new_map = map.clone();
                    let failed_comp_nid: Vec<&NodeId> = map
                        .iter()
                        .filter_map(
                            |(nid, status)| if status.is_failed() { Some(nid) } else { None },
                        )
                        .collect();
                    let unusable_nids: Vec<&NodeId> = failed_comp_nid
                        .iter()
                        .chain(&st.cuc_spares())
                        .cloned()
                        .collect();
                    // Iterate over the st.
                    for (nid, status) in map.iter() {
                        match status {
                            // If is a CuC(cuc_nid), and the cuc_nid is failed
                            Status::CuC(cuc_nid)
                                if self.nodes[*cuc_nid].is_failed(cuc_nid, st, self) =>
                            {
                                // Extension must happen after checking that the Currently used Child is failed, i.e. is in the failed nid list.
                                let new_status = if let Some(new_cuc_nid) =
                                    &self.nodes[*nid].next_cuc(&unusable_nids)
                                {
                                    // Search for next usable child: Activation happens before, just insert into the state.
                                    self.get_child_from_node(new_cuc_nid)
                                        .iter()
                                        .for_each(|c_id| {
                                            new_map.entry(*c_id).and_modify(|e| e.set_active(0));
                                        });
                                    Status::CuC(*new_cuc_nid)
                                } else {
                                    Status::Failed
                                };
                                new_map.insert(*nid, new_status);
                            }
                            // If is not a CuC, do nothing.
                            _ => {}
                        }
                    }

                    State::Transient(new_map)
                }
                // If there are no restrictions, in particular, there are no spares.
                // So we dont have to find a new CuC for spares if there is none.
                // TODO: Can we somehow skip this clone?
                State::Transient(map) => State::Transient(map.to_owned()),
            }
        };
        #[allow(clippy::let_and_return)]
        s
    }

    pub fn _under_state_mut(&self, st: &mut State) {
        let cl = st.clone();
        if self.is_failed(st) {
            *st = State::Absorbing;
        } else if self.can_fail_safe() && self.is_fail_safe(st) {
            *st = State::Locked;
        } else {
            let cuc_nid_on_spares = st.cuc_spares_owned();
            match st {
                State::Transient(mut_map) if self.has_spares() => {
                    // This case only falls here if we have restrictions like PAND, SEQ, or Spares.
                    // let mut new_map = map.clone();
                    let failed_comp_nid = mut_map
                        .iter()
                        .filter_map(
                            |(nid, status)| if status.is_failed() { Some(*nid) } else { None },
                        )
                        .collect_vec();
                    let unusable_nids: Vec<&NodeId> = failed_comp_nid
                        .iter()
                        .chain(&cuc_nid_on_spares)
                        .unique()
                        .collect_vec();
                    mut_map.iter_mut().for_each(|(nid, status)| {
                        if let Status::CuC(cuc_nid) = status
                            && self.nodes[(*cuc_nid)].is_failed(cuc_nid, &cl, self)
                        {
                            if let Some(new) = &self.nodes[*nid].next_cuc(&unusable_nids) {
                                *cuc_nid = *new;
                            } else {
                                *status = Status::Failed;
                            };
                        }
                    });
                }
                // If there are no restrictions, in particular, there are no spares.
                // So we dont have to find a new CuC for spares if there is none.
                _ => {} // State::Transient(map) => State::Transient(map),
                        // State::Absorbing => State::Absorbing,
                        // State::Locked => State::Locked,
            }
        }
    }
}
