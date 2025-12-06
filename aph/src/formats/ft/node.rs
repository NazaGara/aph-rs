use std::{collections::HashMap, fmt::Display};

// use itertools::Itertools;

use crate::{
    aph::{BidiagonalAph, new_erl, new_exp, new_hypexp},
    formats::ft::{
        State,
        nodetree::NodeTree,
        parser::{GateType, NodeId, NodeKind, TriggerID},
        state::Token,
    },
    linalg::fields::PseudoField,
};

#[derive(Debug, Clone)]
pub struct AphNode<F: PseudoField>(BidiagonalAph<F>);

impl<F: PseudoField> AphNode<F> {
    #[allow(unused)]
    pub fn from_node_kind(event: &NodeKind) -> Option<Self> {
        match event {
            NodeKind::Gate(_, _) => None,
            // NodeKind::BE(be) => {
            NodeKind::TriggeredBE(be, _) | NodeKind::BE(be) => {
                let ph = if be.is_erlang() {
                    AphNode(new_erl(be.phases.unwrap(), be.lambdas[0]))
                } else if be.is_hypoexp() {
                    AphNode(new_hypexp(&be.lambdas))
                } else {
                    AphNode(new_exp(be.lambdas[0]))
                };
                Some(ph)
            }
        }
    }

    pub fn repr(self) -> BidiagonalAph<F> {
        self.0
    }
}

/// Status for entries in the state.
#[derive(Debug, Clone, Copy, Eq, PartialEq, PartialOrd, Hash)]
pub enum Status {
    // Working(u16),
    Active(u16),
    Passive(u16),
    FailSafe,
    Failed,
    CuC(NodeId),
}

impl Default for Status {
    fn default() -> Self {
        Status::Active(0)
    }
}

#[derive(Debug, Clone, Eq, Ord, PartialEq, PartialOrd, Hash)]
pub struct StatusCode(pub Token);

impl Display for StatusCode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[allow(unused)]
impl Status {
    pub fn new_spare(nid: NodeId) -> Self {
        Status::CuC(nid)
    }

    pub fn code(&self) -> StatusCode {
        let c = match &self {
            Status::FailSafe => Token::L,
            Status::Failed => Token::F,
            Status::Active(ph) | Status::Passive(ph) => Token::W(*ph),
            Status::CuC(nid) => Token::W(nid.raw() as u16), // supicious.
        };
        StatusCode(c)
    }

    pub fn next_phase(&self) -> Self {
        match self {
            Status::FailSafe => Status::FailSafe,
            Status::Failed => Status::Failed,
            // Status::Working(v) => Status::Working(*v + 1),
            Status::Active(ph) => Status::Active(*ph + 1),
            Status::Passive(ph) => Status::Passive(*ph), // in theory should not be called.
            Status::CuC(k) => Status::CuC(*k),
        }
    }

    // pub fn to_working(mut self, phase: u16) {
    //     self = Self::Working(phase);
    // }
    pub fn set_active(&mut self, phase: u16) {
        *self = match self {
            Status::Passive(_) => Status::Active(phase),
            _ => *self,
        };
    }

    pub fn set_passive(&mut self, phase: u16) {
        *self = match self {
            Status::Active(_) => Status::Passive(phase),
            _ => *self,
        };
    }

    pub fn to_fail(mut self) {
        self = Self::Failed;
    }

    pub fn cuc_inner(&self) -> Option<&NodeId> {
        match self {
            Status::CuC(id) => Some(id),
            _ => None,
        }
    }

    pub fn working_inner(&self) -> Option<u16> {
        match self {
            Status::Passive(ph) | Status::Active(ph) => Some(*ph),
            _ => None,
        }
    }

    pub fn to_fail_safe(mut self) {
        self = Self::FailSafe;
    }

    pub fn is_failed(&self) -> bool {
        matches!(self, Status::Failed)
    }

    pub fn is_failsafe(&self) -> bool {
        matches!(self, Status::FailSafe)
    }

    pub fn is_active(&self) -> bool {
        matches!(self, Status::Active(_))
    }

    pub fn is_passive(&self) -> bool {
        matches!(self, Status::Passive(_))
    }
}

/// Represents a Node in the Tree.
/// It can either be a Gate, which failure depends on its children, or a Component.
/// A Component can be either a Basic Event, or an already processed submodule of the tree,
/// on which the failure distribution is given by an APH distribution.
#[derive(Debug, Clone)]
pub(crate) enum FTNode<F: PseudoField> {
    // ComponentNode(BidiagonalAph<F>),
    Component(BidiagonalAph<F>, F),
    Gate(GateType, Vec<NodeId>),
}

impl<F: PseudoField> FTNode<F> {
    pub fn children(&self) -> Option<Vec<NodeId>> {
        match self {
            // FTNode::Component(_) => None,
            FTNode::Component(_, _) => None,
            FTNode::Gate(gate_type, children) => {
                let mut children = children.clone();
                match gate_type {
                    GateType::Csp(TriggerID::ID(trigger))
                    | GateType::Hsp(TriggerID::ID(trigger))
                    | GateType::Wsp(TriggerID::ID(trigger))
                    | GateType::Pand(TriggerID::ID(trigger)) => children.push(*trigger),
                    _ => {}
                }
                Some(children)
            }
        }
    }

    pub fn get_component(&self) -> Option<&BidiagonalAph<F>> {
        match self {
            FTNode::Gate(_, _) => None,
            // FTNode::Component(aph) => Some(aph),
            FTNode::Component(aph, _) => Some(aph),
        }
    }

    /// If there is one, it returns the next available child for the spare gate: [NodeId].
    /// If there is no child available, or is not a spare gate, returns [None].
    pub fn next_cuc(&self, failed_states: &[&NodeId]) -> Option<NodeId> {
        match &self {
            FTNode::Gate(GateType::Hsp(TriggerID::ID(_failed_nid)), children)
            | FTNode::Gate(GateType::Wsp(TriggerID::ID(_failed_nid)), children)
            | FTNode::Gate(GateType::Csp(TriggerID::ID(_failed_nid)), children) => children
                .iter()
                .find(|y| !failed_states.contains(y))
                .copied(),
            _ => None,
        }
    }

    pub fn component_size(&self) -> usize {
        match &self {
            // FTNode::Component(aph) => aph.size(),
            FTNode::Component(aph, _) => aph.size(),
            FTNode::Gate(_, _) => 0,
        }
    }

    pub fn from_gate(
        gate_type: &GateType,
        children: &[String],
        lookup_table: &HashMap<String, NodeId>,
    ) -> Self {
        let children: Vec<NodeId> = children
            .iter()
            .map(|c_name| lookup_table.get(c_name).unwrap().to_owned())
            .collect();
        FTNode::Gate(gate_type.clone(), children)
    }

    pub fn from_node_kind(nk: &NodeKind, lookup_table: &HashMap<String, NodeId>) -> Self {
        match nk {
            // NodeKind::BE(_) => FTNode::Component(AphNode::<F>::from_node_kind(nk).unwrap().repr()),
            // NodeKind::BE(be) => FTNode::Component(
            //     AphNode::<F>::from_node_kind(nk).unwrap().repr(),
            //     F::from(be.dorm),
            // ),
            NodeKind::TriggeredBE(be, _) | NodeKind::BE(be) => FTNode::Component(
                AphNode::<F>::from_node_kind(nk).unwrap().repr(),
                F::from(be.dorm),
            ),
            NodeKind::Gate(gate_type, children) if matches!(gate_type, GateType::Pand(_)) => {
                if let GateType::Pand(TriggerID::Name(name)) = gate_type {
                    assert!(
                        !children.contains(name),
                        "The event {name} can't be locked, it is the trigger of the PAND gate."
                    );
                    FTNode::Gate(
                        GateType::Pand(TriggerID::ID(lookup_table.get(name).unwrap().to_owned())),
                        children
                            .iter()
                            .map(|c_name| lookup_table.get(c_name).unwrap().to_owned())
                            .collect(),
                    )
                } else {
                    panic!("This should not happen...")
                }
            }
            NodeKind::Gate(GateType::Csp(TriggerID::Name(name)), children) => {
                assert!(
                    !children.contains(name),
                    "The event {name} can't be a spare component, is the currently in use child."
                );
                FTNode::Gate(
                    GateType::Csp(TriggerID::ID(lookup_table.get(name).unwrap().to_owned())),
                    children
                        .iter()
                        .map(|c_name| lookup_table.get(c_name).unwrap().to_owned())
                        .collect(),
                )
            }
            NodeKind::Gate(GateType::Wsp(TriggerID::Name(name)), children) => {
                assert!(
                    !children.contains(name),
                    "The event {name} can't be a spare component, is the currently in use child."
                );
                FTNode::Gate(
                    GateType::Wsp(TriggerID::ID(lookup_table.get(name).unwrap().to_owned())),
                    children
                        .iter()
                        .map(|c_name| lookup_table.get(c_name).unwrap().to_owned())
                        .collect(),
                )
            }
            NodeKind::Gate(GateType::Hsp(TriggerID::Name(name)), children) => {
                assert!(
                    !children.contains(name),
                    "The event {name} can't be a spare component, is the currently in use child."
                );
                FTNode::Gate(
                    GateType::Hsp(TriggerID::ID(lookup_table.get(name).unwrap().to_owned())),
                    children
                        .iter()
                        .map(|c_name| lookup_table.get(c_name).unwrap().to_owned())
                        .collect(),
                )
            }
            NodeKind::Gate(GateType::FDep(TriggerID::Name(name)), children) => {
                assert!(
                    !children.contains(name),
                    "The event {name} can not be triggered by itself."
                );
                FTNode::Gate(
                    GateType::FDep(TriggerID::ID(lookup_table.get(name).unwrap().to_owned())),
                    children
                        .iter()
                        .map(|c_name| lookup_table.get(c_name).unwrap().to_owned())
                        .collect(),
                )
            }
            NodeKind::Gate(gate_type, children) => {
                Self::from_gate(gate_type, children, lookup_table)
            }
        }
    }

    pub fn is_fail_safe(&self, state: &State, ft: &NodeTree<F>) -> bool {
        match &state {
            State::Absorbing => false,
            State::Locked => true,
            State::Transient(_) => {
                if let FTNode::Gate(gate_type, children) = self {
                    matches!(&gate_type, GateType::Pand(TriggerID::ID(t_id)) if {
                        if children
                            .iter()
                            .all(|nid| ft.nodes[*nid].is_failed(nid, state, ft))
                        {
                            !ft.nodes[*t_id].is_failed(t_id, state, ft)
                        } else {
                            false
                        }
                    })
                } else {
                    false
                }
            }
        }
    }

    // Recursively iterates over the FT to mark all the failed nodes.
    pub fn is_failed(&self, self_id: &NodeId, state: &State, ft: &NodeTree<F>) -> bool {
        match &state {
            State::Absorbing => true,
            State::Locked => false,
            State::Transient(transient_state_map) => {
                if let FTNode::Gate(gate_type, children) = self {
                    match &gate_type {
                        GateType::Hsp(TriggerID::Name(_)) | GateType::Csp(TriggerID::Name(_)) => {
                            panic!("Unprocessed gate {gate_type:?} with Id: {self_id:?}.");
                        }
                        GateType::Or => children
                            .iter()
                            .any(|nid| ft.nodes[*nid].is_failed(nid, state, ft)),
                        GateType::And => children
                            .iter()
                            .all(|nid| ft.nodes[*nid].is_failed(nid, state, ft)),
                        GateType::KofN(k) => {
                            children
                                .iter()
                                .filter(|nid| ft.nodes[**nid].is_failed(nid, state, ft))
                                .count() as u32
                                >= *k
                        }
                        GateType::FDep(TriggerID::ID(_t_id)) => {
                            eprintln!(
                                "WARN: FDEP gates do not fail. They trigger the failure of the components if the trigger node is failed."
                            );
                            false
                        }
                        GateType::Pand(TriggerID::ID(trigger_id)) => {
                            // state.state_of_id(trigger_id).is_failed()
                            ft.nodes[*trigger_id].is_failed(trigger_id, state, ft)
                                && children
                                    .iter()
                                    .all(|nid| ft.nodes[*nid].is_failed(nid, state, ft))
                        }
                        // A spare gate fails if the CuC is failed and the childrens are taken or failed.
                        GateType::Wsp(TriggerID::ID(_))
                        | GateType::Hsp(TriggerID::ID(_))
                        | GateType::Csp(TriggerID::ID(_)) => {
                            // We can discard the triggerID from the gatetype, because the in use trigger id is encoded in the state.
                            // Check on the state which one is being used, not in the gatetype.
                            if let Some(in_use_trigger_id) = state.cuc_in_use_at(self_id) {
                                let in_use_nodes = state.cuc_spares();

                                ft.nodes[*in_use_trigger_id].is_failed(in_use_trigger_id, state, ft)
                                    && children.iter().all(|nid| {
                                        in_use_nodes.contains(&nid)
                                            || ft.nodes[*nid].is_failed(nid, state, ft)
                                    })
                            } else {
                                // If there is no node in use, the gate is failed.
                                true
                            }
                        }
                        GateType::Seq => {
                            // A Seq gate fails if all children have failed in order. Restrictions make sure that the order is preserved, so the gates becomes an AND.
                            children
                                .iter()
                                .all(|nid| ft.nodes[*nid].is_failed(nid, state, ft))
                        }
                        _ => {
                            todo!("Gates of type {gate_type:?} not yet supported.")
                        }
                    }
                } else {
                    // If the required component is not in the map (i.e. because the dft is malformed) we assume
                    // the component is failed.
                    transient_state_map
                        .get(self_id)
                        .unwrap_or(&Status::Failed)
                        .is_failed()
                }
            }
        }
    }
}

impl<F: PseudoField> From<BidiagonalAph<F>> for FTNode<F> {
    fn from(value: BidiagonalAph<F>) -> Self {
        FTNode::Component(value, F::one())
    }
}
