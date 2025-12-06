use std::{
    cmp::Ordering,
    collections::BTreeMap,
    fmt::{Debug, Display},
};

use itertools::Itertools;

use crate::{
    formats::ft::{
        node::{FTNode, Status},
        nodetree::NodeTree,
        parser::NodeId,
    },
    linalg::fields::PseudoField,
};

/// Tokens encode the status of a component in a state.
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub enum Token {
    /// Working
    W(u16),
    /// Failed
    F,
    /// Locked (FailSafe)
    L,
    /// Absorbed
    A,
}

impl Ord for Token {
    fn cmp(&self, other: &Self) -> Ordering {
        use Token::*;
        match (self, other) {
            (A, A) => Ordering::Equal,
            (A, _) => Ordering::Greater,
            (_, A) => Ordering::Less,
            (L, W(_)) => Ordering::Greater,
            (W(_), L) => Ordering::Less,
            (L, L) => Ordering::Equal,
            (F, F) => Ordering::Equal,
            (F, _) => Ordering::Greater,
            (_, F) => Ordering::Less,
            (W(a), W(b)) => a.cmp(b),
        }
    }
}

impl PartialOrd for Token {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Display for Token {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self {
            Token::F => write!(f, "X"),
            Token::L => write!(f, "FS"),
            Token::A => write!(f, "X"),
            Token::W(v) => write!(f, "{v}"),
        }
    }
}

// --- State Encoding.

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct StateCode(Vec<Token>);

impl Display for StateCode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0.iter().join(","))
    }
}

impl StateCode {
    // pub fn raw(&self) -> &str {
    //     &self.0
    // }

    pub fn is_abs(&self) -> bool {
        self.0.len() == 1 && *self.0.first().unwrap() == Token::A
    }

    pub fn is_fs(&self) -> bool {
        self.0.len() == 1 && *self.0.first().unwrap() == Token::L
    }

    pub fn tokens(&self) -> Vec<&Token> {
        if self.is_abs() {
            return vec![&Token::A];
        }
        self.0.iter().collect()
    }
}

impl Ord for StateCode {
    fn cmp(&self, other: &Self) -> Ordering {
        if self.is_abs() && other.is_abs() {
            return Ordering::Equal;
        } else if self.is_abs() {
            return Ordering::Greater;
        } else if other.is_abs() {
            return Ordering::Less;
        }

        let mut a_tokens = self.tokens();
        let mut b_tokens = other.tokens();

        // Compare from right to left
        while let (Some(a), Some(b)) = (a_tokens.pop(), b_tokens.pop()) {
            let ord = a.cmp(b);
            if ord != Ordering::Equal {
                return ord;
            }
        }

        // FIXME: All states have the same size...
        a_tokens.len().cmp(&b_tokens.len())
    }
}

impl PartialOrd for StateCode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

// --- State Representation.
#[derive(Debug, Clone, PartialEq)]
pub enum State {
    Transient(BTreeMap<NodeId, Status>), // BTreeMap outperforms IndexMap.
    Absorbing,
    Locked,
}

impl State {
    #[allow(unused)]
    pub fn is_transient(&self) -> bool {
        matches!(self, State::Transient(_))
    }

    pub fn is_absorbing(&self) -> bool {
        matches!(self, State::Absorbing)
    }

    #[allow(unused)]
    pub fn is_failsafe(&self) -> bool {
        matches!(self, State::Locked)
    }

    pub fn cuc_in_use_at(&self, sp_id: &NodeId) -> Option<&NodeId> {
        match &self {
            State::Locked | State::Absorbing => None,
            State::Transient(map) => {
                assert!(
                    map.contains_key(sp_id),
                    "Spare node with id {sp_id:?} must be in the state vector."
                );
                map.get(sp_id).unwrap().cuc_inner()
            }
        }
    }

    /// Return the [NodeId] that are in use by the Spare gates.
    pub fn cuc_spares(&self) -> Vec<&NodeId> {
        match &self {
            State::Absorbing | State::Locked => vec![],
            State::Transient(map) => map
                .values()
                .filter_map(|a| match a {
                    Status::CuC(cuc_nid) => Some(cuc_nid),
                    _ => None,
                })
                .collect_vec(),
        }
    }

    /// Return the [NodeId] that are in use by the Spare gates.
    pub fn cuc_spares_owned(&self) -> Vec<NodeId> {
        match &self {
            State::Absorbing | State::Locked => vec![],
            State::Transient(map) => map
                .values()
                .filter_map(|a| match a {
                    Status::CuC(cuc_nid) => Some(*cuc_nid),
                    _ => None,
                })
                .collect_vec(),
        }
    }

    /// Compute the initial probability of starting in state [self].
    pub(crate) fn compute_initial<F: PseudoField>(&self, ft: &NodeTree<F>) -> F {
        match &self {
            State::Absorbing | State::Locked => F::zero(),
            Self::Transient(map) => {
                let mut p = F::one();
                map.iter().for_each(|(nid, status)| {
                    let other_ini = match status {
                        Status::Failed | Status::FailSafe => &F::zero(),
                        Status::CuC(_) => &F::one(),
                        Status::Active(phase) | Status::Passive(phase) => {
                            if let FTNode::Component(aph, _) = &ft.nodes[*nid] {
                                &aph.initial()[(*phase).into()]
                            } else {
                                &F::one()
                            }
                        }
                    };
                    p.mul_assign(other_ini)
                });
                p
            }
        }
    }
}

impl Display for State {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.to_code())
    }
}

impl State {
    /// Advance the component indexed by [nid] to the next phase in the [self] state.
    /// The [aph_size] parameter specifies if the next status of the component should be [Status::Failed], or continue working.
    /// Thus, we can use it to activate children from sequence enforcers.
    pub(crate) fn take_transition_from(&self, nid: &NodeId, aph_size: u16) -> Self {
        match &self {
            State::Locked => State::Locked,
            State::Absorbing => State::Absorbing,
            State::Transient(status) => {
                let mut new_status = status.clone(); // Clone the whole state.
                let bind = new_status.get(nid).unwrap().next_phase(); // Modify the status of the component nodes.
                match &bind {
                    // It becomes failed if the phase number is greater than the number of phases of the component.
                    // Status::Working(phase) if *phase >= aph_size => {
                    //     new_status.insert(*nid, Status::Failed);
                    // }
                    Status::Active(phase) if *phase >= aph_size => {
                        new_status.insert(*nid, Status::Failed);
                    }

                    // Status::FailSafe | Status::Failed | Status::Working(_) => {
                    Status::FailSafe | Status::Failed | Status::Active(_) | Status::Passive(_) => {
                        new_status.insert(*nid, bind);
                    }
                    Status::CuC(_cuc_nid) => {
                        // if the cuc_nid matches the parameter, then, the spare gate should search for a new one.
                        // Is later sanitized by the [sanitize] funciton.
                        new_status.insert(*nid, bind);
                    }
                }
                State::Transient(new_status)
            }
        }
    }

    pub(crate) fn to_code(&self) -> StateCode {
        StateCode(match &self {
            // State::Transient(status) => status.values().map(|s| s.code()).join(","),
            // State::Absorbing => "ABS".to_owned(),
            // State::FailSafe => "FS".to_owned(),
            State::Transient(status) => status.values().map(|s| s.code().0).collect_vec(),
            State::Absorbing => vec![Token::A],
            State::Locked => vec![Token::L],
        })
    }
}
