use index_vec::{IndexSlice, IndexVec};
use std::{collections::HashMap, fmt::Debug};

use crate::formats::ft::parser::{GateType, NodeId, NodeKind, ParsedTree, TriggerID};

/// Helper struct for the modularisation algorithm.
#[derive(Clone, Debug)]
pub struct DFSNode {
    visited: bool,
    t_fst_visit: u16,
    t_snd_visit: u16,
    t_lst_visit: u16,
    t_max_desc: u16,
    t_min_desc: u16,
}

impl Default for DFSNode {
    fn default() -> Self {
        Self {
            visited: false,
            t_fst_visit: 0,
            t_snd_visit: 0,
            t_lst_visit: 0,
            t_max_desc: u16::MIN,
            t_min_desc: u16::MAX,
        }
    }
}

impl DFSNode {
    fn is_visited(&self) -> bool {
        self.visited
    }

    fn is_module(&self) -> bool {
        self.t_min_desc > self.t_fst_visit && self.t_max_desc < self.t_snd_visit
    }

    fn snd_dfs_visited(&self) -> bool {
        self.t_min_desc == u16::MAX
    }

    fn update_t_desc(&mut self, min_c: u16, max_c: u16) {
        // if self.snd_dfs_visited() {
        //     self.t_min_desc = min_c;
        //     self.t_max_desc = max_c;
        // } else {
        self.t_min_desc = std::cmp::min(self.t_min_desc, min_c);
        self.t_max_desc = std::cmp::max(self.t_max_desc, max_c);
        // };
    }
}

fn fst_dfs(
    nodes: &mut IndexSlice<NodeId, [DFSNode]>,
    children: &IndexSlice<NodeId, [Vec<NodeId>]>,
    curr_idx: NodeId,
    time: &mut u16,
) {
    // Increase Time
    *time += 1;
    // Take current node using the idx
    let curr_node = &mut nodes[curr_idx];
    let curr_children = &children[curr_idx];

    curr_node.t_lst_visit = *time;
    if curr_children.is_empty() {
        if !curr_node.is_visited() {
            curr_node.visited = true;
            curr_node.t_fst_visit = *time;
            curr_node.t_snd_visit = *time;
        }
    } else if !curr_node.is_visited() {
        curr_node.visited = true;
        // On first visit to gate, send for DFS of children.
        curr_node.t_fst_visit = *time;
        // Take immediate children of node and continue the DFS.
        for &child_nid in curr_children {
            fst_dfs(nodes, children, child_nid, time);
        }
        // Come back to the current node, use one more visit, and mark second visit.
        fst_dfs(nodes, children, curr_idx, time);
        // Then update the max and min times.
    } else if curr_node.t_snd_visit == 0 {
        curr_node.t_snd_visit = *time;
    }
}

type DecendantsTimes = (u16, u16);

fn snd_dfs(
    nodes: &mut IndexSlice<NodeId, [DFSNode]>,
    children: &IndexSlice<NodeId, [Vec<NodeId>]>,
    curr_idx: NodeId,
) -> DecendantsTimes {
    // Take current node using the idx
    let curr_node = &nodes[curr_idx];
    // If I already know the pair, return it.
    if !curr_node.snd_dfs_visited() {
        return (curr_node.t_fst_visit, curr_node.t_lst_visit);
    }
    // Save the first and last time of the nodes
    let t_fst_node = curr_node.t_fst_visit;
    let t_lst_node = curr_node.t_lst_visit;

    // Take one-step children
    for &child_nid in &children[curr_idx] {
        if child_nid == curr_idx {
            continue;
        } else {
            // Get the pair, if is a BE, just the times
            let (d_min, d_max) = snd_dfs(nodes, children, child_nid);
            // Update the data on the modularizer
            nodes[curr_idx].update_t_desc(d_min, d_max);
        }
    }

    let DFSNode {
        t_min_desc: curr_min,
        t_max_desc: curr_max,
        ..
    } = nodes[curr_idx];

    // Compare current times with the descendency times
    (
        std::cmp::min(curr_min, t_fst_node),
        std::cmp::max(curr_max, t_lst_node),
    )
}

/// Modularization algorithm based on: Dutuit, Y., & Rauzy, A. (1996). A linear-time algorithm to find modules of fault trees. IEEE transactions on Reliability, 45(3), 422-425.
/// Requires 2 dfs runs. The first one to take the time of visit of each node, and the second one to apply the formula for indentifying the modules.
/// Each dfs run is recursively implemented.
pub fn get_modules(ft: &ParsedTree) -> Vec<NodeId> {
    let root = ft.rootid;
    let mut nodes: IndexVec<NodeId, DFSNode> =
        IndexVec::from_vec(vec![DFSNode::default(); ft.nodes.len()]);

    let _lookup = ft
        .lookup_table
        .clone()
        .into_iter()
        .map(|(k, v)| (v, k))
        .collect::<HashMap<NodeId, String>>();

    // TODO: What do we do if a Gate is triggered? Usually is not allowed.

    let children: IndexVec<NodeId, Vec<NodeId>> = ft
        .nodes
        .iter()
        .map(|n| match n {
            NodeKind::Gate(GateType::Pand(TriggerID::Name(name)), children)
            | NodeKind::Gate(GateType::Hsp(TriggerID::Name(name)), children)
            | NodeKind::Gate(GateType::FDep(TriggerID::Name(name)), children)
            | NodeKind::Gate(GateType::Csp(TriggerID::Name(name)), children) => {
                let mut vec = vec![*ft.lookup_table.get(name).unwrap()];
                children
                    .iter()
                    .for_each(|name| vec.push(ft.lookup_table[name]));
                vec
            }
            NodeKind::Gate(_, children) => {
                children.iter().map(|name| ft.lookup_table[name]).collect()
            }
            NodeKind::TriggeredBE(this, trigger_name) => {
                vec![ft.lookup_table[&this.name], ft.lookup_table[trigger_name]]
            }
            NodeKind::BE(_) => vec![],
        })
        .collect();

    let mut time = 0;
    fst_dfs(&mut nodes, &children, root, &mut time);
    snd_dfs(&mut nodes, &children, root);

    let module_ids: Vec<NodeId> = nodes
        .iter_enumerated()
        .filter_map(|(nid, node)| {
            if node.is_module()
                && !children[nid].is_empty()
                && nid != root
                // had to add it because of the fdep and how I decided to treat them.
                && ft.nodes[nid].is_gate()
            {
                Some(nid)
            } else {
                None
            }
        })
        .collect();

    module_ids
}
