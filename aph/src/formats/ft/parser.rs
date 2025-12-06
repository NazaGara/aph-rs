// TODO:
// - Remove basic events with lambda=0 if they are not triggered.
// - Simplify gates when #Children == 1.
use std::collections::HashMap;

use index_vec::IndexVec;
use itertools::Itertools;

use crate::formats::ParseError;

index_vec::define_index_type! {
    pub struct NodeId = u32;
    MAX_INDEX = usize::MAX;
    DISABLE_MAX_INDEX_CHECK = cfg!(not(debug_assertions));

}

/// Helper struct to parse the basic events accordingly
#[derive(Debug, Clone, PartialEq)]
pub struct BasicEvent {
    pub name: String,
    pub lambdas: Vec<f64>,
    pub phases: Option<usize>,
    pub dorm: f64,
    // TODO: Deal with discrete probs?
}

#[allow(unused)]
impl BasicEvent {
    pub(crate) fn is_erlang(&self) -> bool {
        self.phases.is_some() && self.lambdas.len() == 1
    }

    pub(crate) fn is_exponential(&self) -> bool {
        self.phases.is_none() && self.lambdas.len() == 1
    }

    pub(crate) fn is_hypoexp(&self) -> bool {
        self.lambdas.len() > 1
    }
}

/// Parses a BE from the .dft file.
/// Parses Exponential, Erlangs and Hypoexponential distributed events.
fn parse_basic_event(line: &str) -> Result<BasicEvent, ParseError> {
    let trimmed_line = line.trim().replace(";", "").replace("\"", "");
    let mut arguments = trimmed_line.split(" ").collect_vec();

    // Name of BE is always at the first position.
    let name = arguments.remove(0).to_string();

    let mut event = BasicEvent {
        name: name.to_string(),
        lambdas: vec![],
        phases: None,
        dorm: 1.0,
    };

    while !arguments.is_empty() {
        let arg = arguments.remove(0);
        if let Some((key, value_str)) = arg.split_once('=') {
            let parse_err = |e| ParseError(format!("Invalid value for key '{key}': '{e}'"));
            let parse_err_int = |e| ParseError(format!("Invalid value for key '{key}': '{e}'"));

            match key {
                "lambda" => {
                    let mut lambdas = Vec::new();
                    for val_part in value_str.split(',') {
                        let val = if let Some((num_str, den_str)) = val_part.trim().split_once('/')
                        {
                            let num: f64 = num_str.parse().map_err(parse_err)?;
                            let den: f64 = den_str.parse().map_err(parse_err)?;
                            if den == 0.0 {
                                return Err(ParseError(format!(
                                    "Division by zero for key '{key}'",
                                )));
                            } else if num == 0.0 {
                                eprintln!(
                                    "WARN: Value of key '{key}' of basic event {name:?} is 0."
                                )
                            }
                            num / den
                        } else {
                            let num = val_part.trim().parse::<f64>().map_err(parse_err)?;
                            if num == 0.0 {
                                eprintln!(
                                    "WARN: Value of key '{key}' of basic event {name:?} is 0."
                                )
                            }
                            num
                        };
                        lambdas.push(-val);
                    }
                    event.lambdas = lambdas;
                }
                "phases" => {
                    event.phases = Some(value_str.parse::<usize>().map_err(parse_err_int)?);
                }
                "dorm" => {
                    let dorm = value_str.parse::<f64>().map_err(parse_err)?;
                    if !(0.0..=1.0).contains(&dorm) {
                        return Err(ParseError(format!(
                            "Dormancy factor of event `{name}` must be between in range [0, 1]."
                        )));
                    }
                    event.dorm = dorm;
                }
                _ => eprintln!("Unknown key '{key}' for basic event '{name}'"),
            }
        }
    }

    Ok(event)
}

#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub enum GateType {
    And,
    Or,
    Seq,
    KofN(u32),
    Hsp(TriggerID),
    Wsp(TriggerID),
    Csp(TriggerID),
    Pand(TriggerID),
    FDep(TriggerID),
}

#[derive(Debug, Clone, PartialEq)]
pub enum TriggerID {
    ID(NodeId),
    Name(String),
}

// --- Internal representation for parsing
#[derive(Debug)]
pub(crate) struct ParsedTree {
    pub(crate) rootid: NodeId,
    pub(crate) lookup_table: HashMap<String, NodeId>,
    pub(crate) nodes: IndexVec<NodeId, NodeKind>,
}

impl ParsedTree {
    pub fn new() -> Self {
        ParsedTree {
            rootid: NodeId { _raw: 0 },
            lookup_table: HashMap::new(),
            nodes: vec![].into(),
        }
    }

    pub fn _replace_node(&mut self, from: &String, new_name: String, by: NodeKind) {
        // Add to nodes and look up table
        let nid = if let Some(nid) = self.lookup_table.get(&new_name) {
            *nid
        } else {
            self.nodes.push(by)
        };
        self.lookup_table.insert(new_name.clone(), nid);

        // replace ocurrences in other nodes.
        self.nodes.iter_mut().for_each(|node| match node {
            NodeKind::Gate(GateType::Hsp(TriggerID::Name(trigger)), _)
            | NodeKind::Gate(GateType::Wsp(TriggerID::Name(trigger)), _)
            | NodeKind::Gate(GateType::Csp(TriggerID::Name(trigger)), _)
            | NodeKind::Gate(GateType::Pand(TriggerID::Name(trigger)), _)
                if from == trigger =>
            {
                *trigger = new_name.clone();
            }
            NodeKind::Gate(_, children) if children.contains(from) => {
                children.retain(|c| c != from);
                children.push(new_name.clone());
            }
            _ => {}
        })
    }

    pub fn _replace_node_with_existing(&mut self, from: &String, new_name: String) {
        // replace ocurrences in other nodes.
        self.nodes.iter_mut().for_each(|node| match node {
            NodeKind::Gate(GateType::Hsp(TriggerID::Name(trigger)), _)
            | NodeKind::Gate(GateType::Wsp(TriggerID::Name(trigger)), _)
            | NodeKind::Gate(GateType::Csp(TriggerID::Name(trigger)), _)
            | NodeKind::Gate(GateType::Pand(TriggerID::Name(trigger)), _)
                if from == trigger =>
            {
                *trigger = new_name.clone();
            }
            NodeKind::Gate(_, children) if children.contains(from) => {
                children.retain(|c| c != from);
                children.push(new_name.clone());
            }
            _ => {}
        })
    }

    fn _replace_inplace(&mut self, from: String, new: NodeKind) {
        let nid = self.lookup_table[&from];
        self.nodes[nid] = new;
    }

    pub(crate) fn make_triggered(&mut self, from: &String, trigger: &String) {
        if let NodeKind::BE(be) = &self.nodes[self.lookup_table[from]] {
            self._replace_node(
                from,
                format!("{from}_t"),
                NodeKind::TriggeredBE(be.clone(), trigger.to_owned()),
            );
        }
    }

    pub(crate) fn revert_triggered_be(&mut self, triggered_name: String) {
        if let NodeKind::TriggeredBE(be, _) = &self.nodes[self.lookup_table[&triggered_name]] {
            self._replace_node_with_existing(&triggered_name, be.name.clone());
        }
    }

    #[allow(unused)]
    fn get_event_name(&self, nid: &NodeId) -> String {
        self.lookup_table
            .iter()
            .filter_map(|(name, id)| if nid == id { Some(name) } else { None })
            .collect_vec()
            .first()
            .unwrap()
            .to_string()
    }

    #[allow(unused)]
    /// Returns all locations ([NodeId]) where the node with the `name` appears.
    pub(crate) fn locations(&self, name: &String) -> Vec<NodeId> {
        self.nodes
            .iter_enumerated()
            .filter_map(|(i, n)| match &n {
                NodeKind::Gate(GateType::FDep(_), _) => None,
                NodeKind::Gate(_, ch) if ch.contains(name) => Some(i),
                _ => None,
            })
            .collect()
    }
}

impl Default for ParsedTree {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub(crate) enum NodeKind {
    Gate(GateType, Vec<String>),
    BE(BasicEvent),
    TriggeredBE(BasicEvent, String),
}

impl NodeKind {
    pub(crate) fn is_gate(&self) -> bool {
        matches!(self, NodeKind::Gate(_, _))
    }

    pub(crate) fn _children(&self) -> Vec<String> {
        match &self {
            NodeKind::BE(_) => vec![],
            NodeKind::Gate(_, ch) => ch.clone(),
            NodeKind::TriggeredBE(be, trigger) => vec![be.name.clone(), trigger.clone()],
        }
    }

    pub fn _add_child(&mut self, new_child: String) {
        match self {
            NodeKind::TriggeredBE(_, _) | NodeKind::BE(_) => {}
            NodeKind::Gate(_, children) => {
                children.push(new_child);
            }
        }
    }

    pub fn _rem_child(&mut self, new_child: String) {
        match self {
            NodeKind::TriggeredBE(_, _) | NodeKind::BE(_) => {}
            NodeKind::Gate(_, children) => {
                children.retain(|c| &new_child != c);
            }
        }
    }

    fn _rename_childrens(&mut self, from: String, to: String) {
        match self {
            NodeKind::Gate(_, ch) => ch
                .iter_mut()
                .filter(|name| **name == from)
                .for_each(|e| *e = to.clone()),
            NodeKind::TriggeredBE(_, _) | NodeKind::BE(_) => {}
        }
    }
}

fn parse_vot_gate(name: &str, k: u32, n: u32, children: Vec<String>, ft: &mut ParsedTree) {
    if n == k {
        let nid = ft.nodes.push(NodeKind::Gate(GateType::And, children));
        ft.lookup_table.insert(name.to_owned(), nid);
    } else if k == 1 {
        let nid = ft.nodes.push(NodeKind::Gate(GateType::Or, children));
        ft.lookup_table.insert(name.to_owned(), nid);
    } else {
        let nid = ft.nodes.push(NodeKind::Gate(GateType::KofN(k), children));
        ft.lookup_table.insert(name.to_owned(), nid);
    }
}

/// Remove the nodes that are children of another node, that has the type `fdep`.
fn remove_fdep_children(ft: &mut ParsedTree) {
    let fdep_ids = ft
        .nodes
        .iter()
        .enumerate()
        .filter_map(|(i, n)| {
            if matches!(n, NodeKind::Gate(GateType::FDep(_), _)) {
                Some(NodeId::from(i))
            } else {
                None
            }
        })
        .collect_vec();
    ft.nodes.iter_mut().for_each(|n| match n {
        NodeKind::BE(_) => {}
        NodeKind::TriggeredBE(_, _) => {}
        NodeKind::Gate(_, children) => {
            children.retain(|c| {
                if let Some(nid) = ft.lookup_table.get(c)
                    && fdep_ids.contains(nid)
                {
                    eprintln!("WARN: Removing children {c}. FDEP gates have no output.");
                    false
                } else {
                    true
                }
            });
        }
    });
    // FIXME: It could happen that by doing this, a node loses all the children.
    // We could have another option that allows for No Output, beyond the FailSafe, Failed and Working status.
}

#[allow(unused)]
pub(crate) fn parse_tree(input: &str) -> Result<ParsedTree, ParseError> {
    let mut ft = ParsedTree::new();

    let mut top_level = String::new();

    for (i, line) in input.lines().enumerate() {
        let trimmed_line = line.trim();
        if trimmed_line.starts_with("//")
            || trimmed_line.is_empty()
            || trimmed_line.starts_with(';')
        {
            continue;
        }

        let parts: Vec<&str> = trimmed_line.split_whitespace().collect();
        if parts.len() < 2 {
            return Err(ParseError(format!("Not enough arguments in line {i}.")));
        }
        let clean_name = |s: &str| s.trim_matches('"').to_string();
        if parts[0] == "toplevel" {
            top_level = clean_name(parts[1].trim_end_matches(';'));
            continue;
        }

        let name = clean_name(parts[0]);

        if ft.lookup_table.contains_key(&name) {
            return Err(ParseError(format!(
                "Node name {name} is already present in the tree."
            )));
        }
        let gate_or_param = parts[1];

        match gate_or_param {
            "or" | "and" | "seq" => {
                let children: Vec<String> = parts[2..]
                    .iter()
                    .map(|s| clean_name(s.trim_end_matches(';')))
                    .collect();
                let gate_type = if gate_or_param == "or" {
                    GateType::Or
                } else if gate_or_param == "and" {
                    GateType::And
                } else {
                    GateType::Seq
                };

                let nid = ft.nodes.push(NodeKind::Gate(gate_type, children));
                ft.lookup_table.insert(name, nid);
            }
            "fdep" | "pand" | "hsp" | "csp" | "wsp" => {
                let all_children: Vec<String> = parts[2..]
                    .iter()
                    .map(|s| clean_name(s.trim_end_matches(';')))
                    .collect();

                let (trigger, children) = all_children.split_first().unwrap();

                let trigger = TriggerID::Name(trigger.clone());
                let gate_type = if gate_or_param == "fdep" {
                    GateType::FDep(trigger)
                } else if gate_or_param == "pand" {
                    GateType::Pand(trigger)
                } else if gate_or_param == "csp" {
                    // Cold spares (csp): spare components do not fail when not in use. Thus, behaves like the convolution of all children nodes.
                    GateType::Csp(trigger)
                } else if gate_or_param == "wsp" {
                    // Warm spares (wsp): works like a hsp, but the actual failure rate decreases according to the dormancy factor of each spare component.
                    GateType::Wsp(trigger)
                } else {
                    // Hot spares (hsp): spare components fail at the same rate as when they are used. Thus, behaves like a `and` gate.
                    GateType::Hsp(trigger)
                };
                let nid = ft.nodes.push(NodeKind::Gate(gate_type, children.to_vec()));
                ft.lookup_table.insert(name.clone(), nid);
            }

            gate if gate.contains("of") => {
                let kofn: Vec<&str> = gate.split("of").collect();
                let k = kofn[0]
                    .parse()
                    .expect("Failed to parse 'k' in <k>of<N> gate");
                let n = kofn[1]
                    .parse()
                    .expect("Failed to parse 'N' in <k>of<N> gate");
                let children = parts[2..]
                    .iter()
                    .map(|s| clean_name(s.trim_end_matches(';')))
                    .collect();

                parse_vot_gate(&name, k, n, children, &mut ft);
            }

            _ if trimmed_line.contains("lambda=") || trimmed_line.contains("prob=") => {
                let be = parse_basic_event(trimmed_line).unwrap();
                let nid = ft.nodes.push(NodeKind::BE(be));
                ft.lookup_table.insert(name, nid);
            }
            _ => eprintln!("Warning: Unrecognized line format: {}", line),
        }
    }

    // Remove the childrens that are FDEP.
    remove_fdep_children(&mut ft);

    ft.rootid = *ft.lookup_table.get(&top_level).unwrap();

    Ok(ft)
}

pub(crate) fn parse_tree_w_fdep(
    input: &str,
) -> Result<(ParsedTree, Vec<(String, String)>), ParseError> {
    let mut ft = ParsedTree::new();
    let mut triggered = vec![];

    let mut top_level = String::new();

    for (i, line) in input.lines().enumerate() {
        let trimmed_line = line.trim();
        if trimmed_line.starts_with("//")
            || trimmed_line.is_empty()
            || trimmed_line.starts_with(';')
        {
            continue;
        }

        let parts: Vec<&str> = trimmed_line.split_whitespace().collect();
        if parts.len() < 2 {
            return Err(ParseError(format!("Not enough arguments in line {i}.")));
        }
        let clean_name = |s: &str| s.trim_matches('"').to_string();
        if parts[0] == "toplevel" {
            top_level = clean_name(parts[1].trim_end_matches(';'));
            continue;
        }

        let name = clean_name(parts[0]);

        if ft.lookup_table.contains_key(&name) {
            return Err(ParseError(format!(
                "Node name {name} is already present in the tree."
            )));
        }
        let gate_or_param = parts[1];

        match gate_or_param {
            "or" | "and" | "seq" => {
                let children: Vec<String> = parts[2..]
                    .iter()
                    .map(|s| clean_name(s.trim_end_matches(';')))
                    .collect();
                let gate_type = if gate_or_param == "or" {
                    GateType::Or
                } else if gate_or_param == "and" {
                    GateType::And
                } else {
                    GateType::Seq
                };

                let nid = ft.nodes.push(NodeKind::Gate(gate_type, children));
                ft.lookup_table.insert(name, nid);
            }
            "fdep" | "pand" | "hsp" | "csp" | "wsp" => {
                let all_children: Vec<String> = parts[2..]
                    .iter()
                    .map(|s| clean_name(s.trim_end_matches(';')))
                    .collect();

                let (trigger_name, children) = all_children.split_first().unwrap();

                let trigger = TriggerID::Name(trigger_name.clone());
                let gate_type = if gate_or_param == "fdep" {
                    triggered.extend_from_slice(
                        &children
                            .iter()
                            .map(|c| (c.to_string(), trigger_name.clone()))
                            .collect::<Vec<(String, String)>>(),
                    );
                    GateType::FDep(trigger)
                } else if gate_or_param == "pand" {
                    GateType::Pand(trigger)
                } else if gate_or_param == "csp" {
                    // Cold spares (csp): spare components do not fail when not in use. Thus, behaves like the convolution of all children nodes.
                    GateType::Csp(trigger)
                } else if gate_or_param == "wsp" {
                    // Warm spares (wsp): works like a hsp, but the actual failure rate decreases according to the dormancy factor of each spare component.
                    GateType::Wsp(trigger)
                } else {
                    // Hot spares (hsp): spare components fail at the same rate as when they are used. Thus, behaves like a `and` gate.
                    GateType::Hsp(trigger)
                };
                let nid = ft.nodes.push(NodeKind::Gate(gate_type, children.to_vec()));
                ft.lookup_table.insert(name.clone(), nid);
            }

            gate if gate.contains("of") => {
                let kofn: Vec<&str> = gate.split("of").collect();
                let k = kofn[0]
                    .parse()
                    .expect("Failed to parse 'k' in <k>of<N> gate");
                let n = kofn[1]
                    .parse()
                    .expect("Failed to parse 'N' in <k>of<N> gate");
                let children = parts[2..]
                    .iter()
                    .map(|s| clean_name(s.trim_end_matches(';')))
                    .collect();

                parse_vot_gate(&name, k, n, children, &mut ft);
            }

            _ if trimmed_line.contains("lambda=") || trimmed_line.contains("prob=") => {
                let be = parse_basic_event(trimmed_line).unwrap();
                let nid = ft.nodes.push(NodeKind::BE(be));
                ft.lookup_table.insert(name, nid);
            }
            _ => eprintln!("Warning: Unrecognized line format: {}", line),
        }
    }

    // Remove the childrens that are FDEP.
    remove_fdep_children(&mut ft);

    ft.rootid = *ft.lookup_table.get(&top_level).unwrap();

    Ok((ft, triggered))
}
