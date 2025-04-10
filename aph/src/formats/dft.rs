use std::collections::HashMap;
use std::fs;
use std::path::Path;

use itertools::Itertools;
use log::info;

use crate::linalg::fields::PseudoField;
use crate::operations::{try_max_phs, try_min_phs};
use crate::representation::Bidiagonal;
use crate::Aph;

#[derive(Debug, Clone)]
pub enum NodeType<F: PseudoField> {
    And,
    Or,
    BasicEventExp(F),
    BasicEventErl(usize, F),
}

#[derive(Debug)]
pub struct Node<F: PseudoField> {
    _name: String,
    node_type: NodeType<F>,
    children: Vec<String>,
}


impl<F: PseudoField> Node<F> {
    fn new(_name: String, node_type: NodeType<F>, children: Vec<String>) -> Self {
        Self {
            _name,
            node_type,
            children,
        }
    }

    pub fn to_aph(&self, nodes: &HashMap<String, Node<F>>) -> (Aph<F, Bidiagonal<F>>, usize) {
        let knowns: HashMap<String, Aph<F, Bidiagonal<F>>> = HashMap::new();
        // I need a depth measure of each node. Then, I spawn k-threads, and they do the reductions on each depth level.
        let mut total_reductions = 0;

        if knowns.keys().contains(&self._name) {
            (knowns.get(&self._name).unwrap().clone(), 0)
        } else {
            match &self.node_type {
                NodeType::BasicEventExp(value) => {
                    (Aph::<F, Bidiagonal<F>>::explicit_exp(value.to_owned()), 0)
                }
                NodeType::BasicEventErl(phases, value) => (
                    Aph::<F, Bidiagonal<F>>::explicit_erl(*phases, value.to_owned()),
                    0,
                ),
                NodeType::And => {
                    let ch_nodes = self
                        .children
                        .clone()
                        .iter()
                        .map(|c| nodes.get(c).unwrap())
                        .collect_vec();

                    info!("- APH of {} childs -", ch_nodes.len(),);
                    let nodes = ch_nodes
                        .iter()
                        .map(|n| {
                            let (res, r) = n.to_aph(nodes);
                            total_reductions += r;
                            res
                        })
                        .collect_vec();
                    let mut aph = try_max_phs(nodes.as_slice()).expect("Something went wrong...");
                    total_reductions += aph.reduce(); 
                    (aph, total_reductions)
                }
                NodeType::Or => {
                    let ch_nodes = self
                        .children
                        .clone()
                        .iter()
                        .map(|c| nodes.get(c).unwrap())
                        .collect_vec();

                    info!("- APH of {} childs -", ch_nodes.len(),);
                    let nodes = ch_nodes
                        .iter()
                        .map(|n| {
                            let (res, r) = n.to_aph(nodes);
                            total_reductions += r;
                            res
                        })
                        .collect_vec();
                    let mut aph = try_min_phs(nodes.as_slice()).expect("Something went wrong...");
                    total_reductions += aph.reduce();
                    (aph, total_reductions)
                }
            }
        }
    }
}








#[derive(Debug)]
pub struct FaultTree<F: PseudoField> {
    pub nodes: HashMap<String, Node<F>>,
    pub toplevel: String,
    pub n_gates: usize,
    pub n_be: usize,
}

impl<F: PseudoField> FaultTree<F> {
    fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            toplevel: String::new(),
            n_gates: 0,
            n_be: 0,
        }
    }

    /// Parse a fault tree from a file.
    pub fn from_file(path: &Path) -> Self {
        let contents = fs::read_to_string(path).expect("Error reading the file");

        let mut tree = FaultTree::new();

        for line in contents.lines() {
            let tokens: Vec<&str> = line.split_whitespace().collect();
            if tokens.is_empty() || line.starts_with("//") {
                continue; // Skip empty lines and comments.
            }

            match tokens[0] {
                "toplevel" => {
                    // Set the toplevel node.
                    tree.toplevel = tokens[1]
                        .trim_matches('"')
                        .trim_end_matches("\";")
                        .to_string();
                }
                _ => {
                    let name = tokens[0].trim_matches('"').to_string();
                    let operator_or_leaf = tokens[1];
                    // Determine the node type and extract details.
                    match operator_or_leaf {
                        "or" | "and" => {
                            let node_type = if operator_or_leaf == "or" {
                                NodeType::<F>::Or
                            } else {
                                NodeType::<F>::And
                            };

                            // Collect child node names.
                            let children = tokens[2..]
                                .iter()
                                .filter(|&s| !s.contains("rate="))
                                .map(|s| {
                                    s.trim_matches('"')
                                        .trim_end_matches("\";")
                                        .trim_end_matches(";")
                                        .to_string()
                                })
                                .collect();

                            // Add the node.
                            tree.nodes
                                .insert(name.clone(), Node::new(name, node_type, children));
                        }
                        _ if operator_or_leaf.starts_with("lambda=") => {
                            let prob_str = operator_or_leaf
                                .strip_prefix("lambda=")
                                .unwrap()
                                .strip_suffix(";")
                                .unwrap();
                            let probability = prob_str.parse::<f64>().ok();

                            if let Some(_prob) = probability {
                                tree.nodes.insert(
                                    name.clone(),
                                    Node::new(
                                        name,
                                        NodeType::BasicEventExp(F::from_rational(
                                            &format!("-{}", prob_str),
                                            "1",
                                        )),
                                        vec![],
                                    ),
                                );
                            } else {
                                eprintln!("Warning: Invalid lambda for {}", name);
                            }
                        }
                        _ if operator_or_leaf.starts_with("phases=") => {
                            let phases_str = operator_or_leaf.strip_prefix("phases=").unwrap();
                            let _phases = phases_str.parse::<usize>().ok();
                            todo!("Parse BE from erlang distributions is not supported yet.");
                        }
                        _ if operator_or_leaf.starts_with("prob=") => {
                            eprintln!("Only BE distributed by exponentials or by erlangs distributions. {}", line);
                        }
                        _ => {
                            eprintln!("Warning: Unknown line format: {}", line);
                        }
                    }
                }
            }
        }

        tree.n_be = tree
            .nodes
            .iter()
            .filter(|(_, n)| n.children.is_empty())
            .count();
        tree.n_gates = tree.nodes.len() - tree.n_be;

        tree
    }
}
