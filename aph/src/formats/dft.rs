use std::collections::HashMap;
use std::fs;
use std::marker::PhantomData;
use std::path::Path;

use itertools::Itertools;

use crate::linalg::fields::PseudoField;
use crate::operations::{test_max_phs, test_min_phs};
use crate::representation::Bidiagonal;
use crate::Aph;

#[derive(Debug, Clone)]
pub enum NodeType {
    And,
    Or,
    BasicEventExp(String, String),
    BasicEventErl(usize, String, String),
}

#[derive(Debug, Clone)]
pub struct Node<F: PseudoField> {
    _name: String,
    node_type: NodeType,
    children: Vec<String>,
    // aph: Option<Aph<F,Bidiagonal<F>>>,
    phantom: PhantomData<F>,
}

impl<F: PseudoField> Node<F> {
    fn new(name: &str, node_type: NodeType, children: Vec<String>) -> Self {
        Self {
            _name: name.to_string(),
            node_type,
            children,
            phantom: PhantomData,
        }
    }
    pub fn to_aph(&self, nodes: &HashMap<String, Node<F>>) -> Aph<F, Bidiagonal<F>> {
        // I need a depth measure of each node. Then, I spawn k-threads, and they do the reductions on each depth level.
        match &self.node_type {
            NodeType::BasicEventExp(numer, denom) => {
                Aph::<F, Bidiagonal<F>>::_new_exp(numer, denom)
            }
            NodeType::BasicEventErl(phases, numer, denom) => {
                Aph::<F, Bidiagonal<F>>::_new_erl(*phases, numer, denom)
            }
            NodeType::And => {
                let ch_nodes = self
                    .children
                    .clone()
                    .iter()
                    .map(|c| nodes.get(c).unwrap())
                    .collect_vec();

                let nodes = ch_nodes.iter().map(|n| n.to_aph(nodes)).collect_vec();

                let mut aph = test_max_phs(nodes.as_slice()).expect("Something went wrong...");
                aph.reduce();

                aph
            }
            NodeType::Or => {
                let ch_nodes = self
                    .children
                    .clone()
                    .iter()
                    .map(|c| nodes.get(c).unwrap())
                    .collect_vec();
                let nodes = ch_nodes.iter().map(|n| n.to_aph(nodes)).collect_vec();

                let mut aph = test_min_phs(nodes.as_slice()).expect("Something went wrong...");
                aph.reduce();
                aph
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
                                NodeType::Or
                            } else {
                                NodeType::And
                            };

                            // Collect child node names.
                            let children = tokens[2..]
                                .iter()
                                .filter(|&s| !s.contains("rate="))
                                .map(|s| s.trim_matches('"').trim_end_matches("\";").to_string())
                                .collect();

                            // Add the node.
                            tree.nodes
                                .insert(name.clone(), Node::new(&name, node_type, children));
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
                                        &name,
                                        NodeType::BasicEventExp(
                                            format!("-{}", prob_str),
                                            String::from("1"),
                                        ),
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
                            todo!();
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
