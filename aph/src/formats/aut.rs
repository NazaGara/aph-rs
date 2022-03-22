//! The `.aut` file format.

use std::collections::VecDeque;

use crate::{linalg::fields::PseudoField, new_triangular, Aph, Triangular};

use super::{Cursor, ParseError};

#[derive(Clone, Debug)]
struct Transition<F: PseudoField> {
    pub target: usize,
    pub rate: F,
}

pub fn parse<F: PseudoField>(input: &str) -> Result<Aph<F, Triangular<F>>, ParseError> {
    let mut cursor = Cursor::new(input);
    cursor.consume_tag("des")?;
    cursor.consume_tag("(")?;
    let initial_state = cursor.consume_usize()?;
    cursor.consume_tag(",")?;
    let num_transitions = cursor.consume_usize()?;
    cursor.consume_tag(",")?;
    let num_states = cursor.consume_usize()? + 1;
    cursor.consume_tag(")")?;

    let mut transitions = vec![Vec::new(); num_states];
    let mut in_degree = vec![0; num_states];

    for _ in 0..num_transitions {
        cursor.consume_tag("(")?;
        let source = cursor.consume_usize()?;
        cursor.consume_tag(",")?;
        cursor.consume_tag("\"")?;
        cursor.consume_tag("rate")?;
        let rate = cursor.consume_rational::<F>()?;
        cursor.consume_tag("\"")?;
        cursor.consume_tag(",")?;
        let target = cursor.consume_usize()?;
        cursor.consume_tag(")")?;
        transitions[source].push(Transition { target, rate });
        in_degree[target] += 1;
    }

    let mut ready: VecDeque<_> = vec![initial_state].into();

    let maybe_initial = in_degree
        .iter()
        .enumerate()
        .filter_map(|(idx, degree)| if *degree == 0 { Some(idx) } else { None })
        .count();

    assert!(ready.len() == 1, "More than one initial state? {:?}", ready);

    let mut order = Vec::with_capacity(num_states);

    while let Some(state) = ready.pop_front() {
        order.push(state);
        for transition in &transitions[state] {
            in_degree[transition.target] -= 1;
            if in_degree[transition.target] == 0 {
                ready.push_back(transition.target);
            }
        }
    }

    assert!(
        order.len() == (num_states - (maybe_initial - 1)),
        "Graph may contain cycles."
    );

    let mut aph = new_triangular::<F>(order.len() - 1);

    for (row, state) in order.iter().enumerate() {
        for transition in &transitions[*state] {
            let column = order
                .iter()
                .enumerate()
                .find(|(_, target)| **target == transition.target)
                .unwrap()
                .0;
            aph.repr_mut().set(row, column, transition.rate.clone());
        }
        assert!(
            !aph.diagonal(row).is_zero() || row == order.len() - 1,
            "Diagonal must not be zero. ({}, {}, {})",
            row,
            state,
            transitions[*state].len()
        );
    }

    Ok(aph)
}
