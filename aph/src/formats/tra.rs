//! The `.tra` file format used by the *Markov Reward Model Checker*.
//!
//! The `.tra` file format is used by the *Markov Reward Model Checker* [^1] for the
//! specification of the rate matrix of continuous time Markov chains.
//!
//! [^1]: [J.-P. Katoen, M. Khattri, and I. S. Zapreevt.
//! "A Markov reward model checker."
//! Second International Conference on the Quantitative Evaluation of Systems (QEST'05).
//! IEEE, 2005.](https://doi.org/10.1109/QEST.2005.2)

use crate::{linalg::fields::PseudoField, new_triangular, Aph, Triangular};

use super::{Cursor, ParseError};

pub fn parse<F: PseudoField>(input: &str) -> Result<Aph<F, Triangular<F>>, ParseError> {
    let mut cursor = Cursor::new(input);
    cursor.consume_tag("STATES")?;
    let states = cursor.consume_usize()?;
    cursor.consume_tag("INITIALS")?;
    let initials = cursor.consume_usize()?;
    cursor.consume_tag("TRANSITIONS")?;
    let transitions = cursor.consume_usize()?;

    let mut aph = new_triangular(states - 1);

    for _ in 0..initials {
        let idx = cursor.consume_usize()?;
        let prob = cursor.consume_rational()?;
        aph.initial_mut()[idx - 1] = prob;
    }
    for _ in 0..transitions {
        let row = cursor.consume_usize()?;
        let column = cursor.consume_usize()?;
        let rate = cursor.consume_rational::<F>()?;
        aph.repr_mut().set(row - 1, column - 1, rate);
    }

    Ok(aph)
}
