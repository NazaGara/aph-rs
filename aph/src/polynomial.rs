use itertools::Itertools;
use std::fmt::Display;

use crate::linalg::fields::PseudoField;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LTerm<F: PseudoField> {
    rate: F,
}

impl<F: PseudoField> Display for LTerm<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "((s+{})/{})", self.rate, self.rate)
        // write!(f, "L({})", self.rate)
    }
}

impl<F: PseudoField> LTerm<F> {
    pub fn new(rate: F) -> Self {
        LTerm { rate }
    }
    pub fn eval(&self, s: &F) -> F {
        let mut numerator = s.clone();
        numerator.add_assign(&self.rate);
        numerator.div_assign(&self.rate);
        numerator
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Polynomial<F: PseudoField> {
    clauses: Vec<(F, Vec<LTerm<F>>)>,
}

impl<F: PseudoField> Display for Polynomial<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            self.clauses
                .iter()
                .map(|(el, l_terms)| format!("{} * {}", el, l_terms.iter().join(" * ")))
                .join(" + ")
                .replace("*  +", "+")
                .trim_end_matches("* ")
        )
    }
}

impl<F: PseudoField> Polynomial<F> {
    pub fn new(clauses: Vec<(F, Vec<LTerm<F>>)>) -> Self {
        Polynomial { clauses }
    }

    pub fn eval(&self, s: &F) -> F {
        let mut result = F::zero();
        for (beta, prod) in self.clauses.iter() {
            let mut cl = beta.to_owned();
            for c in prod {
                cl.mul_assign(&c.eval(s));
            }
            result.add_assign(&cl);
        }
        result
    }

    pub fn eval_until(&self, top: usize, s: &F) -> F {
        let mut result = F::zero();
        for (i, (beta, prod)) in self.clauses.iter().enumerate() {
            if i >= top {
                break;
            }
            let mut cl = beta.to_owned();
            for c in prod {
                cl.mul_assign(&c.eval(s));
            }
            result.add_assign(&cl);
        }
        result
    }
}
