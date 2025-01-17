use std::{
    fmt::Display,
    ops::{Add, Mul, Sub},
};

use super::{rational::Rational, FromRational, PseudoField, Round, SparseField, ToRational};
use num_rational::Ratio;

/// A [`PseudoField`] using interval arithmetic to track the rounding imprecision of
/// a [`SparseField`].
#[derive(Clone, PartialEq, Debug)]
pub struct IntervalField<F: SparseField> {
    lower: F,
    upper: F,
}

impl<F: SparseField> IntervalField<F> {
    pub fn lower(&self) -> F {
        self.lower.clone()
    }
    pub fn upper(&self) -> F {
        self.upper.clone()
    }
}

impl<F: SparseField> Display for IntervalField<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{:.8}, {:.8}]", self.lower, self.upper)
    }
}

impl<F: SparseField> PseudoField for IntervalField<F> {
    fn neg_assign(&mut self) {
        self.lower.neg_assign();
        self.upper.neg_assign();
    }

    fn abs_assign(&mut self) {
        self.lower.abs_assign();
        self.upper.abs_assign();
    }

    fn add_assign(&mut self, rhs: &Self) {
        self.lower.add_assign(&rhs.lower, Round::Down); //Round::Down);
        self.upper.add_assign(&rhs.upper, Round::Up); //Round::Nearest);
    }

    fn sub_assign(&mut self, rhs: &Self) {
        self.lower.sub_assign(&rhs.lower, Round::Down); //Round::Down);
        self.upper.sub_assign(&rhs.upper, Round::Up); //Round::Nearest);
    }

    fn mul_assign(&mut self, rhs: &Self) {
        self.lower.mul_assign(&rhs.lower, Round::Down); //Round::Down);
        self.upper.mul_assign(&rhs.upper, Round::Up); //Round::Nearest);
    }

    fn div_assign(&mut self, rhs: &Self) {
        self.lower.div_assign(&rhs.lower, Round::Down); //Round::Down);
        self.upper.div_assign(&rhs.upper, Round::Up); //Round::Nearest);
    }

    fn inv(&mut self) {
        self.lower.inv();
        self.upper.inv();
    }
    fn to_string(&self) -> String {
        // format!("[{:?}, {:?}]", self.lower.to_string(), self.upper.to_string())
        let two = F::one() + F::one();
        let mut mid = self.upper.clone();
        mid.add_assign(&self.lower, Round::Nearest);
        mid.div_assign(&two, Round::Nearest);
        format!("{}", mid.to_string())
    }
}

impl<F: SparseField> FromRational for IntervalField<F> {
    fn from_rational(nominator: &str, denominator: &str) -> Self {
        let lower = F::from_rational(nominator, denominator);
        let upper = F::from_rational(nominator, denominator);
        IntervalField { lower, upper }
    }
}

impl<F: SparseField> ToRational for IntervalField<F> {
    fn to_rational(&self) -> (String, String) {
        let two = F::one() + F::one();
        let mut mid = self.upper.clone();
        mid.add_assign(&self.lower, Round::Nearest);
        mid.div_assign(&two, Round::Nearest);
        let mid: f64 = mid.into();
        let ratio = Ratio::from_float(mid)
            .expect("Something went wrong when converting the Interval to Rational.");
        (ratio.numer().to_string(), ratio.denom().to_string())
    }
}

impl<F: SparseField> num_traits::One for IntervalField<F> {
    fn one() -> Self {
        Self {
            lower: F::one(),
            upper: F::one(),
        }
    }

    fn is_one(&self) -> bool {
        // self.lower.is_one() || self.upper.is_one()
        let mut l = self.lower.clone();
        l.add_assign(&self.upper, Round::Nearest);
        l.is_one()
    }

    fn set_one(&mut self) {
        *self = Self {
            lower: F::one(),
            upper: F::one(),
        }
    }
}

impl<F: SparseField> num_traits::Zero for IntervalField<F> {
    fn zero() -> Self {
        Self {
            lower: F::zero(),
            upper: F::zero(),
        }
    }

    fn is_zero(&self) -> bool {
        self.lower.is_zero() || self.upper.is_zero()
        // let mut l = self.lower.clone();
        // l.add_assign(&self.upper, Round::Zero);
        // l.is_zero()
    }
    fn set_zero(&mut self) {
        *self = Self {
            lower: F::zero(),
            upper: F::zero(),
        }
    }
}

impl<F: SparseField> Add for IntervalField<F> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Self {
            lower: self.lower + rhs.lower,
            upper: self.upper + rhs.upper,
        }
    }
}

impl<F: SparseField> Mul for IntervalField<F> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        Self {
            lower: self.lower * rhs.lower,
            upper: self.upper * rhs.upper,
        }
    }
}

impl<F: SparseField> PartialOrd for IntervalField<F> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        use std::cmp::Ordering::*;
        match (
            self.upper.partial_cmp(&other.lower)?,
            self.lower.partial_cmp(&other.upper)?,
        ) {
            (Equal, Equal) => Some(Equal),
            (Less, Less) => Some(Less),
            (Greater, Greater) => Some(Greater),
            (Equal, Less) => None,
            (Greater, Less) => None,
            (Greater, Equal) => None,
            _ => unreachable!(),
        }
    }
}

impl<F: SparseField> From<&Rational> for IntervalField<F> {
    fn from(value: &Rational) -> Self {
        Self::from_rational(&value.numer(), &value.denom())
    }
}

impl <F: SparseField> Sub for IntervalField<F> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        let mut neg_rhs = rhs.clone();
        neg_rhs.neg_assign();
        self.add(neg_rhs)
    }
}