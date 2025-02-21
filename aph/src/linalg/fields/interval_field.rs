use core::f64;
use std::{
    fmt::Display,
    ops::{Add, Mul, Sub},
};

use super::{
    rational::Rational, Almost, ContFraction, FromCF, FromRational, PseudoField, Round,
    SparseField, ToRational,
};
use num_rational::Ratio;
use num_traits::{One, Zero};

/// A [`PseudoField`] using interval arithmetic to track the rounding imprecision of
/// a [`SparseField`].
#[derive(Clone, PartialEq, Debug, Eq, Hash)]
pub struct IntervalField<F: SparseField> {
    lower: F,
    upper: F,
}

impl<F: SparseField> Almost for IntervalField<F> {
    fn is_almost_zero(&self) -> bool {
        let mut self_abs = self.clone();
        self_abs.abs_assign();

        self_abs.lower.is_almost_zero() && self_abs.upper.is_almost_zero()
    }

    fn cmp_eq(&self, other: &Self) -> bool {
        // self.eq(&other)
        let abstol = Self::from_rational("1", "10000000");
        let epsi = Self::from_rational("1", "10000");

        if self.eq(&other) {
            return true;
        }
        let mut diff = self.clone().sub(other.clone());
        diff.abs_assign();
        if diff.le(&abstol) {
            return true;
        }

        let mut self_abs = self.clone();
        self_abs.abs_assign();
        let mut other_abs = other.clone();
        other_abs.abs_assign();

        let reltol = (if self_abs.ge(&other_abs) {
            self_abs
        } else {
            other_abs
        })
        .mul(epsi.clone());

        diff.le(&reltol)
    }

    fn is_almost_zero_and_correct(&mut self) {
        let mut self_abs = self.clone();
        self_abs.abs_assign();
        let epsilon = Self::from_rational(&format!("{:?}", f64::EPSILON), "1");

        if self_abs.le(&epsilon) {
            self.set_zero();
        };
    }
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
        self.lower.add_assign(&rhs.lower, Round::Down);
        self.upper.add_assign(&rhs.upper, Round::Up);
    }

    fn sub_assign(&mut self, rhs: &Self) {
        self.lower.sub_assign(&rhs.lower, Round::Down);
        self.upper.sub_assign(&rhs.upper, Round::Up);
    }

    // Other approach is to instead of multiplying, just categorize the Interval as M, P, N, Z.
    fn mul_assign(&mut self, rhs: &Self) {
        let comb = vec![
            self.lower.clone() * rhs.lower(),
            self.lower.clone() * rhs.upper(),
            self.upper.clone() * rhs.lower(),
            self.upper.clone() * rhs.upper(),
        ];
        self.lower = comb
            .iter()
            .min()
            .expect(&format!("Something went wrong computing min of {:?}", comb))
            .clone();
        self.upper = comb
            .iter()
            .max()
            .expect(&format!("Something went wrong computing max of {:?}", comb))
            .clone();
        // self.lower.mul_assign(&rhs.lower, Round::Down);
        // self.upper.mul_assign(&rhs.upper, Round::Up);
    }

    // This is wrong
    fn div_assign(&mut self, rhs: &Self) {
        self.lower.div_assign(&rhs.lower, Round::Down);
        self.upper.div_assign(&rhs.upper, Round::Up);
    }

    fn inv_assign(&mut self) {
        self.lower.inv();
        self.upper.inv();
    }

    fn to_string(&self) -> String {
        format!(
            "[{:?}, {:?}]",
            self.lower.to_string(),
            self.upper.to_string(),
        )
        // format!("{}", self.upper.to_string())
        // let two = F::one() + F::one();
        // let mut mid = self.upper.clone();
        // mid.add_assign(&self.lower, Round::Nearest);
        // mid.div_assign(&two, Round::Nearest);
        // format!("{}", mid.to_string())
    }
}

impl<F: SparseField> FromCF for IntervalField<F> {
    fn from_cont_fraction(cf: &mut ContFraction) -> Self {
        if cf.values.len() <= 1 {
            return IntervalField::<F>::from_rational(&format!("{:?}", cf.values[0]), "1");
        } else {
            let val = IntervalField::<F>::from_rational(&format!("{:?}", cf.values.remove(0)), "1");
            let mut rgt = IntervalField::<F>::one();
            rgt.div_assign(&IntervalField::<F>::from_cont_fraction(&mut ContFraction {
                values: cf.values.clone(),
            }));
            return val + rgt;
        }
    }
}

impl<F: SparseField> FromRational for IntervalField<F> {
    fn from_rational(numerator: &str, denominator: &str) -> Self {
        let lower = F::from_rational(numerator, denominator);
        let upper = F::from_rational(numerator, denominator);
        IntervalField { lower, upper }
    }
    fn from_rational_modular(numerator: &str, denominator: &str, _moduler: f64) -> Self {
        Self::from_rational(numerator, denominator)
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
        self.lower.is_one() || self.upper.is_one()
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
    // fn mul(self, rhs: Self) -> Self::Output {
    //     Self {
    //         lower: self.lower * rhs.lower,
    //         upper: self.upper * rhs.upper,
    //     }
    // }
    fn mul(self, rhs: Self) -> Self::Output {
        let comb = vec![
            self.lower.clone() * rhs.lower(),
            self.lower.clone() * rhs.upper(),
            self.upper.clone() * rhs.lower(),
            self.upper.clone() * rhs.upper(),
        ];
        Self {
            lower: comb.iter().min().expect(&format!("Something went wrong computing min of {:?}", comb)).clone(),
            upper: comb.iter().max().expect(&format!("Something went wrong computing max of {:?}", comb)).clone()
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

impl<F: SparseField> Sub for IntervalField<F> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        let mut neg_rhs = rhs.clone();
        neg_rhs.neg_assign();
        self.add(neg_rhs)
    }
}
