use core::panic;
use std::{
    fmt::Display,
    ops::{Add, AddAssign, DivAssign, Mul, MulAssign, Sub, SubAssign},
};

use super::{Almost, FromRational, PseudoField, ToRational};
use log::warn;
use ndarray::ScalarOperand;
use rug::{
    Complete,
    ops::{CompleteRound, NegAssign},
};

/// An arbitrary-precision rational number implementing [`Field`].
///
/// Currently this is based on [`rug::Rational`].
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub struct Rational(rug::Rational);

impl Rational {
    pub fn numer(&self) -> String {
        self.0.numer().to_string()
    }
    pub fn denom(&self) -> String {
        self.0.denom().to_string()
    }

    // Method to check that if the rational has a large size in his string
    // form, which can lead to Nan or Inf values when parsing to float.
    pub fn is_long(&self) -> bool {
        self.numer().chars().count() > 8 && self.denom().chars().count() > 8
    }

    pub fn size(&self) -> (usize, usize) {
        (self.numer().len(), self.denom().len())
    }
}

impl Almost for Rational {
    fn cmp_eq(&self, other: &Self) -> bool {
        self.eq(other)
    }
}

impl Display for Rational {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.denom() == *"1" {
            write!(f, "{}", self.numer())
        } else {
            write!(f, "{}/{}", self.numer(), self.denom())
        }
    }
}

impl From<Rational> for f64 {
    fn from(rational: Rational) -> Self {
        rational.0.to_f64()
    }
}

impl num_traits::One for Rational {
    fn one() -> Self {
        Self(rug::Rational::from_f64(1.0).unwrap())
    }

    fn is_one(&self) -> bool {
        self.0 == 1.0
    }
    fn set_one(&mut self) {
        *self = Self(rug::Rational::from_f64(1.0).unwrap())
    }
}

impl num_traits::Zero for Rational {
    fn zero() -> Self {
        Self(rug::Rational::from_f64(0.0).unwrap())
    }

    fn is_zero(&self) -> bool {
        self.0 == 0.0
    }
    fn set_zero(&mut self) {
        *self = Self(rug::Rational::from_f64(0.0).unwrap())
    }
}

impl Add for Rational {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0.add(rhs.0))
    }
}

impl Sub for Rational {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        Self(self.0.sub(rhs.0))
    }
}

impl Mul for Rational {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        Self(self.0.mul(rhs.0))
    }
}

impl From<f64> for Rational {
    fn from(value: f64) -> Self {
        Self(rug::Rational::from_f64(value).unwrap())
    }
}

impl FromRational for Rational {
    fn from_rational(numerator: &str, denominator: &str) -> Self {
        let result = rug::Rational::parse(format!("{numerator}/{denominator}"))
            .map(|incomplete| incomplete.complete());
        if result.is_err() && denominator == "1" {
            Rational(
                numerator
                    .parse()
                    .map(|value| rug::Rational::from_f64(value).unwrap())
                    .unwrap(),
            )
        } else {
            Rational(result.unwrap())
        }
    }
}

impl ToRational for Rational {
    fn to_rational(&self) -> (String, String) {
        (self.numer(), self.denom())
    }
}

impl PseudoField for Rational {
    fn neg_assign(&mut self) {
        self.0.neg_assign();
    }

    fn abs_assign(&mut self) {
        self.0.abs_mut();
    }

    fn add_assign(&mut self, rhs: &Self) {
        self.0.add_assign(&rhs.0);
    }

    fn sub_assign(&mut self, rhs: &Self) {
        self.0.sub_assign(&rhs.0);
    }

    fn mul_assign(&mut self, rhs: &Self) {
        self.0.mul_assign(&rhs.0);
    }

    fn div_assign(&mut self, rhs: &Self) {
        self.0.div_assign(&rhs.0);
    }

    fn inv_assign(&mut self) {
        *self = Self::from_rational(&self.0.denom().to_string(), &self.0.numer().to_string());
    }

    fn to_string(&self) -> String {
        let value = {
            let numer = rug::Float::parse(self.numer())
                .unwrap_or_else(|e| panic!("{}. Numerator can't be parsed: {}", e, self.numer()))
                .complete(256);
            let denom = rug::Float::parse(self.denom())
                .unwrap_or_else(|e| panic!("{}\nDenominator can't be parsed: {}", e, self.denom()))
                .complete(256);
            (numer / denom).to_f64()
        };
        if value.is_nan() {
            warn!(
                "Value is NaN. Numer: {:?}. Denom: {:?}",
                self.numer(),
                self.denom()
            )
        }
        format!("{:?}", value)
    }
}

impl ScalarOperand for Rational {}
