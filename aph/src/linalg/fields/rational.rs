use std::{
    fmt::Display,
    ops::{Add, AddAssign, DivAssign, Mul, MulAssign, Sub, SubAssign},
};

use super::{FromRational, PseudoField, ToRational};
use ndarray::ScalarOperand;
use rug::{ops::NegAssign, Complete};

/// An arbitrary-precision rational number implementing [`Field`].
///
/// Currently this is based on [`rug::Rational`].
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct Rational(rug::Rational);

impl Rational {
    pub fn numer(&self) -> String {
        self.0.numer().to_string()
    }
    pub fn denom(&self) -> String {
        self.0.denom().to_string()
    }
}

impl Display for Rational {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}/{}", self.numer(), self.denom())
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

impl From<&z3::ast::Real<'_>> for Rational {
    fn from(value: &z3::ast::Real) -> Self {
        let real_str = value
            .to_string()
            .replace("(", "")
            .replace(")", "")
            .replace(".0", "");
        if let Some((_, num_and_denom_str)) = real_str.split_once("/ ") {
            let (num_str, denom_str) = num_and_denom_str
                .split_once(" ")
                .expect("Something went wrong");
            Rational::from_rational(num_str.trim(), denom_str.trim())
        } else {
            Rational(
                rug::Rational::parse(real_str)
                    .map(|incomplete| incomplete.complete())
                    .unwrap(),
            )
        }
    }
}

impl FromRational for Rational {
    fn from_rational(nominator: &str, denominator: &str) -> Self {
        let result = rug::Rational::parse(format!("{nominator}/{denominator}"))
            .map(|incomplete| incomplete.complete());
        if result.is_err() && denominator == "1" {
            Rational(
                nominator
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

    fn inv(&mut self) {
        *self = Self::from_rational(&self.0.denom().to_string(), &self.0.numer().to_string());
    }
    fn to_string(&self) -> String {
        let numer: f64 = self
            .numer()
            .replace("\"", "")
            .parse::<f64>()
            .expect(&format!("Value: '{:?}' could not be parsed.", self.numer()));
        let denom: f64 = self
            .denom()
            .replace("\"", "")
            .parse::<f64>()
            .expect(&format!("Value: '{:?}' could not be parsed.", self.denom()));
        // format!("{}/{}", self.numer(), self.denom())
        format!("{:?}", numer/denom)
    }
}


impl ScalarOperand for Rational{}