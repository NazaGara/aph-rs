use std::{
    fmt::Display,
    ops::{Add, AddAssign, DivAssign, Mul, MulAssign, Sub, SubAssign},
};

use super::{Almost, ContFraction, FromCF, FromRational, PseudoField, ToRational};
use ndarray::ScalarOperand;
use num_traits::{One, Zero};
use rug::{ops::NegAssign, Complete};

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
    // With rationals I can use exact comparison.
    fn almost_one(&self) -> bool {
        self.is_one()
    }
    fn almost_zero(&self) -> bool {
        self.is_zero()
    }
    fn cmp_eq(&self, other: &Self) -> bool {
        self.eq(other)
    }
}

impl FromCF for Rational {
    fn from_cont_fraction(cf: &mut ContFraction) -> Self {
        let mut numer = Self::one();
        let mut pre_numer = Self::zero();
        let mut denom = Self::zero();
        let mut pre_denom = Self::one();
        for e in cf.values.iter() {
            let a_i = Self::from_rational(&format!("{:?}", e), "1");
            let new_numer = numer.clone() * a_i.clone() + pre_numer;
            let new_denom = denom.clone() * a_i.clone() + pre_denom;
            pre_denom = denom;
            pre_numer = numer;
            denom = new_denom;
            numer = new_numer;
        }
        Self::from_rational(&format!("{:?}", numer), &format!("{:?}", denom))
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
        let value = if self.is_long() {
            let cf = ContFraction::from_explicit_with_precision(&self.numer(), &self.denom(), 15);
            let mut numer = 1_i128;
            let mut pre_numer = 0_i128;
            let mut denom = 0_i128;
            let mut pre_denom = 1_i128;
            for e in cf.values.iter() {
                let a_i = e.to_i128_wrapping();
                // We still can have some problems here with overflow in the multiplication.
                // TODO:  Search for un option multiplication, otherwise use a try-catch like struct.
                let new_numer = numer * a_i + pre_numer;
                let new_denom = denom * a_i + pre_denom;
                pre_denom = denom;
                pre_numer = numer;

                denom = new_denom;
                numer = new_numer;
            }

            numer as f64 / denom as f64
        } else {
            let numer: f64 = self
                .numer()
                .replace("\"", "")
                .parse::<f64>()
                .unwrap_or_else(|_| panic!("Value: '{:?}' could not be parsed.", self.numer()));
            let denom: f64 = self
                .denom()
                .replace("\"", "")
                .parse::<f64>()
                .unwrap_or_else(|_| panic!("Value: '{:?}' could not be parsed.", self.denom()));
            numer / denom
        };

        if value.is_nan() {
            panic!(
                "Value is NaN. Numer: {:?}. Denom: {:?}",
                self.numer(),
                self.denom()
            )
        }
        format!("{:?}", value)
    }
}

impl ScalarOperand for Rational {}
