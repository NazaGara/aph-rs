//! Floating Point Representation with 64 bits
//! Double-precision binary floating-point is a commonly used format on PCs, due to its wider range over single-precision
//! floating point, in spite of its performance and bandwidth cost. It is commonly known simply as double.
//! The IEEE 754 standard specifies a binary64 as having:
//! - Sign bit: 1 bit
//! - Exponent width: 11 bits
//! - Significand precision: 53 bits (52 explicitly stored)
//!
//!

use std::{
    fmt::{Debug, Display},
    ops::{Add, Div, Mul, Sub},
};

use super::{Almost, FromRational, Round, SparseField, ToRational, round_to_rug_round};
use num_rational::Ratio;
use rug::{
    Float,
    float::OrdFloat,
    ops::{
        AddAssignRound, CompleteRound, DivAssignRound, MulAssignRound, NegAssign, SubAssignRound,
    },
};

/// A 64-bit floating-point type implementing [`SparseField`].
///
/// Currently this is based on [`rug::Float`] but in the future we may implement this
/// type using architecture specific features.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Float64(rug::float::OrdFloat);

impl Debug for Float64 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0.as_float())
    }
}

impl Float64 {
    pub fn is_nan(&self) -> bool {
        self.0.as_float().is_nan()
    }
    pub fn is_infinite(&self) -> bool {
        self.0.as_float().is_infinite()
    }
}

impl From<rug::Float> for Float64 {
    fn from(value: rug::Float) -> Self {
        Self(OrdFloat::from(value))
    }
}

impl Almost for Float64 {
    fn cmp_eq(&self, other: &Self) -> bool {
        let abstol = Self::abstol();
        let epsilon = Self::reltol();

        if self.eq(other) {
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
        .mul(epsilon);

        diff.le(&reltol)
    }
}

impl Display for Float64 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut val = self.0.as_float().clone();
        val.set_prec_round_64(53, rug::float::Round::Nearest);
        write!(f, "{:.25}", val)
    }
}

impl FromRational for Float64 {
    fn from_rational(numerator: &str, denominator: &str) -> Self {
        let num = Float::parse(numerator).unwrap().complete(53);
        let den = Float::parse(denominator).unwrap().complete(53);
        Self(rug::Float::with_val(53, num / den).into())
    }
}

impl ToRational for Float64 {
    fn to_rational(&self) -> (String, String) {
        let ratio = Ratio::from_float(self.0.as_float().to_f64_round(rug::float::Round::Nearest))
            .expect("Something went wrong when converting the Float to Rational.");
        (ratio.numer().to_string(), ratio.denom().to_string())
    }
}

impl num_traits::Zero for Float64 {
    fn set_zero(&mut self) {
        *self = Float64(rug::Float::with_val(53, 0.0).into());
    }

    fn zero() -> Self {
        Self(rug::Float::with_val(53, 0.0).into())
    }

    fn is_zero(&self) -> bool {
        // *self.0.as_float() == 0.0
        let mut self_abs = self.clone();
        self_abs.abs_assign();
        self_abs.le(&Self::epsilon())
    }
}

impl num_traits::One for Float64 {
    fn is_one(&self) -> bool {
        *self.0.as_float() == 1.0
    }

    fn one() -> Self {
        Self(rug::Float::with_val(53, 1.0).into())
    }
    fn set_one(&mut self) {
        *self = Float64(rug::Float::with_val(53, 1.0).into());
    }
}

impl From<f64> for Float64 {
    fn from(value: f64) -> Self {
        Self(rug::Float::with_val_64(53, value).into())
    }
}

impl From<Float64> for f64 {
    fn from(float: Float64) -> Self {
        float.0.as_float().to_f64()
    }
}

impl SparseField for Float64 {
    fn neg_assign(&mut self) {
        self.0.as_float_mut().neg_assign()
    }

    fn abs_assign(&mut self) {
        self.0.as_float_mut().abs_mut();
    }

    fn add_assign(&mut self, rhs: &Self, round: Round) {
        self.0
            .as_float_mut()
            .add_assign_round(rhs.0.as_float(), round_to_rug_round(round));
    }

    fn sub_assign(&mut self, rhs: &Self, round: Round) {
        self.0
            .as_float_mut()
            .sub_assign_round(rhs.0.as_float(), round_to_rug_round(round));
    }

    fn mul_assign(&mut self, rhs: &Self, round: Round) {
        self.0
            .as_float_mut()
            .mul_assign_round(rhs.0.as_float(), round_to_rug_round(round));
    }

    fn div_assign(&mut self, rhs: &Self, round: Round) {
        self.0
            .as_float_mut()
            .div_assign_round(rhs.0.as_float(), round_to_rug_round(round));
    }

    fn inv_assign(&mut self) {
        *self = Self(rug::Float::with_val_64(53, 1.0.div(rug::Float::from(self.0.clone()))).into())
    }

    fn to_string(&self) -> String {
        let mut val = self.0.as_float().clone();
        if val.is_nan() {
            let (numer, denom) = self.to_rational();
            panic!("Value is NaN. Numer: {:?}. Denom: {:?}", numer, denom)
        }
        val.set_prec_round(53, round_to_rug_round(Round::Nearest));
        format!("{:.53}", val)
    }
}

impl Add for Float64 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Self(OrdFloat::from(
            self.0.as_float().add(rhs.0.as_float()).complete(53),
        ))
    }
}

impl Mul for Float64 {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        Self(OrdFloat::from(
            self.0.as_float().mul(rhs.0.as_float()).complete(53),
        ))
    }
}

impl Sub for Float64 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        let mut neg_rhs = rhs.clone();
        neg_rhs.neg_assign();
        Self(OrdFloat::from(
            self.0.as_float().add(neg_rhs.0.as_float()).complete(53),
        ))
    }
}

impl Div for Float64 {
    type Output = Self;
    fn div(self, rhs: Self) -> Self::Output {
        Self(OrdFloat::from(
            self.0.as_float().div(rhs.0.as_float()).complete(53),
        ))
    }
}
