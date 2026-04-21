use std::{
    fmt::{Debug, Display},
    ops::{Add, Div, Mul, Sub},
};

use super::{FromRational, Round, SparseField, ToRational, round_to_rug_round};
use num_rational::Ratio;
use rug::{
    Float,
    float::OrdFloat,
    ops::{
        AddAssignRound, CompleteRound, DivAssignRound, MulAssignRound, NegAssign, SubAssignRound,
    },
};

const SIGNIFICAND: u32 = 256;

/// A 64-bit floating-point type implementing [`SparseField`].
///
/// Currently this is based on [`rug::Float`] but in the future we may implement this
/// type using architecture specific features.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CustomFloat(rug::float::OrdFloat);

impl sprs::MulAcc for CustomFloat {
    fn mul_acc(&mut self, a: &Self, b: &Self) {
        self.add_assign(&(a.clone() * b.clone()), Round::Nearest)
    }
}

impl Debug for CustomFloat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0.as_float())
    }
}

impl CustomFloat {
    pub fn is_nan(&self) -> bool {
        self.0.as_float().is_nan()
    }
    pub fn is_infinite(&self) -> bool {
        self.0.as_float().is_infinite()
    }
}

impl From<rug::Float> for CustomFloat {
    fn from(value: rug::Float) -> Self {
        Self(OrdFloat::from(value))
    }
}

impl Display for CustomFloat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut val = self.0.as_float().clone();
        val.set_prec_round_64(SIGNIFICAND as u64, rug::float::Round::Nearest);
        write!(f, "{:.25}", val)
    }
}

impl FromRational for CustomFloat {
    fn from_rational(numerator: &str, denominator: &str) -> Self {
        let num = Float::parse(numerator).unwrap().complete(SIGNIFICAND);
        let den = Float::parse(denominator).unwrap().complete(SIGNIFICAND);
        Self(rug::Float::with_val(SIGNIFICAND, num / den).into())
    }
}

impl ToRational for CustomFloat {
    fn to_rational(&self) -> (String, String) {
        let ratio = Ratio::from_float(self.0.as_float().to_f64_round(rug::float::Round::Nearest))
            .expect("Something went wrong when converting the Float to Rational.");
        (ratio.numer().to_string(), ratio.denom().to_string())
    }
}

impl num_traits::Zero for CustomFloat {
    fn set_zero(&mut self) {
        *self = CustomFloat(rug::Float::with_val(SIGNIFICAND, 0.0).into());
    }

    fn zero() -> Self {
        Self(rug::Float::with_val(SIGNIFICAND, 0.0).into())
    }

    fn is_zero(&self) -> bool {
        *self.0.as_float() == 0.0
    }
}

impl num_traits::One for CustomFloat {
    fn is_one(&self) -> bool {
        *self.0.as_float() == 1.0
    }

    fn one() -> Self {
        Self(rug::Float::with_val(SIGNIFICAND, 1.0).into())
    }
    fn set_one(&mut self) {
        *self = CustomFloat(rug::Float::with_val(SIGNIFICAND, 1.0).into());
    }
}

impl From<f64> for CustomFloat {
    fn from(value: f64) -> Self {
        Self(rug::Float::with_val_64(SIGNIFICAND as u64, value).into())
    }
}

impl From<CustomFloat> for f64 {
    fn from(float: CustomFloat) -> Self {
        float.0.as_float().to_f64()
    }
}

impl SparseField for CustomFloat {
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
        self.0.as_float_mut().recip_mut();
    }

    fn to_string(&self) -> String {
        let mut val = self.0.as_float().clone();
        if val.is_nan() {
            let (numer, denom) = self.to_rational();
            panic!("Value is NaN. Numer: {:?}. Denom: {:?}", numer, denom)
        }
        val.set_prec_round(SIGNIFICAND, round_to_rug_round(Round::Nearest));
        format!("{:.8}", val)
    }
}

impl Add for CustomFloat {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Self(OrdFloat::from(
            self.0
                .as_float()
                .add(rhs.0.as_float())
                .complete(SIGNIFICAND),
        ))
    }
}

impl Mul for CustomFloat {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        Self(OrdFloat::from(
            self.0
                .as_float()
                .mul(rhs.0.as_float())
                .complete(SIGNIFICAND),
        ))
    }
}

impl Sub for CustomFloat {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        let mut neg_rhs = rhs.clone();
        neg_rhs.neg_assign();
        Self(OrdFloat::from(
            self.0
                .as_float()
                .add(neg_rhs.0.as_float())
                .complete(SIGNIFICAND),
        ))
    }
}

impl Div for CustomFloat {
    type Output = Self;
    fn div(self, rhs: Self) -> Self::Output {
        Self(OrdFloat::from(
            self.0
                .as_float()
                .div(rhs.0.as_float())
                .complete(SIGNIFICAND),
        ))
    }
}
