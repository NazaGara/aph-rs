use std::{
    fmt::Display,
    ops::{Add, Div, Mul, Sub},
};

use super::{FromRational, Round, SparseField, ToRational};
use num_rational::Ratio;
use rug::{
    float::OrdFloat,
    ops::{
        AddAssignRound, CompleteRound, DivAssignRound, MulAssignRound, NegAssign, SubAssignRound,
    },
};


/// A 64-bit floating-point type implementing [`SparseField`].
///
/// Currently this is based on [`rug::Float`] but in the future we may implement this
/// type using architecture specific features.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct Float64(rug::float::OrdFloat);

impl Display for Float64 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut val = self.0.as_float().clone();
        val.set_prec_round_64(52, rug::float::Round::Nearest);
        write!(f, "{:.8}", val)
    }
}

impl FromRational for Float64 {
    fn from_rational(nominator: &str, denominator: &str) -> Self {
        let nominator = nominator.parse::<f64>().unwrap();
        let denominator = denominator.parse::<f64>().unwrap();
        Self(rug::Float::with_val(52, nominator / denominator).into())
    }
}

impl ToRational for Float64 {
    fn to_rational(&self) -> (String, String) {
        let ratio = Ratio::from_float(self.0.as_float().to_f64_round(rug::float::Round::Nearest))
            .expect("Something went wrong when converting the Float to Rational.");
        (ratio.numer().to_string(), ratio.denom().to_string())
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

impl num_traits::Zero for Float64 {
    fn set_zero(&mut self) {
        *self = Float64(rug::Float::with_val(52, 0.0).into());
    }

    fn zero() -> Self {
        Self(rug::Float::with_val(52, 0.0).into())
    }

    fn is_zero(&self) -> bool {
        self.0.as_float().is_zero()
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

impl num_traits::One for Float64 {
    fn is_one(&self) -> bool
    where
        Self: PartialEq,
    {
        *self.0.as_float() == 1.0
    }
    fn one() -> Self {
        Self(rug::Float::with_val(52, 1.0).into())
    }
    fn set_one(&mut self) {
        *self = Float64(rug::Float::with_val(52, 1.0).into());
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

fn round_to_rug_round(round: Round) -> rug::float::Round {
    match round {
        Round::Nearest => rug::float::Round::Nearest,
        Round::Zero => rug::float::Round::Zero,
        Round::Up => rug::float::Round::Up,
        Round::Down => rug::float::Round::Down,
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
    fn inv(&mut self) {
        *self = Self(rug::Float::with_val_64(53, 1.0.div(rug::Float::from(self.0.clone()))).into())
    }
    fn to_string(&self)-> String {
        let mut val = self.0.as_float().clone();
        val.set_prec_round(53, rug::float::Round::Nearest);
        format!("{:.53}", val)
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