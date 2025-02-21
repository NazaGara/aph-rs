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
    marker::PhantomData,
    ops::{Add, Div, Mul, Sub},
};

use super::{
    round_to_rug_round, Almost, ContFraction, FromCF, FromRational, Round, Rounding, SparseField,
    ToRational,
};
use num_rational::Ratio;
use num_traits::{One, Zero};
use rug::{
    float::OrdFloat,
    ops::{
        AddAssignRound, CompleteRound, DivAssignRound, MulAssignRound, NegAssign, SubAssignRound,
    },
    Float, Integer,
};

/// A 64-bit floating-point type implementing [`SparseField`].
///
/// Currently this is based on [`rug::Float`] but in the future we may implement this
/// type using architecture specific features.
/// Uses Continued Fractions to teransition between ratiuonal numbers and Floats.
#[derive(Clone, PartialEq, PartialOrd, Ord, Eq, Hash)]
pub struct Float64RoundCF<R: Rounding> {
    value: rug::float::OrdFloat,
    round: PhantomData<R>,
}

impl<R: Rounding> Debug for Float64RoundCF<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.value)
    }
}

impl<R: Rounding> Almost for Float64RoundCF<R> {
    fn is_almost_zero(&self) -> bool {
        // Option 1: Strict comparison
        self.is_zero()

        // Option 2: use the 1-st convergent of F64::epsilon/2, which is a good approximation of 1/2 ULP.
        // let mut self_abs = self.clone();
        // self_abs.abs_assign();
        // let (numer, denom) = self_abs.to_rational();
        // let eps_cf = Integer::from(9007199254740992 as i128); // 4503599627370496 for f64::eps
        // let self_cf = ContFraction::from_explicit_with_precision(&numer, &denom, 4);
        // if self_cf.values.len() == 1 {
        //     self_cf.values[0].is_zero() // if is an integer, see if is not zero.
        // } else {
        //     self_cf.values[1] > eps_cf // BCS is index 1 (odd), the comparison must be inverted
        // }
    }

    /// Compare values using Continued Fractions.
    /// Given $[a_0,a_1,...,a_m]$ and $[n_0,n_1,...,n_m]$,, we see the first $i$ such that $a_i \neq b_i$.
    /// If $i$ is odd, we swap the sign.
    /// Thus: a <= b -> CF(a) - CF(b) <= CF(eps) -> CF(a-b) <= CF(eps) -> CF(a-b)_1 <= CF(eps)_1
    fn cmp_eq(&self, other: &Self) -> bool {
        let mut diff = self.clone().sub(other.clone());
        diff.abs_assign();
        let (numer, denom) = diff.to_rational();
        let diff_cf = ContFraction::from_explicit_with_precision(&numer, &denom, 3);
        let eps_cf = Integer::from(10000000 as i128);
        if match diff_cf.values.len() {
            0 => unreachable!("Continued Fraction can not be empty."),
            1 => diff_cf.values[0].ge(&rug::Integer::from_f64(0.0).unwrap()),
            _ => diff_cf.values[1] > eps_cf, // Because is index 1 (odd), the comparison must be inverted
        } {
            return true;
        }

        let eps = Self::from_rational("1", "10000");
        let mut self_abs = self.clone();
        self_abs.abs_assign();
        let mut other_abs = other.clone();
        other_abs.abs_assign();
        let reltol = (if self_abs.ge(&other_abs) {
            self_abs
        } else {
            other_abs
        })
        .mul(eps.clone());

        diff.le(&reltol)
    }

    fn is_almost_zero_and_correct(&mut self) {
        if self.is_almost_zero() {
            self.set_zero();
        };
    }
}

impl<R: Rounding> Display for Float64RoundCF<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut val = self.value.as_float().clone();
        val.set_prec_round_64(53, round_to_rug_round(R::rounding()));
        write!(f, "{:.8}", val)
    }
}

impl<R: Rounding> FromCF for Float64RoundCF<R> {
    fn from_cont_fraction(cf: &mut ContFraction) -> Self {
        if cf.values.len() <= 1 {
            return Float64RoundCF::from(cf.values[0].to_i128_wrapping() as f64);
        } else {
            let val = Float64RoundCF::from(cf.values.remove(0).to_i128_wrapping() as f64);
            let mut rgt = Float64RoundCF::one();
            rgt.div_assign(
                &Float64RoundCF::from_cont_fraction(&mut ContFraction {
                    values: cf.values.clone(),
                }),
                R::rounding(),
            );
            return val + rgt;
        }
    }
}

impl<R: Rounding> FromRational for Float64RoundCF<R> {
    fn from_rational(numerator: &str, denominator: &str) -> Self {
        let (num, _) = Float::parse(numerator)
            .unwrap()
            .complete_round(53, round_to_rug_round(R::rounding()));
        let (den, _) = Float::parse(denominator)
            .unwrap()
            .complete_round(53, round_to_rug_round(R::rounding()));
        let at = match R::rounding() {
            Round::Down | Round::Zero => 4, //8,
            Round::Up => 5,                 //7,
            Round::Nearest => 25,
        };
        let ratio =
            Ratio::from_float((num.div(den)).to_f64_round(round_to_rug_round(R::rounding())))
                .expect("");

        let mut cf = ContFraction::from_explicit_with_precision(
            &ratio.numer().to_string(),
            &ratio.denom().to_string(),
            at,
        );

        let mut val = Float64RoundCF::from_cont_fraction(&mut cf);
        val.is_almost_zero_and_correct();
        val
    }
    fn from_rational_modular(numerator: &str, denominator: &str, _moduler: f64) -> Self {
        Self::from_rational(numerator, denominator)
    }
}

impl<R: Rounding> ToRational for Float64RoundCF<R> {
    fn to_rational(&self) -> (String, String) {
        let ratio = Ratio::from_float(
            self.value
                .as_float()
                .to_f64_round(round_to_rug_round(R::rounding())),
        )
        .expect("Something went wrong when converting the Float to Rational.");
        (ratio.numer().to_string(), ratio.denom().to_string())
    }
}

impl<R: Rounding> Add for Float64RoundCF<R> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Self {
            value: self
                .value
                .as_float()
                .add(rhs.value.as_float())
                .complete_round(53, round_to_rug_round(R::rounding()))
                .0
                .into(),
            round: PhantomData,
        }
    }
}

impl<R: Rounding> num_traits::Zero for Float64RoundCF<R> {
    fn set_zero(&mut self) {
        *self = Float64RoundCF {
            value: rug::Float::with_val(53, 0.0).into(),
            round: PhantomData,
        };
    }

    fn zero() -> Self {
        Self {
            value: rug::Float::with_val(53, 0.0).into(),
            round: PhantomData,
        }
    }

    fn is_zero(&self) -> bool {
        self.value.as_float().is_zero()
    }
}

impl<R: Rounding> Mul for Float64RoundCF<R> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        Self {
            value: self
                .value
                .as_float()
                .mul(rhs.value.as_float())
                .complete_round(53, round_to_rug_round(R::rounding()))
                .0
                .into(),
            round: PhantomData,
        }
    }
}

impl<R: Rounding> num_traits::One for Float64RoundCF<R> {
    fn is_one(&self) -> bool
    where
        Self: PartialEq,
    {
        *self.value.as_float() == 1.0
    }
    fn one() -> Self {
        Self {
            value: rug::Float::with_val(53, 1.0).into(),
            round: PhantomData,
        }
    }
    fn set_one(&mut self) {
        *self = Float64RoundCF {
            value: rug::Float::with_val(53, 1.0).into(),
            round: PhantomData,
        };
    }
}

impl<R: Rounding> From<f64> for Float64RoundCF<R> {
    fn from(value: f64) -> Self {
        Float64RoundCF {
            value: rug::Float::with_val(53, value).into(),
            round: PhantomData,
        }
    }
}

impl<R: Rounding> From<Float64RoundCF<R>> for f64 {
    fn from(float: Float64RoundCF<R>) -> Self {
        float.value.as_float().to_f64()
    }
}

impl<R: Rounding> SparseField for Float64RoundCF<R> {
    fn neg_assign(&mut self) {
        self.value.as_float_mut().neg_assign()
    }

    fn abs_assign(&mut self) {
        self.value.as_float_mut().abs_mut();
    }

    fn add_assign(&mut self, rhs: &Self, _round: Round) {
        self.value
            .as_float_mut()
            .add_assign_round(rhs.value.as_float(), round_to_rug_round(R::rounding()));
    }

    fn sub_assign(&mut self, rhs: &Self, _round: Round) {
        self.value
            .as_float_mut()
            .sub_assign_round(rhs.value.as_float(), round_to_rug_round(R::rounding()));
    }

    fn mul_assign(&mut self, rhs: &Self, _round: Round) {
        self.value
            .as_float_mut()
            .mul_assign_round(rhs.value.as_float(), round_to_rug_round(R::rounding()));
    }

    fn div_assign(&mut self, rhs: &Self, _round: Round) {
        self.value
            .as_float_mut()
            .div_assign_round(rhs.value.as_float(), round_to_rug_round(R::rounding()));
    }

    fn inv(&mut self) {
        *self = Self {
            value: rug::Float::with_val_64(53, 1.0.div(rug::Float::from(self.value.clone())))
                .into(),
            round: PhantomData,
        }
    }

    fn to_string(&self) -> String {
        let mut val = self.value.as_float().clone();
        val.set_prec_round(53, round_to_rug_round(R::rounding()));
        format!("{:?}", val)
    }
}

impl<R: Rounding> Sub for Float64RoundCF<R> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        let mut neg_rhs = rhs.clone();
        neg_rhs.neg_assign();
        Self {
            value: OrdFloat::from(
                self.value
                    .as_float()
                    .add(neg_rhs.value.as_float())
                    .complete(53),
            ),
            round: PhantomData,
        }
    }
}
