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
    multiple_of, round_to_rug_round, Almost, ContFraction, FromCF, FromRational, Round, Rounding,
    SparseField, ToRational,
};
use num_rational::Ratio;
use num_traits::{One, Zero};
use rug::{
    float::OrdFloat,
    ops::{
        AddAssignRound, CompleteRound, DivAssignRound, MulAssignRound, NegAssign, Pow, SubAssignRound
    },
    Float,
};

/// A 64-bit floating-point type implementing [`SparseField`].
///
/// Currently this is based on [`rug::Float`] but in the future we may implement this
/// type using architecture specific features.
#[derive(Clone, PartialEq, PartialOrd, Ord, Eq, Hash)]
pub struct Float64Round<R: Rounding> {
    value: rug::float::OrdFloat,
    round: PhantomData<R>,
}

impl<R: Rounding> Debug for Float64Round<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.value)
    }
}

impl<R: Rounding> Almost for Float64Round<R> {
    fn is_almost_zero(&self) -> bool {
        let mut self_abs = self.clone();
        self_abs.abs_assign();
        let epsilon = Self::from_rational(&format!("{:?}", f64::EPSILON), "2");
        self_abs.le(&epsilon)
    }

    /// -- listing 3 from https://www.accu.org/journals/overload/31/173/floyd/
    fn cmp_eq(&self, other: &Self) -> bool {
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
        if self.is_almost_zero() {
            self.set_zero();
        };
    }
}

impl<R: Rounding> Display for Float64Round<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut val = self.value.as_float().clone();
        val.set_prec_round_64(53, round_to_rug_round(R::rounding()));
        write!(f, "{:.8}", val)
    }
}

impl<R: Rounding> FromCF for Float64Round<R> {
    fn from_cont_fraction(cf: &mut ContFraction) -> Self {
        if cf.values.len() <= 1 {
            return Float64Round::from(cf.values[0].to_i128_wrapping() as f64);
        } else {
            let val = Float64Round::from(cf.values.remove(0).to_i128_wrapping() as f64);
            let mut rgt = Float64Round::one();
            rgt.div_assign(
                &Float64Round::from_cont_fraction(&mut ContFraction {
                    values: cf.values.clone(),
                }),
                R::rounding(),
            );
            return val + rgt;
        }
    }
}

impl<R: Rounding> FromRational for Float64Round<R> {
    fn from_rational(numerator: &str, denominator: &str) -> Self {
        let (num, _) = Float::parse(numerator)
            .unwrap()
            .complete_round(53, round_to_rug_round(R::rounding()));
        let (den, _) = Float::parse(denominator)
            .unwrap()
            .complete_round(53, round_to_rug_round(R::rounding()));
        Self {
            value: rug::Float::with_val(53, num / den).into(),
            round: PhantomData,
        }
    }
    fn from_rational_modular(numerator: &str, denominator: &str, _moduler: f64) -> Self {
        let num = Float::parse(numerator).unwrap().complete(53);
        let den = Float::parse(denominator).unwrap().complete(53);
        let moduler = Float::with_val(53, 1e8);
        match R::rounding() {
            Round::Up => {
                let new_num = multiple_of(&num, &moduler, Round::Up);
                let new_den = multiple_of(&den, &moduler, Round::Down);

                let lala = Float::ceil(new_num.clone().log2() - new_den.clone().log2());
                if lala.is_infinite() || lala.is_nan(){
                    Self::from_rational(numerator, denominator)
                }else{
                    let lala = Float::with_val(53, 2.0).pow(lala);
                    Self{
                        value: lala.into(),
                        round : PhantomData
                    }
                }
                // Self {
                //     value: rug::Float::with_val(53, new_num / new_den).into(),
                //     round: PhantomData,
                // }
            }
            Round::Down | Round::Zero => {
                let new_num = multiple_of(&num, &moduler, Round::Down);
                let new_den = multiple_of(&den, &moduler, Round::Up);
                Self {
                    value: rug::Float::with_val(53, new_num / new_den).into(),
                    round: PhantomData,
                }
                // let new_num = multiple_of(&num, &moduler, Round::Up);
                // let new_den = multiple_of(&den, &moduler, Round::Down);

                // let lala = Float::floor(new_num.clone().log2() - new_den.clone().log2());
                // if lala.is_infinite() || lala.is_nan(){
                //     Self::from_rational(numerator, denominator)
                // }else{
                //     let lala = Float::with_val(53, 2.0).pow(lala);
                //     Self{
                //         value: lala.into(),
                //         round : PhantomData
                //     }
                // }
            }
            Round::Nearest => Self::from_rational(numerator, denominator),
        }
    }
}

impl<R: Rounding> ToRational for Float64Round<R> {
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

impl<R: Rounding> Add for Float64Round<R> {
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

impl<R: Rounding> num_traits::Zero for Float64Round<R> {
    fn set_zero(&mut self) {
        *self = Float64Round {
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

impl<R: Rounding> Mul for Float64Round<R> {
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

impl<R: Rounding> num_traits::One for Float64Round<R> {
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
        *self = Float64Round {
            value: rug::Float::with_val(53, 1.0).into(),
            round: PhantomData,
        };
    }
}

impl<R: Rounding> From<f64> for Float64Round<R> {
    fn from(value: f64) -> Self {
        Float64Round {
            value: rug::Float::with_val(53, value).into(),
            round: PhantomData,
        }
    }
}

impl<R: Rounding> From<Float64Round<R>> for f64 {
    fn from(float: Float64Round<R>) -> Self {
        float.value.as_float().to_f64()
    }
}

impl<R: Rounding> SparseField for Float64Round<R> {
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

impl<R: Rounding> Sub for Float64Round<R> {
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
