//! Traits and data structures for different varieties of *fields*.
//!
//! Mathematically speaking, a *field* is a set $F$ with two binary operators
//! $\mathord{\cdot} : F \times F \to F$ and $\mathord{+} : F \times F \to F$, called
//! *multiplication* and *addition* respectively, such that the following *field axioms*
//! are satisfied:
//!
//! 1. **Associativity**: $x + (y + z) = (x + y) + z$ and
//!    $x \cdot (y \cdot z) = (x \cdot y) \cdot z$.
//! 2. **Commutativity**: $x + y = y + x$ and $x \cdot y = y \cdot x$.
//! 3. **Distributivity**: $x \cdot (y + z) = (x \cdot y) + (x \cdot z)$.
//! 4. **Neutral Elements**: There exists $0$ and $1$ such that $x + 0 = x$ and
//!    $x \cdot 1 = x$.
//! 5. **Inverse Elements**: For each $x$, there exist $-x$ and $x^{-1}$ such that
//!    $x + -x = 0$ and $x \cdot x^{-1} = 1$.
//!
//! In practice, however, any computer is constrained by its finite memory necessitating
//! finite and thus approximate representations of numbers. For instance, floating-point
//! numbers do not satisfy the field axioms as evidenced by:
//!
//! ```
//! assert!(10e-100 + (-1.0 + 1.0) != (10e-100 + -1.0) + 1.0)
//! ```
//!
//! Here, the associativity axiom is violated.
//!
//! This module provides three traits with varying guarantees regarding the field axioms:
//!
//! - A [`Field`] is a numeric type satisfying all field axioms, e.g.,
//!   arbitrary-precision rational numbers as provided by [`rug`].
//!
//! - A [`SparseField`] is a numeric type supporting the usual arithmetic operators but
//!   unable to precisely represent some results of some of these operators. For instance,
//!   the floating-point numbers form a [`SparseField`]. As some results cannot be
//!   precisely represented, they need to be *rounded* to some number which can be
//!   precisely represented. To this end, [`SparseField`] supports a *rounding mode*
//!   [`Round`] for all arithmetic operators specifying how results shall be rounded.
//!
//! - Finally, [`PseudoField`] is a numeric type supporting the usual arithmetic
//!   operators but without any guarantee that any of the field axioms are satisfied and
//!   without the possibility to specify a rounding mode.
//!
//! Every [`Field`] is a [`PseudoField`]. In fact, [`Field`] is a mere marker trait on
//! top of [`PseudoField`]. Every [`SparseField`] is also a [`PseudoField`] by rounding
//! the results of operations to the nearest representable number.
//!
//! An [`IntervalField`] is a [`PseudoField`] which tracks the rounding errors occurring
//! with a [`SparseField`] by using interval arithmetic. In a nutshell, it tracks an
//! upper and a lower bound within which the actual result lies.
//!
//! A type should never implement both [`Field`] and [`SparseField`].
//!
//! Note that the traits are defined with heap-allocated numeric types in mind.

use std::{
    fmt::Display,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, SubAssign},
};

// use super::{One, Zero};
use rug::{
    float::OrdFloat,
    ops::{
        AddAssignRound, CompleteRound, DivAssignRound, MulAssignRound, NegAssign, SubAssignRound,
    },
    Complete,
};
use std::fmt::Debug;

pub trait FromRational: Sized {
    fn from_rational(nominator: &str, denominator: &str) -> Self;
}

/// A [`Field`] is a numeric type satisfying all field axioms, e.g., arbitrary-precision
/// rational numbers.
pub trait Field: PseudoField {}

/// A *rounding mode* for the operations on [`SparseField`].
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Round {
    /// Round towards the nearest representable number.
    Nearest,
    /// Round towards zero.
    Zero,
    /// Round towards positive infinity.
    Up,
    /// Round towards negative infinity.
    Down,
}

/// A [`SparseField`] is a numeric type supporting the usual arithmetic operators of
/// fields but unable to exactly represent some results of some of these operators, e.g.,
/// floating-point numbers.
pub trait SparseField:
    // Debug + Sized + Clone + Ord + Eq + Zero + One + FromRational + Into<f64>
    Debug + Sized + Clone + Ord + Eq + FromRational + Into<f64> + num_traits::Zero + num_traits::One + Display // + num_traits::Inv
{
    fn neg_assign(&mut self);
    fn abs_assign(&mut self);

    fn add_assign(&mut self, rhs: &Self, round: Round);
    fn sub_assign(&mut self, rhs: &Self, round: Round);
    fn mul_assign(&mut self, rhs: &Self, round: Round);
    fn div_assign(&mut self, rhs: &Self, round: Round);

    fn inv(&mut self);
}

/// A [`PseudoField`] is a numeric type supporting the usual arithmetic operators of
/// fields but without any guarantee that the field axioms are satisfied, e.g.,
/// rational numbers, floating-point numbers, or intervals.
pub trait PseudoField:
    // Debug + Sized + Clone + PartialOrd + PartialEq + Zero + One + FromRational
    Debug + Sized + Clone + PartialOrd + PartialEq + num_traits::Zero + num_traits::One + FromRational + Display // + num_traits::Inv
{
    fn neg_assign(&mut self);
    fn abs_assign(&mut self);

    fn add_assign(&mut self, rhs: &Self);
    fn sub_assign(&mut self, rhs: &Self);
    fn mul_assign(&mut self, rhs: &Self);
    fn div_assign(&mut self, rhs: &Self);

    fn inv(&mut self);
}

impl<T: SparseField> PseudoField for T {
    fn neg_assign(&mut self) {
        self.neg_assign()
    }

    fn abs_assign(&mut self) {
        self.abs_assign()
    }

    fn add_assign(&mut self, rhs: &Self) {
        self.add_assign(rhs, Round::Nearest)
    }

    fn sub_assign(&mut self, rhs: &Self) {
        self.sub_assign(rhs, Round::Nearest)
    }

    fn mul_assign(&mut self, rhs: &Self) {
        self.mul_assign(rhs, Round::Nearest)
    }

    fn div_assign(&mut self, rhs: &Self) {
        self.div_assign(rhs, Round::Nearest)
    }

    fn inv(&mut self) {
        self.inv();
    }
}

/// A 64-bit floating-point type implementing [`SparseField`].
///
/// Currently this is based on [`rug::Float`] but in the future we may implement this
/// type using architecture specific features.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct Float64(rug::float::OrdFloat);

impl Display for Float64 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut val = self.0.as_float().clone();
        val.set_prec_round_64(4, rug::float::Round::Up);
        write!(f, "{}", val)
    }
}

impl FromRational for Float64 {
    fn from_rational(nominator: &str, denominator: &str) -> Self {
        let nominator = nominator.parse::<f64>().unwrap();
        let denominator = denominator.parse::<f64>().unwrap();
        Self(rug::Float::with_val(53, nominator / denominator).into())
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
        *self = Float64(rug::Float::with_val(53, 0.0).into());
    }

    fn zero() -> Self {
        Self(rug::Float::with_val(53, 0.0).into())
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
        Self(rug::Float::with_val(53, 1.0).into())
    }
    fn set_one(&mut self) {
        *self = Float64(rug::Float::with_val(53, 1.0).into());
    }
}

// impl num_traits::Inv for Float64{
//     type Output = Self;

//     fn inv(self) -> Self::Output {
//         Self(rug::Float::with_val_64(53, 1.0.div(rug::Float::from(self.0))).into())
//     }
// }

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
}

/// An arbitrary-precision rational number implementing [`Field`].
///
/// Currently this is based on [`rug::Rational`].
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct Rational(rug::Rational);

impl Display for Rational {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0.to_f64())
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

// impl num_traits::Inv for Rational{
//     type Output = Self;

//     fn inv(self) -> Self::Output {
//         Self(rug::Rational::from_f64(1.0.div(self.0.to_f64())).unwrap())
//     }
// }

impl Add for Rational {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0.add(rhs.0))
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

impl From<z3::ast::Real<'_>> for Rational {
    fn from(value: z3::ast::Real) -> Self {
        let real_str = value.to_string().replace("(", "").replace(")", "").replace(".0", "");
        if let Some((_, num_and_denom_str)) = real_str.split_once("/ ") {
            let (num_str, denom_str) = num_and_denom_str
                .split_once(" ")
                .expect("Something went wrong");
            Rational::from_rational(
                num_str.trim(),
                denom_str.trim(),
            )
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
        *self = Self(rug::Rational::from_f64(1.0.div(self.0.to_f64())).unwrap())
    }
}

/// A [`PseudoField`] using interval arithmetic to track the rounding imprecision of
/// a [`SparseField`].
#[derive(Clone, PartialEq, Debug)]
pub struct IntervalField<F: SparseField> {
    lower: F,
    upper: F,
}

impl<F: SparseField> num_traits::One for IntervalField<F> {
    fn one() -> Self {
        Self {
            lower: F::one(),
            upper: F::one(),
        }
    }

    fn is_one(&self) -> bool {
        self.lower.is_one() && self.upper.is_one()
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
        self.lower.is_zero() && self.upper.is_zero()
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
