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

use ndarray::Array2;
use num_rational::Ratio;
use std::fmt::{Debug, Display};
use std::hash::Hash;

pub mod float64;
pub mod inari_int;
pub mod interval_field;
pub mod rational;

pub trait FromRational: Sized {
    fn from_rational(numerator: &str, denominator: &str) -> Self;
}

pub trait ToRational: Sized {
    fn to_rational(&self) -> (String, String);
}

pub trait Almost: Sized + num_traits::One + num_traits::Zero {
    fn cmp_eq(&self, other: &Self) -> bool;
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
    // /// Round towards negative infinity.
    Down,
}

pub trait Rounding: Debug + Sized + Clone + Ord + Eq + Hash {
    fn rounding() -> Round;
}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Eq, Ord, Hash)]
pub struct Up;
impl Rounding for Up {
    fn rounding() -> Round {
        Round::Up
    }
}
impl Display for Up {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "(UP)")
    }
}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Eq, Ord, Hash)]
pub struct Near;
impl Rounding for Near {
    fn rounding() -> Round {
        Round::Nearest
    }
}
impl Display for Near {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "(Near)")
    }
}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Eq, Ord, Hash)]
pub struct Down;
impl Rounding for Down {
    fn rounding() -> Round {
        Round::Down
    }
}
impl Display for Down {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "(DOWN)")
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

/// A [`SparseField`] is a numeric type supporting the usual arithmetic operators of
/// fields but unable to exactly represent some results of some of these operators, e.g.,
/// floating-point numbers.
pub trait SparseField:
    Debug
    + Sized
    + Clone
    + Ord
    + Eq
    + Into<f64>
    + FromRational
    + ToRational
    + num_traits::Zero
    + num_traits::One
    + Display
    + Hash
    + Almost
    + From<f64>
{
    /// Machine epsilon value for F, based on f64::EPSILON.
    /// This is the difference between 1.0 and the next larger representable number.
    fn epsilon() -> Self {
        let ratio = Ratio::from_float(f64::EPSILON).unwrap();
        Self::from_rational(&ratio.numer().to_string(), &ratio.denom().to_string())
    }

    /// Absolute tolerance threshold to use when comparing bwetween two no-zero values.
    fn abstol() -> Self {
        let ratio = Ratio::from_float(1e-16).unwrap();
        Self::from_rational(&ratio.numer().to_string(), &ratio.denom().to_string())
    }

    /// Relative tolerance threshold to use when comparing bwetween two no-zero values.
    fn reltol() -> Self {
        let ratio = Ratio::from_float(1e-4).unwrap();
        Self::from_rational(&ratio.numer().to_string(), &ratio.denom().to_string())
    }

    fn inf() -> Self {
        let ratio = Ratio::from_float(1.0_f64.powf(-300.0)).unwrap();
        Self::from_rational(&ratio.numer().to_string(), &ratio.denom().to_string())
    }

    fn neg_assign(&mut self);
    fn abs_assign(&mut self);
    fn inv_assign(&mut self);

    fn add_assign(&mut self, rhs: &Self, round: Round);
    fn sub_assign(&mut self, rhs: &Self, round: Round);
    fn mul_assign(&mut self, rhs: &Self, round: Round);
    fn div_assign(&mut self, rhs: &Self, round: Round);

    fn to_string(&self) -> String;
}

/// A [`PseudoField`] is a numeric type supporting the usual arithmetic operators of
/// fields but without any guarantee that the field axioms are satisfied, e.g.,
/// rational numbers, floating-point numbers, or intervals.
pub trait PseudoField:
    Debug
    + Sized
    + Clone
    + PartialOrd
    + PartialEq
    + FromRational
    + ToRational
    + num_traits::Zero
    + num_traits::One
    + Display
    + Eq
    + Hash
    + Almost
    + From<f64>
{
    fn neg_assign(&mut self);
    fn abs_assign(&mut self);
    fn inv_assign(&mut self);

    fn add_assign(&mut self, rhs: &Self);
    fn sub_assign(&mut self, rhs: &Self);
    fn mul_assign(&mut self, rhs: &Self);
    fn div_assign(&mut self, rhs: &Self);

    fn to_string(&self) -> String;
    fn max_value() -> Self {
        Self::from(f64::MAX)
    }

    fn min_value() -> Self {
        Self::from(f64::MIN)
    }
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

    fn inv_assign(&mut self) {
        self.inv_assign();
    }

    fn to_string(&self) -> String {
        self.to_string()
    }
}

// ! Matrix operations
// Matrixes A and B are upper triangular.
// pub fn dot_product<F: PseudoField>(a: &Array2<F>, b: &Array2<F>, dst: &mut Array2<F>){
pub fn dot_product<F: PseudoField>(a: &Array2<F>, b: &Array2<F>) -> Array2<F> {
    // Check if the dimensions are compatible
    assert!(
        a.shape()[1] == b.shape()[0],
        "Incompatible dimensions for matrix multiplication."
    );
    // Get dimensions of the result array
    let rows = a.shape()[0];
    let cols = b.shape()[1];
    // New array for result
    let mut dst: ndarray::ArrayBase<ndarray::OwnedRepr<F>, ndarray::Dim<[usize; 2]>> =
        Array2::zeros((rows, cols));
    for j in 0..cols {
        for i in 0..rows {
            for k in 0..a.shape()[1] {
                let mut val = a[(i, k)].clone();
                val.mul_assign(&b[(k, j)]);
                dst[(i, j)].add_assign(&val);
            }
        }
    }

    dst
}

pub fn matrix_power<F: PseudoField>(matrix: &Array2<F>, power: usize) -> Array2<F> {
    // Check if the matrix is square
    let n = matrix.shape()[0];
    if matrix.shape()[1] != n {
        panic!("Matrix must be square to compute power.");
    }
    // Start with an identity matrix of the same size
    let mut result: ndarray::ArrayBase<ndarray::OwnedRepr<F>, ndarray::Dim<[usize; 2]>> =
        Array2::eye(n);

    for _ in 0..power {
        result = dot_product(&result, matrix);
        // result = strassen(&result, &matrix);
    }
    result
}
