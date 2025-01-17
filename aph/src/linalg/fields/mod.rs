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
use std::fmt::Debug;
use std::fmt::Display;
// use std::ops::Add;
// use std::ops::Sub;

pub mod float32;
pub mod float64;
pub mod interval_field;
pub mod rational;
pub mod matrix; 

pub trait FromRational: Sized {
    fn from_rational(nominator: &str, denominator: &str) -> Self;
}

pub trait ToRational: Sized {
    fn to_rational(&self) -> (String, String);
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
    // + Add
    // + Sub
{
    fn neg_assign(&mut self);
    fn abs_assign(&mut self);

    fn add_assign(&mut self, rhs: &Self, round: Round);
    fn sub_assign(&mut self, rhs: &Self, round: Round);
    fn mul_assign(&mut self, rhs: &Self, round: Round);
    fn div_assign(&mut self, rhs: &Self, round: Round);

    fn inv(&mut self);
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
    // + Add
    // + Sub
{
    fn neg_assign(&mut self);
    fn abs_assign(&mut self);

    fn add_assign(&mut self, rhs: &Self);
    fn sub_assign(&mut self, rhs: &Self);
    fn mul_assign(&mut self, rhs: &Self);
    fn div_assign(&mut self, rhs: &Self);

    fn inv(&mut self);
    fn to_string(&self) -> String;
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
    fn to_string(&self) -> String {
        self.to_string()
    }
}

// ! Matrix operations
// Matrixes A and B are upper triangular.
// pub fn dot_product<F: PseudoField>(a: &Array2<F>, b: &Array2<F>, dst: &mut Array2<F>){
    pub fn dot_product<F: PseudoField>(a: &Array2<F>, b: &Array2<F>) -> Array2<F>{
    // Check if the dimensions are compatible
    assert!(
        a.shape()[1] == b.shape()[0],
        "Incompatible dimensions for matrix multiplication."
    );
    // Get dimensions of the result array
    let rows = a.shape()[0];
    let cols = b.shape()[1];
    // Create a new array for the result
    let mut dst: ndarray::ArrayBase<ndarray::OwnedRepr<F>, ndarray::Dim<[usize; 2]>> =
        Array2::zeros((rows, cols));
    // Perform manual dot product calculation
    // for i in 0..rows {
    //     for j in 0..cols {
    //         if j > i {
    //             continue;
    //         }
    //         for k in 0..a.shape()[1] {
    //             if k > i || j > k {
    //                 continue;
    //             }
    //             let mut val = a[(i, k)].clone();
    //             val.mul_assign(&b[(k, j)]);
    //             result[(i, j)].add_assign(&val);
    //         }
    //     }
    // }
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
        result = dot_product(&result, &matrix);
        // result = strassen(&result, &matrix);
    }
    result
}

// fn _add_matrices(a: &Array2<Rational>, b: &Array2<Rational>) -> Array2<Rational> {
//     a + b
// }
// /// Subtract two matrices
// fn _subtract_matrices(a: &Array2<Rational>, b: &Array2<Rational>) -> Array2<Rational> {
//     a - b
// }
// /// Strassen's matrix multiplication algorithm
// pub fn _strassen(a: &Array2<Rational>, b: &Array2<Rational>) -> Array2<Rational> {
//     let n = a.shape()[0];
//     // Base case for small matrices
//     if n == 1 {
//         return Array2::from_elem((1, 1), a[[0, 0]].clone() * b[[0, 0]].clone());
//     }
//     if n <= 32 {
//         return dot_product(a, b);
//     }
//     // Split matrices into four submatrices
//     let mid = n / 2;
//     let low = Array2::<Rational>::zeros((n - mid, mid));
//     let a11 = a.slice(s![0..mid, 0..mid]).to_owned();
//     let a12 = a.slice(s![0..mid, mid..n]).to_owned();

//     // let a21 = a.slice(s![mid..n, 0..mid]).to_owned();
//     // let a21 = Array2::<Rational>::zeros((n - mid, mid)); //a.slice(s![mid..n, 0..mid]).to_owned();

//     let a22 = a.slice(s![mid..n, mid..n]).to_owned();

//     let b11 = b.slice(s![0..mid, 0..mid]).to_owned();
//     let b12 = b.slice(s![0..mid, mid..n]).to_owned();

//     // let b21 = b.slice(s![mid..n, 0..mid]).to_owned();
//     // let b21 = Array2::<Rational>::zeros((n - mid, mid)); //b.slice(s![mid..n, 0..mid]).to_owned();

//     let b22 = b.slice(s![mid..n, mid..n]).to_owned();
//     // Compute the seven products (Strassen's method)
//     let m1 = _strassen(&_add_matrices(&a11, &a22), &_add_matrices(&b11, &b22));
//     // let m2 = _strassen(&_add_matrices(&a21, &a22), &b11);
//     let m2 = _strassen(&a22, &b11);
//     let m3 = _strassen(&a11, &_subtract_matrices(&b12, &b22));
//     let m4 = _strassen(&a22, &_subtract_matrices(&low, &b11));
//     let m5 = _strassen(&_add_matrices(&a11, &a12), &b22);
//     let m6 = _strassen(&_subtract_matrices(&low, &a11), &_add_matrices(&b11, &b12));
//     // let m7 = _strassen(&_subtract_matrices(&a12, &a22), &_add_matrices(&b21, &b22));
//     let m7 = _strassen(&_subtract_matrices(&a12, &a22), &b22);

//     // Combine submatrices into the resulting matrix
//     let c11: ndarray::ArrayBase<ndarray::OwnedRepr<Rational>, ndarray::Dim<[usize; 2]>> =
//         &m1 + &m4 - &m5 + &m7;
//     let c12: ndarray::ArrayBase<ndarray::OwnedRepr<Rational>, ndarray::Dim<[usize; 2]>> = &m3 + &m5;
//     let c21: ndarray::ArrayBase<ndarray::OwnedRepr<Rational>, ndarray::Dim<[usize; 2]>> = &m2 + &m4;
//     let c22: ndarray::ArrayBase<ndarray::OwnedRepr<Rational>, ndarray::Dim<[usize; 2]>> =
//         &m1 - &m2 + &m3 + &m6;
//     // Assemble the result matrix
//     let mut result = Array2::<Rational>::zeros((n, n));
//     result.slice_mut(s![0..mid, 0..mid]).assign(&c11);
//     result.slice_mut(s![0..mid, mid..n]).assign(&c12);
//     result.slice_mut(s![mid..n, 0..mid]).assign(&c21);
//     result.slice_mut(s![mid..n, mid..n]).assign(&c22);
//     result
// }
