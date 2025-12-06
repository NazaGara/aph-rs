use aph::{Aph, BidiagonalAph};
use linalg::{Vector, fields::PseudoField};
use representation::{
    bidiagonal::Bidiagonal,
    triangular::{Triangular, TriangularArray},
};

pub mod aph;
pub mod coxian;
pub mod formats;
pub mod linalg;
pub mod operations;
mod polynomial;
pub mod representation;
pub mod round_op;
pub mod utils;
pub mod voting;

/// Constructs a new [Bidiagonal] APH distribution.
/// Constructs an initial probability vector of the given size.
///
/// The probability for the first state is set to $1$ and the rest to $0$.
pub fn new_bidiagonal<F: PseudoField>(size: usize) -> Aph<F, Bidiagonal<F>> {
    assert!(size > 0, "Size must be greater than zero.");
    Aph::new(Vector::one_and_zeros(0, size), Bidiagonal::new(size))
}

/// Constructs a new [TriangularArray] APH distribution.
/// Constructs an initial probability vector of the given size.
///
/// The probability for the first state is set to $1$ and the rest to $0$.
pub fn new_triangular_array<F: PseudoField>(size: usize) -> Aph<F, TriangularArray<F>> {
    assert!(size > 0, "Size must be greater than zero.");
    Aph::new(Vector::one_and_zeros(0, size), TriangularArray::new(size))
}

/// Constructs a new [Triangular] APH distribution.
/// Constructs an initial probability vector of the given size.
///
/// The probability for the first state is set to $1$ and the rest to $0$.
pub fn new_triangular<F: PseudoField>(size: usize) -> Aph<F, Triangular<F>> {
    assert!(size > 0, "Size must be greater than zero.");
    Aph::new(Vector::one_and_zeros(0, size), Triangular::new(size))
}
