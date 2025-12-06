//! Data structures and algorithms for the representation of the matrix.
use std::fmt::Display;

use linalg::{Vector, fields::PseudoField};
use ndarray::Array;

use crate::linalg;
use crate::representation::Representation;
use crate::representation::triangular::TriangularArray;

/// Bidiagonal matrix representation.
///
/// A Bidiagonal Representation is a pair
/// $(\overrightarrow{\alfa}, \mathbf{Bi}(\lambda_1, \lambda_2, \dots, \lambda_n))$
/// where $n$ is the size of the APH and the matrix generator has the shape:
/// $$
/// \begin{pmatrix}
/// -\lambda_1 & \lambda_1 & 0 & \cdots & 0     \\\\
/// 0 & -\lambda_2 & \lambda_2 & \cdots & 0     \\\\
/// \vdots & \vdots & \vdots & \ddots & \vdots  \\\\
/// 0 & 0 & 0 & \cdots & \lambda_{N-1}          \\\\
/// 0 & 0 & 0 & \cdots & -\lambda_N             \\\\
/// \end{pmatrix}
/// $$
#[derive(Clone, Debug)]
pub struct Bidiagonal<F>(pub Box<[F]>);

impl<F: PseudoField> Bidiagonal<F> {
    pub fn new(size: usize) -> Self {
        Self(vec![F::zero(); size].into())
    }

    pub fn push(&mut self, value: F) {
        let mut elems = self.0.to_vec();
        elems.push(value);
        *self = Self(elems.into_boxed_slice());
    }

    pub fn set(&mut self, row: usize, value: F) -> &mut Self {
        self.0[row] = value;
        self
    }

    pub fn into_ordered(&mut self) {
        self.0.sort_by(|x, y| {
            y.partial_cmp(x)
                .unwrap_or_else(|| panic!("Could not sort the values: {:?} and {:?}.", y, x))
        })
    }

    pub fn eye(size: usize) -> Self {
        Self(vec![F::one(); size].into())
    }
}

impl<F: PseudoField> From<Bidiagonal<F>> for Vec<F> {
    fn from(value: Bidiagonal<F>) -> Self {
        value.0.to_vec()
    }
}

impl<F: PseudoField> From<Vector<F>> for Bidiagonal<F> {
    fn from(value: Vector<F>) -> Self {
        Bidiagonal(value.elements)
    }
}

impl<F: PseudoField> From<Bidiagonal<F>> for TriangularArray<F> {
    fn from(value: Bidiagonal<F>) -> Self {
        TriangularArray::new_from_diagonal(&Array::from_vec(value.0.to_vec()))
    }
}

impl<F: PseudoField> Display for Bidiagonal<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", TriangularArray::from(self.to_owned()).matrix)
    }
}

impl<F: PseudoField> Representation<F> for Bidiagonal<F> {
    fn size(&self) -> usize {
        self.0.len()
    }

    fn row_sum(&self, _idx: usize) -> F {
        F::zero()
    }

    fn get(&self, row: usize, column: usize) -> F {
        if row < self.0.len() {
            if row == column {
                self.0[row].clone()
            } else if row == column - 1 {
                let mut diagonal = self.0[row].clone();
                diagonal.neg_assign();
                diagonal
            } else {
                F::zero()
            }
        } else {
            F::zero()
        }
    }

    fn set(&mut self, row: usize, _column: usize, value: F) -> &mut Self {
        self.set(row, value)
    }

    fn to_array_repr(&self) -> TriangularArray<F> {
        TriangularArray::from(self.clone())
    }

    fn kron_prod(&self, _other: &Bidiagonal<F>) -> Bidiagonal<F> {
        todo!();
    }

    fn kron_sum(&self, _other: &Bidiagonal<F>) -> Bidiagonal<F> {
        todo!();
    }

    fn diagonal(&self, row: usize) -> F {
        self.0[row].clone()
    }

    fn remove_state(&mut self, idx: usize) {
        assert!(
            idx <= self.size(),
            "Attemppting to remove index {:?} out of bounds (size: {:?})",
            idx,
            self.size()
        );

        // Split at idx + 1.
        let (pre, post) = self.0.split_at_mut(idx + 1);
        // Remove the last element from pre.
        let (_removed, pre) = pre.split_last().expect("Something went  wrong");
        // Merge pre withouth the last one with the post.
        self.0 = ([pre, post]).concat().into_boxed_slice();
    }

    fn to_absorbing(&self) -> Vector<F> {
        let mut rate = self.0[self.0.len() - 1].clone();
        rate.neg_assign();
        let mut vector = Vector::zeros(self.size());
        if let Some(last) = vector.elements.get_mut(self.size() - 1) {
            *last = rate;
        }
        vector
    }
}
