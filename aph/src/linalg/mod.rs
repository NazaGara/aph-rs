//! Data structures and algorithms for linear algebra.

use std::fmt::Display;

use itertools::Itertools;
use ndarray::{Array, Array1};

// use crate::representation::Triangular;

use self::fields::PseudoField;

pub mod fields;

/// A type with a *zero element*.
pub trait Zero {
    /// The zero element of the type.
    fn zero() -> Self;

    /// Checks whether `self` is the zero element.
    fn is_zero(&self) -> bool;
}

/// A type with a *one element*.
pub trait One {
    /// The one element of the type.
    fn one() -> Self;

    /// Checks whether `self` is the one element.
    fn is_one(&self) -> bool;
}

/// A vector over a [`PseudoField`].
#[derive(Clone, Debug)]
pub struct Vector<F: PseudoField> {
    pub elements: Box<[F]>,
}

impl<F: PseudoField> Display for Vector<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", Array::from(self.elements.to_vec()))
    }
}

impl<F: PseudoField> From<Vec<F>> for Vector<F> {
    fn from(value: Vec<F>) -> Self {
        Vector {
            elements: value.into_boxed_slice(),
        }
    }
}

impl<F: PseudoField> Vector<F> {
    pub fn len(&self) -> usize {
        self.elements.iter().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn sum(&self) -> F {
        let mut accum = F::zero();
        let _ = self
            .elements
            .iter()
            .map(|e| accum.add_assign(e))
            .collect_vec();
        accum
    }

    pub fn rest(&self) -> F {
        let mut one = F::one();
        one.sub_assign(&self.sum());
        one
    }

    pub fn mult_by_scalar(&self, scalar: &F) -> Self {
        Vector::from(
            self.elements
                .iter()
                .map(|e| scalar.clone() * e.clone())
                .collect_vec(),
        )
    }

    pub fn zeros(size: usize) -> Self {
        Self {
            elements: vec![F::zero(); size].into(),
        }
    }

    pub fn unit(size: usize) -> Self {
        Self {
            elements: vec![F::one(); size].into(),
        }
    }

    pub fn one_and_zeros(idx: usize, size: usize) -> Self {
        let mut vector = Vector::zeros(size);
        if let Some(elem) = vector.elements.get_mut(idx) {
            *elem = F::one();
        }
        vector
    }

    pub fn size(&self) -> usize {
        self.elements.len()
    }

    pub fn scalar_product(&self, other: &Self) -> F {
        let mut result = F::zero();
        assert!(
            self.elements.len() == other.elements.len(),
            "Vectors must have the same dimension."
        );
        for (lhs, rhs) in self.elements.iter().zip(other.elements.iter()) {
            if !lhs.is_zero() {
                let mut product = lhs.clone();
                product.mul_assign(rhs);
                result.add_assign(&product);
            }
        }
        result
    }

    pub fn div_assign(&mut self, scalar: &F) {
        for element in self.elements.iter_mut() {
            element.div_assign(scalar)
        }
    }

    pub fn kron_prod(&self, other: &Vector<F>) -> Vector<F> {
        let result = boxed_kronecker_op(
            &self.elements,
            self.elements.len(),
            &other.elements,
            other.elements.len(),
            |lhs: &mut F, rhs: &F| lhs.mul_assign(rhs),
            F::one(),
        );
        Vector { elements: result }
    }

    pub fn kron_sum(&self, other: &Vector<F>) -> Vector<F> {
        let result = boxed_kronecker_op(
            &self.elements,
            self.elements.len(),
            &other.elements,
            other.elements.len(),
            |lhs: &mut F, rhs: &F| lhs.add_assign(rhs),
            F::zero(),
        );
        Vector { elements: result }
    }

    pub fn remove_idxs(&mut self, idxs: &[usize]) {
        *self = Vector::from(
            (0..self.size())
                .zip(self.elements.iter())
                .filter_map(|(i, r)| {
                    if idxs.contains(&i) {
                        None
                    } else {
                        Some(r.clone())
                    }
                })
                .collect_vec(),
        );
    }

    pub fn remove_last(&self) -> Vector<F> {
        let mut elems = self.elements.to_vec();
        let _ = elems.pop();
        Vector::from(elems)
    }
}

impl<F: PseudoField> std::ops::Index<usize> for Vector<F> {
    type Output = F;

    fn index(&self, index: usize) -> &Self::Output {
        self.elements.index(index)
    }
}

impl<F: PseudoField> std::ops::IndexMut<usize> for Vector<F> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.elements.index_mut(index)
    }
}

impl<F: PseudoField> FromIterator<F> for Vector<F> {
    fn from_iter<T: IntoIterator<Item = F>>(iter: T) -> Self {
        Self {
            elements: Vec::from_iter(iter).into(),
        }
    }
}

impl<F: PseudoField> From<Vector<F>> for Vec<F> {
    fn from(value: Vector<F>) -> Self {
        value.elements.to_vec()
    }
}

impl<F: PseudoField> PartialEq for Vector<F> {
    fn eq(&self, other: &Self) -> bool {
        self.elements
            .iter()
            .zip(other.elements.iter())
            .all(|(e1, e2)| e1 == e2)
    }
}

impl<F: PseudoField> From<Vector<F>> for Array1<F> {
    fn from(value: Vector<F>) -> Self {
        Array1::from_iter(value.elements)
    }
}

/// Given 1D arrays (Vectors) $\mathbf{A}$ and $\mathbf{B}$ it computes the
/// $\mathbf{A} \otimes \mathbf{B}$ or $\mathbf{A} \oplus \mathbf{B}$
/// depending on the passed clausure. Takes no assumption over the ['neutral'] parameter.
pub fn boxed_kronecker_op<F: PseudoField, T: Fn(&mut F, &F)>(
    matrix_a: &[F],
    size_a: usize,
    matrix_b: &[F],
    size_b: usize,
    op: T,
    neutral: F,
) -> Box<[F]> {
    let size_c = size_a * size_b;
    let mut result = vec![F::zero(); size_c].into_boxed_slice();

    for (i, a_i) in matrix_a.iter().enumerate().take(size_a) {
        for (j, b_j) in matrix_b.iter().enumerate().take(size_b) {
            let mut elem = neutral.clone();
            op(&mut elem, a_i);
            op(&mut elem, b_j);

            // New index is given by: (i * size_b) + j // Scale up the row number, and move the column.
            let idx_c = i * size_b + j;
            result[idx_c] = elem;
        }
    }

    result
}
