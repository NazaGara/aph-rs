//! Data structures and algorithms for linear algebra.

use std::fmt::Display;

use ndarray::{Array, Array1};

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

impl<F: PseudoField> Vector<F> {
    pub fn len(&self) -> usize {
        self.elements.iter().len()
    }

    pub fn zero(size: usize) -> Self {
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
        let mut vector = Vector::zero(size);
        if let Some(first) = vector.elements.get_mut(idx) {
            *first = F::one(); // set the first element to 1
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
            let mut product = lhs.clone();
            product.mul_assign(rhs);
            result.add_assign(&product);
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

impl<F: PseudoField> From<Vector<F>> for Array1<F> {
    fn from(value: Vector<F>) -> Self {
        Array1::from_iter(value.elements)
    }
}

impl<F: PseudoField> From<Vec<F>> for Vector<F> {
    fn from(vector: Vec<F>) -> Self {
        Self {
            elements: vector.into(),
        }
    }
}

/// Given 1D arrays (Vectors) $\mathbf{A}$ and $\mathbf{B}$ it computes the
/// $\mathbf{A} \otimes \mathbf{B}$ or $\mathbf{A} \oplus \mathbf{B}$
/// depending on the passed clausure. Takes no assumption over the ['neutral'] parameter.
pub fn boxed_kronecker_op<F: PseudoField, T: Fn(&mut F, &F) -> ()>(
    matrix_a: &Box<[F]>,
    size_a: usize,
    matrix_b: &Box<[F]>,
    size_b: usize,
    op: T,
    neutral: F,
) -> Box<[F]> {
    let size_c = size_a * size_b;
    let mut result = vec![F::zero(); size_c].into_boxed_slice();

    for i in 0..size_a {
        let a_i = &matrix_a[i]; // Element in diagonal matrix A at (i, i)
        for j in 0..size_b {
            let b_j = &matrix_b[j];
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
