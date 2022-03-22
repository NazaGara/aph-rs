//! Data structures and algorithms for linear algebra.

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
    elements: Box<[F]>,
}

impl<F: PseudoField> Vector<F> {
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

impl<F: PseudoField> From<Vec<F>> for Vector<F> {
    fn from(vector: Vec<F>) -> Self {
        Self {
            elements: vector.into(),
        }
    }
}
