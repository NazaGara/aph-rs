//! Data structures and algorithms for the representation of the matrix.
use linalg::{Vector, fields::PseudoField};
use ndarray::Array2;
use std::collections::HashMap;

use crate::linalg;
use crate::representation::triangular::TriangularArray;
pub mod bidiagonal;
pub mod sparse;
pub mod triangular;

pub trait Representation<F: PseudoField> {
    /// The size $N$ of the APH representation.
    fn size(&self) -> usize;

    /// Returns the value at the given row and column.
    ///
    /// # Panics
    ///
    /// Panics if the access is out-of-bounds.
    // TODO: Add a view method that does not clone in each get. Wasteful.
    fn get(&self, row: usize, column: usize) -> F;

    /// Sets a value at the the given row and column.
    ///
    /// # Panics
    ///
    /// Panics if the access is out-of-bounds.
    fn set(&mut self, row: usize, column: usize, value: F) -> &mut Self;

    /// Returns the value at the diagonal of the given row.
    fn diagonal(&self, row: usize) -> F {
        self.get(row, row)
    }

    /// The dimension of the generator matrix.
    fn dimension(&self) -> usize {
        self.size() + 1
    }

    fn kron_prod(&self, _other: &Self) -> Self;

    fn kron_sum(&self, _other: &Self) -> Self;

    fn to_array_repr(&self) -> TriangularArray<F>;

    fn row_sum(&self, idx: usize) -> F;

    fn to_absorbing(&self) -> Vector<F> {
        let mut vector = Vector::zeros(self.size());
        for row in 0..self.size() {
            let mut row_sum = self.row_sum(row);
            row_sum.neg_assign();
            vector[row] = row_sum;
        }
        vector
    }

    fn remove_state(&mut self, idx: usize);

    fn is_bidiagonal(&self) -> bool {
        for row in 0..self.size() - 1 {
            let val = self.get(row, row);
            let mut next = self.get(row, row + 1);
            next.neg_assign();
            if val != next {
                return false;
            }
        }
        true
    }

    fn adjacency_map(&self) -> HashMap<usize, Vec<usize>> {
        let size = self.size();
        let mut transitions: Vec<(usize, usize)> = vec![];
        for i in 0..size {
            for j in i + 1..size {
                if !self.get(i, j).is_zero() {
                    transitions.push((i, j));
                }
            }
        }
        let mut map: HashMap<usize, Vec<usize>> = HashMap::new();
        transitions.iter().for_each(|(from, to)| {
            if *to != size + 1 {
                map.entry(*from)
                    .and_modify(|elems| elems.push(*to))
                    .or_insert(vec![*to]);
            }
        });
        map
    }

    fn rate_count(&self) -> HashMap<F, u32> {
        let mut counts = HashMap::new();

        for num in (0..self.size()).map(|i| self.diagonal(i)) {
            *counts.entry(num).or_insert(0) += 1;
        }
        counts
    }
}

/// Computes the [Kronecker Product](https://en.wikipedia.org/wiki/Kronecker_product)
pub fn kronecker_product_array<F: PseudoField>(
    matrix_a: &Array2<F>,
    rows_a: usize,
    cols_a: usize,
    matrix_b: &Array2<F>,
    rows_b: usize,
    cols_b: usize,
    are_triangular: bool,
) -> Array2<F> {
    let rows_c = rows_a * rows_b;
    let cols_c = cols_a * cols_b;
    let mut result = Array2::zeros((rows_c, cols_c));

    for n in 0..rows_a {
        for m in 0..cols_a {
            if n > m && are_triangular {
                continue;
            }
            let a_nm = matrix_a.get((n, m)).unwrap(); // Element in matrix A at (i, j)
            for k in 0..rows_b {
                for l in 0..cols_b {
                    if k > l && are_triangular {
                        continue;
                    }
                    // Position in matrix B at (k, l)
                    let b_kl = matrix_b.get((k, l)).unwrap();
                    let row_c = n * rows_b + k;
                    let col_c = m * cols_b + l;
                    let mut val = a_nm.clone();
                    val.mul_assign(b_kl);
                    *result.get_mut((row_c, col_c)).unwrap() = val;
                }
            }
        }
    }

    result
}
