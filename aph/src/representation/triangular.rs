//! Data structures and algorithms for the representation of the matrix.
use linalg::fields::PseudoField;
use ndarray::{Array, Array1, Array2, Axis};
use std::io::Write;
use std::{fmt::Display, fs::File};

use crate::linalg;
use crate::representation::bidiagonal::Bidiagonal;
use crate::representation::{Representation, kronecker_product_array};

// This representation has to contain the true matrix representation according to the thesis, not maxis efficient one.
#[derive(Clone, Debug)]
pub struct TriangularArray<F> {
    pub size: usize,
    pub matrix: Array2<F>,
}

impl<F: PseudoField> Display for TriangularArray<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.matrix)
    }
}

impl<F: PseudoField + Sized> TriangularArray<F> {
    pub fn export_to_csv(&self, filename: &str) -> std::io::Result<()> {
        let mut file = File::create(filename)?;
        write!(file, "[")?;
        for row in self.matrix.outer_iter() {
            let row_str: Vec<String> = row.iter().map(|x| format!("{},", x.to_string())).collect();
            writeln!(file, "[{}],", row_str.join(" "))?;
        }
        write!(file, "]")?;
        Ok(())
    }

    /// Creates a new triangular representation of the given *size* filled with zeros.
    pub fn new(size: usize) -> Self {
        // We can skip the last row, the last state is an absorbing one
        let matrix = Array2::from_elem((size, size + 1), F::zero());
        Self { size, matrix }
    }

    pub fn new_from_diagonal(diagonal: &Array1<F>) -> Self {
        let size = diagonal.len();
        let mut matrix = Array::from_diag(diagonal);

        for row in 0..size - 1 {
            let mut val = matrix.get((row, row)).unwrap().clone();
            val.neg_assign();
            *matrix.get_mut((row, row + 1)).unwrap() = val;
        }

        TriangularArray { size, matrix }
    }

    pub fn set_diagonal(&mut self) {
        for row in 0..self.size {
            let mut d_r = F::zero();
            for column in row + 1..self.size + 1 {
                d_r.sub_assign(&self.matrix[[row, column]]);
            }
            self.set(row, row, d_r);
        }
        self.matrix.remove_index(Axis(1), self.size);
    }

    /// Computes the index into the matrix array for the given *row* and *column*.
    ///
    /// # Panics
    ///
    /// Panics if the access is out-of-bounds.
    fn idx(&self, row: usize, column: usize) -> [usize; 2] {
        assert!(row < self.size && column <= self.size);
        [row, column]
    }

    /// Sets the value at the given row and column.
    ///
    /// # Panics
    ///
    /// Panics if the access is out-of-bounds.
    pub fn set(&mut self, row: usize, column: usize, value: impl Into<F>) -> &mut Self {
        self.matrix[[row, column]] = value.into();
        self
    }

    /// Returns the value at the given row and column.
    ///
    /// # Panics
    ///
    /// Panics if the access is out-of-bounds.
    pub fn get(&self, row: usize, column: usize) -> F {
        self.matrix[self.idx(row, column)].clone()
    }

    /// Computes the value of the diagonal of the given *row*.
    ///
    /// # Panics
    ///
    /// Panics if the access is out-of-bounds
    pub fn diagonal(&self, row: usize) -> F {
        self.matrix.diag()[row].clone()
    }
}

impl<F: PseudoField> Representation<F> for TriangularArray<F> {
    fn size(&self) -> usize {
        self.size
    }

    fn set(&mut self, row: usize, column: usize, value: F) -> &mut Self {
        self.set(row, column, value)
    }

    fn row_sum(&self, idx: usize) -> F {
        let mut sum = F::zero();
        for i in idx..self.size() {
            sum = sum + self.get(idx, i);
        }
        sum
    }

    fn get(&self, row: usize, column: usize) -> F {
        self.matrix[[row, column]].clone()
    }

    fn to_array_repr(&self) -> Self {
        self.clone()
    }

    fn kron_prod(&self, other: &TriangularArray<F>) -> TriangularArray<F> {
        let result = kronecker_product_array(
            &self.matrix,
            self.size,
            self.size,
            &other.matrix,
            other.size,
            other.size,
            true,
        );
        TriangularArray {
            size: self.size * other.size,
            matrix: result,
        }
    }

    fn kron_sum(&self, other: &TriangularArray<F>) -> TriangularArray<F> {
        let eye_1 = Array::eye(other.size());
        let eye_2 = Array::eye(self.size);
        let mut result1 = kronecker_product_array(
            &self.matrix,
            self.size,
            self.size,
            &eye_1,
            other.size,
            other.size,
            true,
        );
        let result2 = kronecker_product_array(
            &eye_2,
            self.size,
            self.size,
            &other.matrix,
            other.size,
            other.size,
            true,
        );
        result1.zip_mut_with(&result2, |l, r| l.add_assign(r));
        TriangularArray {
            size: self.size * other.size,
            matrix: result1,
        }
    }

    fn diagonal(&self, row: usize) -> F {
        self.get(row, row)
    }
    fn remove_state(&mut self, idx: usize) {
        assert!(
            idx <= self.size(),
            "Attemppting to remove index {:?} out of bounds (size: {:?})",
            idx,
            self.size()
        );
        let new_diagonal = &mut self.matrix.diag_mut().to_owned();
        new_diagonal.remove_index(Axis(0), idx);
        *self = TriangularArray::new_from_diagonal(new_diagonal);
    }
}

/// --------
#[derive(Clone, Debug)]
pub struct Triangular<F> {
    pub size: usize,
    pub matrix: Box<[F]>,
}

impl<F: PseudoField> Triangular<F> {
    pub fn new(size: usize) -> Self {
        let matrix = vec![F::zero(); (size * (size + 1)) / 2].into();
        Self { size, matrix }
    }

    pub fn eye(size: usize) -> Self {
        let mut tri = Self::new(size);
        for row in 0..size {
            tri.set(row, row, F::one());
        }
        tri
    }

    pub fn new_with_diagonal(diag: &[F]) -> Self {
        let mut ta = Triangular::new(diag.len());
        #[allow(clippy::needless_range_loop)]
        for i in 0..ta.size - 1 {
            let mut elem = diag[i].clone();
            ta.set(i, i, elem.clone());
            elem.neg_assign();
            ta.set(i, i + 1, elem);
        }
        ta.set(ta.size - 1, ta.size - 1, diag[ta.size - 1].clone());
        ta
    }

    pub fn new_with(size: usize, values: Vec<F>) -> Self {
        Triangular {
            size,
            matrix: values.into_boxed_slice(),
        }
    }

    pub fn idx(&self, row: usize, column: usize) -> usize {
        assert!(
            row <= column,
            "Invalid access to lower triangular element in upper triangular matrix"
        );
        assert!(
            column < self.size,
            "Column index out of bounds. Column: {:?}, size: {:?}",
            column,
            self.size
        );
        assert!(
            row < self.size,
            "Row index out of bounds. Index: {:?}, size: {:?}",
            row,
            self.size
        );
        // Calculate the linear index
        (row * (2 * self.size - row + 1)) / 2 + (column - row)
    }

    pub fn set(&mut self, row: usize, column: usize, value: impl Into<F>) -> &mut Self {
        self.matrix[self.idx(row, column)] = value.into();
        self
    }

    pub fn get(&self, row: usize, column: usize) -> F {
        self.matrix[self.idx(row, column)].clone()
    }

    pub fn row_sum_sliced_from(&self, from_idx: usize, to_idx: usize) -> Vec<F> {
        assert!(
            to_idx <= self.size(),
            "Slice index can not be larger than the size of the representation."
        );
        let mut res = vec![F::zero(); to_idx];
        #[allow(clippy::needless_range_loop)]
        for row in from_idx..to_idx {
            let mut sum = F::zero();
            for column in row..to_idx {
                if column >= to_idx {
                    continue;
                } else {
                    sum.add_assign(&self.get(row, column));
                };
            }
            res[row] = sum;
        }
        res
    }
    pub fn row_sum_sliced(&self, idx: usize) -> Vec<F> {
        assert!(
            idx <= self.size(),
            "Slice index can not be larger than the size of the representation."
        );
        let mut res = vec![F::zero(); idx];
        #[allow(clippy::needless_range_loop)]
        for row in 0..idx {
            let mut sum = F::zero();
            for column in row..idx {
                if column >= idx {
                    continue;
                } else {
                    sum.add_assign(&self.get(row, column));
                };
            }
            res[row] = sum;
        }
        res
    }

    pub fn set_diagonal(&mut self) {
        for row in 0..self.size {
            let mut d_r = F::zero();
            for column in row + 1..self.size {
                d_r.sub_assign(&self.get(row, column));
            }
            self.set(row, row, d_r);
        }

        self.remove_state(self.size - 1);
    }

    pub fn dot_product(&self, other: &Triangular<F>) -> Triangular<F> {
        assert!(
            other.size == self.size,
            "Sizes do not match. {:?} vs {:?}.",
            other.size,
            self.size
        );

        let size = self.size;
        let mut dst: Triangular<F> = Triangular::new(size);

        for i in 0..size {
            for j in i..size {
                for k in i..=j {
                    let mut val = self.get(i, k).clone();
                    let mut dst_val = dst.get(i, j);
                    val.mul_assign(&other.get(k, j));
                    dst_val.add_assign(&val);
                    dst.set(i, j, dst_val);
                }
            }
        }
        dst
    }

    pub fn matrix_power(&self, power: usize) -> Triangular<F> {
        let mut result = Triangular::eye(self.size);
        for _ in 0..power {
            result = result.dot_product(self);
        }
        result
    }

    fn add_elementwise(&mut self, other: &Triangular<F>) {
        assert!(
            other.size == self.size,
            "Sizes do not match. {:?} vs {:?}",
            other.size,
            self.size
        );
        for (i, e) in self.matrix.iter_mut().enumerate() {
            e.add_assign(&other.matrix[i]);
        }
    }

    pub fn slice(&self, idx: usize) -> Triangular<F> {
        assert!(
            idx <= self.size,
            "Slice index has to be large equal that the size of the representation."
        );
        let mut sliced = self.clone();
        for i in (idx..self.size).rev() {
            sliced.remove_state(i);
        }
        sliced
    }

    pub fn from_self_sliced(bidi: &Bidiagonal<F>, index: usize) -> Self {
        Triangular::new_with_diagonal(bidi.0.split_at(index).0)
    }
}

impl<F: PseudoField> From<&Bidiagonal<F>> for Triangular<F> {
    fn from(value: &Bidiagonal<F>) -> Self {
        Triangular::new_with_diagonal(&value.0)
    }
}

impl<F: PseudoField> Representation<F> for Triangular<F> {
    fn diagonal(&self, row: usize) -> F {
        self.get(row, row)
    }

    fn get(&self, row: usize, column: usize) -> F {
        self.get(row, column)
    }

    fn set(&mut self, row: usize, column: usize, value: F) -> &mut Self {
        self.set(row, column, value)
    }

    fn row_sum(&self, idx: usize) -> F {
        let mut sum = F::zero();
        for i in idx..self.size() {
            sum.add_assign(&self.get(idx, i));
        }
        sum
    }

    fn size(&self) -> usize {
        self.size
    }

    fn remove_state(&mut self, idx: usize) {
        assert!(
            idx <= self.size(),
            "Attemppting to remove index {:?} out of bounds (size: {:?})",
            idx,
            self.size()
        );
        // New size after removing the column
        let new_size = self.size - 1;
        let mut new_matrix = Vec::with_capacity(new_size * (new_size + 1) / 2);

        for row in 0..self.size {
            for column in row..self.size {
                if column == idx {
                    // Skip elements in the column to be removed
                    continue;
                }
                if column > idx {
                    // Shift column index left by one for elements after the removed column
                    new_matrix.push(self.get(row, column).clone());
                } else {
                    // Directly push elements before the removed column
                    new_matrix.push(self.get(row, column).clone());
                }
            }
        }

        // Update size and matrix storage
        self.size = new_size;
        self.matrix = new_matrix.into_boxed_slice();
    }

    fn to_array_repr(&self) -> TriangularArray<F> {
        // This can be done faster.
        let mut ta = TriangularArray::new(self.size);

        for n in 0..self.size {
            for m in 0..self.size {
                if n > m {
                    continue;
                }
                ta.set(n, m, self.get(n, m));
            }
        }

        ta
    }
    // FIXME: Out of index
    fn kron_prod(&self, other: &Triangular<F>) -> Triangular<F> {
        let size_c = self.size() * self.size();
        let mut result = Triangular::new(size_c);
        for n in 0..self.size {
            for m in 0..self.size {
                if n > m {
                    continue;
                }
                let a_nm = self.get(n, m); // Element in matrix A at (i, j)
                for k in 0..other.size() {
                    for l in 0..other.size() {
                        if k > l {
                            continue;
                        }
                        // Position in matrix B at (k, l)
                        let b_kl = other.get(k, l);
                        let row_c = n * other.size() + k;
                        let col_c = m * other.size() + l;
                        let mut val = a_nm.clone();
                        val.mul_assign(&b_kl);
                        result.set(row_c, col_c, val);
                    }
                }
            }
        }
        result
    }

    fn kron_sum(&self, other: &Triangular<F>) -> Triangular<F> {
        let eye_1 = Triangular::eye(other.size());
        let eye_2 = Triangular::eye(self.size);
        let mut result1 = self.kron_prod(&eye_1);
        let result2 = eye_2.kron_prod(other);
        result1.add_elementwise(&result2);
        result1
    }
}
