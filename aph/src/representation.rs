//! Data structures and algorithms for the representation of the matrix.
use linalg::{Vector, fields::PseudoField};
use ndarray::{Array, Array1, Array2, Axis};
use std::collections::HashMap;
use std::io::Write;
use std::{fmt::Display, fs::File};

use crate::linalg;

pub enum RepresentationType {
    Bidiagonal,
    TriangularArray,
    Triangular,
}
pub trait Representation<F: PseudoField> {
    /// The size $N$ of the APH representation.
    fn size(&self) -> usize;

    /// Returns the value at the given row and column.
    ///
    /// # Panics
    ///
    /// Panics if the access is out-of-bounds.
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

    fn to_ta_repr(&self) -> Triangular<F>;

    fn row_sum(&self, idx: usize) -> F;

    fn to_absorbing(&self) -> Vector<F>;

    fn remove_state(&mut self, idx: usize) {
        assert!(
            idx <= self.size(),
            "Attemppting to remove index {:?} out of bounds (size: {:?})",
            idx,
            self.size()
        );
        todo!()
    }
    fn is_type(&self) -> RepresentationType;

    fn is_bidiagonal(&self) -> bool {
        match self.is_type() {
            RepresentationType::Bidiagonal => true,
            RepresentationType::Triangular | RepresentationType::TriangularArray => {
                for row in 0..self.size() - 1 {
                    let val = self.get(row, row);
                    let mut neg_val = self.get(row, row + 1);
                    neg_val.neg_assign();
                    if val != neg_val {
                        return false;
                    }
                }
                true
            }
        }
    }

    fn rate_count(&self) -> HashMap<F, usize> {
        let mut counts = HashMap::new();

        for num in (0..self.size()).map(|i| self.diagonal(i)) {
            *counts.entry(num).or_insert(0) += 1;
        }
        counts
    }
}

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

fn find_next_larger<F: PseudoField>(arr: &[F], value: &F) -> F {
    let mut ret = None;
    for n in arr {
        if n.lt(value) && (ret.is_none() || n.ge(&ret.clone().unwrap())) {
            ret = Some(n.clone());
        }
    }
    ret.unwrap()
}

fn find_next_smaller<F: PseudoField>(arr: &[F], value: &F) -> F {
    let mut ret = None;
    for n in arr {
        if n.gt(value) && (ret.is_none() || n.le(&ret.clone().unwrap())) {
            ret = Some(n.clone());
        }
    }
    ret.unwrap()
}

impl<F: PseudoField> Bidiagonal<F> {
    pub fn new(size: usize) -> Self {
        Self(vec![F::zero(); size].into())
    }

    /// Returns [Some(\lambda_d, \lambda, \lambda_u)] if there exists \lambda, \lambda_d, \lambda_u, such that:
    /// \lambda_d < \lambda < \lambda_u. If there are none, then it gives [None].
    pub fn mid_values(&self) -> Option<(F, F, F)> {
        let rates = &self.0.iter().as_slice();
        let mid = self.0.get(self.size() / 2)?.to_owned();
        let inf = find_next_smaller(rates, &mid);
        let sup = find_next_larger(rates, &mid);
        let mut dif_i = mid.clone();
        dif_i.sub_assign(&inf);
        let mut dif_s = sup.clone();
        dif_s.sub_assign(&mid);
        if mid.eq(&inf) || mid.eq(&sup) || dif_i > rates[0] || dif_s > rates[0] {
            None
        } else {
            Some((inf, mid, sup))
        }
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

    fn is_type(&self) -> RepresentationType {
        RepresentationType::Bidiagonal
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
        if let Some(first) = vector.elements.get_mut(self.size() - 1) {
            *first = rate;
        }
        vector
    }

    fn to_ta_repr(&self) -> Triangular<F> {
        Triangular::from(self)
    }
}

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
        for row in self.matrix.outer_iter() {
            let row_str: Vec<String> = row.iter().map(|x| x.to_string()).collect();
            writeln!(file, "{}", row_str.join(" "))?;
        }
        Ok(())
    }

    // pub fn spa_with_explicit

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
    fn is_type(&self) -> RepresentationType {
        RepresentationType::TriangularArray
    }

    fn get(&self, row: usize, column: usize) -> F {
        self.matrix[[row, column]].clone()
    }

    fn to_array_repr(&self) -> Self {
        self.clone()
    }

    fn to_ta_repr(&self) -> Triangular<F> {
        todo!()
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

    fn to_absorbing(&self) -> Vector<F> {
        let mut vector = Vector::zeros(self.size);
        for row in 0..self.size() {
            let mut row_sum = self.row_sum(row);
            row_sum.neg_assign();
            vector[row] = row_sum;
        }
        vector
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

    fn is_type(&self) -> RepresentationType {
        RepresentationType::Triangular
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

    fn to_absorbing(&self) -> Vector<F> {
        let mut vector = Vector::zeros(self.size);
        for row in 0..self.size() {
            let mut row_sum = self.row_sum(row);
            row_sum.neg_assign();
            vector[row] = row_sum;
        }
        vector
    }

    fn to_ta_repr(&self) -> Triangular<F> {
        self.to_owned()
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
