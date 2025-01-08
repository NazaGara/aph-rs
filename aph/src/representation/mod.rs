//! Data structures and algorithms for the representation of the matrix.
use std::fmt::Display;

use linalg::{fields::PseudoField, Vector};
use ndarray::{Array, Array1, Array2, Axis};

use crate::linalg;

pub enum RepresentationType {
    Bidiagonal,
    Triangular,
    TriangularArray,
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

    /// Returns the value at the diagonal of the given row.
    fn diagonal(&self, row: usize) -> F {
        self.get(row, row)
    }

    /// The dimension of the generator matrix.
    fn dimension(&self) -> usize {
        self.size() + 1
    }

    fn kron_prod(&self, _other: &Self) -> TriangularArray<F> {
        todo!()
    }
    fn kron_sum(&self, _other: &Self) -> TriangularArray<F> {
        todo!()
    }

    fn to_absorbing(&self) -> Vector<F>;

    fn to_array_repr(&self) -> TriangularArray<F> {
        todo!()
    }
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
                return true;
            }
        }
    }
}

/// Triangular matrix representation.
///
/// As the Markov chain underlying an APH distribution has no cycles, its infinitesimal
/// generator matrix has the form
///
/// $$
/// \begin{pmatrix}
/// d(1) & \lambda_{1,2} & \cdots & \lambda_{1,N} & \lambda_{1,N+1} \\\\
/// 0 & d(2) & \cdots  & \lambda_{2,N} & \lambda_{2,N+1} \\\\
/// \vdots & \vdots & \ddots & \vdots & \vdots \\\\
/// 0 & 0 & \cdots & d(N) & \lambda_{N,N+1} \\\\
/// 0 & 0 & \cdots & 0 & 0
/// \end{pmatrix}
/// $$
///
/// where the diagonal is given by:
///
/// $$
/// d(i) := -\sum_{k=i + 1}^{N} \lambda_{i,k}
/// $$
///
/// To store the matrix, it thus suffices to store the values for all $\lambda_{i,j}$
/// with $1 \leq i < j \leq N + 1$.
///
/// ⚠️ Rows and columns are 0-indexed in the implementation.
#[derive(Clone, Debug)]
pub struct Triangular<F> {
    pub size: usize,
    pub matrix: Box<[F]>,
}

impl<F: PseudoField> From<Vec<F>> for Triangular<F> {
    fn from(value: Vec<F>) -> Self {
        let size = (-1.0 + f64::floor(f64::sqrt((1 + 8 * value.len()) as f64)) / 2.0) as usize;
        let mut t: Triangular<F> = Triangular::new(size);
        for row in 0..size {
            for col in row + 1..size {
                let elem = value[row * col + col].clone();
                t.set(row, col, elem);
            }
        }
        t
    }
}

impl<F: PseudoField> Triangular<F> {
    /// Creates a new triangular representation of the given *size*.
    pub fn new(size: usize) -> Self {
        let matrix = vec![F::zero(); (size * (size + 1)) / 2].into();
        Self { size, matrix }
    }

    /// Computes the index into the matrix array for the given *row* and *column*.
    ///
    /// # Panics
    ///
    /// Panics if *column* is not greater than *row*.
    fn idx(&self, row: usize, column: usize) -> usize {
        assert!(
            column > row,
            "The *column* must be greater than the *row*. (Row: {}, Column: {})",
            row,
            column
        );
        row * self.size + column - (row + 1) * row / 2 - 1
    }

    /// Sets the value at the given row and column.
    ///
    /// # Panics
    ///
    /// Panics if *column* is not greater than *row* or if the access is out-of-bounds.
    pub fn set(&mut self, row: usize, column: usize, value: impl Into<F>) -> &mut Self {
        self.matrix[self.idx(row, column)] = value.into();
        self
    }

    /// Returns the value at the given row and column.
    ///
    /// # Panics
    ///
    /// Panics if the access is out-of-bounds.
    pub fn get(&self, row: usize, column: usize) -> F {
        if row <= column {
            if row == column {
                self.diagonal(row)
            } else {
                self.matrix[self.idx(row, column)].clone()
            }
        } else {
            assert!(
                row <= self.size && column <= self.size,
                "Out-of-bounds access."
            );
            F::zero()
        }
    }

    /// Computes the value of the diagonal of the given *row*.
    ///
    /// # Panics
    ///
    /// Panics if the access is out-of-bounds
    pub fn diagonal(&self, row: usize) -> F {
        assert!(row <= self.size, "Out-of-bounds access.");
        let mut result = F::zero();
        for column in row + 1..self.size + 1 {
            result.sub_assign(&self.matrix[self.idx(row, column)]);
        }
        result
    }
}

impl<F: PseudoField> Representation<F> for Triangular<F> {
    fn size(&self) -> usize {
        self.size
    }
    fn is_type(&self) -> RepresentationType {
        RepresentationType::Triangular
    }

    fn get(&self, row: usize, column: usize) -> F {
        if row <= column {
            if row == column {
                self.diagonal(row)
            } else {
                self.matrix[row * self.size + column - (row + 1) * row / 2 - 1].clone()
            }
        } else {
            assert!(
                row <= self.size && column <= self.size,
                "Out-of-bounds access."
            );
            F::zero()
        }
    }

    fn to_absorbing(&self) -> Vector<F> {
        todo!()
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

impl<F: PseudoField> Bidiagonal<F> {
    pub fn new(size: usize) -> Self {
        Self(vec![F::zero(); size].into())
    }

    pub fn set(&mut self, row: usize, value: F) -> &mut Self {
        self.0[row] = value;
        self
    }

    pub fn into_ordered(&mut self) {
        self.0.sort_by(|x, y| y.partial_cmp(x).unwrap())
    }
}

impl<F: PseudoField> From<Bidiagonal<F>> for TriangularArray<F> {
    fn from(value: Bidiagonal<F>) -> Self {
        TriangularArray::new_from_diagonal(&Array::from_vec(value.0.to_vec()))
    }
}

impl<F: PseudoField> Display for Bidiagonal<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "size: {}.\nrepr: {}",
            self.0.len(),
            TriangularArray::from(self.to_owned()).matrix
        )
    }
}

impl<F: PseudoField> Representation<F> for Bidiagonal<F> {
    fn size(&self) -> usize {
        self.0.len()
    }
    fn is_type(&self) -> RepresentationType {
        RepresentationType::Bidiagonal
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

    fn to_array_repr(&self) -> TriangularArray<F> {
        TriangularArray::from(self.clone())
    }

    fn kron_prod(&self, other: &Bidiagonal<F>) -> TriangularArray<F> {
        self.to_array_repr().kron_prod(&other.to_array_repr())
    }

    fn kron_sum(&self, other: &Bidiagonal<F>) -> TriangularArray<F> {
        self.to_array_repr().kron_sum(&other.to_array_repr())
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
        let (pre, post) = self.0.split_at_mut(idx);
        let (_e, post) = post.split_first().expect("Something went  wrong");
        self.0 = ([pre, post]).concat().into_boxed_slice();
    }

    fn to_absorbing(&self) -> Vector<F> {
        let mut rate = self.0[self.0.len() - 1].clone();
        rate.neg_assign();
        let mut vector = Vector::zero(self.size());
        if let Some(first) = vector.elements.get_mut(self.size() - 1) {
            *first = rate;
        }
        vector
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
    fn is_type(&self) -> RepresentationType {
        RepresentationType::TriangularArray
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

    fn to_absorbing(&self) -> Vector<F> {
        let mut vector = Vector::zero(self.size);
        for row in 0..self.size() {
            let mut row_sum = F::zero();
            for col in row..self.size() {
                row_sum.add_assign(&self.get(row, col));
            }
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
                    val.mul_assign(&b_kl);
                    *result.get_mut((row_c, col_c)).unwrap() = val;
                }
            }
        }
    }

    result
}