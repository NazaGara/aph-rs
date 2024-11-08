pub mod formats;
pub mod linalg;

use itertools::Itertools;
use linalg::{fields::PseudoField, Vector};
use ndarray::{Array, Array2, Axis, Order};

// TODO:
// - [ ] Put kronecker product and sum as part of the Representation Trait
// - [ ] On the APH struct, build the max and min operators.

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

    //fn kron_prod(&self, other: &Self) -> Self;
    //fn kron_sum(&self, other: &Self) -> Self;
}

/// An APH distribution.
#[derive(Debug)]
pub struct Aph<F: PseudoField, R: Representation<F>> {
    initial: Vector<F>,
    repr: R,
}

impl<F: PseudoField, R: Representation<F>> Aph<F, R> {
    pub fn new(initial: impl Into<Vector<F>>, repr: R) -> Self {
        let initial = initial.into();
        assert!(
            initial.size() == repr.size(),
            "Initial probability vector has wrong size."
        );
        Self { initial, repr }
    }

    pub fn into_tuple(self) -> (Vector<F>, R) {
        (self.initial, self.repr)
    }

    pub fn diagonal(&self, row: usize) -> F {
        self.repr.diagonal(row)
    }

    /// Returns the Initial Distribution of the APH
    pub fn initial(&self) -> &Vector<F> {
        &self.initial
    }

    pub fn initial_mut(&mut self) -> &mut Vector<F> {
        &mut self.initial
    }

    /// Returns the Initial Distribution of the APH
    pub fn repr(&self) -> &R {
        &self.repr
    }

    pub fn repr_mut(&mut self) -> &mut R {
        &mut self.repr
    }

    pub fn size(&self) -> usize {
        self.repr.size()
    }

    /// Computes
    ///
    /// $$
    /// \frac{1}{\lambda}(\mathbf{A} + \lambda'\mathbf{I})\vec{x}
    /// $$
    fn inplace_multiply_with_vec(
        &self,
        vector: &Vector<F>,
        result: &mut Vector<F>,
        divide: &F,
        add: &F,
    ) {
        assert!(
            vector.size() == result.size(),
            "Dimensions of vectors do not match."
        );
        for row in 0..result.size() {
            result[row] = F::zero();
            for column in row..vector.size() {
                let factor = if row == column {
                    let mut result = self.repr.get(row, column);
                    result.add_assign(add);
                    result
                } else {
                    self.repr.get(row, column)
                };
                let mut add = factor;
                add.mul_assign(&vector[column]);
                result[row].add_assign(&add);
            }
            result[row].div_assign(divide);
        }
    }

    /// Converts the APH into an ordered bidiagonal representation.
    pub fn to_bidiagonal(&self) -> Aph<F, Bidiagonal<F>> {
        // Step 1️⃣: Construct ordered bidiagonal generator matrix.
        let mut bidiagonal = Bidiagonal::new(self.size());
        for row in 0..self.size() {
            // The eigenvalues are the values on the diagonal of the original
            // generator matrix because it is triangular.
            bidiagonal.set(row, self.repr.get(row, row));
        }
        let bidiagonal = bidiagonal.into_ordered();

        // Step 2️⃣: Compute new initial probability vector.
        let mut initial = Vector::zero(self.initial.size());

        let mut vector = Vector::unit(self.size());
        let mut result = Vector::zero(self.size());

        self.inplace_multiply_with_vec(
            &vector,
            &mut result,
            &bidiagonal.diagonal(self.size() - 1),
            &F::zero(),
        );

        initial[self.size() - 1] = self.initial.scalar_product(&result);

        std::mem::swap(&mut vector, &mut result);

        for row in (0..self.size() - 1).rev() {
            let mut divide = bidiagonal.diagonal(row);
            divide.neg_assign();
            let mut add = bidiagonal.diagonal(row + 1);
            add.neg_assign();
            self.inplace_multiply_with_vec(&vector, &mut result, &divide, &add);
            initial[row] = self.initial.scalar_product(&result);
            std::mem::swap(&mut result, &mut vector);
        }

        Aph::new(initial, bidiagonal)
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
        // println!("K: {:?} ==> n: {:?}", value.len(), size);
        let mut t: Triangular<F> = Triangular::new(size);
        for row in 0..size {
            for col in row + 1..size {
                let elem = value[row * col + col].clone();
                // println!("{:?} {:?} --> {:?}", row, col, elem);
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

    fn get(&self, row: usize, column: usize) -> F {
        if row <= column {
            if row == column {
                self.diagonal(row)
            } else {
                // self.matrix[self.idx(row, column)].clone()
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

    pub fn into_ordered(mut self) -> Self {
        self.0.sort_by(|x, y| y.partial_cmp(x).unwrap());
        self
    }

    pub fn kron_product(&self, other: &Bidiagonal<F>) -> Bidiagonal<F> {
        let m1 = self.0.clone();
        let m2 = self.0.clone();
        let result = diag_kronecker_op(
            &m1,
            self.0.len(),
            &m2,
            other.0.len(),
            |lhs: &mut F, rhs: &F| {
                lhs.mul_assign(rhs);
                // lhs.to_owned()
            },
            F::one(),
        );
        Bidiagonal(result)
    }

    pub fn kron_sum(&self, other: &Bidiagonal<F>) -> Bidiagonal<F> {
        let m1 = self.0.clone();
        let m2 = self.0.clone();
        let result = diag_kronecker_op(
            &m1,
            self.0.len(),
            &m2,
            other.0.len(),
            |lhs: &mut F, rhs: &F| {
                lhs.add_assign(rhs);
                // lhs.to_owned()
            },
            F::zero(),
        );
        Bidiagonal(result)
    }
}

impl<F: PseudoField> Representation<F> for Bidiagonal<F> {
    fn size(&self) -> usize {
        self.0.len()
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
}

/// Constructs an initial probability vector of the given size.
///
/// The probability for the first state is set to $1$ and the rest to $0$.
fn new_initial<F: PseudoField>(size: usize) -> Vector<F> {
    let mut initial = Vector::zero(size);
    initial[0] = F::one();
    initial
}

/// Constructs a new [Triangular] APH distribution.
pub fn new_triangular<F: PseudoField>(size: usize) -> Aph<F, Triangular<F>> {
    assert!(size > 0, "Size must be greater than zero.");
    Aph::new(new_initial(size), Triangular::new(size))
}

/// Constructs a new [Bidiagonal] APH distribution.
pub fn new_bidiagonal<F: PseudoField>(size: usize) -> Aph<F, Bidiagonal<F>> {
    assert!(size > 0, "Size must be greater than zero.");
    Aph::new(new_initial(size), Bidiagonal::new(size))
}

pub fn new_triangular_array<F: PseudoField>(size: usize) -> Aph<F, TriangularArray<F>> {
    assert!(size > 0, "Size must be greater than zero.");
    Aph::new(new_initial(size), TriangularArray::new(size))
}

// This representation has to contain the true matrix representation according to the thesis, not maxis efficient one.
#[derive(Clone, Debug)]
pub struct TriangularArray<F> {
    pub size: usize,
    pub matrix: Array2<F>,
}

impl<F: PseudoField + Sized> TriangularArray<F> {
    /// Creates a new triangular representation of the given *size* filled with zeros.
    pub fn new(size: usize) -> Self {
        // We can skip the last row, the last state is an absorbing one
        let matrix = Array2::from_elem((size, size + 1), F::zero());
        Self { size, matrix }
    }

    pub fn set_diagonal(&mut self) {
        for row in 0..self.size {
            // let d_r = self.diagonal(row);
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
        // assert!(row <= self.size, "Out-of-bounds access.");
        // let mut result = F::zero();
        // // for column in row + 1..self.size + 1 {
        //     for column in row + 1..self.size+1 {
        //     result.sub_assign(&self.matrix[[row, column]]);
        // }
        // result
        self.matrix.diag()[row].clone()
    }

    pub fn kron_product(&self, other: &TriangularArray<F>) -> TriangularArray<F> {
        let new_size = self.size * other.size;
        let m1 = self
            .matrix
            .to_shape(((self.size * self.size), Order::RowMajor))
            .unwrap()
            .to_vec()
            .into_boxed_slice();
        let m2 = self
            .matrix
            .to_shape(((other.size * other.size), Order::RowMajor))
            .unwrap()
            .to_vec()
            .into_boxed_slice();
        let result = kronecker_product(&m1, self.size, self.size, &m2, other.size, other.size);
        let matrix_result = Array::from(result)
            .to_shape(((new_size, new_size), Order::RowMajor))
            .unwrap()
            .to_owned();

        TriangularArray {
            size: new_size,
            matrix: matrix_result,
        }
    }

    pub fn kron_sum(&self, other: &TriangularArray<F>) -> TriangularArray<F> {
        let new_size = self.size * other.size;
        let eye_1 = vec![F::zero(); other.size()].into_boxed_slice();
        let eye_2 = vec![F::zero(); self.size()].into_boxed_slice();
        let m1 = self
            .matrix
            .to_shape(((self.size * self.size), Order::RowMajor))
            .unwrap()
            .to_vec()
            .into_boxed_slice();
        let m2 = self
            .matrix
            .to_shape(((other.size * other.size), Order::RowMajor))
            .unwrap()
            .to_vec()
            .into_boxed_slice();

        let result1 = kronecker_product(&m1, self.size, self.size, &eye_1, other.size, other.size);
        let result2 = kronecker_product(&eye_2, self.size, self.size, &m2, other.size, other.size);

        let result: Box<[F]> = result1
            .iter()
            .zip(result2)
            .map(|(r1, r2)| {
                r1.clone().add_assign(&r2);
                r1.to_owned()
            })
            .collect_vec()
            .into_boxed_slice();

        let matrix_result = Array::from(result)
            .to_shape(((new_size, new_size), Order::RowMajor))
            .unwrap()
            .to_owned();

        TriangularArray {
            size: new_size,
            matrix: matrix_result,
        }
    }
}

impl<F: PseudoField> Representation<F> for TriangularArray<F> {
    fn size(&self) -> usize {
        self.size
    }

    fn get(&self, row: usize, column: usize) -> F {
        self.matrix[[row, column]].clone()
    }
}

/// Computes the [Kronecker Product](https://en.wikipedia.org/wiki/Kronecker_product)
fn kronecker_product<F: PseudoField>(
    matrix_a: &Box<[F]>,
    rows_a: usize,
    cols_a: usize,
    matrix_b: &Box<[F]>,
    rows_b: usize,
    cols_b: usize,
) -> Box<[F]> {
    let rows_c = rows_a * rows_b;
    let cols_c = cols_a * cols_b;
    let mut result = vec![F::zero(); rows_c * cols_c].into_boxed_slice();

    for n in 0..rows_a {
        for m in 0..cols_a {
            if n > m {
                continue;
            }
            let a_nm = &matrix_a[n * cols_a + m]; // Element in matrix A at (i, j)
            for k in 0..rows_b {
                for l in 0..cols_b {
                    if k > l {
                        continue;
                    }
                    // Position in matrix B at (k, l)
                    let b_kl = &matrix_b[k * cols_b + l];
                    let row_c = n * rows_b + k;
                    let col_c = m * cols_b + l;
                    result[row_c * cols_c + col_c] = F::one();
                    result[row_c * cols_c + col_c].mul_assign(&a_nm);
                    result[row_c * cols_c + col_c].mul_assign(&b_kl);
                }
            }
        }
    }

    result
}

/// Computes the [Kronecker Sum](https://en.wikipedia.org/wiki/Kronecker_product#Relations_to_other_matrix_operations)
pub fn diag_kronecker_op<F: PseudoField, T: Fn(&mut F, &F) -> ()>(
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
            let mut elem = neutral.clone(); //F::one();
            op(&mut elem, a_i);
            op(&mut elem, b_j);
            // elem.mul_assign(a_i);
            // elem.mul_assign(b_j);
            println!("{:?}", elem);

            // New index is given by:
            // (i * size_b) + j // Scale up the row number, and move the column.
            let idx_c = i * size_b + j;
            result[idx_c] = elem;
        }
    }

    result
}
