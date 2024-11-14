pub mod formats;
pub mod linalg;

use std::fmt::Display;

use linalg::{fields::PseudoField, Vector};
use ndarray::{Array, Array2, Axis};

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

    fn to_array_repr(&self) -> TriangularArray<F>;

    fn to_absorbing(&self) -> Vector<F>;

    fn kron_prod(&self, other: &Self) -> TriangularArray<F>;

    fn kron_sum(&self, other: &Self) -> TriangularArray<F>;
}

/// An APH distribution.
#[derive(Debug)]
pub struct Aph<F: PseudoField, R: Representation<F>> {
    initial: Vector<F>,
    repr: R,
}

impl<F: PseudoField, R: Representation<F>> Display for Aph<F, R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "initial: {}. representation: {}", self.initial, self.repr.to_array_repr())
    }
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
    /// result vector each column of the $\mathbf{P}$ matrix.
    /// The initial distribution is then obtained by self.initial.dot($\mathbf{P}$).
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

/// Given two PH distributions $(\overrightarrow{\alpha}, \mathbf{A})$, $(\overrightarrow{\beta}, \mathbf{B})$ of size $m$ and $n$ respectively. Then:
/// the minimum between them is the PH: $(\overrightarrow{\alpha} \oplus \overrightarrow{\beta}, \mathbf{A} \otimes \mathbf{B})$
///
/// Computes $min\{[self], [other]\}$
pub fn min_ph<F: PseudoField, R: Representation<F>>(
    ph1: &Aph<F, R>,
    ph2: &Aph<F, R>,
) -> Aph<F, TriangularArray<F>> {
    let size = ph1.size() * ph2.size();
    let delta = ph1.initial.kron_prod(&ph2.initial);
    let repr_d = ph1
        .repr()
        .to_array_repr()
        .kron_prod(&ph2.repr.to_array_repr());

    assert!(
        repr_d.matrix.shape() == &[size, size],
        "Shape of matrix is {:?}, but Size is {:?}",
        repr_d.matrix.shape(),
        size
    );
    Aph {
        initial: delta,
        repr: repr_d,
    }
}

/// Given two PH distributions $(\overrightarrow{\alpha}, \mathbf{A})$, $(\overrightarrow{\beta}, \mathbf{B})$ of size $m$ and $n$ respectively. Then:
/// the maximum between them is the PH: $( \[ \overrightarrow{\alpha} \otimes \overrightarrow{\beta}, \overrightarrow{\beta}_{n+1}\overrightarrow{\alpha}, \overrightarrow{\alpha}_{m+1}\overrightarrow{\beta} \], \mathbf{D})$
/// where:
/// $$
/// \mathbf{D} =
/// \begin{bmatrix}
/// \mathbf{A} \oplus \mathbf{B} & \mathbf{I}_{m} \otimes \overrightarrow{B} & \overrightarrow{A} \otimes \mathbf{I}_n   \\
/// \mathbf{0}                   & \mathbf{A}                                & \mathbf{0}                                \\
/// \mathbf{0}                   & \mathbf{0}                                & \mathbf{B}
/// \end{bmatrix}
/// $$
///
/// Computes $max\{[self], [other]\}$
pub fn max_ph<F: PseudoField, R: Representation<F>>(
    ph1: &Aph<F, R>,
    ph2: &Aph<F, R>,
) -> Aph<F, TriangularArray<F>> {
    let size = ph1.size() * ph2.size() + ph1.size() + ph2.size();

    let mut delta = ph1.initial.kron_prod(&ph2.initial).elements.to_vec();
    delta.append(&mut vec![F::zero(); ph1.initial.size() + ph2.size()]);

    let kron = ph1
        .repr()
        .to_array_repr()
        .kron_sum(&ph2.repr().to_array_repr());
    let binding = Array::from(ph2.repr().to_absorbing());
    let eye_b = kronecker_product_array(
        &Array::eye(ph1.size()),
        ph1.size(),
        ph1.size(),
        &binding.to_shape((ph2.size(), 1)).unwrap().to_owned(),
        ph2.size(),
        1,
        false,
    );
    let binding = Array::from(ph1.repr().to_absorbing());
    let eye_a = kronecker_product_array(
        &binding.to_shape((ph1.size(), 1)).unwrap().to_owned(),
        ph1.size(),
        1,
        &Array::eye(ph2.size()),
        ph2.size(),
        ph2.size(),
        false,
    );

    let top = ndarray::concatenate(Axis(1), &[kron.matrix.view(), eye_b.view(), eye_a.view()])
        .expect("Something went wrong when the `top` part of the matrix.");

    let mid = ndarray::concatenate(
        Axis(1),
        &[
            Array::zeros((ph1.size(), ph1.size() * ph2.size())).view(),
            ph1.repr().to_array_repr().matrix.view(),
            Array::zeros((ph1.size(), ph2.size())).view(),
        ],
    )
    .expect("Something went wrong when the `mid` part of the matrix.");

    let bot = ndarray::concatenate(
        Axis(1),
        &[
            Array::zeros((ph2.size(), ph1.size() * ph2.size())).view(),
            Array::zeros((ph2.size(), ph1.size())).view(),
            ph2.repr().to_array_repr().matrix.view(),
        ],
    )
    .expect("Something went wrong when the `bot` part of the matrix.");

    let matrix = ndarray::concatenate(Axis(0), &[top.view(), mid.view(), bot.view()])
    .expect("Something went wrong when assembling the arrays `top`, `mid` and `bot`. Please check that the sizes are correct.");

    assert!(
        matrix.shape() == &[size, size],
        "Shape of matrix is {:?}, but Size is {:?}",
        matrix.shape(),
        size
    );

    Aph {
        initial: delta.into(),
        repr: TriangularArray { size, matrix },
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
    fn kron_prod(&self, _other: &Self) -> TriangularArray<F> {
        todo!()
    }
    fn kron_sum(&self, _other: &Self) -> TriangularArray<F> {
        todo!()
    }
    fn to_absorbing(&self) -> Vector<F> {
        todo!()
    }
    fn to_array_repr(&self) -> TriangularArray<F> {
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

    pub fn into_ordered(mut self) -> Self {
        self.0.sort_by(|x, y| y.partial_cmp(x).unwrap());
        self
    }
}

impl<F: PseudoField> From<Bidiagonal<F>> for TriangularArray<F> {
    fn from(value: Bidiagonal<F>) -> Self {
        let size = value.size();
        let mut matrix = Array::from_diag(&Array::from_vec(value.0.to_vec()));

        for row in 0..size - 1 {
            let mut val = matrix.get((row, row)).unwrap().clone();
            val.neg_assign();
            *matrix.get_mut((row, row + 1)).unwrap() = val;
        }

        TriangularArray { size: size, matrix }
    }
}

impl<F: PseudoField> Display for Bidiagonal<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "size: {}. repr: {}",
            self.0.len(),
            TriangularArray::from(self.to_owned()).matrix
        )
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

    fn to_array_repr(&self) -> TriangularArray<F> {
        TriangularArray::from(self.clone())
    }

    fn kron_prod(&self, other: &Bidiagonal<F>) -> TriangularArray<F> {
        self.to_array_repr().kron_prod(&other.to_array_repr())
    }

    // FIX!
    fn kron_sum(&self, other: &Bidiagonal<F>) -> TriangularArray<F> {
        self.to_array_repr().kron_sum(&other.to_array_repr())
    }

    fn diagonal(&self, row: usize) -> F {
        self.0[row].clone()
    }

    fn to_absorbing(&self) -> Vector<F> {
        let mut rate = self.0[self.0.len() - 1].clone();
        rate.neg_assign();
        let mut vector = Vector::zero(self.size());
        if let Some(first) = vector.elements.get_mut(self.size()) {
            *first = rate;
        }
        vector
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

impl<F: PseudoField> Display for TriangularArray<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "size: {}. repr: {}", self.size, self.matrix)
    }
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
        self.matrix.diag()[row].clone()
    }
}

impl<F: PseudoField> Representation<F> for TriangularArray<F> {
    fn size(&self) -> usize {
        self.size
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
}

/// Computes the [Kronecker Product](https://en.wikipedia.org/wiki/Kronecker_product)
fn _kronecker_product<F: PseudoField>(
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

// pub fn _kronecker_product_bidi(
//     matrix_a: &Array2<f64>,
//     rows_a: usize,
//     matrix_b: &Array2<f64>,
//     rows_b: usize,
//     cols_b: usize,
// ) -> Array2<f64> {
//     let mut chunks: ndarray::ArrayBase<ndarray::OwnedRepr<f64>, ndarray::Dim<[usize; 2]>> =
//         Array2::zeros((1, rows_a * rows_b));
//     let mut pre = (rows_b, 0);
//     let mut post = (rows_b, cols_b);
//     for row in 0..rows_a {
//         let shift = Array2::zeros(pre);
//         let shift2 = Array2::zeros(post);
//         let d_r = matrix_a.get((row, row)).unwrap();
//         let elems = ndarray::concatenate(
//             Axis(1),
//             &[
//                 shift.view(),
//                 matrix_b.mapv(|e| e * d_r).view(),
//                 matrix_b.mapv(|e| e * -d_r).view(),
//                 shift2.view(),
//             ],
//         )
//         .unwrap();

//         chunks = ndarray::concatenate(
//             Axis(0),
//             &[chunks.view(), elems.slice(s![.., ..rows_b * rows_a])],
//         )
//         .unwrap();
//         pre.1 += cols_b;
//         post.1 = usize::min(0, (post.1 as i32 - cols_b as i32) as usize);
//     }

//     chunks.slice(s![1.., ..]).to_owned()
// }
