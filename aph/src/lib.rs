pub mod formats;
pub mod linalg;

use linalg::{fields::PseudoField, Vector};

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

    pub fn initial(&self) -> &Vector<F> {
        &self.initial
    }

    pub fn initial_mut(&mut self) -> &mut Vector<F> {
        &mut self.initial
    }

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
    size: usize,
    matrix: Box<[F]>,
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
}

/// Bidiagonal matrix representation.
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
