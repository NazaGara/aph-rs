use ndarray::iter::{AxisIter, Iter};
use ndarray::{s, Array1, Array2, Axis, Ix1, Ix2};
use num_traits::One;

use std::fmt::Debug;
use std::ops::{Add, Sub};

use super::float64::Float64;
use super::rational::Rational;
use super::PseudoField;

#[derive(Clone)]
pub struct Matrix<F> {
    pub rows: usize,
    pub cols: usize,
    pub values: Array2<F>,
}

impl<F: PseudoField> Matrix<F> {
    pub fn from_raw_vec(rows: usize, cols: usize, raw_vec: Vec<F>) -> Matrix<F> {
        let data = Array2::from_shape_vec((rows, cols), raw_vec).unwrap();
        return Matrix {
            rows,
            cols,
            values: data,
        };
    }

    pub fn is_upper_triangular(&self) -> bool {
        for j in 0..self.cols - 1 {
            for i in j + 1..self.rows {
                if !self.get(i, j).is_zero() {
                    return false;
                }
            }
        }
        true
    }

    pub fn from_diagonal(diag: &Vec<F>) -> Matrix<F> {
        let size = diag.len();
        let mut mat = Matrix::zeros(size, size);
        for i in 0..size - 1 {
            let mut elem = diag[i].clone();
            mat.set(i, i, elem.clone());
            elem.neg_assign();
            mat.set(i, i + 1, elem);
        }
        mat.set(size - 1, size - 1, diag[size - 1].clone());
        mat
    }

    pub fn new(rows: usize, cols: usize, values: Array2<F>) -> Matrix<F> {
        let shape = values.shape();
        if shape[0] != rows || shape[1] != cols {
            panic!(
                "Wrong array sizes as input {}:{} with {}:{}",
                cols, rows, shape[0], shape[1]
            )
        }
        return Matrix { rows, cols, values };
    }

    pub fn zeros(rows: usize, cols: usize) -> Matrix<F> {
        return Matrix {
            rows,
            cols,
            values: Array2::zeros((rows, cols)),
        };
    }

    pub fn eye(size: usize) -> Matrix<F> {
        return Matrix {
            rows: size,
            cols: size,
            values: Array2::eye(size),
        };
    }

    pub fn dot(&self, other: &Matrix<F>, triangulars: bool) -> Matrix<F> {
        self.check_same_shape(&other);
        let rows = self.rows;
        let cols = other.cols;
        let mut dst: ndarray::ArrayBase<ndarray::OwnedRepr<F>, ndarray::Dim<[usize; 2]>> =
            Array2::zeros((rows, cols));
        if triangulars {
            for i in 0..rows {
                for j in i..cols {
                    // This opt is not better, produces more cache misses.
                    // for j in 0..cols {
                    //     for i in 0..=j {
                    for k in i..=j {
                        let mut val = self.get(i, k).clone();
                        val.mul_assign(&other.get(k, j));
                        dst[(i, j)].add_assign(&val);
                    }
                }
            }
        } else {
            for i in 0..rows {
                for j in 0..cols {
                    for k in 0..self.cols {
                        let mut val = self.get(i, k).clone();
                        val.mul_assign(&other.get(k, j));
                        dst[(i, j)].add_assign(&val);
                    }
                }
            }
        }
        Matrix::new(rows, cols, dst)
    }

    pub fn matrix_power(&self, power: usize, triangulars: bool) -> Matrix<F> {
        if self.rows != self.cols {
            panic!("Matrix must be square to compute power.");
        }
        let mut result = Matrix::eye(self.rows);
        for _ in 0..power {
            result = result.dot(self, triangulars);
        }
        result
    }

    pub fn slice(&self, index: usize) -> Matrix<F> {
        Self::new(
            index,
            index,
            self.values.slice(s![..index, ..index]).to_owned(),
        )
    }

    pub fn sum_axis_sliced(&self, index: usize, axis: usize) -> Array1<F> {
        self.values.slice(s![..index, ..index]).sum_axis(Axis(axis))
    }

    /// Pad the matrix to the next power of two if needed
    pub fn pad_to(&self, new_size: usize) -> Matrix<F> {
        if self.rows == new_size && self.cols == new_size {
            return self.to_owned();
        } else {
            // Create a padded matrix with the new size
            let mut padded_values = Array2::zeros((new_size, new_size));
            padded_values
                .slice_mut(s![..self.rows, ..self.cols])
                .assign(&self.values);

            Matrix::new(new_size, new_size, padded_values)
        }
    }

    /// Remove padding from the matrix to return to original size
    pub fn remove_padding(&self, original_rows: usize, original_cols: usize) -> Matrix<F> {
        let unpadded_values = self
            .values
            .slice(s![..original_rows, ..original_cols])
            .to_owned();
        Matrix::new(original_rows, original_cols, unpadded_values)
    }

    /// Computes the [Kronecker Product](https://en.wikipedia.org/wiki/Kronecker_product)
    pub fn kron(&self, other: &Matrix<F>, are_triangular: bool) -> Matrix<F> {
        let rows_c = self.rows * other.rows;
        let cols_c = self.cols * other.cols;
        let mut result = Self::zeros(rows_c, cols_c);
        for n in 0..self.rows {
            for m in 0..self.cols {
                if n > m && are_triangular {
                    continue;
                }
                // Element in matrix A at (n, m)
                let a_nm = self.get(n, m);
                for k in 0..other.rows {
                    for l in 0..other.cols {
                        if k > l && are_triangular {
                            continue;
                        }
                        // Position in matrix B at (k, l)
                        let b_kl = other.get(k, l);
                        let row_c = n * other.rows + k;
                        let col_c = m * other.cols + l;
                        let mut val = a_nm.clone();
                        val.mul_assign(&b_kl);
                        *result.values.get_mut((row_c, col_c)).unwrap() = val;
                    }
                }
            }
        }
        result
    }

    pub fn kron_add(&self, other: &Matrix<F>) -> Matrix<F> {
        let eye_1 = Matrix::eye(other.rows);
        let eye_2 = Matrix::eye(self.rows);
        let mut result1 = self.kron(&eye_1, true);
        let result2 = eye_2.kron(&other, true);
        result1
            .values
            .zip_mut_with(&result2.values, |l, r| l.add_assign(r));
        result1
    }

    pub fn get(&self, i: usize, j: usize) -> F {
        self.check_args(i, j);
        return self.values.get((i, j)).unwrap().clone();
    }

    pub fn set(&mut self, i: usize, j: usize, value: F) {
        self.check_args(i, j);
        *self.values.get_mut((i, j)).unwrap() = value;
    }

    pub fn iter_all(&self) -> Iter<'_, F, Ix2> {
        return self.values.iter();
    }

    pub fn iter_rows(&self) -> AxisIter<'_, F, Ix1> {
        return self.values.axis_iter(Axis(0));
    }

    pub fn iter_columns(&self) -> AxisIter<'_, F, Ix1> {
        return self.values.axis_iter(Axis(1));
    }

    fn check_args(&self, i: usize, j: usize) {
        if i > self.rows {
            panic!("Overflow of rows: {} {}", i, self.rows)
        }
        if j > self.cols {
            panic!("Overflow of cols: {} {}", j, self.cols)
        }
    }

    fn check_same_cols(&self, rhs: &Self) {
        if self.cols != rhs.cols {
            panic!("Different cols: {} {}", self.cols, rhs.cols)
        }
    }

    fn _check_cols_eq(&self, eq: usize) {
        if self.cols != eq {
            panic!("Different cols: {} {}", self.cols, eq)
        }
    }

    fn check_same_rows(&self, rhs: &Self) {
        if self.rows != rhs.rows {
            panic!("Different rows: {} {}", self.rows, rhs.rows)
        }
    }

    fn _check_rows_eq(&self, eq: usize) {
        if self.rows != eq {
            panic!("Different rows: {} {}", self.rows, eq)
        }
    }

    fn check_same_shape(&self, rhs: &Self) {
        self.check_same_cols(rhs);
        self.check_same_rows(rhs);
    }
}

impl<F: PseudoField> Debug for Matrix<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Matrix [{}:{}] {:?} -> \n{}",
            self.cols,
            self.rows,
            self.values.shape(),
            &self.values,
        )
    }
}

impl Matrix<Rational> {
    pub fn add(&self, b: &Matrix<Rational>) -> Matrix<Rational> {
        self.check_same_cols(&b);
        let r = (&self.values).add(&b.values);
        return Matrix {
            cols: self.cols,
            rows: self.rows,
            values: r,
        };
    }

    pub fn sub(&self, b: &Matrix<Rational>) -> Matrix<Rational> {
        self.check_same_cols(&b);
        let r = (&self.values).sub(&b.values);
        return Matrix {
            cols: self.cols,
            rows: self.rows,
            values: r,
        };
    }

    pub fn strassen(&self, other: &Matrix<Rational>) -> Matrix<Rational> {
        let og_size = self.rows.clone();
        // Pad it to an even number.
        let matrix = if self.rows % 2 != 0 {
            self.pad_to(self.rows + 1)
        } else {
            self.to_owned()
        };
        let result = matrix._strassen(other, true);
        result.remove_padding(og_size, og_size);
        result
    }
    /// Strassen's matrix multiplication algorithm
    fn _strassen(&self, other: &Matrix<Rational>, is_ut: bool) -> Matrix<Rational> {
        let n = self.rows;
        // Base case for small matrices
        if n == 1 {
            let values = Array2::from_elem(
                (1, 1),
                self.values[[0, 0]].clone() * other.values[[0, 0]].clone(),
            );
            return Matrix::new(1, 1, values);
        }
        if n <= 32 {
            return self.dot(&other, is_ut);
        }

        // Split matrix
        let new_size = n / 2;
        if !is_ut {
            let a11 = self.values.slice(s![..new_size, ..new_size]).to_owned();
            let a12 = self.values.slice(s![..new_size, new_size..]).to_owned();
            let a21 = self.values.slice(s![new_size.., ..new_size]).to_owned();
            let a22 = self.values.slice(s![new_size.., new_size..]).to_owned();
            let b11 = other.values.slice(s![..new_size, ..new_size]).to_owned();
            let b12 = other.values.slice(s![..new_size, new_size..]).to_owned();
            let b21 = other.values.slice(s![new_size.., ..new_size]).to_owned();
            let b22 = other.values.slice(s![new_size.., new_size..]).to_owned();

            // Recursive calls for Strassen's algorithm
            let m1 = Matrix::new(new_size, new_size, a11.clone() + a22.clone())._strassen(
                &Matrix::new(new_size, new_size, b11.clone() + b22.clone()),
                false,
            );
            let m2 = Matrix::new(new_size, new_size, a21.clone() + a22.clone())
                ._strassen(&Matrix::new(new_size, new_size, b11.clone()), false);
            let m3 = Matrix::new(new_size, new_size, a11.clone())._strassen(
                &Matrix::new(new_size, new_size, b12.clone() - b22.clone()),
                false,
            );
            let m4 = Matrix::new(new_size, new_size, a22.clone())._strassen(
                &Matrix::new(new_size, new_size, b21.clone() - b11.clone()),
                false,
            );
            let m5 = Matrix::new(new_size, new_size, a11.clone() + a12.clone())
                ._strassen(&Matrix::new(new_size, new_size, b22.clone()), false);
            let m6 = Matrix::new(new_size, new_size, a21 - a11)
                ._strassen(&Matrix::new(new_size, new_size, b11 + b12), false);
            let m7 = Matrix::new(new_size, new_size, a12 - a22)
                ._strassen(&Matrix::new(new_size, new_size, b21 + b22), false);

            // Combine results into quadrants
            let c11 = &m1.values + &m4.values - &m5.values + &m7.values;
            let c12 = &m3.values + &m5.values;
            let c21 = &m2.values + &m4.values;
            let c22 = &m1.values - &m2.values + &m3.values + &m6.values;

            // Merge the quadrants into a single result matrix
            let mut result = Array2::zeros((n, n));
            result.slice_mut(s![..new_size, ..new_size]).assign(&c11);
            result.slice_mut(s![..new_size, new_size..]).assign(&c12);
            result.slice_mut(s![new_size.., ..new_size]).assign(&c21);
            result.slice_mut(s![new_size.., new_size..]).assign(&c22);

            Matrix::new(n, n, result)
        } else {
            let mut minus_one = Rational::one();
            minus_one.neg_assign();

            let a11 = self.values.slice(s![..new_size, ..new_size]).to_owned();
            let a12 = self.values.slice(s![..new_size, new_size..]).to_owned();
            let a22 = self.values.slice(s![new_size.., new_size..]).to_owned();
            let b11 = other.values.slice(s![..new_size, ..new_size]).to_owned();
            let b12 = other.values.slice(s![..new_size, new_size..]).to_owned();
            let b22 = other.values.slice(s![new_size.., new_size..]).to_owned();

            // Recursive calls for Strassen's algorithm
            let m1 = Matrix::new(new_size, new_size, &a11 + &a22)
                ._strassen(&Matrix::new(new_size, new_size, &b11 + &b22), false);
            let m2 = Matrix::new(new_size, new_size, a22.clone())
                ._strassen(&Matrix::new(new_size, new_size, b11.clone()), false);
            let m3 = Matrix::new(new_size, new_size, a11.clone())
                ._strassen(&Matrix::new(new_size, new_size, &b12 - &b22), false);
            let m4 = Matrix::new(new_size, new_size, a22.clone())._strassen(
                &Matrix::new(new_size, new_size, &b11 * minus_one.clone()),
                false,
            );
            let m5 = Matrix::new(new_size, new_size, &a11 + &a12)
                ._strassen(&Matrix::new(new_size, new_size, b22.clone()), false);
            let m6 = Matrix::new(new_size, new_size, &a11 * minus_one)
                ._strassen(&Matrix::new(new_size, new_size, &b11 + &b12), false);
            let m7 = Matrix::new(new_size, new_size, &a12 - &a22)
                ._strassen(&Matrix::new(new_size, new_size, b22), false);

            // Combine results into quadrants
            let c11 = &m1.values + &m4.values - &m5.values + &m7.values;
            let c12 = &m3.values + &m5.values;
            let c21 = &m2.values + &m4.values;
            let c22 = &m1.values - &m2.values + &m3.values + &m6.values;

            // Merge the quadrants into a single result matrix
            let mut result = Array2::zeros((n, n));
            result.slice_mut(s![..new_size, ..new_size]).assign(&c11);
            result.slice_mut(s![..new_size, new_size..]).assign(&c12);
            result.slice_mut(s![new_size.., ..new_size]).assign(&c21);
            result.slice_mut(s![new_size.., new_size..]).assign(&c22);

            Matrix::new(n, n, result)
        }
    }

    pub fn matrix_power_strassen(&self, power: usize) -> Matrix<Rational> {
        if self.rows != self.cols {
            panic!("Matrix must be square to compute power.");
        }
        let mut result: Matrix<Rational> = Matrix::eye(self.rows);
        for _ in 0..power {
            result = result._strassen(&self, false);
        }
        result
    }
}

impl Matrix<Float64> {
    pub fn add(&self, b: &Matrix<Float64>) -> Matrix<Float64> {
        self.check_same_cols(&b);
        let r = (&self.values).add(&b.values);
        return Matrix {
            cols: self.cols,
            rows: self.rows,
            values: r,
        };
    }

    pub fn sub(&self, b: &Matrix<Float64>) -> Matrix<Float64> {
        self.check_same_cols(&b);
        let r = (&self.values).sub(&b.values);
        return Matrix {
            cols: self.cols,
            rows: self.rows,
            values: r,
        };
    }

    pub fn strassen(&self, other: &Matrix<Float64>) -> Matrix<Float64> {
        let og_size = self.rows.clone();
        // Pad it to an even number.
        let matrix = if self.rows % 2 != 0 {
            self.pad_to(self.rows + 1)
        } else {
            self.to_owned()
        };
        let result = matrix._strassen(other);
        result.remove_padding(og_size, og_size);
        result
    }
    /// Strassen's matrix multiplication algorithm
    fn _strassen(&self, other: &Matrix<Float64>) -> Matrix<Float64> {
        let n = self.rows;
        // Base case for small matrices
        if n == 1 {
            let values = Array2::from_elem(
                (1, 1),
                self.values[[0, 0]].clone() * other.values[[0, 0]].clone(),
            );
            return Matrix::new(1, 1, values);
        }
        if n <= 32 {
            return self.dot(&other, false);
        }
        let new_size = n / 2;

        let a11 = self.values.slice(s![..new_size, ..new_size]).to_owned();
        let a12 = self.values.slice(s![..new_size, new_size..]).to_owned();
        let a21 = self.values.slice(s![new_size.., ..new_size]).to_owned();
        let a22 = self.values.slice(s![new_size.., new_size..]).to_owned();

        let b11 = other.values.slice(s![..new_size, ..new_size]).to_owned();
        let b12 = other.values.slice(s![..new_size, new_size..]).to_owned();
        let b21 = other.values.slice(s![new_size.., ..new_size]).to_owned();
        let b22 = other.values.slice(s![new_size.., new_size..]).to_owned();

        // Recursive calls for Strassen's algorithm
        let m1 = Matrix::new(new_size, new_size, a11.clone() + a22.clone())
            ._strassen(&Matrix::new(new_size, new_size, b11.clone() + b22.clone()));
        let m2 = Matrix::new(new_size, new_size, a21.clone() + a22.clone())
            ._strassen(&Matrix::new(new_size, new_size, b11.clone()));
        let m3 = Matrix::new(new_size, new_size, a11.clone())._strassen(&Matrix::new(
            new_size,
            new_size,
            b12.clone() - b22.clone(),
        ));
        let m4 = Matrix::new(new_size, new_size, a22.clone())._strassen(&Matrix::new(
            new_size,
            new_size,
            b21.clone() - b11.clone(),
        ));
        let m5 = Matrix::new(new_size, new_size, a11.clone() + a12.clone())
            ._strassen(&Matrix::new(new_size, new_size, b22.clone()));
        let m6 = Matrix::new(new_size, new_size, a21.clone() - a11.clone())
            ._strassen(&Matrix::new(new_size, new_size, b11.clone() + b12.clone()));
        let m7 = Matrix::new(new_size, new_size, a12.clone() - a22.clone())
            ._strassen(&Matrix::new(new_size, new_size, b21.clone() + b22.clone()));

        // Combine results into quadrants
        let c11 = &m1.values + &m4.values - &m5.values + &m7.values;
        let c12 = &m3.values + &m5.values;
        let c21 = &m2.values + &m4.values;
        let c22 = &m1.values - &m2.values + &m3.values + &m6.values;

        // Merge the quadrants into a single result matrix
        let mut result = Array2::zeros((n, n));
        result.slice_mut(s![..new_size, ..new_size]).assign(&c11);
        result.slice_mut(s![..new_size, new_size..]).assign(&c12);
        result.slice_mut(s![new_size.., ..new_size]).assign(&c21);
        result.slice_mut(s![new_size.., new_size..]).assign(&c22);

        Matrix::new(n, n, result)
    }

    pub fn matrix_power_strassen(&self, power: usize) -> Matrix<Float64> {
        if self.rows != self.cols {
            panic!("Matrix must be square to compute power.");
        }
        let mut result: Matrix<Float64> = Matrix::eye(self.rows);
        for _ in 0..power {
            result = result._strassen(&self);
        }
        result
    }
}
