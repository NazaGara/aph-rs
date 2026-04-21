use ndarray::Array2;
use sprs::TriMat;

use crate::{
    linalg::fields::PseudoField,
    representation::{Representation, triangular::TriangularArray},
};

#[derive(Debug)]
pub struct Sparse<F: PseudoField> {
    pub size: usize,       // Matrix size
    pub matrix: TriMat<F>, // Representation
}

impl<F: PseudoField> Sparse<F> {
    pub fn new() -> Self {
        Self {
            size: 0,
            matrix: TriMat::new((0, 0)),
        }
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
}

impl<F: PseudoField> Default for Sparse<F> {
    fn default() -> Self {
        Self {
            size: 0,
            matrix: TriMat::new((0, 0)),
        }
    }
}

impl<F: PseudoField> Representation<F> for Sparse<F> {
    fn size(&self) -> usize {
        self.size
    }

    fn get(&self, row: usize, col: usize) -> F {
        if let Some(idx) = self.matrix.find_locations(row, col).first() {
            self.matrix.data()[idx.0].clone()
        } else {
            F::zero()
        }
    }

    fn set(&mut self, row: usize, col: usize, value: F) -> &mut Self {
        if let Some(idx) = self.matrix.find_locations(row, col).first() {
            self.matrix.set_triplet(*idx, row, col, value);
        } else {
            self.matrix.add_triplet(row, col, value);
        }
        self
    }

    fn kron_prod(&self, other: &Self) -> Self {
        let n = self.size * other.size;
        let mut tri = TriMat::new((n, n));

        for (a_val, (a_r, a_c)) in self.matrix.view() {
            for (b_val, (b_r, b_c)) in other.matrix.view() {
                let mut val = a_val.clone();
                val.mul_assign(b_val);
                let row = a_r * other.size() + b_r;
                let col = a_c * other.size() + b_c;
                tri.add_triplet(row, col, val);
            }
        }

        Self {
            size: n,
            matrix: tri,
        }
    }

    fn kron_sum(&self, other: &Self) -> Self {
        let n = self.size() * other.size();
        let mut tri = TriMat::new((n, n));

        // fst sum
        for (a_val, (a_r, a_c)) in self.matrix.view() {
            for i in 0..other.size() {
                let row = a_r * other.size() + i;
                let col = a_c * other.size() + i;
                tri.add_triplet(row, col, a_val.clone());
            }
        }

        // snd sum
        for (b_val, (b_r, b_c)) in other.matrix.view() {
            for i in 0..self.size() {
                let row = i * other.size() + b_r;
                let col = i * other.size() + b_c;
                tri.add_triplet(row, col, b_val.clone());
            }
        }

        Self {
            size: n,
            matrix: tri,
        }
    }

    fn row_sum(&self, idx: usize) -> F {
        let mut sum = F::zero();
        for (v, (r, _c)) in self.matrix.view() {
            if r == idx {
                sum.add_assign(v);
            }
        }
        sum
    }

    fn remove_state(&mut self, _idx: usize) {
        todo!(
            "The Sparse representation is for intermediate computations, it does not remove entries."
        )
    }

    fn to_array_repr(&self) -> TriangularArray<F> {
        let (rows, cols) = self.matrix.shape();
        let mut matrix = Array2::<F>::zeros((rows, cols));

        for (val, (row, col)) in self.matrix.view() {
            matrix[(row, col)].add_assign(val); // add in case of duplicate entries
        }

        TriangularArray {
            size: self.size,
            matrix,
        }
    }
}
