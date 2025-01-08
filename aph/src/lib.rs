pub mod formats;
pub mod linalg;
pub mod representation;

use std::{fmt::Display, time::Instant};

use itertools::Itertools;
use linalg::{
    fields::{dot_product_custom, matrix_power, FromRational, PseudoField, Rational},
    Vector,
};
use log::info;
use ndarray::{s, Array, Array1, Axis};
use std::fs::File;
use std::io::{self, Write};

use representation::{
    kronecker_product_array, Bidiagonal, Representation, Triangular, TriangularArray,
};
use z3::{
    ast::{Ast, Real},
    SatResult, Solver,
};

#[derive(Debug, Clone)]
struct LTerm<F: PseudoField> {
    rate: F,
}

impl<F: PseudoField> Display for LTerm<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // write!(f, "((s + {}) / {})", self.rate, self.rate)
        write!(f, "L({})", self.rate)
    }
}

impl<F: PseudoField> LTerm<F> {
    pub fn new(rate: F) -> Self {
        LTerm { rate }
    }
    pub fn eval(&self, s: &F) -> F {
        let mut numerator = s.clone();
        numerator.add_assign(&self.rate);
        let mut denominator: F = self.rate.clone();
        denominator.inv();
        numerator.mul_assign(&denominator);
        numerator
    }
}

#[derive(Debug, Clone)]
pub struct Polynomial<F: PseudoField> {
    clauses: Vec<(F, Vec<LTerm<F>>)>,
}

impl<F: PseudoField> Display for Polynomial<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for (el, l_terms) in self.clauses.iter() {
            let _ = write!(f, "{} * {} + ", el, l_terms.into_iter().join(" * "));
        }
        write!(f, "\n")
    }
}

impl<F: PseudoField> Polynomial<F> {
    fn new(clauses: Vec<(F, Vec<LTerm<F>>)>) -> Self {
        Polynomial { clauses }
    }
    fn eval(&self, s: &F) -> F {
        let mut result = F::zero();
        for (beta, prod) in self.clauses.iter() {
            let mut cl = beta.to_owned();
            for c in prod {
                cl.mul_assign(&c.eval(s));
            }
            result.add_assign(&cl);
        }
        result
    }
    fn _to_closure(self) -> impl Fn(&F) -> F {
        move |s: &F| self.eval(s)
    }
}

/// An APH distribution.
#[derive(Debug)]
pub struct Aph<F: PseudoField, R: Representation<F>> {
    initial: Vector<F>,
    repr: R,
}

impl<F: PseudoField, R: Representation<F>> Display for Aph<F, R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "initial:\n{}.\nrepresentation:\n{}",
            self.initial,
            self.repr.to_array_repr()
        )
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

    pub fn dump_to_tra(&self, filepath: &str) -> io::Result<()> {
        let ph = self.spa();
        let mut file = File::create(filepath)?;

        let ini_not_zero: Vec<(usize, F)> = ph
            .initial
            .elements
            .to_vec()
            .into_iter()
            .enumerate()
            .filter(|(_, e)| !e.is_zero())
            .collect();
        let n_states = ph.size() + 1; // Size + absorbing state
        let n_transitions = ph.size();
        let n_initials = ini_not_zero.len();

        writeln!(file, "STATES {}", n_states)?;
        writeln!(file, "INITIALS {}", n_initials)?;
        writeln!(file, "TRANSITIONS {}", n_transitions)?;

        // Write initial distributions
        for (i, value) in ini_not_zero.iter() {
            writeln!(file, "{} {}", i + 1, value)?;
        }

        // Write transitions
        for (i, value) in ph.repr.0.to_vec().iter().enumerate() {
            writeln!(file, "{} {} {}", i + 1, i + 2, value)?;
        }

        Ok(())
    }

    pub fn _new_exp(nominator: &str, denom: &str) -> Aph<F, Bidiagonal<F>> {
        let mut repr = Bidiagonal::new(1);
        repr.set(0, F::from_rational(nominator, denom));
        Aph::new(Vector::one_and_zeros(0, 1), repr)
    }

    pub fn _new_erl(phases: usize, nominator: &str, denom: &str) -> Aph<F, Bidiagonal<F>> {
        let mut repr = Bidiagonal::new(phases);
        (0..phases).for_each(|i| {
            repr.set(i, F::from_rational(nominator, denom));
        });
        Aph::new(Vector::one_and_zeros(0, phases), repr)
    }

    /// Determines if an specific state can be removed according to Formula 3.11, with the polynomial
    /// $$R_i(s) = \beta_1 + \beta_2 L(\lambda_1) = \cdots + \beta_i L(\lambda_1) \dots L(\lambda_{i-1})$$
    /// Polynomial $R(s)$ is divisible by $L(\lambda_i)$ if $R(-\lambda_i) = 0$.
    /// Additionally, we also consider the note that specifies that states with the
    /// same outgoing rate that the one from the initial state can not be reduced
    pub fn removable(&self, polynomial: &impl Fn(&F) -> F, index: usize) -> bool {
        if index >= self.size() || self.repr().get(index, index) == self.repr().get(0, 0) {
            // info!("idx: {:?}. Size: {:?}", index, self.size());
            false
        } else {
            info!(
                // "R({}) = {:?} ({})",
                "R({}) = {} ({})",
                self.repr.get(index, index),
                polynomial(&self.repr.get(index, index)),
                polynomial(&self.repr.get(index, index)).is_zero()
            );
            polynomial(&self.repr.get(index, index)).is_zero()
        }
    }

    pub fn eq3_14(&self, index: usize) -> Option<Vector<F>> {
        info!("Solving eq3_14 for index {:?} of {:?}.", index, self.size());
        assert!(2 <= index && index <= self.size(), "wrong index");
        // Initialize all for z3
        let cfg = z3::Config::new();
        let ctx = z3::Context::new(&cfg);
        let matrix = self.repr.to_array_repr().matrix;
        // let trim_matrix = matrix.slice(s![..index, ..index]).to_owned();

        let (to_reduce_ini, rest_ini) = self.initial().elements.split_at(index);

        let ini_z3 = to_reduce_ini
            .iter()
            .map(|value| {
                let (numer, denom) = value.to_rational();
                Real::from_real_str(&ctx, &numer, &denom).expect("Something went wrong")
            })
            .collect_vec();

        // Make placeholders for the ne winitial distribution values
        let deltas: Vec<Real<'_>> = (0..index - 1)
            .into_iter()
            .map(|v| Real::new_const(&ctx, format!("d_{}", v + 1)))
            .collect();

        // Set up the linar equation system
        let solver = Solver::new(&ctx);
        let time_start = Instant::now();

        // unroll the case j=0
        let ones_lhs = vec![(Real::from_real_str(&ctx, "1", "1")).expect("msg"); index - 1];
        let ones_rhs = vec![(Real::from_real_str(&ctx, "1", "1")).expect("msg"); index];

        let mut lh_accum = Real::from_real(&ctx, 0, 1);
        let mut rh_accum = Real::from_real(&ctx, 0, 1);

        deltas
            .iter()
            .enumerate()
            .map(|(i, d)| lh_accum += (d * &ones_lhs[i]).simplify())
            .collect_vec();
        lh_accum.simplify();

        ini_z3
            .iter()
            .enumerate()
            .map(|(i, val)| rh_accum += val * &ones_rhs[i].simplify())
            .collect_vec();
        rh_accum.simplify();

        solver.assert(&lh_accum._eq(&rh_accum));
        let trim_matrix = matrix.slice(s![..index, ..index]).to_owned();
        // for j in 0..index - 1 {
        for j in 1..index - 1 {
            // Overwriting the value improves runtime.
            let trim_matrix = matrix_power(&trim_matrix, j);

            let row_sum_lhs = trim_matrix
                .slice(s![..index - 1, ..index - 1])
                .sum_axis(Axis(1));
            let row_sum_rhs = trim_matrix.sum_axis(Axis(1));

            let row_sum_lhs_z3 = row_sum_lhs
                .into_iter()
                .map(|value| {
                    let (numer, denom) = value.to_rational();
                    Real::from_real_str(&ctx, &numer, &denom).expect("Something went wrong")
                })
                .collect_vec();

            let row_sum_rhs_z3 = row_sum_rhs
                .into_iter()
                .map(|value| {
                    let (numer, denom) = value.to_rational();
                    Real::from_real_str(&ctx, &numer, &denom).expect("Something went wrong")
                })
                .collect_vec();

            let mut lh_accum = Real::from_real(&ctx, 0, 1);
            let mut rh_accum = Real::from_real(&ctx, 0, 1);

            deltas
                .iter()
                .enumerate()
                .map(|(i, d)| lh_accum += (d * &row_sum_lhs_z3[i]).simplify())
                .collect_vec();
            lh_accum.simplify();

            ini_z3
                .iter()
                .enumerate()
                .map(|(i, val)| rh_accum += val * &row_sum_rhs_z3[i].simplify())
                .collect_vec();
            rh_accum.simplify();

            solver.assert(&lh_accum._eq(&rh_accum));
        }

        if solver.check() == SatResult::Sat {
            let elapsed = time_start.elapsed();
            info!(
                "Time Elapsed 'eq3_14': {:?}. (IDX: {:?}) SAT",
                elapsed, index
            );
            let model = solver.get_model().unwrap();

            // Retrieve values from model
            let model = deltas
                .iter()
                .map(|delta| model.eval(delta, true).unwrap())
                .collect_vec();

            // Make sure that values are positive and that the initial probabilities are well distributed.
            // Is easier and more precise to just call z3 than to parse everything.
            //TODO: Measure this and test if is faster with internal representation.
            let time_start = Instant::now();
            let mini_solver = Solver::new(&ctx);
            let mut old_accum = Real::from_real(&ctx, 0, 1);
            let mut new_accum = Real::from_real(&ctx, 0, 1);

            let _ = (0..index)
                .into_iter()
                .map(|i| old_accum += &ini_z3[i])
                .collect_vec();
            let _ = model
                .iter()
                .map(|delta| {
                    new_accum += delta;
                    mini_solver.assert(&delta.ge(&Real::from_real(&ctx, 0, 1)));
                })
                .collect_vec();

            mini_solver.assert(&old_accum._eq(&new_accum));
            let res_mini_solver = mini_solver.check();
            let elapsed = time_start.elapsed();
            info!(
                "Time Elapsed 'minisolver': {:?}. ({:?})",
                elapsed, res_mini_solver
            );
            if res_mini_solver != SatResult::Sat {
                return None;
            }
            // If is SAT, then we have to extend the new initial probabilities to match
            // let mut new_ini = model.iter().map(|e| Rational::from(e)).collect_vec();
            let mut new_ini = model
                .iter()
                .map(|e| {
                    let rat = Rational::from(e);
                    F::from_rational(&rat.numer(), &rat.denom())
                })
                .collect_vec();
            new_ini.extend(rest_ini.to_vec());
            Some(Vector {
                elements: new_ini.into_boxed_slice(),
            })
        } else {
            let elapsed = time_start.elapsed();
            info!(
                "Time Elapsed 'eq3_14': {:?}. (IDX: {:?}) UNSAT",
                elapsed, index
            );
            None
        }
    }

    /// Implementation of Algorithm 3.13 from Reza Pulungan's PhD thesis.
    pub fn reduce(&mut self) {
        let mut s_id = 2;
        // let ph = self.spa();
        let old_size = self.size();
        let mut n = self.size();
        let mut polynomial = self.polynomial();
        while s_id <= n {
            // info!("Trying: {:?}", s_id);
            if self.removable(&polynomial, s_id) {
                if let Some(new_ini) = self.eq3_14(s_id) {
                    self.initial = new_ini;
                    // compute new matrix
                    self.repr.remove_state(s_id);
                    polynomial = self.polynomial();
                    n -= 1;
                } else {
                    s_id += 1;
                }
            } else {
                s_id += 1;
            }
        }
        info!("Reduced size from {:?} to {:?}", old_size, self.size());
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

    pub fn polynomial_v2(&self) -> Polynomial<F> {
        let lambdas: Vec<F> = (0..self.size())
            .into_iter()
            .map(|r| {
                let mut v = self.repr.get(r, r);
                v.neg_assign();
                v
            })
            .collect();

        let mut clauses: Vec<(F, Vec<LTerm<F>>)> = vec![];
        for i in 0..self.size() {
            let b_i = self.initial[i].clone();
            if b_i.eq(&F::zero()) {
                continue;
            }
            let mut prod = vec![];
            for j in (0..i).rev() {
                prod.push(LTerm::new(lambdas.get(j).unwrap().to_owned()));
            }
            clauses.push((b_i, prod));
        }
        // println!("Clauses: {:?}", clauses);
        Polynomial::new(clauses)
    }

    /// Computes
    ///
    /// $$
    /// R(s) = \vv{\beta}_1 + \vv{\beta}_2 L(\lambda_1) + \dots + \vv{\beta}_i L(\lambda_1) \dots L(\lambda_{i_1})
    /// $$
    ///
    /// Assumes that the representation is Bidiagonal.
    pub fn polynomial(&self) -> impl Fn(&F) -> F {
        let lambdas: Vec<F> = (0..self.size())
            .into_iter()
            .map(|r| {
                let mut v = self.repr.get(r, r);
                v.neg_assign();
                v
            })
            .collect();

        let mut clauses: Vec<(F, Vec<LTerm<F>>)> = vec![];
        for i in 0..self.size() {
            let b_i = self.initial[i].clone();
            if b_i.eq(&F::zero()) {
                continue;
            }
            let mut prod = vec![];
            for j in (0..i).rev() {
                prod.push(LTerm::new(lambdas.get(j).unwrap().to_owned()));
            }
            clauses.push((b_i, prod));
        }
        // println!("Clauses: {:?}", clauses);
        let poly = Polynomial::new(clauses);
        move |s: &F| poly.eval(s)
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
    pub fn spa(&self) -> Aph<F, Bidiagonal<F>> {
        info!("Transforming to Bidiagonal via SPA");
        if self.repr().is_bidiagonal() {
            let mut bidiagonal = Bidiagonal::new(self.size());
            for row in 0..self.size() {
                bidiagonal.set(row, self.repr.get(row, row));
            }
            bidiagonal.into_ordered();
            Aph::new(self.initial.clone(), bidiagonal)
        } else {
            // Step 1️⃣: Construct ordered bidiagonal generator matrix.
            let mut bidiagonal = Bidiagonal::new(self.size());
            for row in 0..self.size() {
                // The eigenvalues are the values on the diagonal of the original
                // generator matrix because it is triangular.
                bidiagonal.set(row, self.repr.get(row, row));
            }
            bidiagonal.into_ordered();

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
}

/// Given two PH distributions $(\overrightarrow{\alpha}, \mathbf{A})$, $(\overrightarrow{\beta}, \mathbf{B})$ of size $m$ and $n$ respectively. Then:
/// the convolution of them is the PH: $(\overrightarrow{\delta}, \mathbf{D})$
///
/// $$
/// \mathbf{D} =
/// \begin{bmatrix}
/// \mathbf{A} & \overrightarrow{A} \beta   \\
/// \mathbf{0} & \mathbf{B}
/// \end{bmatrix}
/// $$
///
/// Computes $con\{ph1, ph2\}$
pub fn con_ph<F: PseudoField, R: Representation<F>>(
    ph1: &Aph<F, R>,
    ph2: &Aph<F, R>,
) -> Aph<F, TriangularArray<F>> {
    let size = ph1.size() + ph2.size();
    let mut delta = ph1.initial.elements.to_vec();
    delta.append(&mut vec![F::zero(); ph2.size()]);

    let prod_a = Array1::from(ph1.repr().to_absorbing())
        .into_shape_clone((ph1.size(), 1))
        .expect("msg");
    let prod_b = Array1::from(ph2.initial().elements.to_vec())
        .into_shape_clone((1, ph2.size()))
        .expect("msg");

    // println!("A: {:?} \nB:{:?}", prod_a, prod_b);

    let prod = dot_product_custom(&prod_a, &prod_b);

    // println!("DOT: {:?}", prod.view());

    let top = ndarray::concatenate(
        Axis(1),
        &[ph1.repr.to_array_repr().matrix.view(), prod.view()],
    )
    .expect("Something went wrong when the `top` part of the matrix.");

    // println!("TOP: {}\n{:?}", top, &[ph1.repr.to_array_repr().matrix.view(), prod.view()]);

    let bot = ndarray::concatenate(
        Axis(1),
        &[
            // matches number of rows of B and number of cols form A.
            Array::zeros((ph2.size(), ph1.size())).view(),
            ph2.repr().to_array_repr().matrix.view(),
        ],
    )
    .expect("Something went wrong when the `bot` part of the matrix.");

    let matrix = ndarray::concatenate(Axis(0), &[top.view(), bot.view()])
    .expect("Something went wrong when assembling the arrays `top`, `mid` and `bot`. Please check that the sizes are correct.");

    assert!(
        matrix.shape() == &[size, size],
        "Shape of matrix is {:?}, but actual size is {:?}",
        matrix.shape(),
        size
    );
    Aph {
        initial: delta.into(),
        repr: TriangularArray { size, matrix },
    }
}

/// Given two PH distributions $(\overrightarrow{\alpha}, \mathbf{A})$, $(\overrightarrow{\beta}, \mathbf{B})$ of size $m$ and $n$ respectively. Then:
/// the minimum between them is the PH: $(\overrightarrow{\alpha} \oplus \overrightarrow{\beta}, \mathbf{A} \otimes \mathbf{B})$
///
/// Computes $min\{ph1, ph2\}$
pub fn min_ph<F: PseudoField, R: Representation<F>>(
    ph1: &Aph<F, R>,
    ph2: &Aph<F, R>,
) -> Aph<F, TriangularArray<F>> {
    let size = ph1.size() * ph2.size();
    let delta = ph1.initial.kron_prod(&ph2.initial);
    let repr_d = ph1
        .repr()
        .to_array_repr()
        .kron_sum(&ph2.repr.to_array_repr());

    assert!(
        repr_d.matrix.shape() == &[size, size],
        "Shape of matrix is {:?}, but actual size is {:?}",
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
/// Computes $max\{ph1, ph2\}$
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
