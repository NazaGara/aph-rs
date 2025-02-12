pub mod formats;
pub mod linalg;
pub mod operations;
pub mod representation;

use std::{fmt::Display, time::Instant};

use itertools::Itertools;
use linalg::{
    fields::{
        float64::Float64, interval_field::IntervalField, matrix::Matrix, rational::Rational,
        FromRational, PseudoField,
    },
    Vector,
};
use log::info;
use std::fs::File;
use std::io::{self, Write};

use representation::{Bidiagonal, Representation, TriangularArray, Triangular};
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

    pub fn get_matrix(&self) -> Matrix<F> {
        Matrix::new(self.size(), self.size(), self.repr.to_array_repr().matrix)
    }

    pub fn export_tra_with_labels(&self, filepath: &str) -> io::Result<()> {
        let mut tra_file = File::create(format!("{:?}.tra", filepath).replace("\"", ""))?;
        let mut lab_file = File::create(format!("{:?}.lab", filepath).replace("\"", ""))?;
        let ini_not_zero: Vec<(usize, F)> = self
            .initial
            .elements
            .to_vec()
            .into_iter()
            .enumerate()
            .filter(|(_, e)| !e.is_zero())
            .collect();
        writeln!(tra_file, "ctmc")?;
        // Write initial distributions
        for (i, value) in ini_not_zero.iter() {
            writeln!(tra_file, "0 {} {}", i + 1, value.to_string())?;
        }
        if self.repr().is_bidiagonal() {
            // Write transitions, is each state to the next, with exception of the last state.
            for i in 0..self.size() {
                let mut v = self.repr.get(i, i);
                v.neg_assign();
                writeln!(tra_file, "{} {} {}", i + 1, i + 2, v.to_string())?;
            }
            // Make last state absorbing
            writeln!(tra_file, "{} {} 1", self.size() + 1, self.size() + 1)?;
        } else {
            // Write non-diagonal non-zero entries in representation
            for i in 0..self.size() {
                for j in i..self.size() {
                    let v = self.repr().get(i, j);
                    if i != j && !v.is_zero() {
                        writeln!(tra_file, "{} {} {}", i + 1, j + 1, v.to_string())?;
                    }
                }
            }
            // Write leftover diagonal entries where rowsum is not zero.
            for i in 0..self.size() {
                if !self.repr().row_sum(i).is_zero() {
                    let mut v = self.repr().row_sum(i);
                    v.neg_assign();
                    writeln!(tra_file, "{} {} {}", i + 1, self.size() + 1, v.to_string())?;
                }
            }
            // Make last state absorbing
            writeln!(tra_file, "{} {} 1", self.size() + 1, self.size() + 1)?;
        }
        // Write label file for storm.
        writeln!(lab_file, "#DECLARATION")?;
        writeln!(lab_file, "init done")?;
        writeln!(lab_file, "#END")?;
        writeln!(lab_file, "0 init")?;
        writeln!(lab_file, "{} done", self.size() + 1)?;
        Ok(())
    }

    pub fn export_to_tra(&self, filepath: &str) -> io::Result<()> {
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

    /// Mini solver to check that the new initial distributions match the total of the old initial distributions
    /// and to make sure that all of the new values are stochastically correct.
    fn minisolver_z3(
        &self,
        index: usize,
        ini_z3: &Vec<Real<'_>>,
        model: &Vec<Real<'_>>,
        ctx: &z3::Context,
    ) -> bool {
        // Make sure that values are positive and that the initial probabilities are well distributed.
        // Is easier and precise to use z3.
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
                if delta.eq(&Real::from_real(&ctx, 0, 1)) {
                } else {
                    new_accum += delta;
                    mini_solver.assert(&delta.ge(&Real::from_real(&ctx, 0, 1)))
                };
            })
            .collect_vec();

        mini_solver.assert(&old_accum._eq(&new_accum));
        let res_mini_solver = mini_solver.check();

        let elapsed = time_start.elapsed();

        if res_mini_solver != SatResult::Sat {
            info!("Time Elapsed 'minisolver': {:?}. UNSAT", elapsed);
            return false;
        }
        info!("Time Elapsed 'minisolver': {:?}. SAT", elapsed);
        true
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

    #[allow(dead_code)]
    // FIXME: Remove this method.
    fn _polynomial(&self) -> Polynomial<F> {
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
        Polynomial::new(clauses)
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
    /// The initial distribution is then obtained by self.initial.dot($\mathbf{P}$).
    pub fn spa(&self) -> Aph<F, Bidiagonal<F>> {
        if self.repr().is_bidiagonal() {
            // If is already a bidiagonal, just order the diagonal.
            let mut bidiagonal = Bidiagonal::new(self.size());
            for row in 0..self.size() {
                bidiagonal.set(row, self.repr.get(row, row));
            }
            bidiagonal.into_ordered();
            Aph::new(self.initial.clone(), bidiagonal)
        } else {
            info!("Transforming to Bidiagonal via SPA. Size: {}", self.size());
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
                // info!("Row: {} of {}", row, self.size());
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

impl<F: PseudoField> Aph<F, Bidiagonal<F>> {
    /// Implementation of Algorithm 3.13 from Reza Pulungan's PhD thesis.
    pub fn reduce(&mut self) {
        let mut s_id = 2;
        // let ph = self.spa();
        let old_size = self.size();
        let mut n = self.size();
        let mut polynomial = self.polynomial();
        while s_id <= n {
            // TODO: So, if for s_i is not reduced, then, for s_{i+1} rate_i+1 == rate_i and ini_i+1 is zero, then it will aslo fail for that one
            // The idea is that the polynomial does not change (so the value remains), but what happens for eq3_14
            // Something like [0.2,0.2,0,0] with rates: [-1,-3,-3,-3]
            if self.removable(&polynomial, s_id) {
                if let Some(new_ini) = self.eq3_14(s_id) {
                    // if let Some(new_ini) = self.eq3_14(s_id) {
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

    pub fn eq3_14(&self, index: usize) -> Option<Vector<F>> {
        info!("Solving eq3_14 for index {:?} of {:?}.", index, self.size());
        assert!(2 <= index && index <= self.size(), "wrong index");
        // Initialize all for z3
        let cfg = z3::Config::new();
        let ctx = z3::Context::new(&cfg);

        let ta = Triangular::from(self.repr());
        // let matrix = Matrix::new(self.size(), self.size(), self.repr.to_array_repr().matrix);

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

        let time_start_set_up = Instant::now();
        // Set up the linar equation system
        let solver = Solver::new(&ctx);

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
        lh_accum = lh_accum.simplify();

        ini_z3
            .iter()
            .enumerate()
            .map(|(i, val)| rh_accum += (val * &ones_rhs[i]).simplify())
            .collect_vec();
        rh_accum = rh_accum.simplify();

        solver.assert(&lh_accum._eq(&rh_accum));

        let trim_ta = ta.slice(index).to_owned();
        // let trim_matrix = matrix.slice(index).to_owned();

        for j in 1..index - 1 {
            // Overwriting the value improves runtime.
            let trim_ta = trim_ta.matrix_power(j);
            // let trim_matrix = trim_matrix.matrix_power(j, true);
            let row_sum_lhs = trim_ta.row_sum_sliced(index - 1);
            // let row_sum_lhs = trim_matrix.sum_axis_sliced(index - 1, 1);

            let row_sum_rhs = (0..index)
                .into_iter()
                .map(|i| trim_ta.row_sum(i))
                .collect_vec();
            // let row_sum_rhs = trim_matrix.values.view().sum_axis(Axis(1));

            let row_sum_lhs_z3 = row_sum_lhs
                .into_iter()
                .map(|value| {
                    let (numer, denom) = value.to_rational();
                    Real::from_real_str(&ctx, &numer, &denom)
                        .expect("Something went wrong")
                        .simplify()
                })
                .collect_vec();

            let row_sum_rhs_z3 = row_sum_rhs
                .into_iter()
                .map(|value| {
                    let (numer, denom) = value.to_rational();
                    Real::from_real_str(&ctx, &numer, &denom)
                        .expect("Something went wrong")
                        .simplify()
                })
                .collect_vec();

            let mut lh_accum = Real::from_real(&ctx, 0, 1);
            let mut rh_accum = Real::from_real(&ctx, 0, 1);

            deltas
                .iter()
                .enumerate()
                .map(|(i, d)| lh_accum += (d * &row_sum_lhs_z3[i]).simplify())
                .collect_vec();
            lh_accum = lh_accum.simplify();

            ini_z3
                .iter()
                .enumerate()
                .map(|(i, val)| rh_accum += (val * &row_sum_rhs_z3[i]).simplify())
                .collect_vec();
            rh_accum = rh_accum.simplify();

            solver.assert(&lh_accum._eq(&rh_accum));
        }
        let elapsed_loop = time_start_set_up.elapsed();


        let time_start = Instant::now();
        if solver.check() == SatResult::Sat {
            let elapsed = time_start.elapsed();
            info!(
                "Time Elapsed: 'set up LES': {:?}. 'z3_LES': {:?}. (IDX: {:?}) SAT",
                elapsed_loop, elapsed, index
            );
            let model = solver.get_model().unwrap();

            // Retrieve values from model
            let model = deltas
                .iter()
                .map(|delta| model.eval(delta, true).unwrap())
                .collect_vec();

            if !self.minisolver_z3(index, &ini_z3, &model, &ctx) {
                return None;
            }

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
                "Time Elapsed: 'set up LES': {:?}. 'z3_LES': {:?}. (IDX: {:?}) UNSAT",
                elapsed_loop, elapsed, index
            );
            None
        }
    }

    /// Computes
    ///
    /// $$
    /// R(s) = \vv{\beta}_1 + \vv{\beta}_2 L(\lambda_1) + \dots + \vv{\beta}_i L(\lambda_1) \dots L(\lambda_{i_1})
    /// $$
    ///
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
        let poly = Polynomial::new(clauses);
        move |s: &F| poly.eval(s)
    }

    /// Determines if an specific state can be removed according to Formula 3.11, with the polynomial
    /// $$R_i(s) = \beta_1 + \beta_2 L(\lambda_1) = \cdots + \beta_i L(\lambda_1) \dots L(\lambda_{i-1})$$
    /// Polynomial $R(s)$ is divisible by $L(\lambda_i)$ if $R(-\lambda_i) = 0$.
    /// Additionally, we also consider the note that specifies that states with the
    /// same outgoing rate that the one from the initial state can not be reduced
    fn removable(&self, polynomial: &impl Fn(&F) -> F, index: usize) -> bool {
        if index >= self.size() || self.repr().get(index, index) == self.repr().get(0, 0) {
            false
        } else {
            let time_start = Instant::now();
            let is_zero = polynomial(&self.repr.get(index, index)).is_zero();
            let elapsed = time_start.elapsed();
            info!(
                "R({}) = {} ({}). Elapsed: {:?}.",
                self.repr.get(index, index),
                polynomial(&self.repr.get(index, index)),
                polynomial(&self.repr.get(index, index)).is_zero(),
                elapsed
            );
            is_zero
        }
    }
}

/// Constructs a new [Bidiagonal] APH distribution.
/// Constructs an initial probability vector of the given size.
///
/// The probability for the first state is set to $1$ and the rest to $0$.
pub fn new_bidiagonal<F: PseudoField>(size: usize) -> Aph<F, Bidiagonal<F>> {
    assert!(size > 0, "Size must be greater than zero.");
    Aph::new(Vector::one_and_zeros(0, size), Bidiagonal::new(size))
}

/// Constructs a new [TriangularArray] APH distribution.
/// Constructs an initial probability vector of the given size.
///
/// The probability for the first state is set to $1$ and the rest to $0$.
pub fn new_triangular_array<F: PseudoField>(size: usize) -> Aph<F, TriangularArray<F>> {
    assert!(size > 0, "Size must be greater than zero.");
    Aph::new(Vector::one_and_zeros(0, size), TriangularArray::new(size))
}

/// Constructs a new [Triangular] APH distribution.
/// Constructs an initial probability vector of the given size.
///
/// The probability for the first state is set to $1$ and the rest to $0$.
pub fn new_triangular<F: PseudoField>(size: usize) -> Aph<F, Triangular<F>> {
    assert!(size > 0, "Size must be greater than zero.");
    Aph::new(Vector::one_and_zeros(0, size), Triangular::new(size))
}

impl From<Aph<Rational, Bidiagonal<Rational>>>
    for Aph<IntervalField<Float64>, Bidiagonal<IntervalField<Float64>>>
{
    fn from(value: Aph<Rational, Bidiagonal<Rational>>) -> Self {
        let ini = Vector::from(
            value
                .initial
                .elements
                .iter()
                .map(|e| IntervalField::<Float64>::from(e))
                .collect_vec(),
        );
        let mut bidiagonal = Bidiagonal::new(value.size());
        for i in 0..value.size() {
            let val = value.repr.get(i, i);
            let elem = IntervalField::<Float64>::from_rational(&val.numer(), &val.denom());
            bidiagonal.set(i, elem);
        }
        Aph::new(ini, bidiagonal)
    }
}
