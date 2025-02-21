pub mod coxian;
pub mod formats;
pub mod linalg;
pub mod operations;
pub mod representation;

use std::{fmt::Display, time::Instant};

use itertools::Itertools;
use linalg::{
    fields::{rational::Rational, PseudoField},
    Vector,
};

use log::{info, warn};
use std::fs::File;
use std::io::{self, Write};

use representation::{Bidiagonal, Representation, Triangular, TriangularArray};
use z3::{
    ast::{Ast, Real},
    SatResult, Solver,
};

#[derive(Debug, Clone, PartialEq, Eq)]
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
        // let mut denominator: F = self.rate.clone();
        numerator.div_assign(&self.rate);
        
        // denominator.inv_assign();
        
        // numerator.mul_assign(&denominator);
        numerator
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
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

    fn _number_of_clauses(&self) -> usize {
        self.clauses.len()
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

    pub fn _export_to_tra(&self, filepath: &str) -> io::Result<()> {
        let mut file = File::create(format!("{:?}_exp.tra", filepath).replace("\"", ""))?;

        let ini_not_zero: Vec<(usize, F)> = self
            .initial
            .elements
            .to_vec()
            .into_iter()
            .enumerate()
            .filter(|(_, e)| !e.is_zero())
            .collect();

        let n_states = self.size() + 1; // Size + absorbing state
        let n_transitions = self.size() + 1; //  + absorbing loop
        let n_initials = ini_not_zero.len();
        writeln!(file, "STATES {}", n_states)?;
        writeln!(file, "INITIALS {}", n_initials)?;
        writeln!(file, "TRANSITIONS {}", n_transitions)?;

        // Write initial distributions
        for (i, value) in ini_not_zero.iter() {
            writeln!(file, "{} {}", i, value.to_string())?;
        }
        if self.repr().is_bidiagonal() {
            // Write transitions, is each state to the next, with exception of the last state.
            for i in 0..self.size() {
                let mut v = self.repr.get(i, i);
                v.neg_assign();
                writeln!(file, "{} {} {}", i, i + 1, v.to_string())?;
            }
            // Make last state absorbing
            writeln!(file, "{} {} 1", self.size(), self.size())?;
        } else {
            // Write non-diagonal non-zero entries in representation
            for i in 0..self.size() {
                for j in i..self.size() {
                    let v = self.repr().get(i, j);
                    if i != j && !v.is_zero() {
                        writeln!(file, "{} {} {}", i, j, v.to_string())?;
                    }
                }
            }
            // Write leftover diagonal entries where rowsum is not zero.
            for i in 0..self.size() {
                if !self.repr().row_sum(i).is_zero() {
                    let mut v = self.repr().row_sum(i);
                    v.neg_assign();
                    writeln!(file, "{} {} {}", i, self.size(), v.to_string())?;
                }
            }
            // Make last state absorbing
            writeln!(file, "{} {} 1", self.size(), self.size())?;
        }
        Ok(())
    }

    pub fn _new_exp(numerator: &str, denom: &str) -> Aph<F, Bidiagonal<F>> {
        let mut repr = Bidiagonal::new(1);
        repr.set(0, F::from_rational(numerator, denom));
        Aph::new(Vector::one_and_zeros(0, 1), repr)
    }

    pub fn _new_erl(phases: usize, numerator: &str, denom: &str) -> Aph<F, Bidiagonal<F>> {
        let mut repr = Bidiagonal::new(phases);
        (0..phases).for_each(|i| {
            repr.set(i, F::from_rational(numerator, denom));
        });
        Aph::new(Vector::one_and_zeros(0, phases), repr)
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
            let time_start = Instant::now();
            // Step 1️⃣: Construct ordered bidiagonal generator matrix.
            let mut bidiagonal = Bidiagonal::new(self.size());
            for row in 0..self.size() {
                // The eigenvalues are the values on the diagonal of the original
                // generator matrix because it is triangular.
                bidiagonal.set(row, self.repr.get(row, row));
            }
            bidiagonal.into_ordered();

            // Step 2️⃣: Compute new initial probability vector.
            let mut initial = Vector::zeros(self.initial.size());

            let mut vector = Vector::unit(self.size());
            let mut result = Vector::zeros(self.size());

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
            let elapsed = time_start.elapsed();
            info!("Time Elapsed 'spa': {:?}. Size: {:?}", elapsed, self.size());
            Aph::new(initial, bidiagonal)
        }
    }
}

impl<F: PseudoField> Aph<F, Bidiagonal<F>> {
    /// Implementation of Algorithm 3.13 from Reza Pulungan's PhD thesis.
    pub fn reduce(&mut self) -> usize {
        // Starts from state at index 1, Thesis does it from 2, but uses 1-indexed states.
        let mut s_id = 1;
        let old_size = self.size();
        let mut n = self.size();
        while s_id < n {
            let polynomial = self.polynomial(s_id);
            let is_removable = {
                let val = self.removable(&polynomial, s_id);
                val.is_almost_zero()
            };
            
            if is_removable {
                // if let Some(new_ini) = self.eq3_14_z3(s_id) {
                if let Some(new_ini) = self.eq3_14_f(s_id) {
                    // Replace initial distribution
                    self.initial = new_ini;
                    // Compute new matrix removing state
                    self.repr.remove_state(s_id);
                    n -= 1;
                } else {
                    s_id += 1;
                }
            } else {
                s_id += 1;
            }
        }
        info!("Reduced size from {:?} to {:?}", old_size, self.size());
        old_size - self.size()
    }

    /// Mini solver to check that the new initial distributions match the total of the old initial distributions
    /// and to make sure that all of the new values are stochastically correct.
    #[allow(dead_code)]
    fn minisolver(&self, ini: &Vec<F>, model: &Vec<F>) -> bool {
        let time_start = Instant::now();
        let mut pre_accum = F::zero();
        let mut post_accum = F::zero();
        let _ = ini
            .into_iter()
            .map(|val| pre_accum.add_assign(&val))
            .collect_vec();

        for delta in model.iter() {
            let mut val = delta.clone();
            val.is_almost_zero_and_correct();
            if delta.lt(&F::zero()) {
                // If field is not rational, then roundoffs can produce negative values, which then may get cancelled.
                // Don't abort but warn.
                warn!("Negative Value: {:?}.", delta);
                // let elapsed = time_start.elapsed();
                // info!("Time Elapsed 'minisolver': {:?}. UNSAT", elapsed);
                // return false;
            }
            post_accum.add_assign(&val);
        }
        let elapsed = time_start.elapsed();


        if !pre_accum.cmp_eq(&post_accum) {
            info!("Time Elapsed 'minisolver': {:?}. UNSAT", elapsed);
            return false;
        }
        info!("Time Elapsed 'minisolver': {:?}. SAT", elapsed);
        true
    }

    /// Compute Eq 3.14 from Reza's Thesis,
    /// Note that the element to be reduced is located at the last position of the variable
    /// ['to_reduce_ini'], located at position ['index+1'] in the representation, thus, we want to see if we can
    /// reduce from ['index+1'] states to ['index'] states.
    pub fn eq3_14_f(&self, index: usize) -> Option<Vector<F>> {
        info!("Solving eq3_14 for index {:?} of {:?}.", index, self.size());
        assert!(1 <= index && index <= self.size(), "wrong index");

        // Take at most the index + 1 size of the matrix.
        let ta = Triangular::from(self.repr()).slice(index + 1);
        let (to_reduce_ini, rest_ini) = self.initial().elements.split_at(index + 1);

        let mut model: Vec<F> = vec![F::zero(); index];

        // Set up the linar equation system
        let time_start = Instant::now();

        let mut mut_ta: Triangular<F> = Triangular::eye(index + 1);

        for _j in 1..index {
            // I have to elevate to the power of j, which means, do a new dot product with the fixed value and overwrite the trimmed matrix.
            mut_ta = mut_ta.dot_product(&ta);

            let row_sum_lhs = mut_ta.row_sum_sliced(index);
            let row_sum_rhs = (0..index + 1) //(0..index)
                .into_iter()
                .map(|i| mut_ta.row_sum(i))
                .collect_vec();

            // Now, next iteration, the delta_idx will say which one I have to solve,
            // Which will be the (initial[delta_idx+1] * row-sum-rhs[delta_idx+1] - sum {initial * delta_values}) / rwo-sum-lhs[delta_idx]

            // Accummulate rhs
            let delta_idx = index - _j;
            let mut new_rhs: F = F::zero();
            to_reduce_ini
                .iter()
                .enumerate()
                .map(|(i, val)| new_rhs.add_assign(&(val.clone() * row_sum_rhs[i].clone())))
                .collect_vec();

            // Substract values from the deltas that we already compute.
            model
                .iter()
                .enumerate()
                .map(|(i, e)| new_rhs.sub_assign(&(row_sum_lhs[i].clone() * e.clone())))
                .collect_vec();

            let mut val = new_rhs.clone();
            val.div_assign(&row_sum_lhs[delta_idx]);
            
            model[delta_idx] = val;
        }

        // Do for the last equation with all deltas, there are no factors for the deltas.
        let delta_idx = 0;
        let mut val: F = F::zero();
        to_reduce_ini
            .iter()
            .map(|ini| val.add_assign(&ini))
            .collect_vec();

        model.iter().map(|e| val.sub_assign(e)).collect_vec();
        model[delta_idx] = val;
        
        let elapsed = time_start.elapsed();
        
        // The LES has solution, by Lemmas 3.8 and 3.9.
        // Check that system will have solution, but I dont know if wee can determine this before seeing the whole LES.

        info!("Time Elapsed: 'LES': {:?}. (IDX: {:?}) SAT", elapsed, index);

        if !self.minisolver(&to_reduce_ini.to_vec(), &model) {
            return None;
        }

        let mut new_ini = model.clone();
        new_ini.extend(rest_ini.to_vec());
        Some(Vector {
            elements: new_ini.into_boxed_slice(),
        })
    }

    /// Mini solver to check that the new initial distributions match the total of the old initial distributions
    /// and to make sure that all of the new values are stochastically correct.
    #[allow(dead_code)]
    fn minisolver_z3(&self, ini: &Vec<F>, model: &Vec<Real<'_>>) -> bool {
        let time_start = Instant::now();
        let mut pre_accum = F::zero();
        let mut post_accum = F::zero();
        let _ = ini
            .into_iter()
            .map(|val| pre_accum.add_assign(&val))
            .collect_vec();

        for delta in model.iter() {
            let rat = Rational::from(delta);
            let mut val = F::from_rational(&rat.numer(), &rat.denom());
            val.is_almost_zero_and_correct();
            if val.lt(&F::zero()) {
                // If field is not rational, then roundoffs can produce negative values, which then gets cancelled.
                warn!("Negative Value: {:?} --> {:?}", rat, val);
                // let elapsed = time_start.elapsed();
                // info!("Time Elapsed 'minisolver': {:?}. UNSAT", elapsed);
                // return false;
            }

            post_accum.add_assign(&val);
        }
        let elapsed = time_start.elapsed();

        if !pre_accum.cmp_eq(&post_accum) {
            info!("Time Elapsed 'minisolver': {:?}. UNSAT", elapsed);
            return false;
        }
        info!("Time Elapsed 'minisolver': {:?}. SAT", elapsed);
        true
    }

    /// Compute Eq 3.14 from Reza's Thesis,
    /// Note that the element to be reduced is located at the last position of the variable
    /// ['to_reduce_ini'], located at position ['index+1'] in the representation, thus, we want to see if we can
    /// reduce from ['index+1'] states to ['index'] states.
    #[allow(dead_code)]
    pub fn eq3_14_z3(&self, index: usize) -> Option<Vector<F>> {
        info!("Solving eq3_14 for index {:?} of {:?}.", index, self.size());

        assert!(1 <= index && index <= self.size(), "wrong index");
        // Initialize all for z3
        let cfg = z3::Config::new();
        let ctx = z3::Context::new(&cfg);

        // Take at most the index + 1 size of the matrix.
        let ta = Triangular::from(self.repr()).slice(index + 1);
        let (to_reduce_ini, rest_ini) = self.initial().elements.split_at(index + 1);

        let ini_z3 = to_reduce_ini
            .iter()
            .map(|value| {
                let (numer, denom) = value.to_rational();
                Real::from_real_str(&ctx, &numer, &denom).expect("Something went wrong")
            })
            .collect_vec();

        // Make placeholders for the new initial distribution
        let deltas: Vec<Real<'_>> = (0..index)
            .into_iter()
            .map(|v| Real::new_const(&ctx, format!("d_{}", v + 1)))
            .collect();

        // Set up the linar equation system
        let time_start_set_up = Instant::now();
        let solver = Solver::new(&ctx);

        let mut mut_ta: Triangular<F> = Triangular::eye(index + 1);

        // unroll the case j=0
        let mut lh_accum = Real::from_real(&ctx, 0, 1);
        let mut rh_accum = Real::from_real(&ctx, 0, 1);

        deltas.iter().map(|d| lh_accum += d).collect_vec();
        lh_accum = lh_accum.simplify();

        ini_z3.iter().map(|val| rh_accum += val).collect_vec();
        rh_accum = rh_accum.simplify();

        solver.assert(&lh_accum._eq(&rh_accum));

        for _j in 1..index {
            // I have to elevate to the power of j, which means, do a new dot product with the fixed value and overwrite the trimmed matrix.
            mut_ta = mut_ta.dot_product(&ta);

            // Set up LES with syhmbolic deltas.
            let row_sum_lhs_z3 = mut_ta
                .row_sum_sliced(index)
                .into_iter()
                .map(|value| {
                    let (numer, denom) = value.to_rational();
                    Real::from_real_str(&ctx, &numer, &denom)
                        .expect("Something went wrong")
                        .simplify()
                })
                .collect_vec();

            let row_sum_rhs_z3 = (0..index + 1) //(0..index)
                .into_iter()
                .map(|i| mut_ta.row_sum(i))
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

            if !self.minisolver_z3(&to_reduce_ini.to_vec(), &model) {
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
    #[allow(dead_code)]
    pub fn polynomial_all(&self) -> impl Fn(&F) -> F {
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

    /// Computes
    ///
    /// $$
    /// R(s) = \vv{\beta}_1 + \vv{\beta}_2 L(\lambda_1) + \dots + \vv{\beta}_i L(\lambda_1) \dots L(\lambda_{i_1})
    /// $$
    ///
    pub fn polynomial(&self, index: usize) -> impl Fn(&F) -> F {
        assert!(
            index < self.size(),
            "Index ({}) can not be larger than the size ({}).",
            index,
            self.size()
        );
        let lambdas: Vec<F> = (0..index)
            .into_iter()
            .map(|r| {
                let mut v = self.repr.get(r, r);
                v.neg_assign();
                v
            })
            .collect();

        let mut clauses: Vec<(F, Vec<LTerm<F>>)> = vec![];
        for i in 0..=index {
            let b_i = self.initial[i].clone();
            // if b_i.eq(&F::zero()) {
            //     continue;
            // }
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
    /// $R_i(s) = \beta_1 + \beta_2 L(\lambda_1) = \cdots + \beta_i L(\lambda_1) \dots L(\lambda_{i-1})$.
    /// Polynomial $R(s)$ is divisible by $L(\lambda_i)$ if $R(-\lambda_i) = 0$.
    /// Additionally, we also consider the note that specifies that states with the
    /// same outgoing rate that the one from the initial state can not be reduced
    fn removable(&self, polynomial: &impl Fn(&F) -> F, index: usize) -> F {
        if index >= self.size()
            || (!self.initial[0].is_zero()
                && self.repr().get(index, index) == self.repr().get(0, 0))
        {
            F::one()
        } else {
            let time_start = Instant::now();
            let value = polynomial(&self.repr.get(index, index));
            let elapsed = time_start.elapsed();
            info!(
                "({}). R({}) = {} ({}). Elapsed: {:?}.",
                index,
                self.repr.get(index, index),
                value,
                value.is_almost_zero(),
                elapsed
            );
            value.clone()
        }
    }

    /// Translate to the equivalent Coxian representation, so it can then be correctly parsed into CTMC models for other tools
    /// Translation according to Theorem 3.5 and
    /// Cumani, A. (1982). On the canonical representation of homogeneous Markov processes modelling failure-time distributions. Microelectronics Reliability, 22(3), 583-602.
    pub fn to_coxian(&self) -> coxian::Coxian<F> {
        // lambdas stay the same, but order is inverted.
        let lambdas = Vector::from(self.repr().0.iter().rev().map(|e| e.clone()).collect_vec());
        // compute the factors according to the following formula
        // x_i = 1 - beta_i \prod_{j=i+1}^{n} 1 / x_j   2 <= i <= n
        let mut factors: Vec<F> = vec![F::one(); self.size()];
        for i in (1..self.size()).rev() {
            let beta_i = self.initial[i].clone();
            if beta_i.is_almost_zero() {
                continue;
            } else {
                let mut accum = F::one();
                for j in i + 1..self.size() {
                    let mut value = factors[j].clone();
                    value.inv_assign();
                    accum.mul_assign(&value);
                }
                factors[i].sub_assign(&(beta_i * accum));
            }
        }
        let factors = Vector::from(factors.iter().rev().map(|e| e.clone()).collect_vec());
        coxian::Coxian { lambdas, factors }
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
