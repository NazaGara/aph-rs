pub mod coxian;
pub mod formats;
pub mod linalg;
pub mod operations;
pub mod representation;
use itertools::{join, Itertools};
use linalg::{
    fields::{float64::Float64, Almost, FromRational, PseudoField},
    Vector,
};

use std::{collections::HashMap, fmt::Display, time::Instant, vec};

use log::{debug, info, warn};
use num_traits::Zero;
use std::fs::File;
use std::io::{self, Write};

use representation::{Bidiagonal, Representation, Triangular, TriangularArray};

#[derive(Debug, Clone, PartialEq, Eq)]
struct LTerm<F: PseudoField> {
    rate: F,
}

impl<F: PseudoField> Display for LTerm<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "(s+{})/{}", self.rate, self.rate)
    }
}

impl<F: PseudoField> LTerm<F> {
    pub fn new(rate: F) -> Self {
        LTerm { rate }
    }
    pub fn eval(&self, s: &F) -> F {
        let mut numerator = s.clone();
        numerator.add_assign(&self.rate);
        numerator.div_assign(&self.rate);
        numerator
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Polynomial<F: PseudoField> {
    clauses: Vec<(F, Vec<LTerm<F>>)>,
}

impl<F: PseudoField> Display for Polynomial<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            self.clauses
                .iter()
                .map(|(el, l_terms)| format!("{} * {}", el, l_terms.iter().join(" * ")))
                .join(" + ")
                .replace("*  +", "+")
                .trim_end_matches("* ")
        )
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

    fn eval_until(&self, top: usize, s: &F) -> F {
        let mut result = F::zero();
        for (i, (beta, prod)) in self.clauses.iter().enumerate() {
            if i >= top {
                break;
            }
            let mut cl = beta.to_owned();
            for c in prod {
                cl.mul_assign(&c.eval(s));
            }
            result.add_assign(&cl);
        }
        result
    }
}

/// An APH distribution.
#[derive(Debug, Clone)]
pub struct Aph<F: PseudoField, R: Representation<F>> {
    pub initial: Vector<F>,
    pub repr: R,
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

        let ini_not_zero = self
            .initial
            .elements
            .iter()
            .cloned()
            .enumerate()
            .filter(|(_, e)| !e.is_zero())
            .collect_vec();

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

    pub fn new_erl(phases: usize, rate: &F) -> Aph<F, Bidiagonal<F>> {
        let mut repr = Bidiagonal::new(phases);
        (0..phases).for_each(|i| {
            repr.set(i, rate.clone());
        });
        Aph::new(Vector::one_and_zeros(0, phases), repr)
    }

    pub fn _new_hypexp(rates: Vec<(&str, &str)>) -> Aph<F, Bidiagonal<F>> {
        let size = rates.len();
        let mut repr = Bidiagonal::new(size);

        rates.iter().enumerate().for_each(|(i, &(n, d))| {
            repr.set(i, F::from_rational(n, d));
        });

        Aph::new(Vector::one_and_zeros(0, size), repr)
    }

    pub fn explicit_exp(value: F) -> Aph<F, Bidiagonal<F>> {
        let mut repr = Bidiagonal::new(1);
        repr.set(0, value);
        Aph::new(Vector::one_and_zeros(0, 1), repr)
    }

    pub fn explicit_erl(phases: usize, value: F) -> Aph<F, Bidiagonal<F>> {
        let mut repr = Bidiagonal::new(phases);
        (0..phases).for_each(|i| {
            repr.set(i, value.clone());
        });
        Aph::new(Vector::one_and_zeros(0, phases), repr)
    }

    pub fn into_tuple(self) -> (Vector<F>, R) {
        (self.initial, self.repr)
    }

    pub fn diagonal(&self, row: usize) -> F {
        self.repr.diagonal(row)
    }

    pub fn all_diagonal(&self) -> Vector<F> {
        (0..self.size())
            .map(|i| self.repr().diagonal(i))
            .collect_vec()
            .into()
    }

    pub fn unique_rates(&self) -> HashMap<F, usize> {
        //HashSet<F> {
        // HashSet::from_iter((0..self.size()).map(|i| self.repr().diagonal(i)))
        let mut counts = HashMap::new();
        for num in (0..self.size()).map(|i| self.repr().diagonal(i)) {
            *counts.entry(num).or_insert(0) += 1;
        }
        counts
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

    pub fn spa_from_explicit(
        pre_initial: &Vector<F>,
        getter: &impl Fn(usize, usize) -> F,
        bidiagonal: Bidiagonal<F>,
    ) -> Aph<F, Bidiagonal<F>> {
        let time_start = Instant::now();
        // Step 1️⃣: Construct ordered bidiagonal generator matrix.
        let large_size = pre_initial.size();
        let final_size = bidiagonal.size();

        // Step 2️⃣: Compute new initial probability vector.
        let mut new_initial = Vector::zeros(final_size);
        let mut vector = Vector::unit(large_size);
        let mut result = Vector::zeros(large_size);

        multiply_w_vec(
            getter,
            &vector,
            &mut result,
            &bidiagonal.diagonal(final_size - 1),
            &F::zero(),
        );
        new_initial[final_size - 1] = pre_initial.scalar_product(&result);
        std::mem::swap(&mut vector, &mut result);

        for row in (0..final_size - 1).rev() {
            let mut divide = bidiagonal.diagonal(row);
            divide.neg_assign();
            let mut add = bidiagonal.diagonal(row + 1);
            add.neg_assign();
            multiply_w_vec(getter, &vector, &mut result, &divide, &add);
            new_initial[row] = pre_initial.scalar_product(&result);
            std::mem::swap(&mut result, &mut vector);
        }

        let elapsed = time_start.elapsed();
        info!(
            "Time Elapsed 'explicit spa': {:?}. Size: {:?}",
            elapsed, final_size
        );
        Aph::new(new_initial, bidiagonal)
    }
}

/// Puts in ['result'] using the getter function instead of explicitly using the generator matrix.
/// $$
/// \frac{1}{\lambda}(\mathbf{A} + \lambda'\mathbf{I})\vec{x}
/// $$
fn multiply_w_vec<F: PseudoField>(
    getter: &impl Fn(usize, usize) -> F,
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
                let mut result = getter(row, column);
                result.add_assign(add);
                result
            } else {
                getter(row, column)
            };
            let mut add = factor;
            add.mul_assign(&vector[column]);
            result[row].add_assign(&add);
        }
        result[row].div_assign(divide);
    }
}

/// Specific algorithms when the APH is in Bidiagonal form.
impl<F: PseudoField> Aph<F, Bidiagonal<F>> {
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
                value.almost_zero(),
                elapsed
            );
            value.clone()
        }
    }

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
                val.almost_zero()
            };

            if is_removable {
                if let Some(new_ini) = self.eq3_14_f(s_id) {
                    // Replace initial distribution
                    self.initial = new_ini;
                    // Compute new matrix removing state at s_id
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

    /// Compute Eq 3.14 from Reza's Thesis,
    /// Note that the element to be reduced is located at the last position of the variable
    /// ['to_reduce_ini'], located at position [index+1] in the representation, thus, we want to see if we can
    /// reduce from ['index+1'] states to ['index'] states.
    pub fn eq3_14_f(&self, index: usize) -> Option<Vector<F>> {
        info!("Solving eq3_14 for index {:?} of {:?}.", index, self.size());
        assert!(1 <= index && index <= self.size(), "wrong index");

        // Take at most the index + 1 size of the matrix.
        // let ta = Triangular::from(self.repr()).slice(index + 1);

        let ta = Triangular::from_self_sliced(self.repr(), index + 1);
        let (to_reduce_ini, rest_ini) = self.initial().elements.split_at(index + 1);

        // ============== Start LES =====================
        let mut model: Vec<F> = vec![F::zero(); index];

        let time_start = Instant::now();
        let mut mut_ta: Triangular<F> = Triangular::eye(index + 1);

        for j in 1..index {
            let delta_idx = index - j;
            // Compute Matrix to the j power, i.e., do dot product with the fixed value and overwrite the trimmed matrix.
            mut_ta = mut_ta.dot_product(&ta);

            let row_sum_lhs = mut_ta.row_sum_sliced_from(delta_idx, index);

            let row_sum_rhs = (0..index + 1)
                .map(|i| {
                    if i > delta_idx {
                        mut_ta.row_sum(i)
                    } else {
                        F::zero()
                    }
                })
                .collect_vec();

            // Accummulate rhs
            let mut new_rhs: F = F::zero();

            // Note that, the LHS/RHS values that are only interesting for us are in the last delta_idx to the last values.
            to_reduce_ini
                .iter()
                .enumerate()
                .map(|(i, val)| {
                    if i >= delta_idx {
                        new_rhs.add_assign(&(val.clone() * row_sum_rhs[i].clone()))
                    }
                })
                .collect_vec();

            // Substract values from the deltas that we already compute.
            // Note that, the LHS/RHS values that are only interesting for us are in the last delta_idx to the last values.
            model
                .iter()
                .enumerate()
                .map(|(i, e)| {
                    if i >= delta_idx {
                        new_rhs.sub_assign(&(row_sum_lhs[i].clone() * e.clone()))
                    }
                })
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
            .map(|ini| val.add_assign(ini))
            .collect_vec();

        model.iter().map(|e| val.sub_assign(e)).collect_vec();
        model[delta_idx] = val;

        let elapsed = time_start.elapsed();
        // ============== END LES =====================

        // The LES has solution, by Lemmas 3.8 and 3.9.
        // Check that system will have solution, but I dont know if wee can determine this before seeing the whole LES.

        info!("Time Elapsed: 'LES': {:?}. (IDX: {:?}) SAT", elapsed, index);
        model.extend(rest_ini.to_vec());

        Some(Vector {
            elements: model.into_boxed_slice(),
        })
    }

    /// Mini solver to check that the new initial distributions match the total of the old initial distributions
    /// and to make sure that all of the new values are stochastically correct.
    #[allow(dead_code)]
    fn minisolver(&self, ini: &[F], model: &[F]) -> bool {
        let time_start = Instant::now();

        // Using Float64 for the comparison, it will also work for Rationals, it will take a bit longer than using the same type bcs of transformations.
        let mut pre_accum = Float64::zero();
        let mut post_accum = Float64::zero();
        let _ = ini
            .iter()
            .map(|val| {
                let (num, den) = val.to_rational();
                pre_accum.add_assign(&Float64::from_rational(&num, &den));
                // pre_accum.add_assign(&val)
            })
            .collect_vec();

        // for delta in model.iter() {
        for delta in model.iter() {
            let mut val = delta.clone();
            if val.almost_zero() {
                val.set_zero();
            }

            // if val.lt(&F::zero()) || val.gt(&F::one()) {
            if (val.lt(&F::zero()) && !val.cmp_eq(&F::zero()))
                || (val.gt(&F::one()) && !val.cmp_eq(&F::one()))
            {
                // If field is not rational, then roundoffs can produce negative values or really large ones, which then may get cancelled.
                // Don't panic but warn.
                warn!("Negative Value: {:?}.", val);
            } else {
                let (num, den) = val.to_rational();
                post_accum.add_assign(&Float64::from_rational(&num, &den));
            }
            // post_accum.add_assign(&val);
        }
        let elapsed = time_start.elapsed();

        debug!(
            "{:?} v {:?} ({:?})",
            pre_accum,
            post_accum,
            pre_accum.cmp_eq(&post_accum)
        );

        if !pre_accum.cmp_eq(&post_accum) {
            info!("Time Elapsed 'minisolver': {:?}. UNSAT", elapsed);
            return false;
        }
        info!("Time Elapsed 'minisolver': {:?}. SAT", elapsed);
        true
    }

    /// Computes
    ///
    /// $$
    /// R(s) = \vv{\beta}_1 + \vv{\beta}_2 L(\lambda_1) + \dots + \vv{\beta}_i L(\lambda_1) \dots L(\lambda_{i_1})
    /// $$
    ///
    #[allow(dead_code)]
    pub fn polynomial_all(&self) -> impl Fn(usize, &F) -> F {
        let lambdas: Vec<F> = (0..self.size())
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
        move |i: usize, s: &F| poly.eval_until(i, s)
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
            .map(|r| {
                let mut v = self.repr.get(r, r);
                v.neg_assign();
                v
            })
            .collect();
        let mut clauses: Vec<(F, Vec<LTerm<F>>)> = vec![];
        for i in 0..=index {
            let b_i = self.initial[i].clone();
            if !b_i.eq(&F::zero()) {
                let mut prod = vec![];
                for j in (0..i).rev() {
                    prod.push(LTerm::new(lambdas.get(j).unwrap().to_owned()));
                }
                clauses.push((b_i, prod));
            }
        }
        let poly = Polynomial::new(clauses);
        move |s: &F| poly.eval(s)
    }

    /// Export to Markov Automata.
    /// Note that adds an initial state that has and empty action to do a probabilistic
    /// jump according to the initial probabilities to the correct state, and also makes the
    /// last state absorbent.
    pub fn export_to_ma(&self, filename: &str) -> io::Result<()> {
        let ini_not_zero: Vec<String> = self
            .initial
            .elements
            .iter()
            .enumerate()
            .filter(|(_, e)| !e.almost_zero())
            .map(|(i, e)| format!("{} : (s'={})", e.to_string(), i + 1))
            .collect();

        let n_states = self.size() + 1; // + the new initial state.

        let mut file = File::create(format!("{:?}.ma", filename).replace("\"", ""))?;
        writeln!(file, "ma\n")?;
        writeln!(file, "module main")?;
        writeln!(file, "s: [0..{}] init 0;", n_states)?;

        writeln!(file, "[] s=0 -> {};", join(ini_not_zero, " + "))?;
        if self.repr().is_bidiagonal() {
            // Write transitions, is each state to the next, with exception of the last state.
            for i in 0..self.size() {
                let mut v = self.repr.get(i, i);
                v.neg_assign();
                writeln!(
                    file,
                    "<> s={} -> {} : (s'={});",
                    i + 1,
                    v.to_string(),
                    i + 1 + 1
                )?;
            }
            // Make last state absorbing
            // writeln!(file, "<> s={} -> 1 : (s'={});", n_states, n_states)?;
        } else {
            todo!("MA export of non bidiaognal APH not yet supported.")
        }

        writeln!(file, "endmodule\n")?;
        // writeln!(file, "label \"s{}\"= (s={});", n_states, n_states)?;
        writeln!(file, "label \"done\"= (s={});", n_states)?;
        info!(
            "Model writed to file: '{}', last state with label: 'done'.",
            filename
        );

        Ok(())
    }

    /// Translate to the equivalent Coxian representation, so it can then be correctly parsed into CTMC models for other tools
    /// Translation according to Theorem 3.5 and
    /// Cumani, A. (1982). On the canonical representation of homogeneous Markov processes modelling failure-time distributions. Microelectronics Reliability, 22(3), 583-602.
    pub fn to_coxian(&self) -> coxian::Coxian<F> {
        // lambdas stay the same, but order is inverted.
        let lambdas = Vector::from(self.repr().0.iter().rev().cloned().collect_vec());
        // compute the factors according to the following formula
        // x_i = 1 - beta_i \prod_{j=i+1}^{n} 1 / x_j   2 <= i <= n
        let mut factors: Vec<F> = vec![F::one(); self.size()];
        for i in (1..self.size()).rev() {
            let beta_i = self.initial[i].clone();
            if beta_i.almost_zero() {
                continue;
            } else {
                let mut accum = F::one();
                // for j in i + 1..self.size() {
                //     accum.div_assign(&factors[j]);
                // }
                for factor in factors.iter().take(self.size()).skip(i + 1) {
                    accum.div_assign(factor);
                }
                factors[i].sub_assign(&(beta_i * accum));
            }
        }

        let factors = Vector::from(factors.iter().rev().cloned().collect_vec());
        coxian::Coxian { lambdas, factors }
    }

    /// Returns a bidiagional representation, but to compute each of the diagonal entry i, we still have to do 1 - bidi[i].
    pub fn uniformize(&self) -> Bidiagonal<F> {
        // Compute max rate: which is in the last element of the bidiagonal representation.
        let mut max_rate = self.repr().get(self.size() - 1, self.size() - 1);
        max_rate.abs_assign();

        let mut bidi: Bidiagonal<F> = self.repr.clone();

        // Create transition rate matrix.
        for e in bidi.0.iter_mut() {
            e.div_assign(&max_rate);
            e.abs_assign();
        }
        bidi
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
