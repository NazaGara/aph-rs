use itertools::Itertools;
use linalg::{Vector, fields::PseudoField};
use ndarray::Array2;
use representation::{Representation, bidiagonal::Bidiagonal, triangular::TriangularArray};
use std::{cmp::Ordering, collections::HashMap};

use crate::{Aph, BidiagonalAph, linalg, representation};

pub fn build_vot_bidiagonal<'a, F: PseudoField + 'a>(
    inputs: Vec<&'a BidiagonalAph<F>>,
    k: u32,
) -> BidiagonalAph<F> {
    let n = inputs.len() as u32;

    let p_list: Vec<u32> = inputs
        .iter()
        .map(|g| g.size().try_into().unwrap())
        .collect();

    // --- Transient states
    let mut transient_states: Vec<Vec<i32>> = Vec::new();

    enumerate_vot_states(0, n, k, 0, &p_list, &mut Vec::new(), &mut transient_states);
    sort_states(&mut transient_states);
    // Map state -> index
    let mut index: HashMap<Vec<i32>, usize> = HashMap::new();
    for (i, s) in transient_states.iter().enumerate() {
        index.insert(s.clone(), i);
    }
    let size = transient_states.len();
    let mut matrix = Array2::<F>::zeros((size, size));

    // Fill transitions.
    for (i, st) in transient_states.iter().enumerate() {
        let _fails = st.iter().filter(|&&x| x == -1).count();
        let mut absorb_rate = F::zero();
        let mut off_diag_sum = F::zero();

        for (j, &phase) in st.iter().enumerate() {
            if phase == -1 {
                continue;
            }
            let phase_usize = phase as usize; // In which step of the component are we in.
            let g = &inputs[j];

            // the rate is given by the phase the component is in, given by `phase`
            let mut rate = g.repr().get(phase_usize, phase_usize);
            rate.neg_assign();

            // The new state depends if we lead to a new component phase, to a component failure, or a gate failure.
            let mut new_st = st.clone();
            if phase_usize == g.size() - 1 {
                // mark the component as failed.
                new_st[j] = -1;
            } else {
                // Only increment the component phase.
                new_st[j] = phase + 1;
            }

            if let Some(&idx) = index.get(&new_st) {
                // If state is indexed, we transition to new transient state.
                // This can be either: another component phase, or component failure.
                matrix[[i, idx]].add_assign(&rate);
                off_diag_sum.add_assign(&rate);
            } else {
                // If state is not transient, is a gate failure.
                absorb_rate.add_assign(&rate)
            }
        }

        let mut exit_rate = off_diag_sum + absorb_rate;
        exit_rate.neg_assign();
        matrix[[i, i]] = exit_rate;
    }

    let initials: Vec<Vector<F>> = inputs
        .iter()
        .map(|ph| {
            let mut ini = (ph.initial()).clone();
            ini.push(F::zero());
            ini
        })
        .collect();
    let delta = vot_initial_distribution(&initials, &p_list, &transient_states);

    Aph {
        initial: delta,
        repr: TriangularArray { size, matrix },
    }
    .spa()
}

#[allow(unused)]
pub fn build_vot_tr<'a, F: PseudoField + 'a>(
    inputs: Vec<&'a BidiagonalAph<F>>,
    k: u32,
) -> Aph<F, TriangularArray<F>> {
    let n = inputs.len() as u32;

    let p_list: Vec<u32> = inputs
        .iter()
        .map(|g| g.size().try_into().unwrap())
        .collect();

    // --- Transient states
    let mut transient_states: Vec<Vec<i32>> = Vec::new();

    enumerate_vot_states(0, n, k, 0, &p_list, &mut Vec::new(), &mut transient_states);
    sort_states(&mut transient_states);

    // Map state -> index
    let mut index: HashMap<Vec<i32>, usize> = HashMap::new();
    for (i, s) in transient_states.iter().enumerate() {
        index.insert(s.clone(), i);
    }
    let size = transient_states.len();
    let mut matrix = Array2::<F>::zeros((size, size));

    // Fill transitions.
    for (i, st) in transient_states.iter().enumerate() {
        let _fails = st.iter().filter(|&&x| x == -1).count();
        let mut absorb_rate = F::zero();
        let mut off_diag_sum = F::zero();

        for (j, &phase) in st.iter().enumerate() {
            if phase == -1 {
                continue;
            }
            let phase_usize = phase as usize; // In which step of the component are we in.
            let g = &inputs[j];

            // the rate is given by the phase the component is in, given by `phase`
            let mut rate = g.repr().get(phase_usize, phase_usize);
            rate.neg_assign();

            // The new state depends if we lead to a new component phase, to a component failure, or a gate failure.
            let mut new_st = st.clone();
            if phase_usize == g.size() - 1 {
                // mark the component as failed.
                new_st[j] = -1;
            } else {
                // Only increment the component phase.
                new_st[j] = phase + 1;
            }

            if let Some(&idx) = index.get(&new_st) {
                // If state is indexed, we transition to new transient state.
                // This can be either: another component phase, or component failure.
                matrix[[i, idx]].add_assign(&rate);
                off_diag_sum.add_assign(&rate);
            } else {
                // If state is not transient, is a gate failure.
                absorb_rate.add_assign(&rate)
            }
        }

        let mut exit_rate = off_diag_sum + absorb_rate;
        exit_rate.neg_assign();
        matrix[[i, i]] = exit_rate;
    }

    let initials: Vec<Vector<F>> = inputs
        .iter()
        .map(|ph| {
            let mut ini = (ph.initial()).clone();
            ini.push(F::zero());
            ini
        })
        .collect();
    let delta = vot_initial_distribution(&initials, &p_list, &transient_states);

    Aph {
        initial: delta,
        repr: TriangularArray { size, matrix },
    }
}

fn combinations<T: Clone>(data: &[Vec<T>], k: usize) -> Vec<Vec<T>> {
    let mut result = Vec::new();
    let n = data.len();

    // Iterate over all k-combinations of indices
    for indices in (0..n).combinations(k) {
        // Select the sublists
        let chosen: Vec<&[T]> = indices.iter().map(|&i| data[i].as_slice()).collect();
        // Cartesian product across sublists
        for combo in chosen.into_iter().multi_cartesian_product() {
            result.push(combo.into_iter().cloned().collect());
        }
    }

    result
}

pub fn build_vot_bidi<F: PseudoField>(inputs: Vec<&BidiagonalAph<F>>, k: u32) -> BidiagonalAph<F> {
    // We fix some rate and amount, check all combinations with other rates from other inputs.
    // I can have combinations of sizes N to N-K.
    // We add all the rates that are combined, and we see each the number of times they appear, minus one.

    let n = inputs.len() as u32;
    let p_list: Vec<u32> = inputs
        .iter()
        .map(|g| g.size().try_into().unwrap())
        .collect();

    // --- Transient states
    let mut transient_states: Vec<Vec<i32>> = Vec::new();

    enumerate_vot_states(0, n, k, 0, &p_list, &mut Vec::new(), &mut transient_states);
    sort_states(&mut transient_states);

    // --- Build Bidiagonal with the core series.
    let mut new_rates: HashMap<F, u32> = HashMap::new();
    let mut rem_rates = inputs
        .iter()
        .map(|aph| aph.unique_rates_vec())
        .collect_vec();

    while rem_rates.len() as u32 > n - k {
        let rc = rem_rates.pop().unwrap();
        // fix one rate
        for (lambda, amount) in rc.iter() {
            // all number of combinations
            for nc in (n - k)..n {
                // all combinations of the chosen size.
                for combo in combinations(&rem_rates, nc as usize) {
                    // We update the new rates.
                    let mut new_rate = lambda.clone();
                    let mut new_amount = *amount;
                    let _: Vec<_> = combo
                        .iter()
                        .map(|(mu, l)| {
                            new_rate.add_assign(mu);
                            new_amount += l - 1;
                        })
                        .collect();
                    new_rates
                        .entry(new_rate)
                        .and_modify(|v| *v = (*v).max(new_amount))
                        .or_insert(new_amount);
                }
            }
        }
    }
    let mut new_diagonal: Vec<F> = vec![];
    new_rates
        .iter()
        .for_each(|(lambda, k)| new_diagonal.extend(vec![lambda.clone(); (*k) as usize]));

    // Sort rates (necessary for SPA) create representation
    new_diagonal.sort_by(|x, y| {
        y.partial_cmp(x)
            .unwrap_or_else(|| panic!("Could not sort the values: {:?} and {:?}.", y, x))
    });

    let bidiagonal = Bidiagonal::<F>::from(Vector::from(new_diagonal));

    // --- Initial distribution
    let initials: Vec<Vector<F>> = inputs
        .iter()
        .map(|ph| {
            let mut ini = (ph.initial()).clone();
            ini.push(F::zero());
            ini
        })
        .collect();
    let delta = vot_initial_distribution(&initials, &p_list, &transient_states);

    // Lazy getter
    let getter = move |i, j| lazy_vot_getter::<F>(&inputs, &transient_states, i, j);

    Aph::<F, Bidiagonal<F>>::spa_with_getter(&delta, &getter, bidiagonal)
}

#[allow(unused)]
fn lazy_vot_getter<F: PseudoField>(
    inputs: &[&BidiagonalAph<F>],
    states: &[Vec<i32>],
    i: usize,
    j: usize,
) -> F {
    if j < i {
        F::zero()
    } else if i == j {
        //TODO:
        // sum of the exit rates of the non finished states.
        let st: Vec<(usize, &i32)> = states[i].iter().enumerate().collect();
        let mut sum = F::zero();

        st.into_iter().for_each(|(idx, &phase)| {
            let g = inputs[idx];
            // Because values are in the diagonal, they are already negative.
            sum.add_assign(&g.repr().get(phase as usize, phase as usize));
        });
        sum
    } else {
        let changes: Vec<(usize, i32, i32)> = states[i]
            .iter()
            .zip(&states[j])
            .enumerate()
            .filter_map(|(i, (f, t))| if (f != t) { Some((i, *f, *t)) } else { None })
            .collect();
        if changes.len() != 1 {
            return F::zero();
        }
        let ((idx, phase, next_phase)) = *(changes.first().unwrap());
        if phase == -1 || next_phase > next_phase + 1 {
            // - Once a component is failed, there is no more transitions.
            // - We can only do 'one' jump at each time.
            panic!("From phase {phase} to phase {next_phase}. is Invalid!");
        }
        let g = inputs[idx];
        if next_phase == -1 && (phase as usize) != g.size() - 1 {
            // In order to fail component i, the phase must be the last one.
            F::zero()
        } else if next_phase == -1 {
            // Failure of component i.
            let mut val = g.repr().get(phase as usize, phase as usize);
            val.neg_assign();
            val
        } else {
            // Next phase for component i.
            g.repr().get(phase as usize, next_phase as usize)
        }
    }
}

fn vot_initial_distribution<F: PseudoField>(
    initials: &[Vector<F>],
    p_list: &[u32],
    states: &[Vec<i32>],
) -> Vector<F> {
    let n = initials.len();

    let mut pi0: Vector<F> = Vector::zeros(states.len()); //vec![F::zero(); states.len()];
    for (idx, s) in states.iter().enumerate() {
        let mut prob = F::one();
        for i in 0..n {
            let alpha = &initials[i];
            let p = p_list[i];
            let val = if s[i] == -1 {
                // failed -> use last entry
                &alpha[p as usize]
            } else {
                &alpha[s[i] as usize]
            };
            prob.mul_assign(val);
            if prob.is_zero() {
                break; // early out
            }
        }
        pi0[idx] = prob;
    }

    pi0
}

fn enumerate_vot_states(
    idx: usize,
    n: u32,
    k: u32,
    fails: u32,
    p_list: &[u32],
    cur: &mut Vec<i32>,
    states: &mut Vec<Vec<i32>>,
) {
    // Recursively build the state space of the VOT gate.
    if idx == n as usize {
        if fails < k {
            states.push(cur.clone());
        }
        return;
    }
    // Case: this component is failed
    if fails < k {
        cur.push(-1);
        enumerate_vot_states(idx + 1, n, k, fails + 1, p_list, cur, states);
        cur.pop();
    }
    // Case: alive in some phase
    for ph in 0..p_list[idx] {
        cur.push(ph as i32);
        enumerate_vot_states(idx + 1, n, k, fails, p_list, cur, states);
        cur.pop();
    }
}

fn sort_states(states: &mut [Vec<i32>]) {
    // first by number of failures (ascending), then lexicographic order, with -1 last.
    states.sort_by(|a, b| {
        let fa = a.iter().filter(|&&x| x == -1).count();
        let fb = b.iter().filter(|&&x| x == -1).count();
        if fa != fb {
            return fa.cmp(&fb);
        }
        // same number of failures: compare lexicographically with -1 treated as very large
        for j in 0..a.len() {
            let ma = if a[j] == -1 { i32::MAX } else { a[j] };
            let mb = if b[j] == -1 { i32::MAX } else { b[j] };
            if ma != mb {
                return ma.cmp(&mb);
            }
        }
        Ordering::Equal
    });
}
