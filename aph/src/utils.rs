use indicatif::{ProgressBar, ProgressStyle};
use itertools::Itertools;
use linalg::Vector;
use log::debug;
use ndarray::Array2;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use std::collections::HashSet;
use std::fmt::Debug;
use std::{collections::HashMap, vec};
use thiserror::Error;

use representation::{Representation, bidiagonal::Bidiagonal};

#[derive(Debug, Error)]
#[error("{0}")]
pub struct MTTFError(String);

use crate::{
    linalg::{
        self,
        fields::{PseudoField, Round},
    },
    representation,
};

/// Computes the formula: $\frac{1}{\lambda}(\mathbf{A} + \lambda'\mathbf{I})\vec{x}$
///
/// # Arguments
/// * `getter`: The getter function to lazily obtain values from the matrix.
/// * `vector`: Vector x.
/// * `result`: Where the final computed value gets stored.
/// * `divide`: The divide factor.
/// * `add`: The additive factor.
#[allow(unused)]
pub fn multiply_w_vec<F: PseudoField>(
    getter: &(impl Fn(usize, usize) -> F + Sync),
    vector: &Vector<F>,
    result: &mut Vector<F>,
    divide: &F,
    add: &F,
) {
    assert!(
        vector.size() == result.size(),
        "Dimensions of vectors do not match."
    );
    let pb = ProgressBar::new((vector.size()) as u64);
    pb.set_style(
            ProgressStyle::with_template(
                "Mul w vec: {spinner:.cyan} [{elapsed_precise}] {wide_bar:.cyan/white} {pos}/{len} ({eta_precise})",
            ).unwrap()
            .progress_chars("#>-"),
        );
    for row in 0..vector.size() {
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
        pb.inc(1);
    }
    pb.finish_and_clear();
}

/// Computes the formula: $\frac{1}{\lambda}(\mathbf{A} + \lambda'\mathbf{I})\vec{x}$
///
/// # Arguments
/// * `getter`: The getter function to lazily obtain values from the matrix.
/// * `vector`: Vector x.
/// * `result`: Where the final computed value gets stored.
/// * `divide`: The divide factor.
/// * `add`: The additive factor.
pub fn multiply_w_vec_boxed<F: PseudoField>(
    getter: &(impl Fn(usize, usize) -> F + Sync),
    vector: &[F],
    result: &mut Vec<F>,
    divide: &F,
    add: &F,
) {
    assert!(
        vector.len() == result.len(),
        "Dimensions of vectors do not match."
    );
    let size = vector.len();
    let new_result: Vec<F> = (0..size)
        .into_par_iter()
        .map(|row| {
            let mut partial_res = F::zero();
            #[allow(clippy::needless_range_loop)]
            for column in row..size {
                let factor = if row == column {
                    let mut result = getter(row, column);
                    result.add_assign(add);
                    result
                } else {
                    getter(row, column)
                };
                let mut add = factor;
                add.mul_assign(&vector[column]);
                partial_res.add_assign(&add);
            }
            partial_res.div_assign(divide);
            partial_res
        })
        .collect();
    *result = new_result;
}

///
/// Computes the formula: $\frac{1}{\lambda}(\mathbf{A} + \lambda'\mathbf{I})\vec{x}$ by using a parallel iterator.
///
/// # Arguments
/// * `getter`: The getter function to lazily obtain values from the matrix.
/// * `vector`: Vector x.
/// * `result`: Where the final computed value gets stored.
/// * `divide`: The divide factor.
/// * `add`: The additive factor.
pub fn multiply_w_vec_par<F: PseudoField>(
    getter: &(impl Fn(usize, usize) -> F + Sync),
    vector: &Vector<F>,
    result: &mut Vector<F>,
    divide: &F,
    add: &F,
) {
    assert!(
        vector.size() == result.size(),
        "Dimensions of vectors do not match."
    );
    let size = vector.size();
    let new_result: Vec<F> = (0..size)
        .into_par_iter()
        .map(|row| {
            let mut partial_res = F::zero();
            for column in row..size {
                let factor = if row == column {
                    let mut result = getter(row, column);
                    result.add_assign(add);
                    result
                } else {
                    getter(row, column)
                };
                let mut add = factor;
                add.mul_assign(&vector[column]);
                partial_res.add_assign(&add);
            }
            partial_res.div_assign(divide);
            partial_res
        })
        .collect();
    *result = new_result.into();
}

/// Computes the inverse of a bidiagonal representation.
pub fn invert_bidiagonal<F: PseudoField>(repr: &Bidiagonal<F>) -> Result<Array2<F>, MTTFError> {
    let n = repr.size();

    // Extract the diagonal values and calculate reciprocals.
    let mut col_vals = vec![F::zero(); n];

    #[allow(clippy::needless_range_loop)]
    for i in 0..n {
        let diag_val = repr.get(i, i);
        let mut value = F::one();
        if diag_val.is_zero() {
            return Err(MTTFError(format!("Diagonal value at entry {i} is zero")));
        }
        value.div_assign(&diag_val);
        col_vals[i] = value;
    }

    let mut inverse = Array2::zeros((n, n));

    // Fill the inverse matrix
    for j in 0..n {
        let val = col_vals[j].clone();
        for i in 0..=j {
            inverse[(i, j)] = val.clone();
        }
    }

    Ok(inverse)
}

/// Multiplies a matrix (Array2) with a vector.
pub fn matrix_vector_mul<F: PseudoField>(
    matrix: Array2<F>,
    vector: &[F],
) -> Result<Vector<F>, MTTFError> {
    let n = matrix.shape()[0];
    if vector.len() != n {
        return Err(MTTFError(String::from("Dimension mismatch")));
    }

    let mut result = vec![vector[0].clone(); n];

    // Last element of the result vector.
    result[n - 1] = matrix[(n - 1, n - 1)].clone() * vector[n - 1].clone();

    // Iterate backwards
    for i in (0..n - 1).rev() {
        let diagonal_element = matrix[(i, i)].clone();
        let vector_element = vector[i].clone();
        result[i] = (diagonal_element * vector_element) + result[i + 1].clone();
    }

    Ok(Vector::from(result))
}

/// Bound the Core Series. It returns a tuple with:
/// (`used`,  `remanents`, `mapper`)
/// where
/// - `used` are the rates that were used to partially construct the cs and obtain the targets.
/// - `remanents` are the rates that were modified to fit the existing targets.
/// - `mapper` is the hashmap that indicates how to modify the rates.
///
#[allow(clippy::type_complexity)]
pub fn _bound_cs<F: PseudoField>(
    rate_count1: &[(F, u32)],
    rate_count2: &[(F, u32)],
    split_index: usize,
    round: Round,
) -> (Vec<(F, u32)>, Vec<(F, u32)>, HashMap<F, F>) {
    let mut mapper = HashMap::new();
    let to_use_vec = rate_count2[..split_index].to_vec();
    let remanent_vec = rate_count2[split_index..].to_vec();
    let mut new_remanents: Vec<(F, u32)> = vec![(F::from(0.0), 0); remanent_vec.len()];

    let mut targets: HashSet<F> = HashSet::new();
    for (lambda, _k) in rate_count1 {
        for (mu, _l) in &to_use_vec {
            let rate = lambda.clone() + mu.clone();
            targets.insert(rate);
        }
    }

    let mut targets = targets.iter().collect_vec();

    targets.sort_by(|x, y| match round {
        // Sort targets from value. Decreasing for round Down and Zero. Increasing for Up.
        // In order to capture the closest goal to the value.
        // TODO: We can choose a rate that comes from a different path from the CS construction. Thus assuring minimal size.
        Round::Down | Round::Zero => x
            .partial_cmp(y)
            .unwrap_or_else(|| panic!("Could not sort the values: {:?} and {:?}.", y, x)),
        Round::Up => y
            .partial_cmp(x)
            .unwrap_or_else(|| panic!("Could not sort the values: {:?} and {:?}.", y, x)),
        _ => panic!("`Nearest` round method not allowed when bounding."),
    });

    for (j, (mu, l)) in remanent_vec.iter().enumerate() {
        // If we can find matches for the remanents, we keep them
        if rate_count1
            .iter()
            .any(|e| targets.contains(&&(e.0.clone() + mu.clone())))
        {
            debug!("Found a match for remanent {}", mu);
            new_remanents[j] = (mu.clone(), *l);
        } else {
            // If there are no matches, we try to force it to match one of the target, starting with the most repeated one.
            for &t in targets.iter() {
                match rate_count1.iter().find(|&e| {
                    let mut candidate = t.clone();
                    candidate.sub_assign(&e.0);
                    match round {
                        // Depends on the rounding method.
                        Round::Down | Round::Zero => candidate.gt(mu) && candidate.lt(&F::zero()),
                        Round::Up | Round::Nearest => candidate.lt(mu) && candidate.lt(&F::zero()),
                    }
                }) {
                    None => {
                        debug!("No match for {}.", mu);
                        continue;
                    }
                    Some((candidate, _)) => {
                        let mut new_rate = t.clone();
                        new_rate.sub_assign(candidate);
                        debug!(
                            "Found match. From {} to {} (+ {} = {}) ",
                            mu, new_rate, candidate, t
                        );
                        // ph2.replace_all_rates(mu, new_rate.clone());
                        mapper.insert(mu.clone(), new_rate.clone());
                        new_remanents[j] = (new_rate, *l);
                        break;
                    }
                }
            }
        }
    }
    (to_use_vec, new_remanents, mapper)
}

/// Searches new rate values for the rate cound of `to_set_rc`, based on the targets generated by the first value from it,
/// when combined with the values from `fixed_rx`.
/// Returns a mapping from rate values to new values
pub fn find_rate_replacement<'a, F: PseudoField>(
    fixed_rc: &'a [(F, u32)],
    to_set_rc: &'a [(F, u32)],
    round: Round,
) -> HashMap<&'a F, F> {
    let mut mapper = HashMap::new();

    // Take the first element, create the initial targets.
    let (fst_rate, c) = to_set_rc.first().expect("Can not be empty!");
    // let mut new_remanents: Vec<(F, u32)> = vec![(F::from(0.0), 0); remanent_vec.len()];
    let mut targets: HashSet<F> = HashSet::new();
    for (lambda, _) in fixed_rc {
        let mut target = fst_rate.clone();
        target.add_assign(lambda);
        targets.insert(target);
    }

    // TODO: Replace new_remanents with only keeping track of the last chosen element.
    // TODO: If there is no reduction, i.e. then we should use the original ones.
    let mut new_remanents: Vec<(F, u32)> = vec![(F::from(0.0), 0); to_set_rc.len()];
    new_remanents[0] = (fst_rate.clone(), *c);

    // For the rest of the remanents:
    //      Check if it has any existing match. If it does, keep it, if it does not, replace it.
    //      For the initial (j==0), the value must be larger (more negative) than the fst rate.
    //      For the rest (j), each value must be larger (more negative) than the previous one (j-1).
    for (j, (mu, l)) in to_set_rc.iter().enumerate() {
        if j == 0 {
            continue;
        } else {
            // If we can find matches for the remanents, we keep them
            if fixed_rc
                .iter()
                .any(|e| targets.contains(&(e.0.clone() + mu.clone())))
            {
                debug!("Existing match for value {mu}. We keep it.");
                new_remanents[j] = (mu.clone(), *l);
            } else {
                // If there are no matches, we try to force it to match one of the targets, starting with the most repeated one.
                let chosen = match round {
                    Round::Up => {
                        targets
                            .iter()
                            .flat_map(|t| {
                                fixed_rc.iter().filter_map(|(lam, _k)| {
                                    let mut candidate = t.clone();
                                    candidate.sub_assign(lam);
                                    if candidate.lt(mu) //should be gt, but they are negative, so inverted
                                            && candidate.lt(&F::zero())
                                            && candidate.le(&new_remanents[j - 1].0)
                                    {
                                        Some(candidate)
                                    } else {
                                        None
                                    }
                                })
                            })
                            // candidates are negative, we take the closest to the original mu.
                            .min_by(|x, y| y.partial_cmp(x).unwrap_or(std::cmp::Ordering::Equal))
                    }
                    Round::Down | Round::Zero => {
                        targets
                            .iter()
                            .flat_map(|t| {
                                fixed_rc.iter().filter_map(|(lam, _k)| {
                                    let mut candidate = t.clone();
                                    candidate.sub_assign(lam);
                                    if candidate.gt(mu) //should be lt, but they are negative, so inverted
                                    && candidate.lt(&F::zero())
                                    && candidate.le(&new_remanents[j - 1].0)
                                    {
                                        Some(candidate)
                                    } else {
                                        None
                                    }
                                })
                            })
                            // candidates are negative, we take the closest to the original mu.
                            .min_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal))
                    }
                    _ => None,
                };
                let chosen = chosen.unwrap_or(mu.clone());

                debug!(
                    "[{j}]: Replacing {} by {:?}.\n",
                    mu.to_string(),
                    chosen.to_string()
                );

                mapper.insert(mu, chosen.clone());
                new_remanents[j] = (chosen, *l);
            }
        }
        update_targets(&mut targets, &new_remanents[j].0, fixed_rc);
    }

    mapper
}

fn update_targets<F: PseudoField>(targets: &mut HashSet<F>, mu: &F, rc1: &[(F, u32)]) {
    let new_tar = rc1
        .iter()
        .map(|(r, _)| r.clone() + mu.clone())
        .filter(|nr| !targets.contains(nr))
        .collect_vec();
    targets.extend(new_tar);
}

///
/// Merges the Vector of tuples of the processed rate counts, when if the same key is found, it increases the number.
///
pub fn merge_tuple_vec<F, T: Fn(u32, u32) -> u32>(
    map1: &[(F, u32)],
    map2: &[(F, u32)],
    op: T,
) -> Vec<(F, u32)>
where
    F: Eq + std::hash::Hash + Clone,
{
    let mut result = map1
        .iter()
        .map(|(e, v)| (e.clone(), *v))
        .collect::<HashMap<F, u32>>();

    for (key, value2) in map2.iter() {
        result
            .entry(key.clone())
            .and_modify(|value1| {
                *value1 = op(*value1, *value2);
            })
            .or_insert(*value2);
    }
    result.into_iter().collect_vec()
}

/// Computes the Longest Core Series based on the computed depths and the values from the diagonal,
/// On all paths, we only can see one rate at a time for each level, thus, we keep unique pairs of
/// (depth, rate). Afterwards, we merge all into a single HashMap to count the occurrences of each rate.
pub fn count_rate_by_levels<U: Eq + std::hash::Hash + Copy, F: PseudoField>(
    depths: &[U],
    diagonal: &[F],
) -> HashMap<F, u32> {
    let unique_pairs: HashSet<(U, F)> = depths
        .iter()
        .zip(diagonal.iter())
        .map(|(&depth, value)| (depth, value.clone()))
        .collect();

    let mut counts = HashMap::new();
    for (_depth, value) in unique_pairs {
        *counts.entry(value).or_insert(0) += 1;
    }
    counts
}
