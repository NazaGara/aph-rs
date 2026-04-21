//! Functions and algorithms for combining APH with bounds.

use itertools::Itertools;
use linalg::fields::PseudoField;
use log::{debug, info};
use ndarray::Array;
use sprs::TriMat;
use std::collections::HashMap;

use crate::linalg::Vector;
use crate::linalg::fields::Round;
use crate::operations::{lazy_kron_sum, lazy_max_block};
use crate::representation::sparse::Sparse;
use crate::representation::{self, bidiagonal::Bidiagonal};
use crate::utils::{bound_cs, find_rate_replacement, merge_tuple_vec};
use crate::{Aph, BidiagonalAph, linalg};
use representation::Representation;

// ---- Rounded Operations ----
pub fn _min_round<F: PseudoField>(
    ph1: &BidiagonalAph<F>,
    ph2: &BidiagonalAph<F>,
    percentage: f64,
    round: Round,
) -> BidiagonalAph<F> {
    debug_assert!(
        percentage <= 1.0 && percentage > 0.0,
        "Percentage must be in (0,1]."
    );
    debug_assert_ne!(round, Round::Nearest, "Round can not be Nearest");
    // Using mut is a bit slower than just cloning the whole thing
    let mut ph2 = ph2.clone();

    let rate_count_ph1 = ph1.unique_rates_vec();
    let rate_count_ph2 = ph2.unique_rates_vec();

    let split_index = usize::max(1, (rate_count_ph2.len() as f64 * percentage) as usize);
    info!("Splitting at {split_index} of {}.", rate_count_ph2.len());

    let (to_use_vec, new_remanents, mapper) =
        bound_cs::<F>(&rate_count_ph1, &rate_count_ph2, split_index, round);

    // Incorporate changes and also run spa if the new rates are disordered.
    let _: Vec<_> = mapper
        .into_iter()
        .map(|(mu, new_rate)| ph2.replace_all_rates(&mu, new_rate))
        .collect();

    let ph2 = ph2.spa();

    let mut new_rates: HashMap<F, u32> = HashMap::new();
    for (lambda, k) in &rate_count_ph1 {
        // for (mu, l) in &merge_tuple_vec(&to_use_vec, &remanent_vec, |x, y| x + y) {
        for (mu, l) in &merge_tuple_vec(&to_use_vec, &new_remanents, |x, y| x + y) {
            let amount = k + l - 1;
            let rate = lambda.clone() + mu.clone();
            new_rates
                .entry(rate)
                .and_modify(|v| *v = (*v).max(amount))
                .or_insert(amount);
        }
    }

    // Create diagonal.
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

    let delta = ph1.initial.kron_prod(&ph2.initial);
    // Run SPA from explicit to obtain the new initial probabilities.
    let getter = |i, j| {
        lazy_kron_sum::<F, Bidiagonal<F>>(ph1.repr(), ph2.repr(), i, j, ph1.size(), ph2.size())
    };
    Aph::<F, Bidiagonal<F>>::spa_with_getter(&delta, &getter, bidiagonal)
}

pub fn _max_round<F: PseudoField>(
    ph1: &BidiagonalAph<F>,
    ph2: &BidiagonalAph<F>,
    percentage: f64,
    round: Round,
) -> BidiagonalAph<F> {
    debug_assert!(
        percentage <= 1.0 && percentage > 0.0,
        "Percentage must be in (0,1]."
    );
    debug_assert_ne!(round, Round::Nearest, "Round can not be Nearest");
    // Using mut is a bit slower than just cloning the whole thing
    let mut ph2 = ph2.clone();

    let rate_count_ph1 = ph1.unique_rates_vec();
    let rate_count_ph2 = ph2.unique_rates_vec();

    let split_index = usize::max(1, (rate_count_ph2.len() as f64 * percentage) as usize);

    let (to_use_vec, remanent_vec, mapper) =
        bound_cs::<F>(&rate_count_ph1, &rate_count_ph2, split_index, round);
    // Incorporate changes and also run spa if the new rates are disordered.
    let _: Vec<_> = mapper
        .into_iter()
        .map(|(mu, new_rate)| ph2.replace_all_rates(&mu, new_rate))
        .collect();

    let ph2_merged = merge_tuple_vec(&to_use_vec, &remanent_vec, |x, y| x + y);
    // We might need to re run spa if the new value is too high.
    let ph2 = ph2.spa();

    let mut new_rates: HashMap<F, u32> = HashMap::new();
    for (lambda, k) in &rate_count_ph1 {
        for (mu, l) in &ph2_merged {
            let amount = k + l - 1;
            let rate = lambda.clone() + mu.clone();
            new_rates
                .entry(rate)
                .and_modify(|v| *v = (*v).max(amount))
                .or_insert(amount);
        }
    }

    // Here, we need to craft back the old rates.
    let old_rates = merge_tuple_vec(&rate_count_ph1, &ph2_merged, |x, y| x.max(y));

    // Create diagonal.
    let mut new_diagonal: Vec<F> = vec![];
    // Merge new and old rates, becasuse there can repeat and only the maximum is needed.
    // put rates in new diaognal
    merge_tuple_vec(&old_rates, &new_rates.into_iter().collect_vec(), |x, y| {
        x.max(y)
    })
    .iter()
    .for_each(|(lambda, k)| new_diagonal.extend(vec![lambda.clone(); (*k) as usize]));

    // sort rates (for SPA) and create new representation
    new_diagonal.sort_by(|x, y| {
        y.partial_cmp(x)
            .unwrap_or_else(|| panic!("Could not sort the values: {:?} and {:?}.", y, x))
    });
    let bidiagonal = Bidiagonal::<F>::from(Vector::from(new_diagonal));

    // Create new initial distribution.
    let mut delta = ph1.initial.kron_prod(&ph2.initial).elements.to_vec();
    delta.append(
        &mut ph1
            .initial
            .mult_by_scalar(&ph2.initial.rest())
            .elements
            .to_vec(),
    );
    delta.append(
        &mut ph2
            .initial
            .mult_by_scalar(&ph1.initial.rest())
            .elements
            .to_vec(),
    );

    // Run SPA from explicit to obtain the new initial probabilities.
    let vec_a = Array::from(ph1.repr().to_absorbing());
    let vec_b = Array::from(ph2.repr().to_absorbing());
    let arr_a = vec_a.to_shape((ph1.size(), 1)).unwrap().to_owned();
    let eye_m = Array::eye(ph2.size());
    let arr_b = vec_b.to_shape((ph2.size(), 1)).unwrap().to_owned();
    let eye_n = Array::eye(ph1.size());

    let getter = |i, j| {
        lazy_max_block::<F, Bidiagonal<F>>(
            ph1.repr(),
            ph2.repr(),
            &arr_a,
            &eye_m,
            &arr_b,
            &eye_n,
            i,
            j,
            ph1.size(),
            ph2.size(),
        )
    };

    Aph::<F, Bidiagonal<F>>::spa_with_getter(&delta.into(), &getter, bidiagonal)
}

pub fn mininimum_round_depth<F: PseudoField>(
    ph1: &BidiagonalAph<F>,
    ph2: &BidiagonalAph<F>,
    round: Round,
) -> BidiagonalAph<F> {
    let n = ph1.size();
    let m = ph2.size();
    let max_depth = n + m - 1;
    let mut diagonal = match round {
        Round::Down | Round::Zero => vec![F::min_value(); max_depth],
        Round::Up => vec![F::max_value(); max_depth],
        _ => todo!(),
    };
    let mut depth_mapper = HashMap::new();

    for k in 0..max_depth {
        for i in 0..k + 1 {
            let j = k - i;
            if j < m && i < n {
                let mut value = ph1.repr().get(i, i);
                value.add_assign(&ph2.repr().get(j, j));
                depth_mapper.insert((i, j), k);
                if let Some(v) = diagonal.get_mut(k) {
                    match round {
                        Round::Down | Round::Zero if (*v).lt(&value) => {
                            *v = value.clone();
                        }
                        Round::Up if (*v).gt(&value) => {
                            *v = value.clone();
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    let size = n * m;
    let getter = |i, j| {
        lazy_kron_sum::<F, Bidiagonal<F>>(ph1.repr(), ph2.repr(), i, j, ph1.size(), ph2.size())
    };
    let mut trimat: sprs::TriMatBase<Vec<usize>, Vec<F>> = TriMat::new((size, size));
    for row in 0..size {
        let (r_a, r_b) = (row / m, row.rem_euclid(m));
        let d = depth_mapper.get(&(r_a, r_b)).unwrap();
        let lambda = diagonal[*d].clone();
        let mu = &getter(row, row);
        if !lambda.eq(mu) {
            debug!("Replacing {mu} by {lambda}.");
        }
        for col in row + 1..size {
            let rate = getter(row, col);
            if !rate.is_zero() {
                let mut val = rate.clone();
                val.div_assign(mu);
                val.mul_assign(&lambda);
                trimat.add_triplet(row, col, val);
            }
        }
        trimat.add_triplet(row, row, lambda);
    }

    diagonal.sort_by(|x, y| {
        y.partial_cmp(x)
            .unwrap_or_else(|| panic!("Could not sort the values: {:?} and {:?}.", y, x))
    });
    let bidiagonal = Bidiagonal::<F>::from(Vector::from(diagonal));
    let delta = ph1.initial.kron_prod(&ph2.initial);

    let aph = Aph {
        initial: delta,
        repr: Sparse {
            size,
            matrix: trimat,
        },
    };
    aph.sparse_spa_w_bidiagonal(bidiagonal)
}

pub fn maximum_round_depth<F: PseudoField>(
    ph1: &BidiagonalAph<F>,
    ph2: &BidiagonalAph<F>,
    round: Round,
) -> BidiagonalAph<F> {
    let n = ph1.size();
    let m = ph2.size();
    let max_depth = n + m;
    let mut diagonal = match round {
        Round::Down | Round::Zero => vec![F::min_value(); max_depth],
        Round::Up => vec![F::max_value(); max_depth],
        _ => todo!(),
    };
    let mut depth_mapper = HashMap::new();
    // Collect diagonal + depths.
    for k in 0..max_depth {
        for i in 0..=k {
            let j = k - i;
            let value = if i < n && j < m {
                let mut value = ph1.repr().get(i, i);
                value.add_assign(&ph2.repr().get(j, j));
                Some(value)
            } else if i < n && j == m {
                Some(ph1.repr().get(i, i))
            } else if j < m && i == n {
                Some(ph2.repr().get(j, j))
            } else {
                None
            };
            depth_mapper.insert((i, j), k);
            if let Some(value) = value
                && let Some(v) = diagonal.get_mut(k)
            {
                match round {
                    Round::Down | Round::Zero if (*v).lt(&value) => {
                        *v = value.clone();
                    }
                    Round::Up if (*v).gt(&value) => {
                        *v = value.clone();
                    }
                    _ => {}
                }
            }
        }
    }

    let vec_a = Array::from(ph1.repr().to_absorbing());
    let vec_b = Array::from(ph2.repr().to_absorbing());
    let arr_a = vec_a.to_shape((ph1.size(), 1)).unwrap().to_owned();
    let eye_m = Array::eye(ph2.size());
    let arr_b = vec_b.to_shape((ph2.size(), 1)).unwrap().to_owned();
    let eye_n = Array::eye(ph1.size());

    let getter = |i, j| {
        lazy_max_block::<F, Bidiagonal<F>>(
            ph1.repr(),
            ph2.repr(),
            &arr_a,
            &eye_m,
            &arr_b,
            &eye_n,
            i,
            j,
            ph1.size(),
            ph2.size(),
        )
    };
    let size = n * m + n + m;

    let mut trimat: sprs::TriMatBase<Vec<usize>, Vec<F>> = TriMat::new((size, size));
    for row in 0..size {
        let (r_a, r_b) = if row < n * m {
            (row / m, row.rem_euclid(m))
        } else if row < n * m + n {
            (row - n * m, ph2.size())
        } else {
            (ph1.size(), row - (n * m + n))
        };

        let d = depth_mapper.get(&(r_a, r_b)).unwrap();
        let lambda = diagonal[*d].clone();
        let mu = &getter(row, row);
        if !lambda.eq(mu) {
            debug!("Replacing {mu} by {lambda}.");
        }
        for col in row + 1..size {
            let rate = getter(row, col);
            if !rate.is_zero() {
                let mut val = rate.clone();
                val.div_assign(mu);
                val.mul_assign(&lambda);
                trimat.add_triplet(row, col, val);
            }
        }
        trimat.add_triplet(row, row, lambda);
    }

    diagonal.sort_by(|x, y| {
        y.partial_cmp(x)
            .unwrap_or_else(|| panic!("Could not sort the values: {:?} and {:?}.", y, x))
    });
    let bidiagonal = Bidiagonal::<F>::from(Vector::from(diagonal));

    let mut delta = ph1.initial.kron_prod(&ph2.initial).elements.to_vec();
    delta.append(
        &mut ph1
            .initial
            .mult_by_scalar(&ph2.initial.rest())
            .elements
            .to_vec(),
    );
    delta.append(
        &mut ph2
            .initial
            .mult_by_scalar(&ph1.initial.rest())
            .elements
            .to_vec(),
    );

    let aph = Aph {
        initial: delta.into(),
        repr: Sparse {
            size,
            matrix: trimat,
        },
    };

    aph.sparse_spa_w_bidiagonal(bidiagonal)
}

#[allow(unused)]
pub fn minimum_round<F: PseudoField>(
    ph1: &BidiagonalAph<F>,
    ph2: &BidiagonalAph<F>,
    round: Round,
) -> BidiagonalAph<F> {
    let rate_count_ph1 = ph1.unique_rates_vec();
    let rate_count_ph2 = ph2.unique_rates_vec();

    let (to_set_rc, mut to_set_ph, fixed_rc, fixed_ph) =
        if rate_count_ph1.len() > rate_count_ph2.len() {
            (rate_count_ph1, ph1.clone(), rate_count_ph2, ph2)
        } else {
            (rate_count_ph2, ph2.clone(), rate_count_ph1, ph1)
        };

    let mapper = find_rate_replacement::<F>(&fixed_rc, &to_set_rc, round);
    // Incorporate changes and also run spa if the new rates are disordered.
    mapper
        .into_iter()
        .for_each(|(mu, new_rate)| to_set_ph.replace_all_rates(mu, new_rate));
    // Values were chosen so that SPA is not needed.
    let to_set_rc = to_set_ph.unique_rates_vec();

    let mut new_rates: HashMap<F, u32> = HashMap::new();
    for (lambda, k) in &fixed_rc {
        for (mu, l) in &to_set_rc {
            let amount = k + l - 1;
            let rate = lambda.clone() + mu.clone();
            new_rates
                .entry(rate)
                .and_modify(|v| *v = (*v).max(amount))
                .or_insert(amount);
        }
    }

    // Create diagonal.
    let mut new_diagonal: Vec<F> = vec![];
    new_rates
        .into_iter()
        .for_each(|(lambda, k)| new_diagonal.extend(vec![lambda; k as usize]));

    // Sort rates (necessary for SPA) create representation
    new_diagonal.sort_by(|x, y| {
        y.partial_cmp(x)
            .unwrap_or_else(|| panic!("Could not sort the values: {:?} and {:?}.", y, x))
    });
    let bidiagonal = Bidiagonal::<F>::from(Vector::from(new_diagonal));

    let delta = fixed_ph.initial.kron_prod(&to_set_ph.initial);

    // Run Lazy SPA to obtain the new initial probabilities.
    let getter = |i, j| {
        lazy_kron_sum::<F, Bidiagonal<F>>(
            fixed_ph.repr(),
            to_set_ph.repr(),
            i,
            j,
            fixed_ph.size(),
            to_set_ph.size(),
        )
    };
    Aph::<F, Bidiagonal<F>>::spa_with_getter(&delta, &getter, bidiagonal)
}

pub fn maximum_round<F: PseudoField>(
    ph1: &BidiagonalAph<F>,
    ph2: &BidiagonalAph<F>,
    round: Round,
) -> BidiagonalAph<F> {
    debug_assert_ne!(round, Round::Nearest, "Round can not be Nearest");

    let rate_count_ph1 = ph1.unique_rates_vec();
    let rate_count_ph2 = ph2.unique_rates_vec();
    let (to_set_rc, mut to_set_ph, fixed_rc, fixed_ph) =
        if rate_count_ph1.len() > rate_count_ph2.len() {
            (rate_count_ph1, ph1.clone(), rate_count_ph2, ph2)
        } else {
            (rate_count_ph2, ph2.clone(), rate_count_ph1, ph1)
        };
    let mapper = find_rate_replacement::<F>(&fixed_rc, &to_set_rc, round);

    // Incorporate changes and also run spa if the new rates are disordered.
    mapper
        .into_iter()
        .for_each(|(mu, new_rate)| to_set_ph.replace_all_rates(mu, new_rate));

    to_set_ph = to_set_ph.spa(); // We must run the SPA algorithm again, it may happen that a replaced value is smaller than one of the original rates.
    let to_set_rc = to_set_ph.unique_rates_vec();

    // Craft back the original rates.
    let old_rates = merge_tuple_vec(&fixed_rc, &to_set_rc, |x, y| x.max(y));

    let mut new_rates: HashMap<F, u32> = HashMap::new();
    for (lambda, k) in &fixed_rc {
        for (mu, l) in &to_set_rc {
            let amount = k + l - 1;
            let rate = lambda.clone() + mu.clone();
            new_rates
                .entry(rate)
                .and_modify(|v| *v = (*v).max(amount))
                .or_insert(amount);
        }
    }

    // Create diagonal.
    let mut new_diagonal: Vec<F> = vec![];
    // Merge new and old rates, becasuse there can repeat and only the maximum is needed.
    // put rates in new diagonal
    merge_tuple_vec(&old_rates, &new_rates.into_iter().collect_vec(), u32::max)
        .into_iter()
        .for_each(|(lambda, k)| new_diagonal.extend(vec![lambda; k as usize]));

    // sort rates (for SPA) and create new representation
    new_diagonal.sort_by(|x, y| {
        y.partial_cmp(x)
            .unwrap_or_else(|| panic!("Could not sort the values: {:?} and {:?}.", y, x))
    });
    let bidiagonal = Bidiagonal::<F>::from(Vector::from(new_diagonal));

    // Create new initial distribution.
    let mut delta = fixed_ph
        .initial
        .kron_prod(&to_set_ph.initial)
        .elements
        .to_vec();
    delta.append(
        &mut fixed_ph
            .initial
            .mult_by_scalar(&to_set_ph.initial.rest())
            .elements
            .to_vec(),
    );
    delta.append(
        &mut to_set_ph
            .initial
            .mult_by_scalar(&fixed_ph.initial.rest())
            .elements
            .to_vec(),
    );

    // Run SPA from explicit to obtain the new initial probabilities.
    let vec_a = Array::from(fixed_ph.repr().to_absorbing());
    let vec_b = Array::from(to_set_ph.repr().to_absorbing());
    let arr_a = vec_a.to_shape((fixed_ph.size(), 1)).unwrap().to_owned();
    let eye_m = Array::eye(to_set_ph.size());
    let arr_b = vec_b.to_shape((to_set_ph.size(), 1)).unwrap().to_owned();
    let eye_n = Array::eye(fixed_ph.size());

    let getter = |i, j| {
        lazy_max_block::<F, Bidiagonal<F>>(
            fixed_ph.repr(),
            to_set_ph.repr(),
            &arr_a,
            &eye_m,
            &arr_b,
            &eye_n,
            i,
            j,
            fixed_ph.size(),
            to_set_ph.size(),
        )
    };

    Aph::<F, Bidiagonal<F>>::spa_with_getter(&delta.into(), &getter, bidiagonal)
}
