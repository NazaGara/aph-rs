use clap::ValueEnum;
use itertools::Itertools;
use log::debug;
use serde::{Deserialize, Serialize};
use sprs::TriMat;
use std::collections::HashMap;
use std::fmt::Debug;

use crate::aph::Aph;
use crate::linalg::fields::{PseudoField, Round};
use crate::representation::sparse::Sparse;

#[derive(Debug, Copy, Clone, ValueEnum, Serialize, Deserialize)]
pub enum RoundMode {
    Depth,
    Mix,
}

#[derive(Debug, Copy, Clone, ValueEnum, Serialize, Deserialize)]
pub enum ConstructionMethod {
    Matrix,
    Sparse,
    CoreSeries,
    Triangular,
}

pub(crate) fn count_rate_by_levels_round<F: PseudoField>(
    depths: &[u32],
    diagonal: &[F],
    aph: &mut Aph<F, Sparse<F>>,
    round: Round,
) -> HashMap<F, u32> {
    let size = aph.size();
    let mut trimat: sprs::TriMatBase<Vec<usize>, Vec<F>> = TriMat::new((size, size));
    let matrix = aph.repr().matrix.to_csr::<usize>();

    let mut counts = HashMap::new();
    // Group by depth, keep the index of each state in the matrix with the value that must be used to
    let mut idx_value_vec = diagonal
        .iter()
        .enumerate()
        .zip(depths)
        .map(|((idx, rate), lvl)| (idx, rate, lvl))
        .into_group_map_by(|a| *a.2)
        .into_values()
        .flat_map(|values| {
            let value = match round {
                Round::Up => *values
                    .iter()
                    .map(|(_, mid, _)| mid)
                    .max_by(|x, y| y.partial_cmp(x).unwrap_or(std::cmp::Ordering::Equal))
                    .unwrap(),
                Round::Down | Round::Zero => *values
                    .iter()
                    .map(|(_, mid, _)| mid)
                    .max_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal))
                    .unwrap(),
                _ => unreachable!("Use the analogous method without rounding."),
            };
            //TODO: To put a condition like relative tolerance (|lambda - mu| < |mu| * e) on the changes, we must also consider the counts.
            // So, it is better to map the changes first, and then do it.
            counts
                .entry(value.clone())
                .and_modify(|c| *c += 1)
                .or_insert(1);

            // create the new vector dropping the third element,
            // but replacing all middle values by the max
            values
                .iter()
                .map(|(a, _, _)| (*a, value.clone()))
                .collect_vec()
        })
        .collect_vec();
    idx_value_vec.sort_by(|x, y| x.0.cmp(&y.0));

    for (row, lambda) in idx_value_vec.into_iter() {
        let mu = diagonal.get(row).unwrap();
        if !lambda.eq(mu) {
            debug!("Replacing {mu} by {lambda}.");
        }
        for col in row + 1..size {
            if let Some(mu_i) = matrix.get(row, col) {
                let mut new_val = mu_i.clone();
                new_val.div_assign(mu);
                new_val.mul_assign(&lambda);
                trimat.add_triplet(row, col, new_val);
            }
        }
        trimat.add_triplet(row, row, lambda.clone());
    }

    let new_aph = Aph {
        initial: aph.initial.clone(),
        repr: Sparse {
            size,
            matrix: trimat,
        },
    };
    *aph = new_aph;
    counts
}

/*
for (row, lambda) in idx_value_vec.into_iter() {
    let mu = diagonal.get(row).unwrap();
    let mut thresh = mu.clone();
    thresh.abs_assign();
    thresh.mul_assign(&F::from(f64::EPSILON));
    let mut diff = mu.clone();
    diff.sub_assign(&lambda);
    diff.abs_assign();

    if diff.gt(&thresh) {
        // debug!("NOT Replacing {mu} by {lambda}.");
        for col in row + 1..size {
            if let Some(mu_i) = matrix.get(row, col) {
                trimat.add_triplet(row, col, mu_i.clone());
            }
        }
        trimat.add_triplet(row, row, mu.clone());
    } else {
        debug!("Replacing {mu} by {lambda}: {diff} <= {thresh}");
        for col in row + 1..size {
            if let Some(mu_i) = matrix.get(row, col) {
                let mut new_val = mu_i.clone();
                new_val.div_assign(mu);
                new_val.mul_assign(&lambda);
                trimat.add_triplet(row, col, new_val);
            }
        }
        trimat.add_triplet(row, row, lambda.clone());
    }
}
*/
