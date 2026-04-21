//! Functions and algorithms for combining APH.

use itertools::Itertools;
use linalg::fields::PseudoField;
use ndarray::{Array, Array1, Array2, Axis};
use std::collections::HashMap;

use crate::formats::ft::utils::RoundMode;
use crate::linalg::Vector;
use crate::linalg::fields::Round;
use crate::representation::{self, bidiagonal::Bidiagonal};
use crate::round_op::*;
use crate::voting::build_vot_bidi;
use crate::{Aph, BidiagonalAph, linalg};
use representation::{Representation, kronecker_product_array, triangular::TriangularArray};

/// Given two PH distributions $(\vec{\alpha}, \mathbf{A})$, $(\vec{\beta}, \mathbf{B})$ of size $m$ and $n$ respectively. Then:
/// the convolution of them is the PH: $(\vec{\delta}, \mathbf{D})$
///
/// $$
/// \mathbf{D} =
/// \begin{bmatrix}
/// \mathbf{A} & \vec{A} \beta   \\
/// \mathbf{0} & \mathbf{B}
/// \end{bmatrix}
/// $$
///
/// Computes $con\{ph1, ph2\}$
pub fn con_ph<F: PseudoField, R: Representation<F>>(
    ph1: &Aph<F, R>,
    ph2: &Aph<F, R>,
) -> BidiagonalAph<F> {
    let size = ph1.size() + ph2.size();
    let mut delta = ph1.initial.elements.clone().to_vec();
    delta.append(
        &mut ph2
            .initial
            .mult_by_scalar(&ph1.initial.rest())
            .elements
            .to_vec(),
    );

    let prod_a = Array1::from(ph1.repr().to_absorbing())
        .into_shape_clone((ph1.size(), 1))
        .expect("msg");
    let prod_b = Array1::from(ph2.initial().elements.to_vec())
        .into_shape_clone((1, ph2.size()))
        .expect("msg");

    let prod = linalg::fields::dot_product(&prod_a, &prod_b);

    let top = ndarray::concatenate(
        Axis(1),
        &[ph1.repr.to_array_repr().matrix.view(), prod.view()],
    )
    .expect("Something went wrong when the `top` part of the matrix.");

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

    debug_assert!(
        matrix.shape() == [size, size],
        "Shape of matrix is {:?}, but actual size is {:?}",
        matrix.shape(),
        size
    );
    let delta: Vector<F> = delta.into();
    Aph {
        initial: delta,
        repr: TriangularArray { size, matrix },
    }
    .spa()
}

/// Given two PH distributions $(\vec{\alpha}, \mathbf{A})$, $(\vec{\beta}, \mathbf{B})$ of size $m$ and $n$ respectively, and $p \in [0,1]$. Then:
/// the $p$-mixture of them is the PH: $([p \vec{\alpha}, (1-p) \vec{beta}], \mathbf{D})$
///
/// $$
/// \mathbf{D} =
/// \begin{bmatrix}
/// \mathbf{A} & \mathbf{0} \beta   \\
/// \mathbf{0} & \mathbf{B}
/// \end{bmatrix}
/// $$
///
/// Computes $mix\{ph1, ph2\}$
pub fn mix_ph<F: PseudoField, R: Representation<F>>(
    ph1: &Aph<F, R>,
    ph2: &Aph<F, R>,
    p: &F,
) -> BidiagonalAph<F> {
    debug_assert!(
        p.ge(&F::zero()) && p.le(&F::one()),
        "`p` should be between one and zero."
    );
    let size = ph1.size() + ph2.size();
    let mut one_minus_p = F::one();
    one_minus_p.sub_assign(p);
    let mut delta = ph1.initial.mult_by_scalar(p).elements.to_vec();
    delta.append(&mut ph2.initial.mult_by_scalar(&one_minus_p).elements.to_vec());

    let top = ndarray::concatenate(
        Axis(1),
        &[
            ph1.repr.to_array_repr().matrix.view(),
            Array::zeros((ph1.size(), ph2.size())).view(),
        ],
    )
    .expect("Something went wrong when the `top` part of the matrix.");

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

    debug_assert!(
        matrix.shape() == [size, size],
        "Shape of matrix is {:?}, but actual size is {:?}",
        matrix.shape(),
        size
    );
    let delta: Vector<F> = delta.into();
    Aph {
        initial: delta,
        repr: TriangularArray { size, matrix },
    }
    .spa()
}

// ---- General Min/Max constructs ----

/// Given two PH distributions $(\vec{\alpha}, \mathbf{A})$, $(\vec{\beta}, \mathbf{B})$ of size $m$ and $n$ respectively. Then:
/// the minimum between them is the PH: $(\vec{\alpha} \oplus \vec{\beta}, \mathbf{A} \otimes \mathbf{B})$
///
/// Computes $min\{ph1, ph2\}$
pub fn min_ph<F: PseudoField, R: Representation<F>>(
    ph1: &Aph<F, R>,
    ph2: &Aph<F, R>,
) -> Aph<F, Bidiagonal<F>> {
    let _size = ph1.size() * ph2.size();
    let delta = ph1.initial.kron_prod(&ph2.initial);

    let repr_d = ph1
        .repr()
        .to_array_repr()
        .kron_sum(&ph2.repr.to_array_repr());

    Aph {
        initial: delta,
        repr: repr_d,
    }
    .spa()
}

/// Given two PH distributions $(\vec{\alpha}, \mathbf{A})$, $(\vec{\beta}, \mathbf{B})$ of size $m$ and $n$ respectively. Then:
/// the maximum between them is the PH: $( \[ \vec{\alpha} \otimes \vec{\beta}, \vec{\beta}_{n+1}\vec{\alpha}, \vec{\alpha}_{m+1}\vec{\beta} \], \mathbf{D})$
/// where:
/// $$
/// \mathbf{D} =
/// \begin{bmatrix}
/// \mathbf{A} \oplus \mathbf{B} & \mathbf{I}_{m} \otimes \vec{B} & \vec{A} \otimes \mathbf{I}_n   \\
/// \mathbf{0}                   & \mathbf{A}                                & \mathbf{0}                                \\
/// \mathbf{0}                   & \mathbf{0}                                & \mathbf{B}
/// \end{bmatrix}
/// $$
///
/// Computes $max\{ph1, ph2\}$
pub fn max_ph<F: PseudoField, R: Representation<F>>(
    ph1: &Aph<F, R>,
    ph2: &Aph<F, R>,
) -> Aph<F, Bidiagonal<F>> {
    let size = ph1.size() * ph2.size() + ph1.size() + ph2.size();
    // let mut delta = ph1.initial.kron_prod(&ph2.initial).elements.to_vec();
    // delta.append(&mut vec![F::zero(); ph1.initial.size() + ph2.size()]);
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
        .expect("Something went wrong on the `top` part of the matrix.");

    let mid = ndarray::concatenate(
        Axis(1),
        &[
            Array::zeros((ph1.size(), ph1.size() * ph2.size())).view(),
            ph1.repr().to_array_repr().matrix.view(),
            Array::zeros((ph1.size(), ph2.size())).view(),
        ],
    )
    .expect("Something went wrong on the `mid` part of the matrix.");

    let bot = ndarray::concatenate(
        Axis(1),
        &[
            Array::zeros((ph2.size(), ph1.size() * ph2.size())).view(),
            Array::zeros((ph2.size(), ph1.size())).view(),
            ph2.repr().to_array_repr().matrix.view(),
        ],
    )
    .expect("Something went wrong on the `bot` part of the matrix.");

    let matrix = ndarray::concatenate(Axis(0), &[top.view(), mid.view(), bot.view()])
    .expect("Something went wrong when assembling the arrays `top`, `mid` and `bot`. Please check that the sizes are correct.");

    debug_assert!(
        matrix.shape() == [size, size],
        "Shape of matrix is {:?}, but Size is {:?}",
        matrix.shape(),
        size
    );

    Aph {
        initial: delta.into(),
        repr: TriangularArray { size, matrix },
    }
    .spa()
}

// ---- Algorithmic Improvement for Min/Max in OBR ----
fn min_bidi<F: PseudoField>(ph1: &BidiagonalAph<F>, ph2: &BidiagonalAph<F>) -> BidiagonalAph<F> {
    let delta = ph1.initial.kron_prod(&ph2.initial);

    let rate_count_ph1 = ph1.unique_rates();
    let rate_count_ph2 = ph2.unique_rates();
    let mut new_diagonal: Vec<F> = vec![];

    // All rates are sum of rates
    let mut new_rates: HashMap<F, u32> = HashMap::new();
    for (lambda, k) in &rate_count_ph1 {
        for (mu, l) in &rate_count_ph2 {
            let amount = k + l - 1;
            let rate = lambda.clone() + mu.clone();
            new_rates
                .entry(rate)
                .and_modify(|v| *v = (*v).max(amount))
                .or_insert(amount);
        }
    }

    new_rates
        .iter()
        .for_each(|(lambda, k)| new_diagonal.extend(vec![lambda.clone(); (*k) as usize]));

    // Sort rates (necessary for SPA) create represetantion
    new_diagonal.sort_by(|x, y| {
        y.partial_cmp(x)
            .unwrap_or_else(|| panic!("Could not sort the values: {:?} and {:?}.", y, x))
    });

    let bidiagonal = Bidiagonal::<F>::from(Vector::from(new_diagonal));

    // Run SPA from explicit to obtain the new initial probabilities.
    let getter = |i, j| {
        lazy_kron_sum::<F, Bidiagonal<F>>(ph1.repr(), ph2.repr(), i, j, ph1.size(), ph2.size())
    };
    Aph::<F, Bidiagonal<F>>::spa_with_getter(&delta, &getter, bidiagonal)
}

///
/// Merges hashmaps of rate counts, when if the same key is found, instead it just keeps track of the maximum.
///
fn merge_maps<F>(map1: &HashMap<F, u32>, map2: &HashMap<F, u32>) -> HashMap<F, u32>
where
    F: Eq + std::hash::Hash + Clone,
{
    let mut result = map1.clone();
    for (key, &value2) in map2 {
        result
            .entry(key.clone())
            .and_modify(|value1| {
                *value1 = u32::max(*value1, value2);
            })
            .or_insert(value2);
    }
    result
}

fn max_bidi<F: PseudoField>(ph1: &BidiagonalAph<F>, ph2: &BidiagonalAph<F>) -> BidiagonalAph<F> {
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

    // Collect old rates, the maximum number of times they appear in the input will stay.
    let rate_count_ph1 = ph1.unique_rates();
    let rate_count_ph2 = ph2.unique_rates();

    let old_rates: HashMap<F, u32> = merge_maps::<F>(&rate_count_ph1, &rate_count_ph2);

    let mut new_diagonal: Vec<F> = vec![];
    let mut new_rates: HashMap<F, u32> = HashMap::new();
    for (lambda, k) in &rate_count_ph1 {
        for (mu, l) in &rate_count_ph2 {
            let amount = k + l - 1;
            let rate = lambda.clone() + mu.clone();
            new_rates
                .entry(rate)
                .and_modify(|v| *v = (*v).max(amount))
                .or_insert(amount);
        }
    }

    // Merge new and old rates, becasuse there can repeat and only the maximum is needed.
    // put rates in new diaognal
    merge_maps::<F>(&old_rates, &new_rates)
        .iter()
        .for_each(|(lambda, k)| new_diagonal.extend(vec![lambda.clone(); (*k) as usize]));

    // sort rates (for SPA) and create new representation
    new_diagonal.sort_by(|x, y| {
        y.partial_cmp(x)
            .unwrap_or_else(|| panic!("Could not sort the values: {:?} and {:?}.", y, x))
    });
    let bidiagonal = Bidiagonal::<F>::from(Vector::from(new_diagonal));

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

// ---- Multiple inputs ----

pub fn apply_many<
    'a,
    I,
    F: PseudoField + 'a,
    T: Fn(&BidiagonalAph<F>, &BidiagonalAph<F>) -> BidiagonalAph<F>,
>(
    mut iter: I,
    operator: T,
) -> BidiagonalAph<F>
where
    I: Iterator<Item = &'a BidiagonalAph<F>>,
{
    let first = iter
        .next()
        .expect("Can not do operations on 0 inputs.")
        .clone();
    iter.fold(first, |acc, item| operator(&acc, item))
}

// ---- Lazy computation of kronecker products. ----

/// Compute the entry [i,j] of the block matrix from the Max of two representations in a lazy manner, i.e. only compute the
/// desired entry and not the entire matrix.
#[allow(clippy::too_many_arguments)]
pub fn lazy_max_block<F: PseudoField, R: Representation<F>>(
    repr_a: &R,
    repr_b: &R,
    arr_a: &Array2<F>,
    eye_m: &Array2<F>,
    arr_b: &Array2<F>,
    eye_n: &Array2<F>,
    i: usize,
    j: usize,
    n: usize,
    m: usize,
) -> F {
    // let n = repr_a.size();
    // let m = repr_b.size();
    debug_assert!(
        usize::max(i, j) < n * m + n + m,
        "Out of bounds for index ({i},{j})",
    );
    let nm = n * m;

    if i > j {
        F::zero()
    } else if i < nm && j < nm {
        // Top lft block, A + B
        lazy_kron_sum(repr_a, repr_b, i, j, n, m)
    } else if i < nm && nm <= j && j < nm + n {
        // Top mid block, I_n x B_abs
        let local_j = j - nm;
        lazy_kron_prod(eye_n, arr_b, i, local_j)
    } else if i < nm && j < nm + n + m {
        // Top rgt block, A_abs x I_m
        let local_j = j - (nm + n);
        lazy_kron_prod(arr_a, eye_m, i, local_j)
    } else if nm <= i && i < nm + n && nm <= j && j < nm + n {
        // Middle-middle block: A
        let local_i = i - nm;
        let local_j = j - nm;
        if local_i <= local_j {
            repr_a.get(local_i, local_j)
        } else {
            F::zero()
        }
    } else if nm + n <= i && i < nm + n + m && nm + n <= j && j < nm + n + m {
        // Bottom-right block: B
        let local_i = i - (nm + n);
        let local_j = j - (nm + n);
        if local_i <= local_j {
            repr_b.get(local_i, local_j)
        } else {
            F::zero()
        }
    } else {
        F::zero()
    }
}

/// Compute the entry [i,j] of the kronecker product of two representations in a lazy manner, i.e. only compute the
/// desired entry and not the entire matrix.
fn lazy_kron_prod<F: PseudoField>(arr_a: &Array2<F>, arr_b: &Array2<F>, i: usize, j: usize) -> F {
    // let n = arr_a.shape();
    let m = arr_b.shape();

    // let rows = n[0] * shape_b[0];
    // let cols = n[1] * shape_b[1];
    // debug_assert!(i <= rows && j <= cols, "Index out of bounds ({},{})", i, j);

    let r_a = i / m[0];
    let c_a = j / m[1];
    let r_b = i.rem_euclid(m[0]);
    let c_b = j.rem_euclid(m[1]);

    let mut res = arr_a.get((r_a, c_a)).unwrap().clone();
    res.mul_assign(arr_b.get((r_b, c_b)).unwrap());
    res
}

/// Compute the entry [i,j] of the kronecker sum of two representations in a lazy manner, i.e. only compute the
/// desired entry and not the entire matrix.
pub fn lazy_kron_sum<F: PseudoField, R: Representation<F>>(
    repr_a: &R,
    repr_b: &R,
    i: usize,
    j: usize,
    n: usize,
    m: usize,
) -> F {
    debug_assert!(
        usize::max(i, j) < n * m,
        "Index ({i},{j}) is out of bounds. Size: ({},{})",
        n * m,
        n * m
    );
    let r_a = i / m;
    let r_b = i.rem_euclid(m);
    let c_a = j / m;
    let c_b = j.rem_euclid(m);
    let mut result = F::zero();

    // From A \otimes I_m
    if r_b == c_b && r_a <= c_a {
        result.add_assign(&repr_a.get(r_a, c_a));
    };

    // From I_n \otimes B
    if r_a == c_a && r_b <= c_b {
        result.add_assign(&repr_b.get(r_b, c_b));
    };

    result
}

/// Given two PH distributions $(\vec{\alpha}, \mathbf{A})$, $(\vec{\beta}, \mathbf{B})$ of size $m$ and $n$ respectively. Then:
/// the convolution of them is the PH: $(\vec{\delta}, \mathbf{D})$
///
/// $$
/// \mathbf{D} =
/// \begin{bmatrix}
/// \mathbf{A} & \vec{A} \beta   \\
/// \mathbf{0} & \mathbf{B}
/// \end{bmatrix}
/// $$
///
/// Computes $con\{ph1, ph2\}$
fn con_bidi<F: PseudoField>(ph1: &BidiagonalAph<F>, ph2: &BidiagonalAph<F>) -> BidiagonalAph<F> {
    let mut delta = ph1.initial.elements.clone().to_vec();
    delta.append(
        &mut ph2
            .initial
            .mult_by_scalar(&ph1.initial.rest())
            .elements
            .to_vec(),
    );

    let mut leftover = ph1.repr().get(ph1.size() - 1, ph1.size() - 1);
    leftover.neg_assign();
    let leftovers = ph2.initial.mult_by_scalar(&leftover);

    let mut repr = Vec::from(ph1.repr().clone());
    repr.append(&mut Vec::from(ph2.repr().clone()));

    let mut bidiagonal = Bidiagonal::from(Vector::from(repr.clone()));
    bidiagonal.into_ordered();

    let getter = |i: usize, j: usize| {
        if i == j {
            repr[i].clone()
        } else if i == ph1.size() - 1 && j >= i {
            leftovers[(j - i) - 1].clone()
        } else if i + 1 == j {
            let mut elem = repr[i].clone();
            elem.neg_assign();
            elem
        } else {
            F::zero()
        }
    };

    Aph::<F, Bidiagonal<F>>::spa_with_getter(&delta.into(), &getter, bidiagonal)
}

#[allow(unused)]
fn mix_bidi<F: PseudoField>(
    ph1: &BidiagonalAph<F>,
    ph2: &BidiagonalAph<F>,
    p: &F,
) -> BidiagonalAph<F> {
    debug_assert!(
        p.ge(&F::zero()) && p.le(&F::one()),
        "`p` should be between one and zero."
    );
    let mut one_minus_p = F::one();
    one_minus_p.sub_assign(p);
    let mut delta = ph1.initial.mult_by_scalar(p).elements.to_vec();
    delta.append(&mut ph2.initial.mult_by_scalar(&one_minus_p).elements.to_vec());

    let mut repr = Vec::from(ph1.repr().clone());
    repr.append(&mut Vec::from(ph2.repr().clone()));

    let mut bidiagonal = Bidiagonal::from(Vector::from(repr.clone()));
    bidiagonal.into_ordered();

    let getter = |i: usize, j: usize| {
        if i == j {
            repr[i].clone()
        } else if i + 1 == j {
            let mut elem = repr[i].clone();
            elem.neg_assign();
            elem
        } else {
            F::zero()
        }
    };

    Aph::<F, Bidiagonal<F>>::spa_with_getter(&delta.into(), &getter, bidiagonal)
}

// Public interfaces
pub fn convolution_bidiagonal<'a, I, F: PseudoField + 'a>(
    inputs: I,
    _round: Round,
) -> BidiagonalAph<F>
where
    I: Iterator<Item = &'a BidiagonalAph<F>>,
{
    apply_many(inputs, |x, y| con_bidi::<F>(x, y))
}

pub fn minimum_bidiagonal<'a, I, F: PseudoField + 'a>(
    inputs: I,
    round: Round,
    mode: RoundMode,
) -> BidiagonalAph<F>
where
    I: Iterator<Item = &'a BidiagonalAph<F>>,
{
    match (round, mode) {
        (Round::Down, RoundMode::Mix) | (Round::Down, RoundMode::MixTLE) => {
            apply_many(inputs, |x, y| minimum_round::<F>(x, y, Round::Down))
        }
        (Round::Down, RoundMode::DepthTLE) | (Round::Down, RoundMode::Depth) => {
            apply_many(inputs, |x, y| mininimum_round_depth::<F>(x, y, Round::Down))
        }
        (Round::Up, RoundMode::MixTLE) | (Round::Up, RoundMode::Mix) => {
            apply_many(inputs, |x, y| minimum_round::<F>(x, y, Round::Up))
        }
        (Round::Up, RoundMode::DepthTLE) | (Round::Up, RoundMode::Depth) => {
            apply_many(inputs, |x, y| mininimum_round_depth::<F>(x, y, Round::Up))
        }
        (Round::Nearest, _) | (Round::Zero, _) => apply_many(inputs, |x, y| min_bidi::<F>(x, y)),
    }
}

pub fn maximum_bidiagonal<'a, I, F: PseudoField + 'a>(
    inputs: I,
    round: Round,
    mode: RoundMode,
) -> BidiagonalAph<F>
where
    I: Iterator<Item = &'a BidiagonalAph<F>>,
{
    match (round, mode) {
        (Round::Down, RoundMode::MixTLE) | (Round::Down, RoundMode::Mix) => {
            apply_many(inputs, |x, y| maximum_round::<F>(x, y, Round::Down))
        }
        (Round::Down, RoundMode::DepthTLE) | (Round::Down, RoundMode::Depth) => {
            apply_many(inputs, |x, y| maximum_round_depth::<F>(x, y, Round::Down))
        }
        (Round::Up, RoundMode::MixTLE) | (Round::Up, RoundMode::Mix) => {
            apply_many(inputs, |x, y| maximum_round::<F>(x, y, Round::Up))
        }
        (Round::Up, RoundMode::DepthTLE) | (Round::Up, RoundMode::Depth) => {
            apply_many(inputs, |x, y| maximum_round_depth::<F>(x, y, Round::Up))
        }
        (Round::Nearest, _) | (Round::Zero, _) => apply_many(inputs, |x, y| max_bidi::<F>(x, y)),
    }
}

pub fn vot_bidiagonal<'a, I, F: PseudoField + 'a>(
    inputs: I,
    k: u32,
    _round: Round,
) -> BidiagonalAph<F>
where
    I: Iterator<Item = &'a BidiagonalAph<F>>,
{
    build_vot_bidi(&inputs.collect_vec(), k)
}
