//! Functions and algorithms for combining APH.

use std::collections::HashMap;

use itertools::Itertools;
use linalg::fields::{PseudoField, dot_product};
use log::info;
use ndarray::{Array, Array1, Array2, Axis};

use representation::{Representation, TriangularArray, kronecker_product_array};

use crate::linalg::Vector;
use crate::representation::{self, Bidiagonal};
use crate::{Aph, BidiagonalAph, linalg};

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
) -> BidiagonalAph<F> {
    let size = ph1.size() + ph2.size();
    info!("New size: {:?} (CON)", size);
    let mut delta = ph1.initial.elements.to_vec();
    delta.append(&mut vec![F::zero(); ph2.size()]);

    let prod_a = Array1::from(ph1.repr().to_absorbing())
        .into_shape_clone((ph1.size(), 1))
        .expect("msg");
    let prod_b = Array1::from(ph2.initial().elements.to_vec())
        .into_shape_clone((1, ph2.size()))
        .expect("msg");

    let prod = dot_product(&prod_a, &prod_b);

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

    assert!(
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

/// Given two PH distributions $(\overrightarrow{\alpha}, \mathbf{A})$, $(\overrightarrow{\beta}, \mathbf{B})$ of size $m$ and $n$ respectively. Then:
/// the minimum between them is the PH: $(\overrightarrow{\alpha} \oplus \overrightarrow{\beta}, \mathbf{A} \otimes \mathbf{B})$
///
/// Computes $min\{ph1, ph2\}$
pub fn min_ph<F: PseudoField, R: Representation<F>>(
    ph1: &Aph<F, R>,
    ph2: &Aph<F, R>,
) -> Aph<F, Bidiagonal<F>> {
    let size = ph1.size() * ph2.size();
    info!("New size: {:?} (MIN)", size);
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
) -> Aph<F, Bidiagonal<F>> {
    let size = ph1.size() * ph2.size() + ph1.size() + ph2.size();
    info!("New size: {:?} (MAX)", size);
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

    assert!(
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

// ---- First Algorithmic Improvement in Min/Max ----
pub fn min_ph_opt<F: PseudoField>(
    ph1: &BidiagonalAph<F>,
    ph2: &BidiagonalAph<F>,
) -> Aph<F, Bidiagonal<F>> {
    let delta = ph1.initial.kron_prod(&ph2.initial);
    let unique_rates_a = ph1.unique_rates();
    let unique_rates_b = ph2.unique_rates();
    let mut bidi: Vec<F> = vec![];

    // All rates are sum of rates
    for (lambda, k) in &unique_rates_a {
        for (mu, l) in &unique_rates_b {
            let amount = k + l - 1;
            let rate = lambda.clone() + mu.clone();
            bidi.extend(vec![rate; amount]);
        }
    }
    bidi.sort_by(|x, y| {
        y.partial_cmp(x)
            .unwrap_or_else(|| panic!("Could not sort the values: {:?} and {:?}.", y, x))
    });
    let bidiagonal = Bidiagonal::<F>::from(Vector::from(bidi));

    info!("New size: {:?} (MIN OPT)", bidiagonal.size());
    let getter = |i, j| lazy_kron_sum::<F, Bidiagonal<F>>(ph1.repr(), ph2.repr(), i, j);

    Aph::<F, Bidiagonal<F>>::spa_from_explicit(&delta, &getter, bidiagonal)
}

pub fn max_ph_opt<F: PseudoField>(
    ph1: &BidiagonalAph<F>,
    ph2: &BidiagonalAph<F>,
) -> Aph<F, Bidiagonal<F>> {
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

    let unique_rates_a = ph1.unique_rates();
    let unique_rates_b = ph2.unique_rates();
    let mut bidi: Vec<F> = vec![];

    // Unique rates from each are repeated the same number of times as before, their number of states with the sums of rates can be reduced.
    let _: Vec<_> = unique_rates_a
        .iter()
        .map(|(lambda, k)| bidi.extend(vec![lambda.clone(); *k]))
        .collect();

    let _: Vec<_> = unique_rates_b
        .iter()
        .map(|(mu, l)| bidi.extend(vec![mu.clone(); *l]))
        .collect();

    for (lambda, k) in &unique_rates_a {
        for (mu, l) in &unique_rates_b {
            let amount = k + l - 1;
            let rate = lambda.clone() + mu.clone();
            bidi.extend(vec![rate; amount]);
        }
    }
    bidi.sort_by(|x, y| {
        y.partial_cmp(x)
            .unwrap_or_else(|| panic!("Could not sort the values: {:?} and {:?}.", y, x))
    });
    let bidiagonal = Bidiagonal::<F>::from(Vector::from(bidi));
    info!("New size: {:?} (MAX OPT)", bidiagonal.size());

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
        )
    };

    Aph::<F, Bidiagonal<F>>::spa_from_explicit(&delta.into(), &getter, bidiagonal)
}

// ---- Second Algorithmic Improvement in Min/Max ----
pub fn min_minimal<F: PseudoField>(
    ph1: &BidiagonalAph<F>,
    ph2: &BidiagonalAph<F>,
) -> BidiagonalAph<F> {
    let delta = ph1.initial.kron_prod(&ph2.initial);

    let rate_count_ph1 = ph1.unique_rates();
    let rate_count_ph2 = ph2.unique_rates();
    let mut new_diagonal: Vec<F> = vec![];

    // All rates are sum of rates
    let mut new_rates: HashMap<F, usize> = HashMap::new();
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
        .for_each(|(lambda, k)| new_diagonal.extend(vec![lambda.clone(); *k]));

    // Sort rates (necessary for SPA) create represetantion
    new_diagonal.sort_by(|x, y| {
        y.partial_cmp(x)
            .unwrap_or_else(|| panic!("Could not sort the values: {:?} and {:?}.", y, x))
    });
    let bidiagonal = Bidiagonal::<F>::from(Vector::from(new_diagonal));
    info!("New size: {:?} (MIN MINIMAL)", bidiagonal.size());

    // Run SPA from explicit to obtain the new initial probabilities.
    let getter = |i, j| lazy_kron_sum::<F, Bidiagonal<F>>(ph1.repr(), ph2.repr(), i, j);
    Aph::<F, Bidiagonal<F>>::spa_from_explicit(&delta, &getter, bidiagonal)
}

// /// Merge two hashmaps,
// fn merge_maps<F>(map1: &HashMap<F, usize>, map2: &HashMap<F, usize>) -> HashMap<F, usize>
// where
//     F: Clone + Eq + std::hash::Hash,
// {
//     let mut result = map1.clone();
//     for (key, value) in map2.iter() {
//         if map1.contains_key(key) {
//             let previous = map1.get(key).unwrap();
//             if previous < value {
//                 *result.get_mut(key).unwrap() += value - previous;
//             }
//         } else {
//             result.insert(key.clone(), *value);
//         }
//     }
//     result
// }

fn merge_maps<F>(map1: &HashMap<F, usize>, map2: &HashMap<F, usize>) -> HashMap<F, usize>
where
    F: Eq + std::hash::Hash + Clone,
{
    let mut result = map1.clone();

    for (key, &value2) in map2 {
        result
            .entry(key.clone())
            .and_modify(|value1| {
                if *value1 < value2 {
                    *value1 += value2 - *value1;
                }
            })
            .or_insert(value2);
    }
    result
}

pub fn max_minimal<F: PseudoField>(
    ph1: &BidiagonalAph<F>,
    ph2: &BidiagonalAph<F>,
) -> BidiagonalAph<F> {
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

    // delta.append(&mut vec![ph2.initial.rest(); ph1.initial.size()]);
    // delta.append(&mut vec![ph1.initial.rest(); ph2.initial.size()]);

    // Collect old rates, the maximum number of times they appear in the input will stay.
    let rate_count_ph1 = ph1.repr().rate_count();
    let rate_count_ph2 = ph2.repr().rate_count();
    let old_rates: HashMap<F, usize> = merge_maps::<F>(&rate_count_ph1, &rate_count_ph2);

    let mut new_diagonal: Vec<F> = vec![];
    let mut new_rates: HashMap<F, usize> = HashMap::new();
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
        .for_each(|(lambda, k)| new_diagonal.extend(vec![lambda.clone(); *k]));

    // sort rates (for SPA) and create new representation
    new_diagonal.sort_by(|x, y| {
        y.partial_cmp(x)
            .unwrap_or_else(|| panic!("Could not sort the values: {:?} and {:?}.", y, x))
    });
    let bidiagonal = Bidiagonal::<F>::from(Vector::from(new_diagonal));
    info!("New size: {:?} (MAX MINIMAL)", bidiagonal.size());

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
        )
    };

    Aph::<F, Bidiagonal<F>>::spa_from_explicit(&delta.into(), &getter, bidiagonal)
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
) -> Option<BidiagonalAph<F>>
where
    I: Iterator<Item = &'a BidiagonalAph<F>>,
{
    let first = iter.next()?.clone(); // Get the first element, return None if empty
    Some(iter.fold(first, |acc, item| operator(&acc, item)))
}

// TODO: Remove this two
pub fn try_max_phs<F: PseudoField, R: Representation<F>>(
    instances: &[Aph<F, R>],
) -> Option<Aph<F, Bidiagonal<F>>> {
    if instances.is_empty() {
        return None;
    }
    let mut result = instances.first().unwrap().spa();
    let instances = instances.iter().map(|aph| aph.spa()).collect_vec();
    for instance in &instances[1..] {
        result = max_ph(&result, instance);
    }
    Some(result)
}

pub fn try_min_phs<F: PseudoField, R: Representation<F>>(
    instances: &[Aph<F, R>],
) -> Option<Aph<F, Bidiagonal<F>>> {
    if instances.is_empty() {
        return None;
    }
    let mut result = instances.first().unwrap().spa();
    let instances = instances.iter().map(|aph| aph.spa()).collect_vec();
    for instance in &instances[1..] {
        result = min_ph(&result, instance);
    }
    Some(result)
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
) -> F {
    let n = repr_a.size();
    let m = repr_b.size();
    assert!(
        usize::max(i, j) < n * m + n + m,
        "Out of bounds for index ({},{})",
        i,
        j
    );
    let nm = n * m;

    if i > j {
        F::zero()
    } else if i < nm && j < nm {
        // Top lft block, A + B
        lazy_kron_sum(repr_a, repr_b, i, j)
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
    let shape_a = arr_a.shape();
    let shape_b = arr_b.shape();

    let rows = shape_a[0] * shape_b[0];
    let cols = shape_a[1] * shape_b[1];
    assert!(i <= rows && j <= cols, "Index out of bounds ({},{})", i, j);

    let row_a = i / shape_b[0];
    let col_a = j / shape_b[1];
    let row_b = i.rem_euclid(shape_b[0]);
    let col_b = j.rem_euclid(shape_b[1]);

    let mut res = arr_a.get((row_a, col_a)).unwrap().clone();
    res.mul_assign(arr_b.get((row_b, col_b)).unwrap());
    res
}

/// Compute the entry [i,j] of the kronecker sum of two representations in a lazy manner, i.e. only compute the
/// desired entry and not the entire matrix.
pub fn lazy_kron_sum<F: PseudoField, R: Representation<F>>(
    repr_a: &R,
    repr_b: &R,
    i: usize,
    j: usize,
) -> F {
    let n = repr_a.size();
    let m = repr_b.size();
    assert!(
        usize::max(i, j) < n * m,
        "Index ({},{}) is out of bounds. Size: ({},{})",
        i,
        j,
        n * m,
        n * m
    );
    let r_a = i / m;
    let r_b = i.rem_euclid(m);
    let c_a = j / m;
    let c_b = j.rem_euclid(m);

    let mut result = F::zero();

    // Contribution from A \otimes I_m
    if r_b == c_b && r_a <= c_a {
        result.add_assign(&repr_a.get(r_a, c_a));
    };

    // Contribution from I_n \otimes B
    if r_a == c_a && r_b <= c_b {
        result.add_assign(&repr_b.get(r_b, c_b));
    };

    result
}
