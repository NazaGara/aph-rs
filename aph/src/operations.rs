//! Functions and algorithms for combining APH.

use itertools::Itertools;
use linalg::fields::{dot_product, PseudoField};
use ndarray::{Array, Array1, Axis};

use representation::{kronecker_product_array, Representation, TriangularArray};

use crate::representation::{self, Bidiagonal};
use crate::{linalg, Aph};

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
) -> Aph<F, Bidiagonal<F>> {
    let size = ph1.size() + ph2.size();
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
        matrix.shape() == &[size, size],
        "Shape of matrix is {:?}, but actual size is {:?}",
        matrix.shape(),
        size
    );
    Aph {
        initial: delta.into(),
        repr: TriangularArray { size, matrix },
    }.spa()
}

/// Given two PH distributions $(\overrightarrow{\alpha}, \mathbf{A})$, $(\overrightarrow{\beta}, \mathbf{B})$ of size $m$ and $n$ respectively. Then:
/// the minimum between them is the PH: $(\overrightarrow{\alpha} \oplus \overrightarrow{\beta}, \mathbf{A} \otimes \mathbf{B})$
///
/// Computes $min\{ph1, ph2\}$
pub fn min_ph<F: PseudoField, R: Representation<F>>(
    ph1: &Aph<F, R>,
    ph2: &Aph<F, R>,
) -> Aph<F, Bidiagonal<F>> {
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
    .spa()
}

pub fn max_phs<F: PseudoField, R: Representation<F>>(
    instances: &[&Aph<F, R>],
) -> Option<Aph<F, Bidiagonal<F>>> {
    if instances.is_empty() {
        return None;
    }
    let mut result = instances.get(0).unwrap().spa();
    let instances = instances.iter().map(|aph| aph.spa()).collect_vec();
    for instance in &instances[1..] {
        result = max_ph(&result, &instance);
    }
    Some(result)
}

pub fn min_phs<F: PseudoField, R: Representation<F>>(
    instances: &[&Aph<F, R>],
) -> Option<Aph<F, Bidiagonal<F>>> {
    if instances.is_empty() {
        return None;
    }
    let mut result = instances.get(0).unwrap().spa();
    let instances = instances.iter().map(|aph| aph.spa()).collect_vec();
    for instance in &instances[1..] {
        result = min_ph(&result, &instance);
    }
    Some(result)
}

pub fn con_phs<F: PseudoField, R: Representation<F>>(
    instances: &[&Aph<F, R>],
) -> Option<Aph<F, Bidiagonal<F>>> {
    if instances.is_empty() {
        return None;
    }
    let mut result = instances.get(0).unwrap().spa();
    let instances = instances.iter().map(|aph| aph.spa()).collect_vec();
    for instance in &instances[1..] {
        result = con_ph(&result, &instance);
    }
    Some(result)
}
