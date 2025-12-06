// --- Fault Tree

use log::info;
use ndarray::Array2;
use sprs::TriMat;
use std::{
    collections::{BTreeMap, HashMap},
    fmt::{Debug, Display},
};

use crate::{
    aph::{Aph, BidiagonalAph},
    formats::ft::{
        ConstructionMethod, parser::NodeId, state::StateCode, utils::count_rate_by_levels_round,
    },
    linalg::{
        Vector,
        fields::{PseudoField, Round},
    },
    representation::{
        bidiagonal::Bidiagonal,
        sparse::Sparse,
        triangular::{Triangular, TriangularArray},
    },
    utils::count_rate_by_levels,
};

#[derive(Debug)]
pub struct Transition<F: PseudoField>(pub StateCode, pub StateCode, pub F);

impl<F: PseudoField> Display for Transition<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}--({})-->{}", self.0, self.1, self.2)
    }
}

#[derive(Debug)]
pub struct CTMCAut<F: PseudoField> {
    pub initial: Vector<F>,
    pub states: Vec<StateCode>,
    // pub transitions: BTreeMap<StateCode, HashMap<StateCode, F>>,
    pub generator: BTreeMap<StateCode, HashMap<StateCode, F>>,
    index: HashMap<StateCode, usize>,
}

impl<F: PseudoField> CTMCAut<F> {
    pub fn new(
        initial: Vector<F>,
        states: Vec<StateCode>,
        transitions: BTreeMap<StateCode, HashMap<StateCode, F>>,
    ) -> Self {
        let index: HashMap<StateCode, usize> = states
            .iter()
            .enumerate()
            .filter_map(|(idx, s)| {
                if !s.is_abs() {
                    Some((s.clone(), idx))
                } else {
                    None
                }
            })
            .collect::<HashMap<StateCode, usize>>();
        CTMCAut {
            initial,
            states,
            generator: transitions,
            index,
        }
    }

    pub fn index_size(&self) -> usize {
        self.index.len()
    }

    pub fn to_aph(
        &self,
        method: ConstructionMethod,
        _root_id: NodeId,
        round: Round,
    ) -> BidiagonalAph<F> {
        match method {
            ConstructionMethod::Sparse => match round {
                Round::Nearest => self.to_aph_sparse_lcs(_root_id),
                _ => self.to_aph_sparse_lcs_round(_root_id, round),
            },
            ConstructionMethod::Triangular => self.to_aph_triangular_lcs(_root_id),
            ConstructionMethod::Matrix => self.to_aph_matrix(_root_id),
            ConstructionMethod::CoreSeries => self.to_aph_cs(),
        }
    }

    fn to_aph_cs(&self) -> BidiagonalAph<F> {
        let getter = move |i, j| self.lazy_getter(i, j);
        let bidi = (0..self.index_size())
            .map(|i| getter(i, i))
            .collect::<Vec<F>>();
        let bidiagonal = self.depth_based_lcs(&bidi, usize::max, usize::MIN);

        Aph::<F, Bidiagonal<F>>::spa_with_getter(&self.initial, &getter, bidiagonal)
    }

    fn lazy_getter(&self, i: usize, j: usize) -> F {
        if i > j {
            F::zero()
        } else if i == j {
            self.generator
                .get(&self.states[i])
                .unwrap()
                .iter()
                .fold(F::zero(), |acc, (_, x)| acc.add(x.clone()))
        } else {
            let mut rate = self
                .generator
                .get(&self.states[i])
                .unwrap()
                .get(&self.states[j])
                .unwrap()
                .clone();
            rate.neg_assign();
            rate
        }
    }

    fn to_aph_matrix(&self, _root_id: NodeId) -> BidiagonalAph<F> {
        let size = self.index_size();

        let mut matrix = Array2::<F>::zeros((size, size));
        for (i, st) in self.states.iter().enumerate() {
            if st.is_abs() {
                continue;
            }
            if st.is_fs() {
                panic!("Fail safe mode is not yet supported, It requires a multi exit phase type.");
            }

            let mut absorb_rate = F::zero();
            let mut off_diag_sum = F::zero();
            for (to, rate) in self.generator[st].iter() {
                if let Some(&idx) = self.index.get(to) {
                    // If state is indexed, we transition to new transient state.
                    matrix[[i, idx]].sub_assign(rate);
                    off_diag_sum.sub_assign(rate);
                } else if to.is_abs() {
                    // If state is not transient, is to abs
                    absorb_rate.sub_assign(rate)
                } else {
                    todo!("PAND not yet supported!.")
                }
            }

            let mut exit_rate = off_diag_sum + absorb_rate;
            exit_rate.neg_assign();
            matrix[[i, i]] = exit_rate;
        }

        if log::log_enabled!(log::Level::Debug) {
            use std::io::Write;
            let filepath = format!("matrix_{:?}.csv", _root_id);
            info!("Writing to: {filepath:?}");
            let mut file = std::fs::File::create(filepath).unwrap();
            for row in matrix.rows() {
                let _ = writeln!(
                    file,
                    "{}",
                    itertools::Itertools::join(&mut row.iter().map(|e| e.to_string()), ",")
                );
            }
        }

        Aph {
            initial: self.initial.clone(),
            repr: TriangularArray { size, matrix },
        }
        .spa()
    }

    #[allow(unused)]
    fn to_aph_sparse(&self) -> BidiagonalAph<F> {
        let size = self.index_size();

        let mut matrix: sprs::TriMatBase<Vec<usize>, Vec<F>> = TriMat::new((size, size));
        for (i, st) in self.states.iter().enumerate() {
            if st.is_abs() {
                continue;
            }
            if st.is_fs() {
                panic!("Fail safe mode is not yet supported, It requires a multi exit phase type.");
            }

            let mut absorb_rate = F::zero();
            let mut off_diag_sum = F::zero();
            for (to, rate) in self.generator[st].iter() {
                if let Some(&idx) = self.index.get(to) {
                    // If state is indexed, we transition to new transient state.
                    // matrix[[i, idx]].sub_assign(rate);
                    let mut neg_rate = rate.clone();
                    neg_rate.neg_assign();
                    matrix.add_triplet(i, idx, neg_rate);
                    off_diag_sum.sub_assign(rate);
                } else if to.is_abs() {
                    // If state is not transient, is to abs
                    absorb_rate.sub_assign(rate)
                } else {
                    todo!("PAND not yet supported!.")
                }
            }

            let mut exit_rate = off_diag_sum + absorb_rate;
            exit_rate.neg_assign();
            // matrix[[i, i]] = exit_rate;
            matrix.add_triplet(i, i, exit_rate);
        }

        Aph {
            initial: self.initial.clone(),
            repr: Sparse { size, matrix },
        }
        .sparse_spa()
    }

    fn to_aph_triangular_lcs(&self, _root_id: NodeId) -> BidiagonalAph<F> {
        let size = self.index_size();
        let mut diagonal = vec![F::zero(); size];
        let mut depths = vec![0_u32; size];
        let mut matrix: Triangular<F> = Triangular::new(size);

        for (i, st) in self.states.iter().enumerate() {
            if st.is_abs() {
                continue;
            }
            if st.is_fs() {
                panic!("Fail safe mode is not yet supported, It requires a multi exit phase type.");
            }

            let my_level = depths[i];

            let mut absorb_rate = F::zero();
            let mut off_diag_sum = F::zero();
            for (to, rate) in self.generator[st].iter() {
                if let Some(&idx) = self.index.get(to) {
                    depths[idx] = u32::max(depths[idx], my_level + 1);

                    // If state is indexed, we transition to new transient state.
                    let mut neg_rate = rate.clone();
                    neg_rate.neg_assign();
                    matrix.set(i, idx, neg_rate);
                    off_diag_sum.sub_assign(rate);
                } else if to.is_abs() {
                    // If state is not transient, is to abs
                    absorb_rate.sub_assign(rate)
                } else {
                    todo!("PAND not yet supported!.")
                }
            }

            let mut exit_rate = off_diag_sum + absorb_rate;
            exit_rate.neg_assign();
            diagonal[i] = exit_rate.clone();
            matrix.set(i, i, exit_rate);
        }
        let aph = Aph {
            initial: self.initial.clone(),
            repr: matrix,
        };

        let counts = count_rate_by_levels(&depths, &diagonal);
        let mut new_diagonal: Vec<F> = vec![];
        counts.iter().for_each(|(lambda, k)| {
            new_diagonal.extend(vec![(*lambda).clone(); (*k).try_into().unwrap()])
        });
        new_diagonal.sort_by(|x, y| {
            y.partial_cmp(x)
                .unwrap_or_else(|| panic!("Could not sort the values: {y:?} and {x:?}."))
        });
        let bidiagonal = Bidiagonal::<F>::from(Vector::from(new_diagonal));
        aph.spa_with_bidiagonal(bidiagonal)
    }

    fn to_aph_sparse_lcs_round(&self, _root_id: NodeId, round: Round) -> BidiagonalAph<F> {
        let size = self.index_size();
        let mut diagonal = vec![F::zero(); size];
        let mut depths = vec![0_u32; size];
        let mut matrix: sprs::TriMatBase<Vec<usize>, Vec<F>> = TriMat::new((size, size));

        for (i, st) in self.states.iter().enumerate() {
            if st.is_abs() {
                continue;
            }
            if st.is_fs() {
                panic!("Fail safe mode is not yet supported, It requires a multi exit phase type.");
            }

            let my_level = depths[i];

            let mut absorb_rate = F::zero();
            let mut off_diag_sum = F::zero();
            for (to, rate) in self.generator[st].iter() {
                if let Some(&idx) = self.index.get(to) {
                    depths[idx] = u32::max(depths[idx], my_level + 1);

                    // If state is indexed, we transition to new transient state.
                    let mut neg_rate = rate.clone();
                    neg_rate.neg_assign();
                    matrix.add_triplet(i, idx, neg_rate);
                    off_diag_sum.sub_assign(rate);
                } else if to.is_abs() {
                    // If state is not transient, is to abs
                    absorb_rate.sub_assign(rate)
                } else {
                    todo!("PAND not yet supported!.")
                }
            }

            let mut exit_rate = off_diag_sum + absorb_rate;
            exit_rate.neg_assign();
            diagonal[i] = exit_rate.clone();
            matrix.add_triplet(i, i, exit_rate);
        }
        let mut aph = Aph {
            initial: self.initial.clone(),
            repr: Sparse { size, matrix },
        };

        let counts = count_rate_by_levels_round(&depths, &diagonal, &mut aph, round);
        let mut new_diagonal: Vec<F> = vec![];
        counts.iter().for_each(|(lambda, k)| {
            new_diagonal.extend(vec![(*lambda).clone(); (*k).try_into().unwrap()])
        });
        new_diagonal.sort_by(|x, y| {
            y.partial_cmp(x)
                .unwrap_or_else(|| panic!("Could not sort the values: {y:?} and {x:?}."))
        });
        let bidiagonal = Bidiagonal::<F>::from(Vector::from(new_diagonal));

        aph.sparse_spa_w_bidiagonal(bidiagonal)
    }

    fn to_aph_sparse_lcs(&self, _root_id: NodeId) -> BidiagonalAph<F> {
        let size = self.index_size();

        let mut diagonal = vec![F::zero(); size];
        let mut depths = vec![0_u32; size];
        let mut matrix: sprs::TriMatBase<Vec<usize>, Vec<F>> = TriMat::new((size, size));

        for (i, st) in self.states.iter().enumerate() {
            if st.is_abs() {
                continue;
            }
            if st.is_fs() {
                panic!("Fail safe mode is not yet supported. Requires Multi Exit PH.");
            }

            let my_level = depths[i];
            let mut absorb_rate = F::zero();
            let mut off_diag_sum = F::zero();
            for (to, rate) in self.generator[st].iter() {
                if let Some(&idx) = self.index.get(to) {
                    depths[idx] = u32::max(depths[idx], my_level + 1);

                    // If state is indexed, we transition to new transient state.
                    let mut neg_rate = rate.clone();
                    neg_rate.neg_assign();
                    matrix.add_triplet(i, idx, neg_rate);
                    off_diag_sum.sub_assign(rate);
                } else if to.is_abs() {
                    // If state is not transient, is to abs
                    absorb_rate.sub_assign(rate)
                } else {
                    todo!("PAND not yet supported!.")
                }
            }

            let mut exit_rate = off_diag_sum + absorb_rate;
            exit_rate.neg_assign();
            diagonal[i] = exit_rate.clone();
            matrix.add_triplet(i, i, exit_rate);
        }
        let aph = Aph {
            initial: self.initial.clone(),
            repr: Sparse { size, matrix },
        };

        log::debug!("Maximum Depth: {:?}", depths.iter().max());
        let counts = count_rate_by_levels(&depths, &diagonal);
        let mut new_diagonal: Vec<F> = vec![];
        counts.iter().for_each(|(lambda, k)| {
            new_diagonal.extend(vec![(*lambda).clone(); (*k).try_into().unwrap()])
        });
        new_diagonal.sort_by(|x, y| {
            y.partial_cmp(x)
                .unwrap_or_else(|| panic!("Could not sort the values: {y:?} and {x:?}."))
        });
        let bidiagonal = Bidiagonal::<F>::from(Vector::from(new_diagonal));
        aph.sparse_spa_w_bidiagonal(bidiagonal)
    }

    #[allow(unused)]
    fn depth_based_lcs(
        &self,
        diagonal: &[F],
        decider: impl Fn(usize, usize) -> usize,
        default_usize: usize,
    ) -> Bidiagonal<F> {
        // Size is the #States + 1 == len(index)
        let size = self.index_size();
        let adj = {
            let mut adj: HashMap<usize, Vec<usize>> = HashMap::new();
            self.generator.iter().for_each(|(from, map)| {
                adj.entry(self.index[from])
                    .and_modify(|elems| elems.extend(map.keys().map(|k| self.index[k])))
                    .or_insert(map.keys().map(|k| self.index[k]).collect());
            });
            adj
        };
        // let diagonal = (0..size).map(|i| getter(i, i)).collect::<Vec<F>>();

        // Depth check.
        fn dfs(
            current: usize,
            adj: &HashMap<usize, Vec<usize>>,
            level: usize,
            depths: &mut [usize],
            decider: &impl Fn(usize, usize) -> usize,
        ) {
            // depths[current] = depths[current].min(level);
            depths[current] = decider(depths[current], level);
            if let Some(froms) = adj.get(&current) {
                for nxt in froms {
                    dfs(*nxt, adj, level + 1, depths, decider);
                }
            }
        }

        let mut depths = vec![default_usize; size];
        dfs(0, &adj, 0, &mut depths, &decider);
        // Then, on each depth level, take all the unique rates.
        // let counts = count_rate_by_levels(&depths, diagonal);
        let counts = count_rate_by_levels(&depths, diagonal);

        let mut new_diagonal: Vec<F> = vec![];
        counts.iter().for_each(|(lambda, k)| {
            new_diagonal.extend(vec![(*lambda).clone(); (*k).try_into().unwrap()])
        });
        new_diagonal.sort_by(|x, y| {
            y.partial_cmp(x)
                .unwrap_or_else(|| panic!("Could not sort the values: {y:?} and {x:?}."))
        });
        Bidiagonal::<F>::from(Vector::from(new_diagonal))
    }
}
