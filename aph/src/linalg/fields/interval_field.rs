use core::{f64, panic};
use std::{
    fmt::Display,
    marker::PhantomData,
    ops::{Add, Mul, Sub},
};

use super::{
    rational::Rational, Almost, ContFraction, FromCF, FromRational, PseudoField, Round, Rounding,
    SparseField, ToRational,
};
use log::warn;
use num_rational::Ratio;
use num_traits::One;

/// A [`PseudoField`] using interval arithmetic to track the rounding imprecision of
/// a [`SparseField`].
#[derive(Clone, PartialEq, Debug, Eq, Hash)]
pub struct IF<F: SparseField, R: Rounding> {
    inf: F,
    sup: F,
    phantom: PhantomData<R>,
}

/// The following codes are used to represent the class of an interval:
///
///  Code | Description
/// ------+---------------
///     E | Empty
///     M | a < 0 < b
///    N0 | a < 0 ∧ b = 0
///    N1 | b < 0
///    P0 | a = 0 ∧ 0 < b
///    P1 | 0 < a
///     Z | a = b = 0
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum IC {
    // Each bit represents b ≤ 0, -a ≤ 0, b ≥ 0 and -a ≥ 0, respectively.
    E,  // empty
    M,  // b > 0 ∧ -a > 0
    N0, // b = 0 ∧ -a > 0
    N1, // b < 0 ∧ -a > 0
    P0, // b > 0 ∧ -a = 0
    P1, // b > 0 ∧ -a < 0
    Z,  // b = 0 ∧ -a = 0
}

impl<F: SparseField, R: Rounding> Almost for IF<F, R> {
    fn almost_one(&self) -> bool {
        if !self.contains(&F::one()) {
            return false;
        }
        // Option 1-3: The interval is contianed in <-eps, +eps>
        let mut low = F::one();
        low.sub_assign(&F::epsilon(), Round::Down);
        let mut upp = F::one();
        upp.add_assign(&F::epsilon(), Round::Up);

        self.contained_in(&IF {
            inf: low,
            sup: upp,
            phantom: PhantomData,
        })
    }

    fn almost_zero(&self) -> bool {
        // Option 2: The middle point is eps close to zero.
        // let mut mid = self.mid();
        // mid.abs_assign();
        // mid.le(&F::epsilon())

        // Option 1: Zero is contained in the interval.
        if !self.contains(&F::zero()) {
            return false;
        }
        // Option 1-3: The interval is contianed in <-eps, +eps>
        let mut eps_lo = F::epsilon();
        eps_lo.neg_assign();
        self.contained_in(&IF {
            inf: eps_lo,
            sup: F::epsilon(),
            phantom: PhantomData,
        })

        // Option 4: The interval is the point interval <0.0,0.0>
        // self.is_zero()

        // Option 5: Middle point is exactly zero.
        // self.mid().is_zero()
    }

    fn cmp_eq(&self, other: &Self) -> bool {
        let abstol = Self::from_rational("1", "10000000");
        let epsi = Self::from_rational("1", "10000");

        if self.eq(other) {
            return true;
        }
        let mut diff = self.clone().sub(other.clone());
        diff.abs_assign();

        if diff.le(&abstol) {
            return true;
        }

        let mut self_abs = self.clone();
        self_abs.abs_assign();
        let mut other_abs = other.clone();
        other_abs.abs_assign();

        let reltol = (if self_abs.ge(&other_abs) {
            self_abs
        } else {
            other_abs
        })
        .mul(epsi.clone());
        diff.le(&reltol)
    }
}

impl<F: SparseField, R: Rounding> IF<F, R> {
    pub fn low(&self) -> F {
        self.inf.clone()
    }

    pub fn inf(&self) -> F {
        self.sup.clone()
    }

    pub fn mid(&self) -> F {
        let mut mid = self.sup.clone();
        let two = F::one() + F::one();
        mid.add_assign(&self.inf, Round::Nearest);
        mid.div_assign(&two, Round::Nearest);
        mid
    }

    pub fn contained_in(&self, other: &IF<F, R>) -> bool {
        self.low().ge(&other.low()) && self.inf().le(&other.inf())
    }

    pub fn contains(&self, value: &F) -> bool {
        self.low().le(value) && self.inf().ge(value)
    }

    fn classify(&self) -> IC {
        match (self.low(), self.inf()) {
            (a, b) if a.lt(&F::zero()) && b.gt(&F::zero()) => IC::M,
            (a, b) if a.lt(&F::zero()) && b.is_zero() => IC::N0,
            (_a, b) if b.lt(&F::zero()) => IC::N1,
            (a, b) if a.is_zero() && b.gt(&F::zero()) => IC::P0,
            (a, _b) if a.gt(&F::zero()) => IC::P1,
            (a, b) if a.is_zero() && b.is_zero() => IC::Z,
            _ => IC::E,
        }
    }

    fn classify2(&self, rhs: &Self) -> (IC, IC) {
        (self.classify(), rhs.classify())
    }
}

impl<F: SparseField, R: Rounding> Display for IF<F, R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}, {}]", self.inf, self.sup)
    }
}

impl<F: SparseField, R: Rounding> PseudoField for IF<F, R> {
    fn neg_assign(&mut self) {
        self.inf.neg_assign();
        self.sup.neg_assign();
    }

    fn abs_assign(&mut self) {
        self.inf.abs_assign();
        self.sup.abs_assign();
    }

    fn add_assign(&mut self, rhs: &Self) {
        self.inf.add_assign(&rhs.inf, Round::Down);
        self.sup.add_assign(&rhs.sup, Round::Up);
    }

    fn sub_assign(&mut self, rhs: &Self) {
        self.inf.sub_assign(&rhs.inf, Round::Down);
        self.sup.sub_assign(&rhs.sup, Round::Up);
    }

    /// [a, b] * [c, d] =
    ///
    ///    |      M     |      N     |      P     |  Z
    /// ---+------------+------------+------------+-----
    ///  M |     *1     | [b*c, a*c] | [a*d, b*d] | {0}
    ///  N | [a*d, a*c] | [b*d, a*c] | [a*d, b*c] | {0}
    ///  P | [b*c, b*d] | [b*c, a*d] | [a*c, b*d] | {0}
    ///  Z |     {0}    |     {0}    |     {0}    | {0}
    /// *1 [min{a*d, b*c}, max{a*c, b*d}]
    fn mul_assign(&mut self, rhs: &Self) {
        #[allow(unreachable_patterns)]
        let (lower, upper) = match self.classify2(rhs) {
            //--- Any Empty Interval
            (IC::E, _) | (_, IC::E) => todo!(),
            //--- Any Zero Interval
            (IC::Z, _) | (_, IC::Z) => (F::zero(), F::zero()),
            //--- With zero and
            (IC::M, IC::M) => {
                let m0 = self.low() * rhs.inf();
                let m1 = self.inf() * rhs.low();
                let m2 = self.low() * rhs.low();
                let m3 = self.inf() * rhs.inf();
                (m0.min(m1), m2.max(m3))
            }
            (IC::M, IC::N0) | (IC::M, IC::N1) => (self.inf() * rhs.low(), self.low() * rhs.low()),
            (IC::M, IC::P0) | (IC::M, IC::P1) => (self.low() * rhs.inf(), self.inf() * rhs.inf()),
            //--- Negative and
            (IC::N0, IC::M) | (IC::N1, IC::M) => (self.low() * rhs.inf(), self.low() * rhs.low()),
            (IC::N0, IC::N0) | (IC::N0, IC::N1) | (IC::N1, IC::N0) | (IC::N1, IC::N1) => {
                (self.inf() * rhs.inf(), self.low() * rhs.low())
            }
            (IC::N0, IC::P0) | (IC::N0, IC::P1) | (IC::N1, IC::P0) | (IC::N1, IC::P1) => {
                (self.low() * rhs.inf(), self.inf() * rhs.low())
            }
            // --- Positive and
            (IC::P0, IC::M) | (IC::P1, IC::M) => (self.inf() * rhs.low(), self.inf() * rhs.inf()),
            (IC::P0, IC::N0) | (IC::P0, IC::N1) | (IC::P1, IC::N0) | (IC::P1, IC::N1) => {
                (self.inf() * rhs.low(), self.low() * rhs.inf())
            }
            (IC::P0, IC::P0) | (IC::P0, IC::P1) | (IC::P1, IC::P0) | (IC::P1, IC::P1) => {
                (self.inf() * rhs.low(), self.low() * rhs.inf())
            }
            (ic1, ic2) => {
                todo!("Still missing this case: {:?} - {:?}", ic1, ic2)
            }
        };
        *self = Self {
            inf: lower,
            sup: upper,
            phantom: PhantomData,
        };
    }

    /// [a, b] / [c, d] =
    ///
    ///    |  M  |     N0    |     N1     |     P0    |     P1     | Z
    /// ---+-----+-----------+------------+-----------+------------+---
    ///  M |  ℝ  |     ℝ     | [b/d, a/d] |     ℝ     | [a/c, b/c] | ∅
    ///  N |  ℝ  | [b/c, +∞] | [b/c, a/d] | [-∞, b/d] | [a/c, b/d] | ∅
    ///  P |  ℝ  | [-∞, a/c] | [b/d, a/c] | [a/d, +∞] | [a/d, b/c] | ∅
    ///  Z | {0} |    {0}    |     {0}    |    {0}    |     {0}    | ∅
    fn div_assign(&mut self, rhs: &Self) {
        // if rhs.contains(&F::zero()) {
        //     panic!("Can't divide over an interval that contains zero.")
        // }
        // self.lower.div_assign(&rhs.upper, Round::Down);
        // self.upper.div_assign(&rhs.lower, Round::Up);
        #[allow(unreachable_patterns)]
        let (lower, upper) = match self.classify2(rhs) {
            //--- Any Empty Interval
            (IC::E, _) | (_, IC::E) => panic!("Cannot divide Empty intervals."),
            //--- Divide by zero
            (_, IC::Z) | (_, IC::M) => panic!("Cannot divide by zero."),
            (IC::Z, _) => (F::zero(), F::zero()),
            //--- With zero and
            (IC::M, IC::N1) => {
                let mut aux_lo = self.inf();
                aux_lo.div_assign(&rhs.sup, Round::Down);
                let mut aux_up = self.low();
                aux_up.div_assign(&rhs.sup, Round::Up);
                (aux_lo, aux_up)
            }
            (IC::M, IC::P1) => {
                let mut aux_lo = self.low();
                aux_lo.div_assign(&rhs.inf, Round::Down);
                let mut aux_up = self.inf();
                aux_up.div_assign(&rhs.inf, Round::Up);
                (aux_lo, aux_up)
            }
            // ---
            (IC::N0, IC::N0) | (IC::N1, IC::N0) => {
                let mut eps_inv = F::inf();
                eps_inv.inv_assign();
                let mut aux = self.inf();
                aux.div_assign(&rhs.low(), Round::Down);
                (aux, eps_inv)
            }
            (IC::N0, IC::N1) | (IC::N1, IC::N1) => {
                let mut aux_lo = self.inf();
                aux_lo.div_assign(&rhs.inf, Round::Down);
                let mut aux_up = self.low();
                aux_up.div_assign(&rhs.sup, Round::Up);
                (aux_lo, aux_up)
            }
            (IC::N0, IC::P0) | (IC::N1, IC::P0) => {
                let mut eps_inv = F::inf();
                eps_inv.inv_assign();
                eps_inv.neg_assign();
                let mut aux_up = self.inf();
                aux_up.div_assign(&rhs.sup, Round::Up);
                (eps_inv, aux_up)
            }
            (IC::N0, IC::P1) | (IC::N1, IC::P1) => {
                let mut aux_lo = self.low();
                aux_lo.div_assign(&rhs.inf, Round::Down);
                let mut aux_up = self.inf();
                aux_up.div_assign(&rhs.sup, Round::Up);
                (aux_lo, aux_up)
            }
            // ----
            (IC::P0, IC::N0) | (IC::P1, IC::N0) => {
                let mut eps_inv = F::inf();
                eps_inv.inv_assign();
                eps_inv.neg_assign();
                let mut aux_up = self.low();
                aux_up.div_assign(&rhs.inf, Round::Up);
                (eps_inv, aux_up)
            }
            (IC::P0, IC::N1) | (IC::P1, IC::N1) => {
                let mut aux_lo = self.inf();
                aux_lo.div_assign(&rhs.sup, Round::Down);
                let mut aux_up = self.low();
                aux_up.div_assign(&rhs.inf, Round::Up);
                (aux_lo, aux_up)
            }
            (IC::P0, IC::P0) | (IC::P1, IC::P0) => {
                let mut aux_lo = self.low();
                aux_lo.div_assign(&rhs.sup, Round::Down);
                let mut eps_inv = F::inf();
                eps_inv.inv_assign();
                (aux_lo, eps_inv)
            }
            (IC::P0, IC::P1) | (IC::P1, IC::P1) => {
                let mut aux_lo = self.low();
                aux_lo.div_assign(&rhs.sup, Round::Down);
                let mut aux_up = self.inf();
                aux_up.div_assign(&rhs.inf, Round::Up);
                (aux_lo, aux_up)
            }
            // ---
            (ic1, ic2) => {
                todo!("Still missing this case: {:?} - {:?}", ic1, ic2)
            }
        };
        *self = Self {
            inf: lower,
            sup: upper,
            phantom: PhantomData,
        };
    }

    fn inv_assign(&mut self) {
        let mut new_sup = if self.low().is_zero() {
            warn!("Upper bound is INFINITY, using 1/eps.");
            F::epsilon()
        } else {
            self.low().clone()
        };
        let mut new_inf = if self.inf().is_zero() {
            warn!("Lower bound is INFINITY, using 1/eps.");
            F::epsilon()
        } else {
            self.inf().clone()
        };
        new_sup.inv_assign();
        new_inf.inv_assign();
        Self {
            inf: new_inf,
            sup: new_sup,
            phantom: PhantomData,
        };
    }

    fn to_string(&self) -> String {
        format!(
            "{}",
            match R::rounding() {
                super::Round::Down | super::Round::Zero => self.low(),
                super::Round::Nearest => self.mid(),
                super::Round::Up => self.inf(),
            }
        )
    }
}

impl<F: SparseField, R: Rounding> FromCF for IF<F, R> {
    fn from_cont_fraction(cf: &mut ContFraction) -> Self {
        if cf.values.len() <= 1 {
            IF::<F, R>::from_rational(&format!("{:?}", cf.values[0]), "1")
        } else {
            let val = IF::<F, R>::from_rational(&format!("{:?}", cf.values.remove(0)), "1");
            let mut rgt = IF::<F, R>::one();
            rgt.div_assign(&IF::<F, R>::from_cont_fraction(&mut ContFraction {
                values: cf.values.clone(),
            }));
            val + rgt
        }
    }
}

impl<F: SparseField, R: Rounding> FromRational for IF<F, R> {
    fn from_rational(numerator: &str, denominator: &str) -> Self {
        let lower = F::from_rational(numerator, denominator);
        let upper = F::from_rational(numerator, denominator);
        IF {
            inf: lower,
            sup: upper,
            phantom: PhantomData,
        }
    }
}

impl<F: SparseField, R: Rounding> ToRational for IF<F, R> {
    fn to_rational(&self) -> (String, String) {
        let mid: f64 = self.mid().into();
        let ratio = Ratio::from_float(mid)
            .expect("Something went wrong when converting the Interval to Rational.");
        (ratio.numer().to_string(), ratio.denom().to_string())
    }
}

impl<F: SparseField, R: Rounding> num_traits::One for IF<F, R> {
    fn one() -> Self {
        Self {
            inf: F::one(),
            sup: F::one(),
            phantom: PhantomData,
        }
    }

    fn is_one(&self) -> bool {
        self.inf.is_one() || self.sup.is_one()
    }

    fn set_one(&mut self) {
        *self = Self {
            inf: F::one(),
            sup: F::one(),
            phantom: PhantomData,
        }
    }
}

impl<F: SparseField, R: Rounding> num_traits::Zero for IF<F, R> {
    fn zero() -> Self {
        Self {
            inf: F::zero(),
            sup: F::zero(),
            phantom: PhantomData,
        }
    }

    fn is_zero(&self) -> bool {
        self.inf.is_zero() || self.sup.is_zero()
    }

    fn set_zero(&mut self) {
        *self = Self {
            inf: F::zero(),
            sup: F::zero(),
            phantom: PhantomData,
        }
    }
}

impl<F: SparseField, R: Rounding> Add for IF<F, R> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Self {
            inf: self.inf + rhs.inf,
            sup: self.sup + rhs.sup,
            phantom: PhantomData,
        }
    }
}

impl<F: SparseField, R: Rounding> Mul for IF<F, R> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        Self {
            inf: self.inf * rhs.inf,
            sup: self.sup * rhs.sup,
            phantom: PhantomData,
        }
    }
    // fn mul(self, rhs: Self) -> Self::Output {
    //     let comb = vec![
    //         self.lower.clone() * rhs.lower(),
    //         self.lower.clone() * rhs.upper(),
    //         self.upper.clone() * rhs.lower(),
    //         self.upper.clone() * rhs.upper(),
    //     ];
    //     Self {
    //         lower: comb
    //             .iter()
    //             .min()
    //             .expect(&format!("Something went wrong computing min of {:?}", comb))
    //             .clone(),
    //         upper: comb
    //             .iter()
    //             .max()
    //             .expect(&format!("Something went wrong computing max of {:?}", comb))
    //             .clone(),
    //         phantom: PhantomData
    //     }
    // }
}

impl<F: SparseField, R: Rounding> PartialOrd for IF<F, R> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        use std::cmp::Ordering::*;
        if self.contained_in(other) {
            Some(Less)
        } else if other.contained_in(self) {
            Some(Greater)
        } else {
            match (
                self.inf.partial_cmp(&other.inf)?,
                self.sup.partial_cmp(&other.sup)?,
                self.sup.partial_cmp(&other.inf)?,
                self.inf.partial_cmp(&other.sup)?,
            ) {
                (_, _, Equal, Equal) => Some(Equal),
                (_, _, Less, Less) => Some(Less),
                (_, _, Equal, Less) => Some(Less),
                (_, _, Greater, Greater) => Some(Greater),
                (_, _, Greater, Equal) => Some(Greater),
                (Greater, Equal, Greater, Less) => Some(Less),
                (Equal, Greater, Greater, Less) => Some(Greater),
                (Equal, Equal, Greater, Less) => Some(Equal),
                (_, _, Greater, Less) => None,
                _ => unreachable!(),
            }
        }
    }
}

impl<F: SparseField, R: Rounding> From<&Rational> for IF<F, R> {
    fn from(value: &Rational) -> Self {
        Self::from_rational(&value.numer(), &value.denom())
    }
}

impl<F: SparseField, R: Rounding> Sub for IF<F, R> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        let mut neg_rhs = rhs.clone();
        neg_rhs.neg_assign();
        self.add(neg_rhs)
    }
}
