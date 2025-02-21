use std::{
    fmt::Display,
    ops::{Add, Mul, Sub},
};

use super::{Almost, ContFraction, FromCF, FromRational, PseudoField, ToRational};
use inari::*;
use log::warn;
use num_rational::Ratio;
use num_traits::{One, Zero};
use rug::{ops::CompleteRound, Float};

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Interval(inari::Interval);

impl PseudoField for Interval {
    fn neg_assign(&mut self) {
        self.mul_assign(&Self(interval_exact!("[-1.0]").unwrap()))
    }

    fn abs_assign(&mut self) {
        *self = Self(self.0.abs())
    }

    fn add_assign(&mut self, rhs: &Self) {
        *self = Self(self.0.add(rhs.0))
    }

    fn sub_assign(&mut self, rhs: &Self) {
        *self = Self(self.0.sub(rhs.0))
    }

    fn mul_assign(&mut self, rhs: &Self) {
        *self = Self(self.0.mul(rhs.0))
    }

    fn div_assign(&mut self, rhs: &Self) {
        *self = Self(self.0 / rhs.0);
    }

    fn inv_assign(&mut self) {
        let new_sup = if self.0.inf().is_infinite() {
            0.0
        } else if self.0.inf().is_zero() {
            warn!("Sup bound is INFINITY.");
            f64::INFINITY
        } else if self.0.inf().is_nan() {
            warn!("Sup bound is NAN.");
            f64::NAN
        } else {
            1.0 / self.0.inf()
        };
        let new_inf = if self.0.sup().is_infinite() {
            0.0
        } else if self.0.sup().is_zero() {
            warn!("Inf bound is INFINITY.");
            f64::INFINITY
        } else if self.0.sup().is_nan() {
            warn!("Inf bound is NAN.");
            f64::NAN
        } else {
            1.0 / self.0.sup()
        };
        self.div_assign(&Self(interval!(new_inf, new_sup).unwrap()));
    }

    fn to_string(&self) -> String {
        format!("<{},{}>", self.0.inf(), self.0.sup())
    }
}

impl Display for Interval {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "<{:.53}, {:.53}>", self.0.inf(), self.0.sup())
    }
}

impl FromRational for Interval {
    fn from_rational(numerator: &str, denominator: &str) -> Self {
        let num = Float::parse(numerator).unwrap().complete(53);
        let den = Float::parse(denominator).unwrap().complete(53);
        let div = (num / den).to_f64();
        Self(interval!(div, div).unwrap())
    }
    fn from_rational_modular(numerator: &str, denominator: &str, _moduler: f64) -> Self {
        Self::from_rational(numerator, denominator)
    }
}

impl ToRational for Interval {
    fn to_rational(&self) -> (String, String) {
        let ratio = Ratio::from_float(self.0.mid())
            .expect("Something went wrong when converting the Interval to Rational.");
        (ratio.numer().to_string(), ratio.denom().to_string())
    }
}

impl FromCF for Interval {
    fn from_cont_fraction(cf: &mut ContFraction) -> Self {
        if cf.values.len() <= 1 {
            return Interval::from_rational(&format!("{:?}", cf.values[0]), "1");
        } else {
            let val = Interval::from_rational(&format!("{:?}", cf.values.remove(0)), "1");
            let mut rgt = Self::one();
            rgt.div_assign(&Interval::from_cont_fraction(&mut ContFraction {
                values: cf.values.clone(),
            }));
            return val + rgt;
        }
    }
}

impl Almost for Interval {
    fn cmp_eq(&self, other: &Self) -> bool {
        let abstol = Self::from_rational("1", "10000000");
        let epsi = Self::from_rational("1", "10000");

        if self.eq(&other) {
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

    fn is_almost_zero(&self) -> bool {
        // Option 1: Strict absolute tolerance in both bounds w.r.t. epsilon.
        // if !self.0.contains(0.0) {
        //     return false;
        // }
        // let mut self_abs = self.clone();
        // self_abs.abs_assign();
        // let epsilon = Self::from_rational(&format!("{:?}", f64::EPSILON), "1");
        // self_abs.le(&epsilon)

        // Option 2: The middle point is eps close to zero.
        self.0.mid().abs() <= f64::EPSILON

        // Option 3: The interval is contianed in <-eps, +eps>
        // self.0
        //     .interior(const_interval!(-f64::EPSILON, f64::EPSILON))

        // Option 4: The interval is the point interval <0.0,0.0>
        // self.is_zero()

        // Option 5: Middle point is exactly zero.
        // self.0.mid().is_zero()
    }

    fn is_almost_zero_and_correct(&mut self) {
        if self.is_almost_zero() {
            self.set_zero();
        };
    }
}

impl num_traits::One for Interval {
    fn one() -> Self {
        Self(const_interval!(1.0, 1.0))
    }

    fn is_one(&self) -> bool {
        self.0.inf().is_one() || self.0.sup().is_one()
    }

    fn set_one(&mut self) {
        *self = Interval::one();
    }
}

impl num_traits::Zero for Interval {
    fn zero() -> Self {
        Self(const_interval!(0.0, 0.0))
    }

    fn is_zero(&self) -> bool {
        self.0.inf().is_zero() || self.0.sup().is_zero()
    }

    fn set_zero(&mut self) {
        *self = Self::zero();
    }
}

impl Add for Interval {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0.add(rhs.0))
    }
}

impl Mul for Interval {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self(self.0.mul(rhs.0))
    }
}

impl Sub for Interval {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        Self(self.0.sub(rhs.0))
    }
}

impl PartialOrd for Interval {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        use std::cmp::Ordering::*;

        match (
            self.0.inf().partial_cmp(&other.0.inf())?,
            self.0.sup().partial_cmp(&other.0.sup())?,
        ) {
            (Equal, Equal) => Some(Equal),
            (Less, Less) => Some(Less),
            (Greater, Greater) => Some(Greater),
            (Less, Greater) => None,
            (Greater, Less) => None,
            (Greater, Equal) => Some(Greater),
            (Equal, Greater) => Some(Greater),
            (Less, Equal) => Some(Less),
            (Equal, Less) => Some(Less),
            // _ => unreachable!(),
        }
    }
}
