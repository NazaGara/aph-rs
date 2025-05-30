use std::{
    fmt::Display,
    marker::PhantomData,
    ops::{Add, Mul, Sub},
};

use super::{Almost, FromRational, PseudoField, Rounding, ToRational};
use inari::*;
use log::warn;
use num_rational::Ratio;
use num_traits::Zero;
use rug::{Float, ops::CompleteRound};

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Interval<R: Rounding>(inari::Interval, PhantomData<R>);

impl<R: Rounding> PseudoField for Interval<R> {
    fn neg_assign(&mut self) {
        self.mul_assign(&Self(interval_exact!("[-1.0]").unwrap(), PhantomData))
    }

    fn abs_assign(&mut self) {
        *self = Self(self.0.abs(), PhantomData);
    }

    fn add_assign(&mut self, rhs: &Self) {
        *self = Self(self.0.add(rhs.0), PhantomData);
    }

    fn sub_assign(&mut self, rhs: &Self) {
        *self = Self(self.0.sub(rhs.0), PhantomData);
    }

    fn mul_assign(&mut self, rhs: &Self) {
        *self = Self(self.0.mul(rhs.0), PhantomData);
    }

    fn div_assign(&mut self, rhs: &Self) {
        *self = Self(self.0 / rhs.0, PhantomData);
    }

    fn inv_assign(&mut self) {
        // Assuming that are both are positive, or both negative
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
        self.div_assign(&Self(interval!(new_inf, new_sup).unwrap(), PhantomData));
    }

    fn to_string(&self) -> String {
        format!(
            "{}",
            match R::rounding() {
                super::Round::Down | super::Round::Zero => self.0.inf(),
                super::Round::Nearest => self.0.mid(),
                super::Round::Up => self.0.sup(),
            }
        )
    }
}

impl<R: Rounding> From<f64> for Interval<R> {
    fn from(value: f64) -> Self {
        Self(interval!(value, value).unwrap(), PhantomData)
    }
}

impl<R: Rounding> Display for Interval<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "<{:?}, {:?}>", self.0.inf(), self.0.sup())
    }
}

impl<R: Rounding> FromRational for Interval<R> {
    fn from_rational(numerator: &str, denominator: &str) -> Self {
        let num = Float::parse(numerator).unwrap().complete(53);
        let den = Float::parse(denominator).unwrap().complete(53);
        let div = (num / den).to_f64();
        Self(interval!(div, div).unwrap(), PhantomData)
    }
}

impl<R: Rounding> ToRational for Interval<R> {
    fn to_rational(&self) -> (String, String) {
        let ratio = Ratio::from_float(self.0.mid())
            .expect("Something went wrong when converting the Interval to Rational.");
        (ratio.numer().to_string(), ratio.denom().to_string())
    }
}

impl<R: Rounding> Almost for Interval<R> {
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

impl<R: Rounding> num_traits::One for Interval<R> {
    fn one() -> Self {
        Self(const_interval!(1.0, 1.0), PhantomData)
    }

    fn is_one(&self) -> bool {
        self.0.inf().is_one() || self.0.sup().is_one()
    }

    fn set_one(&mut self) {
        *self = Interval::one();
    }
}

impl<R: Rounding> num_traits::Zero for Interval<R> {
    fn zero() -> Self {
        Self(const_interval!(0.0, 0.0), PhantomData)
    }

    fn is_zero(&self) -> bool {
        // Option 2: The middle point is eps close to zero.
        // self.0.mid().abs() <= f64::EPSILON

        // Option 1-3: The interval is contianed in <-eps, +eps>
        if !self.0.contains(0.0) {
            return false;
        }
        self.0
            .interior(const_interval!(-f64::EPSILON, f64::EPSILON))

        // Option 4: The interval is the point interval <0.0,0.0>
        // self.0.inf().is_zero() && self.0.sup().is_zero()

        // Option 5: Middle point is exactly zero.
        // self.0.mid().is_zero()
    }

    fn set_zero(&mut self) {
        *self = Self::zero();
    }
}

impl<R: Rounding> Add for Interval<R> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0.add(rhs.0), PhantomData)
    }
}

impl<R: Rounding> Mul for Interval<R> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self(self.0.mul(rhs.0), PhantomData)
    }
}

impl<R: Rounding> Sub for Interval<R> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        Self(self.0.sub(rhs.0), PhantomData)
    }
}

impl<R: Rounding> PartialOrd for Interval<R> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        if self.0.strict_less(other.0) {
            Some(std::cmp::Ordering::Less)
        } else if other.0.strict_less(self.0) {
            Some(std::cmp::Ordering::Greater)
        } else {
            Some(std::cmp::Ordering::Equal)
        }
    }
}
