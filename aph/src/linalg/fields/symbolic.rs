use std::ops::Sub;
use std::{
    fmt::Display,
    ops::{Add, Mul},
};

use super::{Almost, FromRational, PseudoField, ToRational};
use ndarray::ScalarOperand;
use num_traits::{One, Zero};

#[derive(Debug, Hash, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Symbol(String);

impl Almost for Symbol {
    fn cmp_eq(&self, other: &Self) -> bool {
        self.eq(other)
    }
}

impl Display for Symbol {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl num_traits::One for Symbol {
    fn one() -> Self {
        Self("1".to_string())
    }

    fn is_one(&self) -> bool {
        self.0 == "1"
    }
    fn set_one(&mut self) {
        *self = Self("1".to_string())
    }
}

impl num_traits::Zero for Symbol {
    fn zero() -> Self {
        Self("0".to_string())
    }

    fn is_zero(&self) -> bool {
        self.0 == "0"
    }

    fn set_zero(&mut self) {
        *self = Self("0".to_string())
    }
}

impl Add for Symbol {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        if self.is_zero() {
            rhs
        } else if rhs.is_zero() {
            self
        } else {
            Self(format!("({}+{})", self.0, rhs.0))
        }
    }
}

impl Sub for Symbol {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        if self.0 == rhs.0 {
            Self::zero()
        } else {
            Self(format!("({}—{})", self.0, rhs.0))
        }
    }
}

impl Mul for Symbol {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        if self.is_zero() || rhs.is_zero() {
            Self::zero()
        } else {
            Self(format!("{}*{}", self.0, rhs.0))
        }
    }
}

impl From<f64> for Symbol {
    fn from(value: f64) -> Self {
        Self(format!("{}", value))
    }
}

impl FromRational for Symbol {
    fn from_rational(numerator: &str, denominator: &str) -> Self {
        if denominator == "1" {
            Self(numerator.to_string())
        } else {
            Self(format!("({}) / ({})", numerator, denominator))
        }
    }
}

impl ToRational for Symbol {
    fn to_rational(&self) -> (String, String) {
        (self.0.clone(), "1".to_owned())
    }
}

impl PseudoField for Symbol {
    fn neg_assign(&mut self) {
        *self = if self.0.starts_with("-") {
            Self(self.0.strip_prefix("-").unwrap().to_string())
        } else {
            Self(format!("-{}", self.0))
        };
    }

    fn abs_assign(&mut self) {
        if !self.is_zero() {
            *self = Self(format!("|{}|", self.0));
        }
    }

    fn add_assign(&mut self, rhs: &Self) {
        // *self = Self(format!("({}+{})", self.0, rhs.0));
        *self = if self.is_zero() {
            rhs.clone()
        } else if rhs.is_zero() {
            self.clone()
        } else {
            Self(format!("({}+{})", self.0, rhs.0))
        };
    }

    fn sub_assign(&mut self, rhs: &Self) {
        *self = if self.0 == rhs.0 {
            Self::zero()
        } else {
            Self(format!("({}—{})", self.0, rhs.0))
        };
    }

    fn mul_assign(&mut self, rhs: &Self) {
        // *self = Self(format!("{}*{}", self.0, rhs.0));
        *self = if self.is_one() {
            rhs.clone()
        } else if rhs.is_one() {
            self.clone()
        } else if self.is_zero() || rhs.is_zero() {
            Self::zero()
        } else {
            Self(format!("({}*{})", self.0, rhs.0))
        };
    }

    fn div_assign(&mut self, rhs: &Self) {
        *self = Self(format!("({})/({})", self.0, rhs.0));
    }

    fn inv_assign(&mut self) {
        *self = if self.0.starts_with("1/") {
            Self(self.0.strip_prefix("1/").unwrap().to_string())
        } else {
            Self(format!("(1/{})", self.0))
        };
    }

    fn to_string(&self) -> String {
        self.0.clone()
    }
}

impl ScalarOperand for Symbol {}
