use std::{
    fmt::Display,
    ops::{Add, AddAssign, DivAssign, Mul, MulAssign, Sub, SubAssign},
};

use super::{Almost, ContFraction, FromCF, FromRational, PseudoField, ToRational};
use ndarray::ScalarOperand;
use num_traits::{One, Zero};
use rug::{ops::NegAssign, Complete};

/// An arbitrary-precision rational number implementing [`Field`].
///
/// Currently this is based on [`rug::Rational`].
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub struct Rational(rug::Rational);

impl Rational {
    pub fn numer(&self) -> String {
        self.0.numer().to_string()
    }
    pub fn denom(&self) -> String {
        self.0.denom().to_string()
    }

    // Method to check that if the rational has a large size in his string
    // form, which can lead to Nan or Inf values when parsing to float.
    pub fn is_long(&self) -> bool {
        self.numer().len() > 12 || self.denom().len() > 12
    }
}

impl Almost for Rational {
    fn is_almost_zero(&self) -> bool {
        // With rationals I can use exact comparison.
        self.is_zero()
    }
    fn cmp_eq(&self, other: &Self) -> bool {
        self.eq(&other)
    }
    fn is_almost_zero_and_correct(&mut self) {}
}

impl FromCF for Rational {
    fn from_cont_fraction(cf: &mut ContFraction) -> Self {
        let mut numer = Rational::one();
        let mut pre_numer = Rational::zero();
        let mut denom = Rational::zero();
        let mut pre_denom = Rational::one();
        for e in cf.values.iter() {
            let a_i = Self::from_rational(&format!("{:?}", e), "1");
            let new_numer = numer.clone() * a_i.clone() + pre_numer;
            let new_denom = denom.clone() * a_i.clone() + pre_denom;

            pre_denom = denom;
            pre_numer = numer;

            denom = new_denom;
            numer = new_numer;
        }

        Self::from_rational(&format!("{:?}", numer), &format!("{:?}", denom))
        // if cf.values.len() <= 1 {
        //     return Self::from_rational(&format!("{:?}", cf.values[0]), "1");
        // } else {
        //     let val = Rational::from_rational(&format!("{:?}", cf.values.remove(0)), "1");
        //     let mut rgt = Rational::one();
        //     rgt.div_assign(&Rational::from_cont_fraction(&mut ContFraction {
        //         values: cf.values.clone(),
        //     }));
        //     return val + rgt;
        // }
    }
}

impl Display for Rational {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}/{}", self.numer(), self.denom())
    }
}

impl From<Rational> for f64 {
    fn from(rational: Rational) -> Self {
        rational.0.to_f64()
    }
}

impl num_traits::One for Rational {
    fn one() -> Self {
        Self(rug::Rational::from_f64(1.0).unwrap())
    }

    fn is_one(&self) -> bool {
        self.0 == 1.0
    }
    fn set_one(&mut self) {
        *self = Self(rug::Rational::from_f64(1.0).unwrap())
    }
}

impl num_traits::Zero for Rational {
    fn zero() -> Self {
        Self(rug::Rational::from_f64(0.0).unwrap())
    }

    fn is_zero(&self) -> bool {
        self.0 == 0.0
    }
    fn set_zero(&mut self) {
        *self = Self(rug::Rational::from_f64(0.0).unwrap())
    }
}

impl Add for Rational {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0.add(rhs.0))
    }
}

impl Sub for Rational {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        Self(self.0.sub(rhs.0))
    }
}

impl Mul for Rational {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        Self(self.0.mul(rhs.0))
    }
}

impl From<f64> for Rational {
    fn from(value: f64) -> Self {
        Self(rug::Rational::from_f64(value).unwrap())
    }
}

impl From<&z3::ast::Real<'_>> for Rational {
    fn from(value: &z3::ast::Real) -> Self {
        let real_str = value
            .to_string()
            .replace("(", "")
            .replace(")", "")
            .replace(".0", "");
        let prefix = if real_str.starts_with("-") { "-" } else { "" };
        if let Some((_, num_and_denom_str)) = real_str.split_once("/ ") {
            let (num_str, denom_str) = num_and_denom_str
                .split_once(" ")
                .expect("Something went wrong");
            let num_str = format!("{}{}", prefix, num_str);
            Rational::from_rational(num_str.trim(), denom_str.trim())
        } else {
            Rational(
                rug::Rational::parse(real_str)
                    .map(|incomplete| incomplete.complete())
                    .unwrap(),
            )
        }
    }
}

impl FromRational for Rational {
    fn from_rational(numerator: &str, denominator: &str) -> Self {
        let result = rug::Rational::parse(format!("{numerator}/{denominator}"))
            .map(|incomplete| incomplete.complete());
        if result.is_err() && denominator == "1" {
            Rational(
                numerator
                    .parse()
                    .map(|value| rug::Rational::from_f64(value).unwrap())
                    .unwrap(),
            )
        } else {
            Rational(result.unwrap())
        }
    }
    fn from_rational_modular(numerator: &str, denominator: &str, _moduler: f64) -> Self {
        Self::from_rational(numerator, denominator)
        // let num = Float::parse(numerator).unwrap().complete(53);
        // let den = Float::parse(denominator).unwrap().complete(53);
        // let moduler = Float::with_val(53, moduler);
        // let new_num = multiple_of(&num, &moduler, Round::Up)
        //     .to_integer()
        //     .expect("")
        //     .to_string()
        //     .replace(".0", "");
        // let new_den = multiple_of(&den, &moduler, Round::Down)
        //     .to_integer()
        //     .expect("")
        //     .to_string()
        //     .replace(".0", "");
        // Self::from_rational(&new_num, &new_den)
    }
}

impl ToRational for Rational {
    fn to_rational(&self) -> (String, String) {
        (self.numer(), self.denom())
    }
}

impl PseudoField for Rational {
    fn neg_assign(&mut self) {
        self.0.neg_assign();
    }

    fn abs_assign(&mut self) {
        self.0.abs_mut();
    }

    fn add_assign(&mut self, rhs: &Self) {
        self.0.add_assign(&rhs.0);
    }

    fn sub_assign(&mut self, rhs: &Self) {
        self.0.sub_assign(&rhs.0);
    }

    fn mul_assign(&mut self, rhs: &Self) {
        self.0.mul_assign(&rhs.0);
    }

    fn div_assign(&mut self, rhs: &Self) {
        self.0.div_assign(&rhs.0);
    }

    fn inv_assign(&mut self) {
        *self = Self::from_rational(&self.0.denom().to_string(), &self.0.numer().to_string());
    }

    fn to_string(&self) -> String {
        let value = if self.is_long() {
            let cf = ContFraction::from_explicit_with_precision(&self.numer(), &self.denom(), 15);
            let mut numer = 1_i128;
            let mut pre_numer = 0_i128;
            let mut denom = 0_i128;
            let mut pre_denom = 1_i128;
            for e in cf.values.iter() {
                let a_i = e.to_i128_wrapping();
                // We still can have some problems here with overflow in the multiplication.
                // TODO:  Search for un option multiplication, otherwise use a try-catch like struct.
                let new_numer = numer * a_i + pre_numer;
                let new_denom = denom * a_i + pre_denom;
                pre_denom = denom;
                pre_numer = numer;

                denom = new_denom;
                numer = new_numer;
            }

            numer as f64 / denom as f64
        } else {
            let numer: f64 = self
                .numer()
                .replace("\"", "")
                .parse::<f64>()
                .expect(&format!("Value: '{:?}' could not be parsed.", self.numer()));
            let denom: f64 = self
                .denom()
                .replace("\"", "")
                .parse::<f64>()
                .expect(&format!("Value: '{:?}' could not be parsed.", self.denom()));
            numer / denom
        };

        if value.is_nan() {
            panic!(
                "Value is NaN. Numer: {:?}. Denom: {:?}",
                self.numer(),
                self.denom()
            )
        }
        format!("{:?}", value)
    }
}

impl ScalarOperand for Rational {}
