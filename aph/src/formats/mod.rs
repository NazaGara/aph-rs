//! Input and output file formats.

use std::str::FromStr;

use thiserror::Error;

use crate::linalg::fields::PseudoField;

pub mod aut;
pub mod tra;

#[derive(Debug, Error)]
#[error("{0}")]
pub struct ParseError(String);

pub(crate) struct Cursor<'i> {
    tail: &'i str,
}

impl<'i> Cursor<'i> {
    pub fn new(input: &'i str) -> Self {
        let mut this = Self { tail: input };
        this.consume_whitespace();
        this
    }

    pub fn consume_whitespace(&mut self) {
        self.tail = self.tail.trim_start()
    }

    pub fn consume_tag(&mut self, tag: impl AsRef<str>) -> Result<(), ParseError> {
        let tag = tag.as_ref();
        self.tail = self
            .tail
            .strip_prefix(tag)
            .ok_or_else(|| ParseError(format!("Expected `{tag}` but found {}.", self.tail)))?;
        self.consume_whitespace();
        Ok(())
    }

    pub fn consume_number(&mut self) -> Result<&'i str, ParseError> {
        let mut chars = self.tail.chars();
        while chars
            .as_str()
            .starts_with(|c| char::is_digit(c, 10) || c == '.' || c == 'e' || c == '-')
        {
            chars.next();
        }
        let length = self.tail.len() - chars.as_str().len();
        if length > 0 {
            let number = &self.tail[..length];
            self.tail = chars.as_str();
            self.consume_whitespace();
            Ok(number)
        } else {
            Err(ParseError("Expected number.".to_owned()))
        }
    }

    pub fn consume_usize(&mut self) -> Result<usize, ParseError> {
        let number = self.consume_number()?;
        usize::from_str(number)
            .map_err(|_| ParseError(format!("Unable to convert number {number} to usize.")))
    }

    pub fn consume_rational<F: PseudoField>(&mut self) -> Result<F, ParseError> {
        let nominator = self.consume_number()?;
        let denominator = if self.consume_tag("/").is_ok() {
            self.consume_number()?
        } else {
            "1"
        };
        //.map_err(|_| ParseError("Unable to construct rational.".to_owned()))
        Ok(F::from_rational(nominator, denominator))
    }
}
