use std::borrow::Cow;
use std::str::FromStr;

use anyhow::{anyhow, Result};
use num_traits::NumCast;

use super::{Dataset, PrimitiveValue};

mod fragments;
pub use fragments::PixelFragments;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Length {
    Defined,
    #[default]
    Undefined,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Sequence {
    pub(crate) items: Vec<Dataset>,
    pub(crate) length: Length,
}

impl Sequence {
    pub fn new(items: Vec<Dataset>, length: Length) -> Self {
        Self { items, length }
    }

    pub fn items(&self) -> &[Dataset] {
        &self.items
    }

    pub fn items_mut(&mut self) -> &mut [Dataset] {
        &mut self.items
    }
}

impl From<Vec<Dataset>> for Sequence {
    fn from(items: Vec<Dataset>) -> Self {
        Self::new(items, Length::Undefined)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Primitive(PrimitiveValue),
    Sequence(Sequence),
    PixelSequence(PixelFragments),
}

impl Value {
    pub fn items(&self) -> Option<&[Dataset]> {
        match self {
            Self::Sequence(sequence) => Some(sequence.items()),
            _ => None,
        }
    }

    pub const fn primitive(&self) -> Option<&PrimitiveValue> {
        match self {
            Self::Primitive(p) => Some(p),
            Self::Sequence(_) | Self::PixelSequence(_) => None,
        }
    }

    fn require_primitive(&self) -> Result<&PrimitiveValue> {
        self.primitive()
            .ok_or_else(|| anyhow!("DICOM {} is not primitive", self.value_type()))
    }

    /// # Errors
    /// Sequences and fragments cannot be converted to primitive text.
    pub fn to_str(&self) -> Result<Cow<'_, str>> {
        Ok(self.require_primitive()?.to_str())
    }

    /// # Errors
    /// Sequences and fragments cannot be converted to primitive bytes.
    pub fn to_bytes(&self) -> Result<Cow<'_, [u8]>> {
        Ok(self.require_primitive()?.to_bytes())
    }

    /// # Errors
    /// The first primitive value must fit the requested type.
    pub fn to_num<T: NumCast + FromStr>(&self) -> Result<T> {
        self.require_primitive()?.to_num()
    }

    /// # Errors
    /// Every primitive component must fit the requested type.
    pub fn to_numbers<T: NumCast + FromStr>(&self) -> Result<Vec<T>> {
        self.require_primitive()?.to_numbers()
    }

    pub fn cardinality(&self) -> usize {
        match self {
            Self::Primitive(p) => p.cardinality(),
            Self::Sequence(_) | Self::PixelSequence(_) => 1,
        }
    }

    pub const fn value_type(&self) -> &'static str {
        match self {
            Self::Primitive(p) => p.value_type(),
            Self::Sequence(_) => "Sequence",
            Self::PixelSequence(_) => "PixelSequence",
        }
    }
}

impl From<PrimitiveValue> for Value {
    fn from(value: PrimitiveValue) -> Self {
        Self::Primitive(value)
    }
}

impl From<Sequence> for Value {
    fn from(value: Sequence) -> Self {
        Self::Sequence(value)
    }
}

impl From<PixelFragments> for Value {
    fn from(value: PixelFragments) -> Self {
        Self::PixelSequence(value)
    }
}
