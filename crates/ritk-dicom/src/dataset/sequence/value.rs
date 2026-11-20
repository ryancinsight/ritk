use std::borrow::Cow;

use anyhow::{anyhow, Result};

use super::{PixelFragments, Sequence};
use crate::dataset::{Dataset, DicomNumber, PrimitiveValue};

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
    pub fn to_num<T: DicomNumber>(&self) -> Result<T> {
        self.require_primitive()?.to_num()
    }

    /// # Errors
    /// Every primitive component must fit the requested type.
    pub fn to_numbers<T: DicomNumber>(&self) -> Result<Vec<T>> {
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
