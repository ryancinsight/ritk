use std::borrow::Cow;

use anyhow::Result;

use super::VR;
use crate::dataset::{Dataset, DicomNumber, Value};
use crate::DicomTag;

#[derive(Debug, Clone, PartialEq)]
pub struct DataElement {
    tag: DicomTag,
    vr: VR,
    value: Value,
}

impl DataElement {
    pub fn new(tag: DicomTag, vr: VR, value: impl Into<Value>) -> Self {
        Self {
            tag,
            vr,
            value: value.into(),
        }
    }

    pub const fn tag(&self) -> DicomTag {
        self.tag
    }

    pub const fn vr(&self) -> VR {
        self.vr
    }

    pub const fn value(&self) -> &Value {
        &self.value
    }

    pub fn value_mut(&mut self) -> &mut Value {
        &mut self.value
    }

    /// # Errors
    /// The element must have a primitive value.
    pub fn to_str(&self) -> Result<Cow<'_, str>> {
        self.value.to_str()
    }

    /// # Errors
    /// Sequences and fragments do not have primitive bytes.
    pub fn to_bytes(&self) -> Result<Cow<'_, [u8]>> {
        self.value.to_bytes()
    }

    pub fn items(&self) -> Option<&[Dataset]> {
        self.value.items()
    }

    /// # Errors
    /// Every value must fit the requested numeric type.
    pub fn to_numbers<T: DicomNumber>(&self) -> Result<Vec<T>> {
        self.value.to_numbers()
    }

    /// # Errors
    /// The first value must be a number representable by the requested type.
    pub fn to_num<T: DicomNumber>(&self) -> Result<T> {
        self.value.to_num()
    }
}
