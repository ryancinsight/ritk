use std::collections::BTreeMap;

use anyhow::{anyhow, Result};

use super::{DataElement, FileMeta, FileMetaBuilder, Length, PrimitiveValue, VR};
use crate::DicomTag;

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Dataset {
    elements: BTreeMap<DicomTag, DataElement>,
    pub(crate) item_length: Length,
}

impl Dataset {
    pub const fn new() -> Self {
        Self {
            elements: BTreeMap::new(),
            item_length: Length::Undefined,
        }
    }

    pub fn from_elements(elements: impl IntoIterator<Item = DataElement>) -> Self {
        let mut dataset = Self::new();
        for element in elements {
            dataset.put(element);
        }
        dataset
    }

    pub fn get(&self, tag: DicomTag) -> Option<&DataElement> {
        self.elements.get(&tag)
    }

    /// # Errors
    /// The tag must occur at this level of the dataset.
    pub fn element(&self, tag: DicomTag) -> Result<&DataElement> {
        self.get(tag)
            .ok_or_else(|| anyhow!("missing DICOM element {tag}"))
    }

    pub fn get_mut(&mut self, tag: DicomTag) -> Option<&mut DataElement> {
        self.elements.get_mut(&tag)
    }

    pub fn put(&mut self, element: DataElement) -> Option<DataElement> {
        self.elements.insert(element.tag(), element)
    }

    pub fn put_value(
        &mut self,
        tag: DicomTag,
        vr: VR,
        value: impl Into<PrimitiveValue>,
    ) -> Option<DataElement> {
        self.put(DataElement::new(tag, vr, value.into()))
    }

    pub fn remove_element(&mut self, tag: DicomTag) -> Option<DataElement> {
        self.elements.remove(&tag)
    }

    pub fn iter(&self) -> impl Iterator<Item = &DataElement> {
        self.into_iter()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut DataElement> {
        self.elements.values_mut()
    }

    /// # Errors
    /// The builder or dataset must supply the required Part 10 identifiers.
    pub fn with_meta(self, builder: FileMetaBuilder) -> Result<super::DicomFile> {
        let meta = FileMeta::from_builder(builder, &self)?;
        Ok(super::DicomFile::new(meta, self))
    }
}

impl<'a> IntoIterator for &'a Dataset {
    type Item = &'a DataElement;
    type IntoIter = std::collections::btree_map::Values<'a, DicomTag, DataElement>;

    fn into_iter(self) -> Self::IntoIter {
        self.elements.values()
    }
}
