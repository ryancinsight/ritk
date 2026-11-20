use crate::dataset::Dataset;

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
