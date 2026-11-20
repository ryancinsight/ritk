use bytes::Bytes;

mod frame;

#[derive(Debug, Clone, PartialEq)]
pub struct PixelFragments {
    pub(crate) offsets: Vec<u32>,
    pub(crate) fragments: Vec<Bytes>,
}

impl PixelFragments {
    pub(crate) fn from_spans(offsets: Vec<u32>, fragments: Vec<Bytes>) -> Self {
        Self { offsets, fragments }
    }

    pub fn new(offsets: Vec<u32>, fragments: impl IntoIterator<Item = Vec<u8>>) -> Self {
        Self {
            offsets,
            fragments: fragments.into_iter().map(Bytes::from).collect(),
        }
    }

    pub fn from_fragments(fragments: impl IntoIterator<Item = Vec<u8>>) -> Self {
        Self::new(Vec::new(), fragments)
    }

    pub fn fragments(&self) -> &[Bytes] {
        &self.fragments
    }
}
