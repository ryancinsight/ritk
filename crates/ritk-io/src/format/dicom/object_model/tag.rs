//! DICOM tag types and private-tag predicate.

use std::fmt;

/// DICOM tag in `(group, element)` form.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DicomTag {
    /// Group number.
    pub group: u16,
    /// Element number.
    pub element: u16,
}

impl DicomTag {
    /// Create a new DICOM tag.
    #[inline]
    pub const fn new(group: u16, element: u16) -> Self {
        Self { group, element }
    }

    /// Return the canonical `(gggg,eeee)` text form.
    #[inline]
    pub fn canonical(self) -> String {
        format!("{:04X},{:04X}", self.group, self.element)
    }
}

impl fmt::Display for DicomTag {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({:04X},{:04X})", self.group, self.element)
    }
}

/// Determine whether a tag is private.
///
/// In DICOM, private tags occupy odd-numbered groups.
#[inline]
pub const fn is_private_tag(tag: DicomTag) -> bool {
    tag.group % 2 == 1
}
