//! Validated block-matching input views.

use anyhow::{bail, Result};

/// Moving-image samples and their optional per-sample validity.
///
/// A `true` validity entry means the corresponding moving sample is available.
/// The matcher excludes a candidate unless every sample in its moving block is
/// valid. This preserves normalized-correlation semantics at internal
/// field-of-view boundaries: padding and variable-size overlaps never become
/// evidence.
///
/// # Examples
///
/// ```
/// use ritk_block_matching::MovingSamples;
///
/// let samples = [1.0_f32, 2.0, 3.0];
/// let validity = [true, false, true];
/// MovingSamples::try_with_validity(&samples, &validity)?;
/// # Ok::<(), anyhow::Error>(())
/// ```
#[derive(Debug, Clone, Copy)]
pub struct MovingSamples<'a, T> {
    values: &'a [T],
    validity: Option<&'a [bool]>,
}

impl<'a, T> MovingSamples<'a, T> {
    /// Borrow a moving image whose samples are all geometrically available.
    #[must_use]
    pub const fn complete(values: &'a [T]) -> Self {
        Self {
            values,
            validity: None,
        }
    }

    /// Borrow a moving image with an explicit availability mask.
    ///
    /// # Errors
    ///
    /// Returns an error when `validity` does not contain exactly one entry per
    /// sample.
    pub fn try_with_validity(values: &'a [T], validity: &'a [bool]) -> Result<Self> {
        if values.len() != validity.len() {
            bail!(
                "moving validity length {} does not match sample length {}",
                validity.len(),
                values.len()
            );
        }
        Ok(Self {
            values,
            validity: Some(validity),
        })
    }

    pub(crate) const fn values(self) -> &'a [T] {
        self.values
    }

    pub(crate) const fn validity(self) -> Option<&'a [bool]> {
        self.validity
    }
}
