use thiserror::Error;

/// Finite resource ceilings for reading one image or an acquisition series.
///
/// The default encoded and decoded byte limits match RITK's established
/// DICOM read budget. The series-volume limit matches the bounded metadata
/// entry count used by the NRRD reader. Format adapters check declared sizes
/// before allocating payload or per-volume storage. Callers may select larger
/// limits for trusted data or smaller limits for constrained applications.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ImageReadBudget {
    encoded_bytes: u64,
    decoded_bytes: u64,
    series_volumes: u64,
}

impl ImageReadBudget {
    /// Default limits for encoded bytes, decoded bytes, and series volumes.
    pub const DEFAULT: Self = Self {
        encoded_bytes: 1024 * 1024 * 1024,
        decoded_bytes: 1024 * 1024 * 1024,
        series_volumes: 65_536,
    };

    /// Creates a budget with explicit nonzero ceilings.
    ///
    /// # Errors
    ///
    /// Returns [`ImageReadBudgetError::ZeroLimit`] when any ceiling is zero.
    pub const fn new(
        max_encoded_bytes: u64,
        max_decoded_bytes: u64,
        max_series_volumes: u64,
    ) -> Result<Self, ImageReadBudgetError> {
        if max_encoded_bytes == 0 {
            return Err(ImageReadBudgetError::ZeroLimit {
                resource: ImageReadResource::EncodedBytes,
            });
        }
        if max_decoded_bytes == 0 {
            return Err(ImageReadBudgetError::ZeroLimit {
                resource: ImageReadResource::DecodedBytes,
            });
        }
        if max_series_volumes == 0 {
            return Err(ImageReadBudgetError::ZeroLimit {
                resource: ImageReadResource::SeriesVolumes,
            });
        }
        Ok(Self {
            encoded_bytes: max_encoded_bytes,
            decoded_bytes: max_decoded_bytes,
            series_volumes: max_series_volumes,
        })
    }

    /// Returns the maximum accepted encoded payload size.
    #[must_use]
    pub const fn max_encoded_bytes(self) -> u64 {
        self.encoded_bytes
    }

    /// Returns the maximum accepted decoded output storage, including samples
    /// and retained metadata represented in the output.
    #[must_use]
    pub const fn max_decoded_bytes(self) -> u64 {
        self.decoded_bytes
    }

    /// Returns the maximum accepted number of volumes in one series.
    #[must_use]
    pub const fn max_series_volumes(self) -> u64 {
        self.series_volumes
    }

    /// Checks whether an observed resource use fits this budget.
    ///
    /// # Errors
    ///
    /// Returns [`ImageReadBudgetError::Exceeded`] when `actual` is greater
    /// than the selected resource ceiling.
    pub const fn check(
        self,
        resource: ImageReadResource,
        actual: u64,
    ) -> Result<(), ImageReadBudgetError> {
        let maximum = match resource {
            ImageReadResource::EncodedBytes => self.encoded_bytes,
            ImageReadResource::DecodedBytes => self.decoded_bytes,
            ImageReadResource::SeriesVolumes => self.series_volumes,
        };
        if actual > maximum {
            return Err(ImageReadBudgetError::Exceeded {
                resource,
                actual,
                maximum,
            });
        }
        Ok(())
    }
}

/// A resource dimension bounded by [`ImageReadBudget`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ImageReadResource {
    /// Bytes read from the encoded payload source.
    EncodedBytes,
    /// Bytes retained for decoded output, including samples and represented metadata.
    DecodedBytes,
    /// Number of images in an acquisition series.
    SeriesVolumes,
}

impl std::fmt::Display for ImageReadResource {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EncodedBytes => formatter.write_str("encoded payload bytes"),
            Self::DecodedBytes => formatter.write_str("decoded output bytes"),
            Self::SeriesVolumes => formatter.write_str("series volumes"),
        }
    }
}

/// A resource budget is invalid or a read exceeded its configured ceiling.
#[derive(Clone, Copy, Debug, Error, Eq, PartialEq)]
pub enum ImageReadBudgetError {
    /// A configured resource ceiling is zero.
    #[error("image read budget for {resource} must be nonzero")]
    ZeroLimit {
        /// Resource whose configured ceiling is zero.
        resource: ImageReadResource,
    },
    /// An input declaration or observed payload exceeds its ceiling.
    #[error("image read requires {actual} {resource}, exceeding the limit of {maximum}")]
    Exceeded {
        /// Resource whose ceiling was exceeded.
        resource: ImageReadResource,
        /// Declared or observed resource use.
        actual: u64,
        /// Configured resource ceiling.
        maximum: u64,
    },
}
