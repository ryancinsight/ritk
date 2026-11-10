use thiserror::Error;

use crate::ImageReadBudgetError;

/// Errors returned by [`super::decode_jpeg2000_fragment`].
///
/// # Examples
///
/// ```
/// use ritk_codecs::{
///     decode_jpeg2000_fragment, ImageReadBudget, Jpeg2000DecodeError, PixelLayout,
///     PixelSignedness,
/// };
///
/// let layout = PixelLayout {
///     rows: 1,
///     cols: 1,
///     samples_per_pixel: 1,
///     bits_allocated: 8,
///     bits_stored: 8,
///     pixel_representation: PixelSignedness::Unsigned,
///     rescale_slope: 1.0,
///     rescale_intercept: 0.0,
/// };
/// let budget = ImageReadBudget::new(1, 1, 1)?;
/// let error = decode_jpeg2000_fragment(&[0xFF, 0x4F, 0x00], layout, budget)
///     .expect_err("three encoded bytes exceed a one-byte budget");
/// assert!(matches!(error, Jpeg2000DecodeError::Budget { .. }));
/// # Ok::<(), ritk_codecs::ImageReadBudgetError>(())
/// ```
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum Jpeg2000DecodeError {
    /// The encoded fragment or decoded sample storage exceeds the read budget.
    #[error(transparent)]
    Budget {
        /// Resource ceiling rejected before parsing or output allocation.
        #[from]
        source: ImageReadBudgetError,
    },
    /// The allocator cannot reserve the `f32` output frame the budget admitted.
    #[error("JPEG 2000 output of {samples} samples cannot be allocated")]
    OutputAllocation {
        /// Samples in the output frame.
        samples: usize,
        /// Allocator refusal.
        #[source]
        source: std::collections::TryReserveError,
    },
    /// The codestream is malformed, truncated, or outside the supported profile.
    ///
    /// The codestream parser reports its failures as `anyhow` chains; the chain
    /// is preserved here until each parser stage has its own variants.
    #[error(transparent)]
    Codestream(#[from] anyhow::Error),
}
