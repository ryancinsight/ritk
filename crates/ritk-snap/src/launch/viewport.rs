//! Validated eframe viewport dimensions used by capture fixtures.

use std::num::NonZeroU16;
use std::str::FromStr;

/// A positive eframe logical viewport size.
///
/// The values are logical points. The operating system may map them to a
/// different number of physical pixels according to the display scale; a
/// capture record must therefore retain both the requested size and the
/// observed image dimensions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EframeViewport {
    width: NonZeroU16,
    height: NonZeroU16,
}

impl EframeViewport {
    /// Construct a viewport from positive dimensions no larger than `u16`.
    ///
    /// # Errors
    /// Returns [`EframeViewportError`] when either dimension is zero or does
    /// not fit the bounded host window representation.
    pub fn new(width: u32, height: u32) -> Result<Self, EframeViewportError> {
        let width = NonZeroU16::new(
            u16::try_from(width).map_err(|_| EframeViewportError::TooLarge { width, height })?,
        )
        .ok_or(EframeViewportError::Zero { width, height })?;
        let height =
            NonZeroU16::new(
                u16::try_from(height).map_err(|_| EframeViewportError::TooLarge {
                    width: u32::from(width.get()),
                    height,
                })?,
            )
            .ok_or(EframeViewportError::Zero {
                width: u32::from(width.get()),
                height,
            })?;
        Ok(Self { width, height })
    }

    /// Return the logical width in points.
    pub const fn width(self) -> u16 {
        self.width.get()
    }

    /// Return the logical height in points.
    pub const fn height(self) -> u16 {
        self.height.get()
    }

    /// Return the dimensions in the form accepted by egui's viewport builder.
    pub fn logical_size(self) -> [f32; 2] {
        [f32::from(self.width()), f32::from(self.height())]
    }
}

impl Default for EframeViewport {
    fn default() -> Self {
        Self::new(1_280, 800).expect("default eframe viewport dimensions are valid")
    }
}

/// Error returned when an eframe viewport string or dimension is invalid.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum EframeViewportError {
    /// A dimension was zero.
    #[error("eframe viewport dimensions must be positive (got {width}x{height})")]
    Zero { width: u32, height: u32 },
    /// A dimension exceeds the bounded host representation.
    #[error("eframe viewport dimensions must fit u16 (got {width}x{height})")]
    TooLarge { width: u32, height: u32 },
    /// The command-line value was not in `WIDTHxHEIGHT` form.
    #[error("eframe viewport must use WIDTHxHEIGHT (got {value:?})")]
    Format { value: String },
    /// One command-line dimension was not an unsigned integer.
    #[error("eframe viewport dimension is not an unsigned integer (got {value:?})")]
    Number { value: String },
}

impl FromStr for EframeViewport {
    type Err = EframeViewportError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let Some((width, height)) = value.split_once('x') else {
            return Err(EframeViewportError::Format {
                value: value.to_owned(),
            });
        };
        let width = width
            .parse::<u32>()
            .map_err(|_| EframeViewportError::Number {
                value: width.to_owned(),
            })?;
        let height = height
            .parse::<u32>()
            .map_err(|_| EframeViewportError::Number {
                value: height.to_owned(),
            })?;
        Self::new(width, height)
    }
}

#[cfg(test)]
mod tests {
    use super::{EframeViewport, EframeViewportError};
    use std::str::FromStr;

    #[test]
    fn parses_positive_dimensions() {
        let viewport = EframeViewport::from_str("1024x640").expect("valid viewport");
        assert_eq!((viewport.width(), viewport.height()), (1024, 640));
        assert_eq!(viewport.logical_size(), [1024.0, 640.0]);
    }

    #[test]
    fn rejects_malformed_dimensions() {
        assert!(matches!(
            EframeViewport::from_str("1024"),
            Err(EframeViewportError::Format { .. })
        ));
        assert!(matches!(
            EframeViewport::from_str("widex640"),
            Err(EframeViewportError::Number { .. })
        ));
        assert!(matches!(
            EframeViewport::from_str("0x640"),
            Err(EframeViewportError::Zero { .. })
        ));
        assert!(matches!(
            EframeViewport::from_str("65536x640"),
            Err(EframeViewportError::TooLarge { .. })
        ));
    }

    #[test]
    fn default_is_the_existing_viewport() {
        assert_eq!(
            (
                EframeViewport::default().width(),
                EframeViewport::default().height()
            ),
            (1280, 800)
        );
    }
}
