//! Stored volumes ordered along one acquisition axis.

use ritk_diffusion_scheme::GradientScheme;
use thiserror::Error;

use crate::StoredVolume;

/// Meaning carried by a non-spatial axis in an image series.
///
/// `Unspecified` records formats that identify an axis but do not define its
/// meaning. `List` identifies an ordered collection. Diffusion metadata reuses
/// RITK's validated gradient scheme in the same volume order.
#[derive(Debug, PartialEq)]
#[non_exhaustive]
pub enum SeriesAxis {
    /// The source is one spatial volume without an acquisition axis.
    SingleVolume,
    /// The source format does not identify the axis meaning.
    Unspecified,
    /// The source identifies an ordered list of volumes.
    List,
    /// The source carries one validated diffusion entry per volume.
    Diffusion(GradientScheme),
}

/// Stored volumes with the semantics of their shared acquisition axis.
///
/// Per-volume geometry remains attached to each [`StoredVolume`], so formats
/// with varying frame positions can retain them without flattening metadata.
#[derive(Debug)]
pub struct StoredSeries {
    volumes: Box<[StoredVolume]>,
    axis: SeriesAxis,
}

impl StoredSeries {
    /// Creates a nonempty series and checks axis-specific metadata lengths.
    ///
    /// # Errors
    ///
    /// Returns [`StoredSeriesError::Empty`] or
    /// [`StoredSeriesError::DiffusionVolumeCountMismatch`] when the diffusion
    /// scheme does not describe the supplied volume sequence.
    pub fn new(volumes: Vec<StoredVolume>, axis: SeriesAxis) -> Result<Self, StoredSeriesError> {
        if volumes.is_empty() {
            return Err(StoredSeriesError::Empty);
        }
        match &axis {
            SeriesAxis::SingleVolume if volumes.len() != 1 => {
                return Err(StoredSeriesError::SingleVolumeCountMismatch {
                    volumes: volumes.len(),
                });
            }
            SeriesAxis::Diffusion(scheme) if scheme.len() != volumes.len() => {
                return Err(StoredSeriesError::DiffusionVolumeCountMismatch {
                    volumes: volumes.len(),
                    gradients: scheme.len(),
                });
            }
            SeriesAxis::SingleVolume
            | SeriesAxis::Unspecified
            | SeriesAxis::List
            | SeriesAxis::Diffusion(_) => {}
        }
        Ok(Self {
            volumes: volumes.into_boxed_slice(),
            axis,
        })
    }

    /// Returns the volumes in acquisition order.
    #[must_use]
    pub fn volumes(&self) -> &[StoredVolume] {
        &self.volumes
    }

    /// Returns the semantic descriptor for the acquisition axis.
    #[must_use]
    pub const fn axis(&self) -> &SeriesAxis {
        &self.axis
    }
}

/// A stored series does not match its acquisition-axis descriptor.
#[derive(Debug, Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum StoredSeriesError {
    /// An acquisition series must contain at least one volume.
    #[error("stored series must contain at least one volume")]
    Empty,
    /// A single-volume descriptor was paired with multiple volumes.
    #[error("single-volume descriptor has {volumes} volumes")]
    SingleVolumeCountMismatch {
        /// Number of supplied volumes.
        volumes: usize,
    },
    /// Diffusion entries do not match the number of stored volumes.
    #[error("stored series has {volumes} volumes but {gradients} diffusion entries")]
    DiffusionVolumeCountMismatch {
        /// Number of stored volumes.
        volumes: usize,
        /// Number of diffusion entries.
        gradients: usize,
    },
}

#[cfg(test)]
mod tests {
    use ritk_codecs::{ByteOrder, SampleBuffer};
    use ritk_diffusion_scheme::{GradientFrame, GradientScheme};
    use ritk_image::ImageMetadata;
    use ritk_spatial::{CoordinateMap, Vector};

    use super::{SeriesAxis, StoredSeries, StoredSeriesError};
    use crate::{IntensityCalibration, StoredVolume};

    fn volume(value: u16) -> StoredVolume {
        StoredVolume::new(
            [1, 1, 1],
            SampleBuffer::from_samples(vec![value]),
            ImageMetadata::default(),
            CoordinateMap::Cartesian,
            IntensityCalibration::Identity,
        )
        .expect("one-sample volume has valid default metadata")
    }

    #[test]
    fn list_axis_keeps_exact_volume_order() {
        let series = StoredSeries::new(vec![volume(11), volume(29)], SeriesAxis::List)
            .expect("nonempty list series");

        assert_eq!(series.volumes().len(), 2);
        assert_eq!(
            series.volumes()[0]
                .samples()
                .encode(ByteOrder::LeastSignificantByteFirst)
                .expect("first sample encoding"),
            vec![11, 0]
        );
        assert_eq!(
            series.volumes()[1]
                .samples()
                .encode(ByteOrder::LeastSignificantByteFirst)
                .expect("second sample encoding"),
            vec![29, 0]
        );
        assert_eq!(series.axis(), &SeriesAxis::List);
    }

    #[test]
    fn empty_series_is_rejected() {
        assert!(matches!(
            StoredSeries::new(Vec::new(), SeriesAxis::Unspecified),
            Err(StoredSeriesError::Empty)
        ));
    }

    #[test]
    fn single_volume_axis_rejects_multiple_frames() {
        assert!(matches!(
            StoredSeries::new(vec![volume(11), volume(29)], SeriesAxis::SingleVolume),
            Err(StoredSeriesError::SingleVolumeCountMismatch { volumes: 2 })
        ));
    }

    #[test]
    fn diffusion_axis_requires_one_entry_per_volume() {
        let scheme = GradientScheme::from_seconds_per_square_millimeter(
            vec![
                (0.0, Vector::new([0.0, 0.0, 0.0])),
                (1000.0, Vector::new([1.0, 0.0, 0.0])),
            ],
            GradientFrame::Lps,
        )
        .expect("valid two-volume scheme");

        assert!(matches!(
            StoredSeries::new(vec![volume(7)], SeriesAxis::Diffusion(scheme)),
            Err(StoredSeriesError::DiffusionVolumeCountMismatch {
                volumes: 1,
                gradients: 2
            })
        ));
    }
}
