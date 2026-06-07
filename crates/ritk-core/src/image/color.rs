//! Multi-component image volumes with physical metadata.
//!
//! `ColorVolume<B, C>` stores C interleaved samples per voxel in a rank-4
//! tensor with shape `[depth, rows, cols, C]`. Spatial metadata remains 3-D.

use anyhow::{bail, Result};
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

use crate::spatial::{Direction, Point, Spacing};

/// 3-D multi-component image volume.
///
/// `C` is the compile-time channel count. The tensor axis order is
/// `[depth, rows, cols, channel]`, so each voxel owns exactly `C` component
/// samples and spatial metadata remains independent of component layout.
#[derive(Debug, Clone)]
pub struct ColorVolume<B: Backend, const C: usize> {
    data: Tensor<B, 4>,
    origin: Point<3>,
    spacing: Spacing<3>,
    direction: Direction<3>,
}

/// RGB color volume with interleaved channels.
pub type RgbVolume<B> = ColorVolume<B, 3>;

impl<B: Backend, const C: usize> ColorVolume<B, C> {
    /// Construct a color volume after verifying the channel axis.
    pub fn try_new(
        data: Tensor<B, 4>,
        origin: Point<3>,
        spacing: Spacing<3>,
        direction: Direction<3>,
    ) -> Result<Self> {
        if C == 0 {
            bail!("ColorVolume channel count C must be positive");
        }
        let shape = data.dims();
        if shape[3] != C {
            bail!(
                "ColorVolume channel axis mismatch: tensor has {}, type requires {}",
                shape[3],
                C
            );
        }
        Ok(Self {
            data,
            origin,
            spacing,
            direction,
        })
    }

    /// Get the image data tensor.
    pub fn data(&self) -> &Tensor<B, 4> {
        &self.data
    }

    /// Get the full tensor shape `[depth, rows, cols, channel]`.
    pub fn shape(&self) -> [usize; 4] {
        self.data.dims()
    }

    /// Get the spatial shape `[depth, rows, cols]`.
    pub fn spatial_shape(&self) -> [usize; 3] {
        let [depth, rows, cols, _channels] = self.shape();
        [depth, rows, cols]
    }

    /// Compile-time channel count.
    pub const fn channels(&self) -> usize {
        C
    }

    /// Get the physical origin.
    pub fn origin(&self) -> &Point<3> {
        &self.origin
    }

    /// Get voxel spacing.
    pub fn spacing(&self) -> &Spacing<3> {
        &self.spacing
    }

    /// Get direction cosines.
    pub fn direction(&self) -> &Direction<3> {
        &self.direction
    }

    /// Consume the volume and return all components.
    pub fn into_parts(self) -> (Tensor<B, 4>, Point<3>, Spacing<3>, Direction<3>) {
        (self.data, self.origin, self.spacing, self.direction)
    }

    /// Extract the underlying f32 tensor data as a `Vec<f32>`.
    ///
    /// # Panics
    /// Panics if the tensor's internal scalar type is not `f32`.
    #[inline]
    pub fn data_vec(&self) -> Vec<f32> {
        self.data
            .clone()
            .into_data()
            .into_vec::<f32>()
            .expect("ColorVolume::data_vec requires f32 backend tensor")
    }

    /// Provide a `&[f32]` view of the volume data to a closure without
    /// allocating a `Vec`.
    ///
    /// # Panics
    /// Panics if the tensor's internal scalar type is not `f32`.
    #[inline]
    pub fn with_data_slice<R>(&self, f: impl FnOnce(&[f32]) -> R) -> R {
        let data = self.data.clone().into_data();
        let slice = data
            .as_slice::<f32>()
            .expect("color image data must be contiguous f32");
        f(slice)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Shape, TensorData};
    use burn_ndarray::NdArray;

    type B = NdArray<f32>;

    fn tensor(shape: [usize; 4]) -> Tensor<B, 4> {
        let n = shape.iter().product();
        let data = vec![0.0; n];
        Tensor::<B, 4>::from_data(
            TensorData::new(data, Shape::new(shape)),
            &Default::default(),
        )
    }

    #[test]
    fn color_volume_preserves_spatial_metadata_and_shape() {
        let origin = Point::new([1.0, 2.0, 3.0]);
        let spacing = Spacing::new([4.0, 5.0, 6.0]);
        let direction = Direction::identity();

        let volume = RgbVolume::<B>::try_new(tensor([2, 3, 4, 3]), origin, spacing, direction)
            .expect("RGB channel count must be valid");

        assert_eq!(volume.shape(), [2, 3, 4, 3]);
        assert_eq!(volume.spatial_shape(), [2, 3, 4]);
        assert_eq!(volume.channels(), 3);
        assert_eq!(volume.origin(), &origin);
        assert_eq!(volume.spacing(), &spacing);
        assert_eq!(volume.direction(), &direction);
    }

    #[test]
    fn color_volume_rejects_wrong_channel_axis() {
        let err = RgbVolume::<B>::try_new(
            tensor([1, 2, 3, 1]),
            Point::origin(),
            Spacing::uniform(1.0),
            Direction::identity(),
        )
        .unwrap_err();

        assert!(
            err.to_string().contains("channel axis"),
            "expected channel-axis error, got {err:#}"
        );
    }
}
