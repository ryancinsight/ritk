//! Multi-component image volumes with physical metadata.
//!
//! `ColorVolume<f32, B, C>` stores C interleaved samples per voxel in a rank-4
//! tensor with shape `[depth, rows, cols, C]`. Spatial metadata remains 3-D.

use std::fmt;

use anyhow::{bail, Result};
use coeus_core::{ComputeBackend, Scalar};
use coeus_tensor::Tensor;

use ritk_spatial::{Direction, Point, Spacing};

/// Coeus-backed color volumes with a non-spatial component axis.
pub mod native;

/// 3-D multi-component image volume.
///
/// `C` is the compile-time channel count. The tensor axis order is
/// `[depth, rows, cols, channel]`, so each voxel owns exactly `C` component
/// samples and spatial metadata remains independent of component layout.
#[derive(Clone)]
pub struct ColorVolume<T, B, const C: usize>
where
    T: Scalar,
    B: ComputeBackend,
{
    data: Tensor<T, B>,
    origin: Point<3>,
    spacing: Spacing<3>,
    direction: Direction<3>,
}

impl<T, B, const C: usize> fmt::Debug for ColorVolume<T, B, C>
where
    T: Scalar,
    B: ComputeBackend,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ColorVolume")
            .field("shape", &self.data.shape())
            .field("origin", &self.origin)
            .field("spacing", &self.spacing)
            .field("direction", &self.direction)
            .finish()
    }
}

/// RGB color volume with interleaved channels.
pub type RgbVolume<T, B> = ColorVolume<T, B, 3>;

impl<T, B, const C: usize> ColorVolume<T, B, C>
where
    T: Scalar,
    B: ComputeBackend,
{
    /// Construct a color volume after verifying the channel axis.
    pub fn try_new(
        data: Tensor<T, B>,
        origin: Point<3>,
        spacing: Spacing<3>,
        direction: Direction<3>,
    ) -> Result<Self> {
        if C == 0 {
            bail!("ColorVolume channel count C must be positive");
        }
        let shape = data.shape();
        if shape.len() != 4 {
            bail!(
                "ColorVolume requires rank-4 tensor, got rank {}",
                shape.len()
            );
        }
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
    pub fn data(&self) -> &Tensor<T, B> {
        &self.data
    }

    /// Get the full tensor shape `[depth, rows, cols, channel]`.
    pub fn shape(&self) -> [usize; 4] {
        self.data
            .shape()
            .try_into()
            .expect("ColorVolume invariant: rank-4 tensor")
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
    pub fn into_parts(self) -> (Tensor<T, B>, Point<3>, Spacing<3>, Direction<3>) {
        (self.data, self.origin, self.spacing, self.direction)
    }
}

impl<T, B, const C: usize> ColorVolume<T, B, C>
where
    T: Scalar,
    B: ComputeBackend + Default,
    B::DeviceBuffer<T>: coeus_core::CpuAddressableStorage<T>,
{
    /// Extract the underlying tensor data as a `Vec<T>`.
    #[inline]
    pub fn data_vec(&self) -> Vec<T> {
        self.data.as_slice().to_vec()
    }

    /// Provide a `&[T]` view of the volume data to a closure without
    /// allocating a `Vec`.
    ///
    /// # Panics
    /// Panics if the tensor is not contiguous or not CPU-addressable.
    #[inline]
    pub fn with_data_slice<R>(&self, f: impl FnOnce(&[T]) -> R) -> R {
        f(self.data.as_slice())
    }

    /// Deinterleave the volume into `C` scalar component buffers, each of length
    /// `depth·rows·cols` in `[depth, rows, cols]` row-major order.
    ///
    /// The interleaved layout is `[depth, rows, cols, channel]` (channel is the
    /// fastest-varying axis), so component `k`'s buffer is `flat[k], flat[k+C],
    /// flat[k+2C], …`.
    pub fn into_component_buffers(&self) -> Vec<Vec<T>>
    where
        T: Clone + Default,
    {
        let [d, r, c, _ch] = self.shape();
        let n = d * r * c;
        self.with_data_slice(|interleaved| {
            let mut comps: Vec<Vec<T>> = (0..C).map(|_| Vec::with_capacity(n)).collect();
            for (i, v) in interleaved.iter().enumerate() {
                comps[i % C].push(*v);
            }
            comps
        })
    }

    /// Rebuild a color volume from `C` scalar component buffers (each in
    /// `[depth, rows, cols]` row-major order) and the spatial shape + metadata.
    ///
    /// # Errors
    /// Returns `Err` if the number of buffers is not `C`, any buffer length is
    /// not `depth·rows·cols`, or `C == 0`.
    pub fn from_component_buffers(
        channels: &[Vec<T>],
        spatial: [usize; 3],
        origin: Point<3>,
        spacing: Spacing<3>,
        direction: Direction<3>,
        backend: &B,
    ) -> Result<Self>
    where
        T: Clone + Default,
    {
        if channels.len() != C {
            bail!(
                "from_component_buffers: expected {C} channels, got {}",
                channels.len()
            );
        }
        let [d, r, c] = spatial;
        let n = d * r * c;
        for (k, buf) in channels.iter().enumerate() {
            if buf.len() != n {
                bail!(
                    "from_component_buffers: channel {k} has length {}, expected {n}",
                    buf.len()
                );
            }
        }
        let mut interleaved = vec![T::default(); n * C];
        for (comp, buf) in channels.iter().enumerate() {
            for (i, v) in buf.iter().enumerate() {
                interleaved[i * C + comp] = *v;
            }
        }
        let tensor = Tensor::<T, B>::from_slice_on([d, r, c, C], &interleaved, backend);
        Self::try_new(tensor, origin, spacing, direction)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use coeus_core::SequentialBackend;

    type B = SequentialBackend;

    fn tensor(shape: [usize; 4]) -> Tensor<f32, B> {
        let n = shape.iter().product();
        let data = vec![0.0_f32; n];
        Tensor::<f32, B>::from_slice_on(shape, &data, &B::default())
    }

    #[test]
    fn color_volume_preserves_spatial_metadata_and_shape() {
        let origin = Point::new([1.0, 2.0, 3.0]);
        let spacing = Spacing::new([4.0, 5.0, 6.0]);
        let direction = Direction::identity();

        let volume = RgbVolume::<f32, B>::try_new(tensor([2, 3, 4, 3]), origin, spacing, direction)
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
        let err = RgbVolume::<f32, B>::try_new(
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

    #[test]
    fn component_buffers_roundtrip_is_identity() {
        let interleaved: Vec<f32> = vec![
            10.0, 100.0, 200.0, 11.0, 101.0, 201.0, 12.0, 102.0, 202.0, 13.0, 103.0, 203.0,
        ];
        let backend = B::default();
        let vol = RgbVolume::<f32, B>::try_new(
            Tensor::<f32, B>::from_slice_on([1, 2, 2, 3], &interleaved, &backend),
            Point::new([1.0, 2.0, 3.0]),
            Spacing::new([4.0, 5.0, 6.0]),
            Direction::identity(),
        )
        .unwrap();

        let comps = vol.into_component_buffers();
        assert_eq!(comps.len(), 3);
        assert_eq!(comps[0], vec![10.0, 11.0, 12.0, 13.0]);
        assert_eq!(comps[1], vec![100.0, 101.0, 102.0, 103.0]);
        assert_eq!(comps[2], vec![200.0, 201.0, 202.0, 203.0]);

        let rebuilt = RgbVolume::<f32, B>::from_component_buffers(
            &comps,
            [1, 2, 2],
            *vol.origin(),
            *vol.spacing(),
            *vol.direction(),
            &backend,
        )
        .unwrap();
        assert_eq!(rebuilt.data_vec(), interleaved);
        assert_eq!(rebuilt.origin(), vol.origin());
        assert_eq!(rebuilt.spacing(), vol.spacing());
    }

    #[test]
    fn from_component_buffers_rejects_wrong_count_and_length() {
        let backend = B::default();
        let two = vec![vec![0.0_f32; 4], vec![0.0; 4]];
        assert!(RgbVolume::<f32, B>::from_component_buffers(
            &two,
            [1, 2, 2],
            Point::origin(),
            Spacing::uniform(1.0),
            Direction::identity(),
            &backend,
        )
        .is_err());
        let bad_len = vec![vec![0.0_f32; 3], vec![0.0; 4], vec![0.0; 4]];
        assert!(RgbVolume::<f32, B>::from_component_buffers(
            &bad_len,
            [1, 2, 2],
            Point::origin(),
            Spacing::uniform(1.0),
            Direction::identity(),
            &backend,
        )
        .is_err());
    }
}
