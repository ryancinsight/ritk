//! Shared single-volume / acquisition-series I/O helpers for the format crates.
//!
//! Every container format (NIfTI, NRRD, MIF, MGH, MetaImage, Analyze, MINC,
//! JPEG) writes one `[Z, Y, X]` voxel block per volume and validates that
//! every volume of a series shares a single spatial grid; every reader that
//! returns one 3-D volume rejects a multi-frame container rather than silently
//! decoding its first frame. That contract lives here once so the format
//! crates keep only their header parsers and axis conventions.
//!
//! The little-endian serializers exist because a `write_all` per voxel is far
//! slower across millions of voxels: they stage a bounded block and issue one
//! `write_all` per block, so peak memory stays flat as the payload grows.

use std::io::Write;

use anyhow::{bail, Result};
use coeus_core::ComputeBackend;

use crate::types::Image;
use ritk_spatial::{Direction, Point, Spacing};

/// Serialize `values` as little-endian samples through a bounded block buffer.
///
/// One `write_all` per element is far slower across millions of voxels. A
/// fixed-size block issues one `write_all` per block, keeps peak memory flat
/// regardless of payload size, and emits exactly the bytes a per-element
/// `to_le_bytes` loop would.
macro_rules! define_write_le {
    ($name:ident, $ty:ty, $doc:literal) => {
        #[doc = $doc]
        pub fn $name<W: Write + ?Sized>(writer: &mut W, values: &[$ty]) -> Result<()> {
            const SAMPLES_PER_BLOCK: usize = 4096;
            const SAMPLE_WIDTH: usize = std::mem::size_of::<$ty>();
            let mut block = [0u8; SAMPLES_PER_BLOCK * SAMPLE_WIDTH];
            for chunk in values.chunks(SAMPLES_PER_BLOCK) {
                for (slot, &value) in block.chunks_exact_mut(SAMPLE_WIDTH).zip(chunk.iter()) {
                    slot.copy_from_slice(&value.to_le_bytes());
                }
                writer.write_all(&block[..chunk.len() * SAMPLE_WIDTH])?;
            }
            Ok(())
        }
    };
}

define_write_le!(
    write_le_f32,
    f32,
    "Write `values` as little-endian `f32` through a bounded block buffer."
);
define_write_le!(
    write_le_u32,
    u32,
    "Write `values` as little-endian `u32` through a bounded block buffer."
);

/// The spatial grid a decoded volume set shares.
///
/// Distinct from [`ritk_spatial::VolumeDims`] because it also carries the
/// physical origin, spacing, and direction that a series places in exactly one
/// header.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VolumeGrid {
    /// `[nz, ny, nx]`, RITK axis order.
    pub dims: [usize; 3],
    /// Physical origin in world coordinates.
    pub origin: Point<3>,
    /// Voxel spacing `[dz, dy, dx]`.
    pub spacing: Spacing<3>,
    /// Direction cosines, columns in `(z, y, x)` order.
    pub direction: Direction<3>,
}

impl VolumeGrid {
    /// Read the grid of a 3-D image.
    pub fn of<B: ComputeBackend>(image: &Image<f32, B, 3>) -> Self {
        Self {
            dims: image.shape(),
            origin: *image.origin(),
            spacing: *image.spacing(),
            direction: *image.direction(),
        }
    }
}

/// Validate that every image in `volumes` shares one spatial grid.
///
/// Returns the grid of volume 0. An empty slice, or any volume whose shape,
/// origin, spacing, or direction differs from volume 0, is rejected: a series
/// container holds exactly one geometry, so a caller that assembled volumes
/// from different images must fail here rather than write a file whose grid
/// silently applies to only some of its content.
///
/// `context` names the caller (e.g. `"write_nifti_series"`) so the message
/// points at the format entry point.
pub fn ensure_single_grid<B: ComputeBackend>(
    context: &str,
    volumes: &[Image<f32, B, 3>],
) -> Result<VolumeGrid> {
    let Some((first, rest)) = volumes.split_first() else {
        bail!("{context}: a series requires at least one volume");
    };

    let grid = VolumeGrid::of(first);
    for (index, volume) in rest.iter().enumerate() {
        let position = index + 1;
        if volume.shape() != grid.dims {
            bail!(
                "{context}: volume {position} shape {:?} differs from volume 0 {:?}; \
                 a series has one spatial grid",
                volume.shape(),
                grid.dims
            );
        }
        if *volume.origin() != grid.origin || *volume.spacing() != grid.spacing {
            bail!(
                "{context}: volume {position} origin or spacing differs from volume 0; \
                 a series has one spatial grid"
            );
        }
        if *volume.direction() != grid.direction {
            bail!(
                "{context}: volume {position} direction differs from volume 0; \
                 a series has one spatial grid"
            );
        }
    }
    Ok(grid)
}

/// The rejection every single-volume reader raises when the container holds a
/// series.
///
/// One producer so the near-verbatim per-format messages cannot drift as the
/// readers are edited. The format supplies its own container name and frame
/// noun, so the diagnostic keeps the format's vocabulary.
pub fn reject_series(container: &str, unit: &str, count: usize) -> anyhow::Error {
    anyhow::anyhow!(
        "{container} declares {count} {unit}; this reader returns one 3-D volume. \
         Use the series reader to decode the acquisition series without discarding \
         {} of its {unit}.",
        count.saturating_sub(1)
    )
}

/// A decoded acquisition series: one flat `[Z, Y, X]` volume per frame plus the
/// grid they share.
///
/// `G` is whatever grid descriptor the owning reader needs (RITK spatial
/// metadata, or that plus a format-specific key/value field). The
/// single-volume-versus-series policy lives here so each reader does not
/// restate it.
#[derive(Debug, Clone, PartialEq)]
pub struct VolumeSet<G> {
    grid: G,
    volumes: Vec<Vec<f32>>,
}

impl<G> VolumeSet<G> {
    /// Assemble a decoded set from its shared grid and per-frame payloads.
    pub fn new(grid: G, volumes: Vec<Vec<f32>>) -> Self {
        Self { grid, volumes }
    }

    /// The shared grid descriptor.
    pub fn grid(&self) -> &G {
        &self.grid
    }

    /// Number of frames (volumes) in the set.
    pub fn volume_count(&self) -> usize {
        self.volumes.len()
    }

    /// Split into the shared grid and the per-frame payloads.
    pub fn into_parts(self) -> (G, Vec<Vec<f32>>) {
        (self.grid, self.volumes)
    }

    /// Take the sole volume of a one-frame set, rejecting a series.
    ///
    /// The single-volume readers carry a `[nz, ny, nx]` contract, so a series
    /// has no correct representation through them; returning frame 0 would
    /// discard the rest of the acquisition while reporting success.
    ///
    /// `container` names the file (e.g. `"NIfTI file"`) and `unit` is the frame
    /// noun (`"volumes"` or `"frames"`) so the message matches the format's own
    /// vocabulary.
    pub fn into_single_volume(self, container: &str, unit: &str) -> Result<(G, Vec<f32>)> {
        let count = self.volumes.len();
        if count != 1 {
            return Err(reject_series(container, unit, count));
        }
        let mut volumes = self.volumes;
        let volume = volumes
            .pop()
            .expect("invariant: length checked to be exactly one above");
        Ok((self.grid, volume))
    }
}

#[cfg(test)]
mod tests {
    #![expect(clippy::unwrap_used, reason = "ratchet RITK-UNWRAP-1")]

    use super::*;
    use coeus_core::SequentialBackend;

    fn image(
        dims: [usize; 3],
        origin: Point<3>,
        spacing: Spacing<3>,
    ) -> Image<f32, SequentialBackend, 3> {
        let voxels = dims[0] * dims[1] * dims[2];
        Image::from_flat_on(
            vec![0.0; voxels],
            dims,
            origin,
            spacing,
            Direction::identity(),
            &SequentialBackend,
        )
        .unwrap()
    }

    #[test]
    fn write_le_f32_matches_a_per_element_loop() {
        // A payload larger than one staging block exercises the block boundary.
        let values: Vec<f32> = (0..4096 * 2 + 7).map(|i| i as f32 * 0.25).collect();
        let mut expected = Vec::new();
        for value in &values {
            expected.extend_from_slice(&value.to_le_bytes());
        }

        let mut actual = Vec::new();
        write_le_f32(&mut actual, &values).unwrap();
        assert_eq!(actual, expected);
    }

    #[test]
    fn write_le_u32_matches_a_per_element_loop() {
        let values: Vec<u32> = (0..10_000).collect();
        let mut expected = Vec::new();
        for value in &values {
            expected.extend_from_slice(&value.to_le_bytes());
        }

        let mut actual = Vec::new();
        write_le_u32(&mut actual, &values).unwrap();
        assert_eq!(actual, expected);
    }

    #[test]
    fn ensure_single_grid_accepts_a_uniform_series() {
        let volumes = vec![
            image(
                [2, 3, 4],
                Point::new([0.0, 0.0, 0.0]),
                Spacing::new([1.0, 1.0, 1.0]),
            ),
            image(
                [2, 3, 4],
                Point::new([0.0, 0.0, 0.0]),
                Spacing::new([1.0, 1.0, 1.0]),
            ),
        ];
        let grid = ensure_single_grid("test", &volumes).unwrap();
        assert_eq!(grid.dims, [2, 3, 4]);
    }

    #[test]
    fn ensure_single_grid_rejects_a_mismatched_shape() {
        let volumes = vec![
            image(
                [2, 3, 4],
                Point::new([0.0, 0.0, 0.0]),
                Spacing::new([1.0, 1.0, 1.0]),
            ),
            image(
                [2, 3, 5],
                Point::new([0.0, 0.0, 0.0]),
                Spacing::new([1.0, 1.0, 1.0]),
            ),
        ];
        let err = ensure_single_grid("test", &volumes).unwrap_err();
        let message = err.to_string();
        assert!(
            message.contains("volume 1") && message.contains("shape"),
            "{message}"
        );
    }

    #[test]
    fn ensure_single_grid_rejects_a_differing_direction() {
        let first = image(
            [2, 3, 4],
            Point::new([0.0, 0.0, 0.0]),
            Spacing::new([1.0, 1.0, 1.0]),
        );
        let mut flipped = Direction::identity();
        flipped[(0, 0)] = -1.0;
        let second = Image::from_flat_on(
            vec![0.0; 24],
            [2, 3, 4],
            Point::new([0.0, 0.0, 0.0]),
            Spacing::new([1.0, 1.0, 1.0]),
            flipped,
            &SequentialBackend,
        )
        .unwrap();
        let err = ensure_single_grid("test", &[first, second]).unwrap_err();
        assert!(err.to_string().contains("direction"), "{err}");
    }

    #[test]
    fn ensure_single_grid_rejects_an_empty_series() {
        let empty: Vec<Image<f32, SequentialBackend, 3>> = Vec::new();
        let err = ensure_single_grid("write_x_series", &empty).unwrap_err();
        assert!(err.to_string().contains("at least one volume"), "{err}");
    }

    #[test]
    fn into_single_volume_names_the_declared_count() {
        let set = VolumeSet::new((), vec![vec![1.0], vec![2.0], vec![3.0]]);
        let err = set.into_single_volume("NIfTI file", "volumes").unwrap_err();
        assert!(err.to_string().contains("3 volumes"), "{err}");
    }

    #[test]
    fn into_single_volume_unwraps_a_one_frame_set() {
        let set = VolumeSet::new("grid", vec![vec![7.0]]);
        let (grid, volume) = set.into_single_volume("NRRD file", "volumes").unwrap();
        assert_eq!(grid, "grid");
        assert_eq!(volume, vec![7.0]);
    }
}
