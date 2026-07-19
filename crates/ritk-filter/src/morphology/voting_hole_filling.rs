//! Voting binary hole-filling filter (`itk::VotingBinaryHoleFillingImageFilter`).
//!
//! # Mathematical Specification
//!
//! A specialisation of voting-binary that **only fills** background voxels and
//! never removes foreground. A background voxel `p` becomes foreground when the
//! number of foreground voxels in its `(2r+1)` neighbourhood reaches the
//! majority threshold:
//!
//! ```text
//! threshold = (W − 1) / 2 + majority_threshold,   W = Π_d (2 r_d + 1)
//! I_out(p) = fg   if I(p) = fg                                   (foreground survives)
//!          = fg   if I(p) = bg  AND  N_fg(p) ≥ threshold         (hole filled)
//!          = I(p) otherwise
//! ```
//!
//! # Boundary
//!
//! Replicate (clamp) — out-of-bounds neighbours take the edge voxel's value, and
//! the neighbourhood size `W` is the **full** `(2r+1)^D` regardless of image
//! extent. On a `z = 1` (2-D) volume the `z` neighbours clamp onto the single
//! plane, so each in-plane pixel is counted three times and `W = 27` for
//! `r = 1`. Pinned against `sitk.VotingBinaryHoleFilling`: a corner background
//! voxel with in-bounds foreground neighbours fills (clamped fg count 15 ≥ 14),
//! which a constant/zero boundary (count 9) would not.
//!
//! # ITK Parity
//!
//! Corresponds to `itk::VotingBinaryHoleFillingImageFilter`. Defaults
//! `Radius = 1`, `MajorityThreshold = 1`, `ForegroundValue = 1`,
//! `BackgroundValue = 0`. Unlike [`super::voting_binary::VotingBinaryImageFilter`]
//! (which uses a shrink window), this clamps the boundary to match ITK on 2-D
//! and bordering voxels.

use ritk_image::tensor::Backend;
use ritk_image::Image;
use ritk_tensor_ops::{extract_vec_infallible, rebuild};

/// Voting binary hole-filling filter (ITK `VotingBinaryHoleFillingImageFilter`).
#[derive(Debug, Clone, Copy)]
pub struct VotingBinaryHoleFillingImageFilter {
    /// Per-axis neighbourhood radii `[rz, ry, rx]`. ITK default `[1, 1, 1]`.
    pub radius: [usize; 3],
    /// Votes above the half-window majority required to fill. ITK default `1`.
    pub majority_threshold: usize,
    /// Foreground intensity. ITK default `1.0`.
    pub foreground_value: f32,
    /// Background intensity. ITK default `0.0`.
    pub background_value: f32,
}

impl VotingBinaryHoleFillingImageFilter {
    /// Construct with explicit parameters.
    pub fn new(
        radius: [usize; 3],
        majority_threshold: usize,
        foreground_value: f32,
        background_value: f32,
    ) -> Self {
        Self {
            radius,
            majority_threshold,
            foreground_value,
            background_value,
        }
    }

    /// Apply the hole-filling pass to a 3-D image.
    pub fn apply<B: Backend>(&self, image: &Image<f32, B, 3>) -> Image<f32, B, 3> {
        let (vals, dims) = extract_vec_infallible(image);
        let out = self.hole_fill_flat(&vals, dims);
        rebuild(out, dims, image)
    }

    /// Coeus-native sister of [`VotingBinaryHoleFillingImageFilter::apply`].
    ///
    /// Runs the identical single-pass majority-vote hole fill via the shared
    /// `hole_fill_flat` host core on the image's
    /// contiguous host buffer, so the result is bitwise-identical to the Burn
    /// path. No Burn tensor is constructed. Spatial metadata is preserved.
    ///
    /// # Errors
    /// Returns an error when the image tensor is not host-addressable/contiguous
    /// or the rebuilt image fails shape validation.
    pub fn apply_native<B>(
        &self,
        image: &ritk_image::Image<f32, B, 3>,
        backend: &B,
    ) -> anyhow::Result<ritk_image::Image<f32, B, 3>>
    where
        B: coeus_core::ComputeBackend,
        B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    {
        crate::native_support::map_flat_image(image, backend, |vals, dims| {
            self.hole_fill_flat(vals, dims)
        })
    }

    /// Substrate-agnostic host core: one majority-vote hole-filling pass
    /// (replicate boundary, full `(2r+1)^D` window) on a flat z-major buffer.
    /// Single source of truth for the Burn [`apply`](Self::apply) and
    /// Coeus-native [`apply_native`](Self::apply_native) paths.
    fn hole_fill_flat(&self, vals: &[f32], dims: [usize; 3]) -> Vec<f32> {
        let [nz, ny, nx] = dims;
        let [rz, ry, rx] = self.radius;
        let fg = self.foreground_value;
        let window = (2 * rz + 1) * (2 * ry + 1) * (2 * rx + 1);
        let threshold = (window - 1) / 2 + self.majority_threshold;

        let (rz, ry, rx) = (rz as isize, ry as isize, rx as isize);
        let (snz, sny, snx) = (nz as isize, ny as isize, nx as isize);
        let slab = ny * nx;
        let bg = self.background_value;
        // PERF-378-01: parallelise over flat voxel index — clamp-boundary window read
        // is read-only from vals; no inter-voxel write dependency; bit-identical to serial.
        moirai::map_collect_index_with::<moirai::Adaptive, _, _>(vals.len(), |flat| {
            if vals[flat] == fg {
                return fg;
            }
            let iz = flat / slab;
            let rem = flat - iz * slab;
            let iy = rem / nx;
            let ix = rem - iy * nx;

            let mut count = 0usize;
            for dz in -rz..=rz {
                let zz = (iz as isize + dz).clamp(0, snz - 1) as usize;
                for dy in -ry..=ry {
                    let yy = (iy as isize + dy).clamp(0, sny - 1) as usize;
                    let base = (zz * ny + yy) * nx;
                    for dx in -rx..=rx {
                        let xx = (ix as isize + dx).clamp(0, snx - 1) as usize;
                        if vals[base + xx] == fg {
                            count += 1;
                        }
                    }
                }
            }
            if count >= threshold {
                fg
            } else {
                bg
            }
        })
    }

    /// Apply the hole-filling pass repeatedly, up to `max_iterations` times,
    /// stopping early when an iteration changes no voxel (ITK
    /// `VotingBinaryIterativeHoleFillingImageFilter`). `max_iterations = 0`
    /// returns the input unchanged.
    pub fn apply_iterative<B: Backend>(
        &self,
        image: &Image<f32, B, 3>,
        max_iterations: usize,
    ) -> Image<f32, B, 3>
    where
        B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    {
        if max_iterations == 0 {
            let (vals, dims) = extract_vec_infallible(image);
            return rebuild(vals, dims, image);
        }
        let mut current = self.apply(image);
        let mut prev = extract_vec_infallible(&current).0;
        for _ in 1..max_iterations {
            let next = self.apply(&current);
            let next_vals = extract_vec_infallible(&next).0;
            let changed = next_vals != prev;
            current = next;
            if !changed {
                break;
            }
            prev = next_vals;
        }
        current
    }

    /// Coeus-native sister of [`VotingBinaryHoleFillingImageFilter::apply_iterative`].
    ///
    /// Iterates the shared `hole_fill_flat` host core on
    /// a flat host buffer up to `max_iterations` times, stopping early when a
    /// pass changes no voxel, so the result is bitwise-identical to the Burn
    /// iterative path. No Burn tensor is constructed. Spatial metadata is
    /// preserved.
    ///
    /// # Errors
    /// Returns an error when the image tensor is not host-addressable/contiguous
    /// or the rebuilt image fails shape validation.
    pub fn apply_iterative_native<B>(
        &self,
        image: &ritk_image::Image<f32, B, 3>,
        max_iterations: usize,
        backend: &B,
    ) -> anyhow::Result<ritk_image::Image<f32, B, 3>>
    where
        B: coeus_core::ComputeBackend,
        B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    {
        let (mut current, dims) = ritk_tensor_ops::native::extract_image_vec(image)?;
        if max_iterations > 0 {
            let mut prev = self.hole_fill_flat(&current, dims);
            current = prev.clone();
            for _ in 1..max_iterations {
                let next = self.hole_fill_flat(&current, dims);
                let changed = next != prev;
                current = next.clone();
                if !changed {
                    break;
                }
                prev = next;
            }
        }
        ritk_tensor_ops::native::rebuild_image(current, dims, image, backend)
    }
}

#[cfg(test)]
#[path = "tests_voting_hole_filling.rs"]
mod tests_voting_hole_filling;

#[cfg(test)]
mod tests_native {
    use super::VotingBinaryHoleFillingImageFilter;
    use crate::native_support::{make_native_image, native_vals};
    use coeus_core::SequentialBackend;
    use ritk_image::test_support as ts;

    fn filter() -> VotingBinaryHoleFillingImageFilter {
        VotingBinaryHoleFillingImageFilter::new([1, 1, 1], 1, 1.0, 0.0)
    }

    /// Differential vs Burn: both `apply` paths share `hole_fill_flat`, so the
    /// single-pass and iterative outputs must be bitwise-identical. Uses a binary
    /// volume with an interior pit (background voxel enclosed by foreground).
    #[test]
    fn matches_coeus() {
        let dims = [3, 3, 3];
        let mut vals = vec![1.0f32; 27];
        vals[13] = 0.0; // center pit
        vals[0] = 0.0; // corner background (should stay bg)

        let burn_img = ts::make_image::<f32, coeus_core::SequentialBackend, 3>(vals.clone(), dims);
        let burn_out = filter().apply(&burn_img);
        let burn_vals = burn_out.data().to_vec();

        let native_out = filter()
            .apply_native(&make_native_image(vals.clone(), dims), &SequentialBackend)
            .expect("native hole fill");
        assert_eq!(native_vals(&native_out), burn_vals);

        // Iterative parity (2 passes).
        let burn_it = filter().apply_iterative(&burn_img, 2);
        let burn_it_vals = burn_it.data().to_vec();
        let native_it = filter()
            .apply_iterative_native(&make_native_image(vals, dims), 2, &SequentialBackend)
            .expect("native iterative hole fill");
        assert_eq!(native_vals(&native_it), burn_it_vals);
    }

    /// Oracle: an interior background pit fully enclosed by foreground is raised
    /// to foreground (26 fg neighbours ≥ threshold 14), and a pre-existing
    /// foreground voxel is never cleared (hole-filling only adds foreground).
    #[test]
    fn oracle_pit_is_raised() {
        let mut vals = vec![1.0f32; 27];
        vals[13] = 0.0; // center pit → should fill
        let out = filter()
            .apply_native(&make_native_image(vals, [3, 3, 3]), &SequentialBackend)
            .expect("native hole fill");
        let out = native_vals(&out);
        assert_eq!(out[13], 1.0, "enclosed pit must be filled");
        assert_eq!(out[1], 1.0, "existing foreground must survive");
    }

    /// Oracle: an all-background volume gains no foreground — every voxel has a
    /// zero foreground count, far below the majority threshold.
    #[test]
    fn oracle_empty_stays_empty() {
        let out = filter()
            .apply_native(
                &make_native_image(vec![0.0f32; 27], [3, 3, 3]),
                &SequentialBackend,
            )
            .expect("native hole fill");
        for &v in &native_vals(&out) {
            assert_eq!(v, 0.0, "all-background volume must stay background");
        }
    }
}
