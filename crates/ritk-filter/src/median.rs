//! Sliding-window median filter for 3-D volumes.
//!
//! # Algorithm
//! For each voxel `(iz, iy, ix)`, collect all values in the axis-aligned cube
//! `[iz ± r, iy ± r, ix ± r]` using replicate (clamp) boundary conditions,
//! sort them, and take the middle element (lower median for even-length
//! neighbourhoods, consistent with standard medical-imaging toolkits).
//!
//! # Complexity
//! O(n * (2r+1)^3) average where n is the total voxel count and r is
//! the neighbourhood half-width, using introselect (select_nth_unstable_by)
//! rather than a full sort.  Parallelised over z-slices via Rayon.

use ritk_core::image::Image;
use ritk_image::tensor::Backend;
use ritk_tensor_ops::{extract_vec, rebuild};

/// Sliding-window median filter for 3-D volumes.
///
/// Replaces each voxel with the median of its `(2r+1)³` axis-aligned
/// neighbourhood. Out-of-bounds positions use replicate (clamp) padding.
pub struct MedianFilter {
    /// Neighbourhood half-width in voxels (default 1 → 3×3×3 cube).
    pub radius: usize,
}

impl MedianFilter {
    /// Create a new median filter with the given neighbourhood half-width.
    ///
    /// A radius of 0 yields identity (each voxel is its own sole neighbour).
    /// A radius of 1 produces a 3×3×3 kernel (27 samples per voxel).
    pub fn new(radius: usize) -> Self {
        Self { radius }
    }

    /// Apply the median filter to a 3-D image.
    ///
    /// Returns a new image with identical shape and spatial metadata (origin,
    /// spacing, direction). The tensor device of the output matches the input.
    ///
    /// # Errors
    /// Returns `Err` if the underlying tensor data cannot be extracted as `f32`.
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> anyhow::Result<Image<B, 3>> {
        let (vals, shape) = extract_vec(image)?;

        let filtered = median_3d(&vals, shape, self.radius);

        Ok(rebuild(filtered, shape, image))
    }

    /// Coeus-native sister of [`MedianFilter::apply`].
    ///
    /// Runs the identical sliding-window lower-median (replicate boundary) via
    /// the shared `median_3d` host core on the image's contiguous host buffer,
    /// so the result is bitwise-identical to the Burn path. No Burn tensor is
    /// constructed. Spatial metadata is preserved.
    ///
    /// # Errors
    /// Returns an error when the image tensor is not host-addressable/contiguous
    /// or the rebuilt tensor fails shape validation.
    pub fn apply_native<B>(
        &self,
        image: &ritk_image::native::Image<f32, B, 3>,
    ) -> anyhow::Result<ritk_image::native::Image<f32, B, 3>>
    where
        B: coeus_core::ComputeBackend + Default,
        B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    {
        let (vals, dims) = ritk_tensor_ops::native::extract_image_vec(image)?;
        let filtered = median_3d(&vals, dims, self.radius);
        ritk_tensor_ops::native::rebuild_image(filtered, dims, image, &B::default()::default())
    }
}

/// Sliding-window median on a 3-D volume stored in flat Z×Y×X order.
///
/// # Arguments
/// * `data`   — flat voxel values in row-major (Z-major) order.
/// * `dims`   — `[nz, ny, nx]`.
/// * `radius` — neighbourhood half-width in voxels.
///
/// # Boundary handling
/// Replicate (clamp) padding: out-of-bounds indices are clamped to the
/// nearest valid index along each axis.
fn median_3d(data: &[f32], dims: [usize; 3], radius: usize) -> Vec<f32> {
    let (nz, ny, nx) = (dims[0], dims[1], dims[2]);
    let r = radius as isize;
    let cap = (2 * radius + 1).pow(3);
    let nz_isize = nz as isize;
    let ny_isize = ny as isize;
    let nx_isize = nx as isize;
    let stride_yx = ny * nx;
    let mut output = vec![0.0_f32; nz * ny * nx];

    moirai::for_each_chunk_mut_enumerated_with::<moirai::Adaptive, _, _>(
        &mut output,
        ny * nx,
        |iz, out_slice| {
            // Allocate once per z-slice (per Rayon thread); reused across all
            // voxels in the slice via clear() to avoid per-voxel heap allocation.
            let mut neighbors: Vec<f32> = Vec::with_capacity(cap);

            // Pre-clamp the Z-plane once per voxel-row: each `dz` maps to a
            // single `zz` regardless of `iy, ix`. Hoisting removes a
            // `(2r+1)²`-fold redundant branch per voxel (PERF-377-01 partial).
            const BUF_CAP: usize = 64;
            // The clamp-buffer holds `2 * radius + 1` clamped indices per axis.
            // Capacity 64 supports radii up to 31 (test-only envelope — production
            // radii are ≪ 8). A larger radius panics here to keep the hot path
            // stack-allocated.
            assert!(
                2 * radius < BUF_CAP,
                "MedianFilter::median_3d: radius {radius} exceeds buffer cap (max 31)"
            );
            let mut zz_buf: [usize; BUF_CAP] = [0; BUF_CAP];
            #[allow(clippy::needless_range_loop)]
            for dz in 0..=(2 * radius) {
                let z_raw = (iz as isize + dz as isize - r).clamp(0, nz_isize - 1);
                zz_buf[dz] = z_raw as usize;
            }

            if radius == 0 {
                // Single sample per voxel — neighborhood is just the voxel.
                for (iy, out_row) in out_slice.chunks_exact_mut(nx).enumerate() {
                    for (ix, cell) in out_row.iter_mut().enumerate() {
                        let idx = iz * stride_yx + iy * nx + ix;
                        *cell = data[idx];
                    }
                }
                return;
            }

            for (iy, out_row) in out_slice.chunks_exact_mut(nx).enumerate() {
                // Pre-clamp the Y-row once per iy: `(2r+1)²`-fold redundant
                // branch elimination when paired with the dz-hoist above.
                let mut yy_buf: [usize; BUF_CAP] = [0; BUF_CAP];
                #[allow(clippy::needless_range_loop)]
                for dy in 0..=(2 * radius) {
                    let y_raw = (iy as isize + dy as isize - r).clamp(0, ny_isize - 1);
                    yy_buf[dy] = y_raw as usize;
                }

                for (ix, cell) in out_row.iter_mut().enumerate() {
                    neighbors.clear();

                    // Collect (2r+1)^3 neighbourhood with replicate (clamp)
                    // padding. The per-axis clamp results are pre-baked in
                    // `zz_buf` / `yy_buf`; only the X-axis clamp is computed
                    // per inner-tick (it depends on `ix`).
                    //
                    // The triple-nested `dz`, `dy`, `dx` loops use indices
                    // to drive the geometry of the cubic neighbourhood;
                    // the iterator-with-enumerate transformation adds a
                    // bounds-check on every tick without paying for the
                    // inner clamp hoist. Single block-level allow per the
                    // precedent set by morphology::window_1d.
                    #[allow(clippy::needless_range_loop)]
                    for dz in 0..=(2 * radius) {
                        let zz_base = zz_buf[dz] * stride_yx;
                        #[allow(clippy::needless_range_loop)]
                        for dy in 0..=(2 * radius) {
                            let yy_base = yy_buf[dy] * nx;
                            let base = zz_base + yy_base;
                            for dx in 0..=(2 * radius) {
                                let xx =
                                    (ix as isize + dx as isize - r).clamp(0, nx_isize - 1) as usize;
                                neighbors.push(data[base + xx]);
                            }
                        }
                    }

                    // Lower median for even-length neighbourhoods.
                    // select_nth_unstable_by is O(N) average (introselect) versus
                    // O(N log N) for a full sort; the value at neighbors[mid] after
                    // the call is identical to sort-then-index.
                    let mid = neighbors.len() / 2;
                    neighbors.select_nth_unstable_by(mid, |a, b| {
                        a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                    });
                    *cell = neighbors[mid];
                }
            }
        },
    );

    output
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "tests_median_native.rs"]
mod tests_native;

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;
    use coeus_core::SequentialBackend;
    use ritk_core::image::Image;
    use ritk_image::native::Image as NativeImage;
    use ritk_image::tensor::{Shape, Tensor, TensorData};
    use ritk_spatial::{Direction, Point, Spacing};

    type B = NdArray<f32>;

    /// Construct a test image from flat values, shape, and optional metadata.
    fn make_image(
        vals: Vec<f32>,
        dims: [usize; 3],
        origin: [f64; 3],
        spacing: [f64; 3],
    ) -> Image<B, 3> {
        let device = Default::default();
        let td = TensorData::new(vals, Shape::new(dims));
        let tensor = Tensor::<B, 3>::from_data(td, &device);
        Image::new(
            tensor,
            Point::new(origin),
            Spacing::new(spacing),
            Direction::identity(),
        )
    }

    /// Extract voxel data as `Vec<f32>` from an image.
    fn extract_vals(img: &Image<B, 3>) -> Vec<f32> {
        img.data_slice().into_owned()
    }

    // ── Test 1: Uniform image is unchanged ────────────────────────────────

    /// A constant image must be invariant under median filtering for any
    /// radius, because the median of identical values equals that value.
    ///
    /// **Proof sketch**: Let I(p) = c for all p. For any neighbourhood N(p),
    /// every element of the sorted list is c, so median = c. ∎
    #[test]
    fn test_uniform_image_unchanged() {
        let dims = [8, 8, 8];
        let val = 42.0_f32;
        let vals = vec![val; dims[0] * dims[1] * dims[2]];
        let img = make_image(vals, dims, [0.0; 3], [1.0; 3]);

        let filter = MedianFilter::new(2);
        let out = filter.apply(&img).unwrap();
        let result = extract_vals(&out);

        assert_eq!(out.shape(), dims);
        for (i, &v) in result.iter().enumerate() {
            assert!((v - val).abs() < 1e-6, "voxel {i}: expected {val}, got {v}");
        }
    }

    #[test]
    fn native_median_removes_an_impulse_and_preserves_metadata() {
        let backend = SequentialBackend;
        let source = NativeImage::from_flat_on(
            vec![0.0, 10.0, 0.0],
            [1, 1, 3],
            Point::new([2.0, 3.0, 4.0]),
            Spacing::new([0.5, 1.0, 2.0]),
            Direction::identity(),
            &backend,
        )
        .unwrap();
        let output = MedianFilter::new(1).apply_native(&source).unwrap();

        assert_eq!(output.data_slice().unwrap(), &[0.0, 0.0, 0.0]);
        assert_eq!(output.shape(), source.shape());
        assert_eq!(output.origin(), source.origin());
        assert_eq!(output.spacing(), source.spacing());
        assert_eq!(output.direction(), source.direction());
    }

    // ── Test 2: Impulse noise removed ─────────────────────────────────────

    /// A single spike voxel (salt-noise) embedded in a constant field must be
    /// eliminated by median filtering with radius ≥ 1, because the spike
    /// constitutes at most 1 out of (2r+1)³ ≥ 27 samples and therefore
    /// cannot be the median of the sorted neighbourhood.
    ///
    /// **Proof**: In a 3×3×3 neighbourhood around the spike, 26 values equal
    /// the background `c` and 1 equals `c + spike`. Sorted, the 14th element
    /// (index 13) is `c`. ∎
    #[test]
    fn test_impulse_noise_removed() {
        let dims = [8, 8, 8];
        let bg = 10.0_f32;
        let spike = 1000.0_f32;
        let n = dims[0] * dims[1] * dims[2];
        let mut vals = vec![bg; n];

        // Place spike at centre voxel (4, 4, 4).
        let spike_idx = 4 * dims[1] * dims[2] + 4 * dims[2] + 4;
        vals[spike_idx] = spike;

        let img = make_image(vals, dims, [0.0; 3], [1.0; 3]);
        let filter = MedianFilter::new(1);
        let out = filter.apply(&img).unwrap();
        let result = extract_vals(&out);

        // The spike location must now hold the background value.
        assert!(
            (result[spike_idx] - bg).abs() < 1e-6,
            "spike not removed: expected {bg}, got {}",
            result[spike_idx]
        );

        // All other voxels in the interior (away from boundaries) must remain
        // at the background value since their entire neighbourhood is constant.
        for iz in 1..dims[0] - 1 {
            for iy in 1..dims[1] - 1 {
                for ix in 1..dims[2] - 1 {
                    let idx = iz * dims[1] * dims[2] + iy * dims[2] + ix;
                    if idx == spike_idx {
                        continue;
                    }
                    // Voxels adjacent to the spike see at most 1 non-bg value
                    // out of 27, so their median is still bg.
                    assert!(
                        (result[idx] - bg).abs() < 1e-6,
                        "interior voxel ({iz},{iy},{ix}): expected {bg}, got {}",
                        result[idx]
                    );
                }
            }
        }
    }

    // ── Test 3: Metadata preserved ────────────────────────────────────────

    /// Origin, spacing, and direction of the output image must be identical
    /// to the input. The median filter operates exclusively on voxel
    /// intensities and must not mutate spatial metadata.
    #[test]
    fn test_metadata_preserved() {
        let dims = [4, 4, 4];
        let origin = [10.0, -5.5, 2.71]; // arbitrary float coordinates
        let spacing = [0.5, 0.75, 1.25];
        let vals = vec![7.0_f32; dims[0] * dims[1] * dims[2]];
        let img = make_image(vals, dims, origin, spacing);

        let filter = MedianFilter::new(1);
        let out = filter.apply(&img).unwrap();

        // Shape.
        assert_eq!(out.shape(), dims);

        // Origin (exact equality; no computation on these values).
        let out_origin = out.origin();
        for d in 0..3 {
            assert!(
                (out_origin[d] - origin[d]).abs() < 1e-12,
                "origin[{d}]: expected {}, got {}",
                origin[d],
                out_origin[d]
            );
        }

        // Spacing.
        let out_spacing = out.spacing();
        for d in 0..3 {
            assert!(
                (out_spacing[d] - spacing[d]).abs() < 1e-12,
                "spacing[{d}]: expected {}, got {}",
                spacing[d],
                out_spacing[d]
            );
        }

        // Direction (identity → identity).
        let out_dir = out.direction();
        let in_dir = img.direction();
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (out_dir[(i, j)] - in_dir[(i, j)]).abs() < 1e-12,
                    "direction[{i},{j}] mismatch"
                );
            }
        }
    }

    // -- Test 4: Identity for radius zero ---------------------------------

    /// With radius = 0 the neighbourhood is a single voxel, so the output
    /// must be bit-identical to the input.
    ///
    /// **Proof**: |N(p)| = 1³ = 1, so median of {I(p)} = I(p). ∎
    #[test]
    fn test_identity_for_radius_zero() {
        let dims = [6, 6, 6];
        let n = dims[0] * dims[1] * dims[2];
        // Deterministic non-trivial pattern: voxel value = flat index.
        let vals: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let img = make_image(vals.clone(), dims, [0.0; 3], [1.0; 3]);

        let filter = MedianFilter::new(0);
        let out = filter.apply(&img).unwrap();
        let result = extract_vals(&out);

        assert_eq!(result.len(), vals.len());
        for (i, (&expected, &actual)) in vals.iter().zip(result.iter()).enumerate() {
            assert!(
                (actual - expected).abs() < 1e-6,
                "voxel {i}: expected {expected}, got {actual}"
            );
        }
    }

    // -- Test 5: Brute-force reference agreement -------------------------

    /// `median_3d` must produce the lower median over the multiset
    /// `{data[clamp(iz + dz), clamp(iy + dy), clamp(ix + dx)] : dz,
    /// dy, dx ∈ [-r, r]}` for every voxel. The brute-force reference below
    /// gathers the full multiset via clamp arithmetic and applies the
    /// same `select_nth_unstable_by(mid)` call; agreement is therefore
    /// bit-equal — `assert_eq!` is the correct (not merely bounded)
    /// oracle. The clamp arithmetic matches the production code: same
    /// `.clamp(0, n - 1)` idiom means identical multisets.
    ///
    /// This test guards against regressions introduced by the
    /// PERF-377-01 clamp-hoisting micro-optimisation (clamp indices are
    /// pre-baked into `zz_buf` / `yy_buf` instead of computed per inner
    /// tick). The hoisting does NOT change the values placed into
    /// `neighbors`; it merely re-orders computation.
    fn median_3d_brute_force(data: &[f32], dims: [usize; 3], radius: usize) -> Vec<f32> {
        let (nz, ny, nx) = (dims[0], dims[1], dims[2]);
        let r = radius as isize;
        let mut output = vec![0.0_f32; nz * ny * nx];

        for iz in 0..nz {
            for iy in 0..ny {
                for ix in 0..nx {
                    let mut nbrs: Vec<f32> = Vec::with_capacity((2 * radius + 1).pow(3));
                    for dz in -r..=r {
                        let zz = (iz as isize + dz).clamp(0, nz as isize - 1) as usize;
                        for dy in -r..=r {
                            let yy = (iy as isize + dy).clamp(0, ny as isize - 1) as usize;
                            for dx in -r..=r {
                                let xx = (ix as isize + dx).clamp(0, nx as isize - 1) as usize;
                                nbrs.push(data[zz * ny * nx + yy * nx + xx]);
                            }
                        }
                    }
                    let mid = nbrs.len() / 2;
                    nbrs.select_nth_unstable_by(mid, |a, b| {
                        a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                    });
                    output[iz * ny * nx + iy * nx + ix] = nbrs[mid];
                }
            }
        }
        output
    }

    #[test]
    fn test_median_3d_matches_brute_force_reference_r1() {
        // 12×12×12 = 1728 voxels; small but non-trivial boundary handling
        // (r=1 windows always touch the clamp boundary on a 12-D axis).
        let dims = [12, 12, 12];
        let n: usize = dims.iter().product();
        let vals: Vec<f32> = (0..n).map(|i| ((i * 31) % 97) as f32 * 0.13).collect();

        let r1 = median_3d_brute_force(&vals, dims, 1);
        let r2 = super::median_3d(&vals, dims, 1);
        assert_eq!(r1.len(), dims.iter().product::<usize>());
        assert_eq!(r2.len(), dims.iter().product::<usize>());
        for (i, (&a, &B::default())) in r1.iter().zip(r2.iter()).enumerate() {
            // `select_nth_unstable_by` is deterministic for a given input
            // and partition scheme, so values MUST match exactly.
            assert!(
                a.to_bits() == b.to_bits(),
                "voxel {i} mismatch: brute={a} (bits={:08x}) hoisted={b} (bits={:08x})",
                a.to_bits(),
                b.to_bits()
            );
        }
    }

    #[test]
    fn test_median_3d_matches_brute_force_reference_r3() {
        // r=3 exercises the larger (2r+1) = 7 cube (343 samples) and
        // surfaces any off-by-one in the clamp hoist.
        let dims = [10, 10, 10];
        let n: usize = dims.iter().product();
        let vals: Vec<f32> = (0..n).map(|i| ((i * 13 + 7) % 53) as f32 - 26.0).collect();

        let r1 = median_3d_brute_force(&vals, dims, 3);
        let r2 = super::median_3d(&vals, dims, 3);
        assert_eq!(r1.len(), n);
        assert_eq!(r2.len(), n);
        for (i, (&a, &B::default())) in r1.iter().zip(r2.iter()).enumerate() {
            assert!(
                a.to_bits() == b.to_bits(),
                "voxel {i} mismatch (r=3): brute={a} hoisted={b}"
            );
        }
    }
}
