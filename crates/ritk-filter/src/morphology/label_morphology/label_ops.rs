//! Label dilation, erosion, opening, and closing for 3-D label volumes.
//!
//! # Mathematical Specification
//!
//! Given a label volume L: [nz x ny x nx] -> N (0=background, k>0 label k),
//! label dilation with radius r expands each labeled region outward:
//!
//! For each background voxel x:
//!   N_r(x) = { y : max|y_i - x_i| <= r }  (cubic neighbourhood)
//!   labels(x) = { L(y) : y in N_r(x), L(y) > 0 }
//!   L_out(x) = min(labels(x))  if labels(x) non-empty, else 0
//!
//! Existing labelled voxels are preserved: L_out(x) = L(x) for L(x) > 0.
//! Conflict resolution: where two label regions meet, the minimum label ID wins.
//!
//! # References
//! - Serra, J. (1982). Image Analysis and Mathematical Morphology. Academic Press.

use coeus_core::{ComputeBackend, CpuAddressableStorage};
use ritk_image::tensor::Backend;
use ritk_image::Image;
use ritk_tensor_ops::{extract_vec, rebuild};

// ════════════════════════════════════════════════════════════════════════════
// LabelDilation
// ════════════════════════════════════════════════════════════════════════════

/// Label dilation for 3-D label volumes.
///
/// Expands each labeled region by radius voxels. For conflicting dilations
/// (two label regions growing into the same background voxel), the label
/// with the minimum ID takes priority.
#[derive(Debug, Clone)]
pub struct LabelDilation {
    pub radius: usize,
}

impl LabelDilation {
    pub fn new(radius: usize) -> Self {
        Self { radius }
    }
    pub fn with_radius(mut self, r: usize) -> Self {
        self.radius = r;
        self
    }

    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> anyhow::Result<Image<B, 3>> {
        let (vals, dims) = extract_vec(image)?;
        let result = dilate_labels(&vals, dims, self.radius);
        Ok(rebuild(result, dims, image))
    }

    /// Apply label dilation to a Coeus-native image.
    pub fn apply_native<B>(
        &self,
        image: &ritk_image::native::Image<f32, B, 3>,
        backend: &B,
    ) -> anyhow::Result<ritk_image::native::Image<f32, B, 3>>
    where
        B: ComputeBackend,
        B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
    {
        ritk_image::native::Image::from_flat_on(
            dilate_labels(image.data_slice()?, image.shape(), self.radius),
            image.shape(),
            *image.origin(),
            *image.spacing(),
            *image.direction(),
            backend,
        )
    }
}

fn dilate_labels(data: &[f32], dims: [usize; 3], radius: usize) -> Vec<f32> {
    let [nz, ny, nx] = dims;
    let r = radius as isize;
    let nz_isize = nz as isize;
    let ny_isize = ny as isize;
    let nx_isize = nx as isize;
    let mut out = data.to_vec();

    // Parallelise over z-slices: each closure call owns a disjoint `ny*nx`
    // chunk of `out`; `data` is captured read-only (Sync) and is safe to
    // share across threads.
    moirai::for_each_chunk_mut_enumerated_with::<moirai::Adaptive, _, _>(
        &mut out,
        ny * nx,
        |iz, out_slice| {
            // Stack-allocated clamp buffers — one clamped z-index per dz
            // offset, computed once per z-slice. Eliminates a (2r+1)²-fold
            // redundant clamp per voxel (matching the pattern in median_3d).
            const BUF_CAP: usize = 64;
            assert!(
                2 * radius < BUF_CAP,
                "LabelDilation: radius {radius} exceeds buffer cap (max 31)"
            );
            let mut zz_buf = [0usize; BUF_CAP];
            #[allow(clippy::needless_range_loop)]
            for dz in 0..=(2 * radius) {
                let z_raw = (iz as isize + dz as isize - r).clamp(0, nz_isize - 1);
                zz_buf[dz] = z_raw as usize;
            }

            for (iy, out_row) in out_slice.chunks_exact_mut(nx).enumerate() {
                // Clamp buffer for the Y axis, hoisted once per y-row.
                let mut yy_buf = [0usize; BUF_CAP];
                #[allow(clippy::needless_range_loop)]
                for dy in 0..=(2 * radius) {
                    let y_raw = (iy as isize + dy as isize - r).clamp(0, ny_isize - 1);
                    yy_buf[dy] = y_raw as usize;
                }

                for (ix, cell) in out_row.iter_mut().enumerate() {
                    // Only expand into background voxels.
                    if data[iz * ny * nx + iy * nx + ix] > 0.5 {
                        continue;
                    }
                    let mut min_label: f32 = 0.0;
                    // Clamped-coordinate inner scan: out-of-bounds offsets
                    // repeat the nearest border voxel, which is always
                    // background (since the voxel being processed is
                    // background), so duplicates never change min_label.
                    // Result is bit-identical to the original skip-OOB scan.
                    #[allow(clippy::needless_range_loop)]
                    for dz in 0..=(2 * radius) {
                        let zz_base = zz_buf[dz] * ny * nx;
                        #[allow(clippy::needless_range_loop)]
                        for dy in 0..=(2 * radius) {
                            let base = zz_base + yy_buf[dy] * nx;
                            for dx in 0..=(2 * radius) {
                                let xx =
                                    (ix as isize + dx as isize - r).clamp(0, nx_isize - 1) as usize;
                                let v = data[base + xx];
                                if v > 0.5 && (min_label < 0.5 || v < min_label) {
                                    min_label = v;
                                }
                            }
                        }
                    }
                    *cell = min_label;
                }
            }
        },
    );

    out
}

// ════════════════════════════════════════════════════════════════════════════
// LabelErosion
// ════════════════════════════════════════════════════════════════════════════

/// Label erosion for 3-D label volumes.
///
/// # Mathematical Specification
///
/// For each voxel x in a label volume L: [nz x ny x nx] -> N (0=background):
/// - If L(x) = 0: L_out(x) = 0  (background remains background)
/// - If L(x) > 0: examine N_r(x) = { y : max_i|y_i - x_i| <= r }
///   - If any y in N_r(x) has L(y) = 0 -> L_out(x) = 0  (border voxel eroded)
///   - Else L_out(x) = L(x)  (interior voxel preserved)
///
/// # Properties
/// - Anti-extensivity: L_out(x) <= L(x) for all x.
/// - r=0 identity: N_0(x) = {x}, so only checks itself; no change.
/// - Dual to LabelDilation under label-complement.
///
/// # References
/// - Serra, J. (1982). *Image Analysis and Mathematical Morphology*. Academic Press.
#[derive(Debug, Clone)]
pub struct LabelErosion {
    pub radius: usize,
}

impl LabelErosion {
    pub fn new(radius: usize) -> Self {
        Self { radius }
    }
    pub fn with_radius(mut self, r: usize) -> Self {
        self.radius = r;
        self
    }

    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> anyhow::Result<Image<B, 3>> {
        let (vals, dims) = extract_vec(image)?;
        let result = erode_labels(&vals, dims, self.radius);
        Ok(rebuild(result, dims, image))
    }

    /// Apply label erosion to a Coeus-native image.
    pub fn apply_native<B>(
        &self,
        image: &ritk_image::native::Image<f32, B, 3>,
        backend: &B,
    ) -> anyhow::Result<ritk_image::native::Image<f32, B, 3>>
    where
        B: ComputeBackend,
        B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
    {
        ritk_image::native::Image::from_flat_on(
            erode_labels(image.data_slice()?, image.shape(), self.radius),
            image.shape(),
            *image.origin(),
            *image.spacing(),
            *image.direction(),
            backend,
        )
    }
}

fn erode_labels(data: &[f32], dims: [usize; 3], radius: usize) -> Vec<f32> {
    let [nz, ny, nx] = dims;
    let r = radius as isize;
    let nz_isize = nz as isize;
    let ny_isize = ny as isize;
    let nx_isize = nx as isize;
    // Initialise output from data so unmodified voxels are preserved.
    let mut out = data.to_vec();

    // Parallelise over z-slices. The inner neighborhood scan intentionally
    // preserves OOB-triggers-erosion semantics: a labeled voxel whose cubic
    // neighborhood extends outside the volume is always eroded (border
    // erosion). Clamping would suppress that behavior, so bounds-checking
    // is kept in the inner loop to guarantee bit-identical output.
    moirai::for_each_chunk_mut_enumerated_with::<moirai::Adaptive, _, _>(
        &mut out,
        ny * nx,
        |iz, out_slice| {
            for (iy, out_row) in out_slice.chunks_exact_mut(nx).enumerate() {
                for (ix, cell) in out_row.iter_mut().enumerate() {
                    if data[iz * ny * nx + iy * nx + ix] < 0.5 {
                        continue; // background stays background
                    }
                    let mut eroded = false;
                    'outer: for dz in -r..=r {
                        for dy in -r..=r {
                            for dx in -r..=r {
                                let (zz, yy, xx) =
                                    (iz as isize + dz, iy as isize + dy, ix as isize + dx);
                                if zz < 0
                                    || zz >= nz_isize
                                    || yy < 0
                                    || yy >= ny_isize
                                    || xx < 0
                                    || xx >= nx_isize
                                {
                                    eroded = true;
                                    break 'outer;
                                }
                                let v =
                                    data[zz as usize * ny * nx + yy as usize * nx + xx as usize];
                                if v < 0.5 {
                                    eroded = true;
                                    break 'outer;
                                }
                            }
                        }
                    }
                    if eroded {
                        *cell = 0.0;
                    }
                }
            }
        },
    );

    out
}

// ════════════════════════════════════════════════════════════════════════════
// LabelOpening
// ════════════════════════════════════════════════════════════════════════════

/// Morphological opening on label volumes: LabelDilation composed with LabelErosion.
///
/// Removes labeled regions smaller than radius r, smooths region boundaries.
/// Opening = Dilation(Erosion(L)).
#[derive(Debug, Clone)]
pub struct LabelOpening {
    pub radius: usize,
}

impl LabelOpening {
    pub fn new(radius: usize) -> Self {
        Self { radius }
    }

    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> anyhow::Result<Image<B, 3>> {
        let eroded = LabelErosion::new(self.radius).apply(image)?;
        LabelDilation::new(self.radius).apply(&eroded)
    }

    /// Apply label opening to a Coeus-native image.
    pub fn apply_native<B>(
        &self,
        image: &ritk_image::native::Image<f32, B, 3>,
        backend: &B,
    ) -> anyhow::Result<ritk_image::native::Image<f32, B, 3>>
    where
        B: ComputeBackend,
        B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
    {
        let eroded = LabelErosion::new(self.radius).apply_native(image, backend)?;
        LabelDilation::new(self.radius).apply_native(&eroded, backend)
    }
}

// ════════════════════════════════════════════════════════════════════════════
// LabelClosing
// ════════════════════════════════════════════════════════════════════════════

/// Morphological closing on label volumes: LabelErosion composed with LabelDilation.
///
/// Fills background holes smaller than radius r within labeled regions.
/// Closing = Erosion(Dilation(L)).
#[derive(Debug, Clone)]
pub struct LabelClosing {
    pub radius: usize,
}

impl LabelClosing {
    pub fn new(radius: usize) -> Self {
        Self { radius }
    }

    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> anyhow::Result<Image<B, 3>> {
        let dilated = LabelDilation::new(self.radius).apply(image)?;
        LabelErosion::new(self.radius).apply(&dilated)
    }

    /// Apply label closing to a Coeus-native image.
    pub fn apply_native<B>(
        &self,
        image: &ritk_image::native::Image<f32, B, 3>,
        backend: &B,
    ) -> anyhow::Result<ritk_image::native::Image<f32, B, 3>>
    where
        B: ComputeBackend,
        B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
    {
        let dilated = LabelDilation::new(self.radius).apply_native(image, backend)?;
        LabelErosion::new(self.radius).apply_native(&dilated, backend)
    }
}
