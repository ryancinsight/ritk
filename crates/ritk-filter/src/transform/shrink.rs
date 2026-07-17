//! Integer downsampling filters.
//!
//! Two distinct downsampling operations live here, matching two distinct ITK
//! filters:
//!
//! - [`ShrinkImageFilter`] — **subsampling** (ITK `ShrinkImageFilter` /
//!   `sitk.Shrink`): keeps one voxel per tile, no averaging, `floor(N/f)` output
//!   size. The kept voxel is offset `f/2` into each tile and the origin is
//!   shifted to that tile's centroid.
//! - [`TileMeanShrinkFilter`] — **tile averaging** (anti-aliased display
//!   downsample): the arithmetic mean of every voxel in each tile, `ceil(N/f)`
//!   output size (trailing partial tiles averaged). This is NOT ITK `Shrink`
//!   (which subsamples); for full-bin averaging with `floor(N/f)` size use
//!   `BinShrinkImageFilter` (ITK `BinShrink`).

use ritk_image::tensor::Backend;
use ritk_image::Image;
use ritk_spatial::Spacing;
use ritk_tensor_ops::{extract_vec_infallible, rebuild_with_metadata};

/// Integer **subsampling** filter — ITK `ShrinkImageFilter` / `sitk.Shrink`.
///
/// # Mathematical Specification
///
/// For input shape `[Nz, Ny, Nx]` and factors `[fz, fy, fx]` (all ≥ 1):
///
/// ```text
/// out_shape[d] = floor(N[d] / f[d])
/// offset[d]    = (N[d] mod f[d] + f[d]) / 2            (integer; centers samples)
/// out(iz,iy,ix) = I(iz·fz + offset_z, iy·fy + offset_y, ix·fx + offset_x)
/// ```
///
/// ITK centers the retained samples within the full extent, so the per-axis
/// offset depends on the trailing remainder `N mod f` (it equals `f/2` only when
/// the axis divides evenly). The output spacing is `in_spacing[d] · f[d]`, and
/// the origin is shifted by the continuous centroid offset
/// `(N mod f + f − 1)/2` voxels: `out_origin = in_origin + Direction · (in_spacing · shift)`.
///
/// No averaging is performed (cf. [`TileMeanShrinkFilter`] / `BinShrink`).
#[derive(Debug, Clone)]
pub struct ShrinkImageFilter {
    /// Subsampling factors per axis `\[fz, fy, fx\]`. All must be ≥ 1. Default \[1,1,1\].
    pub shrink_factors: [usize; 3],
}

impl ShrinkImageFilter {
    /// Construct with the given per-axis subsampling factors (0 is treated as 1).
    pub fn new(shrink_factors: [usize; 3]) -> Self {
        Self { shrink_factors }
    }
}

impl Default for ShrinkImageFilter {
    fn default() -> Self {
        Self::new([1, 1, 1])
    }
}

impl ShrinkImageFilter {
    /// Apply the subsampling shrink to a 3-D image.
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> anyhow::Result<Image<B, 3>> {
        let [nz, ny, nx] = image.shape();
        let [fz, fy, fx] = [
            self.shrink_factors[0].max(1),
            self.shrink_factors[1].max(1),
            self.shrink_factors[2].max(1),
        ];

        // Output shape: floor(N/f).
        let (oz, oy, ox) = (nz / fz, ny / fy, nx / fx);
        // ITK centers the retained samples: offset = (N mod f + f) / 2.
        let (offz, offy, offx) = ((nz % fz + fz) / 2, (ny % fy + fy) / 2, (nx % fx + fx) / 2);

        let (vals, _) = extract_vec_infallible(image);
        let mut out = vec![0.0f32; oz * oy * ox];
        for iz in 0..oz {
            let kz = iz * fz + offz;
            for iy in 0..oy {
                let ky = iy * fy + offy;
                for ix in 0..ox {
                    let kx = ix * fx + offx;
                    out[(iz * oy + iy) * ox + ix] = vals[(kz * ny + ky) * nx + kx];
                }
            }
        }

        let in_s = image.spacing();
        let out_spacing = Spacing::new([
            in_s[0] * fz as f64,
            in_s[1] * fy as f64,
            in_s[2] * fx as f64,
        ]);

        // Origin shifts by the continuous centroid offset (N mod f + f − 1)/2 voxels.
        let delta = [
            in_s[0] * ((nz % fz) as f64 + fz as f64 - 1.0) / 2.0,
            in_s[1] * ((ny % fy) as f64 + fy as f64 - 1.0) / 2.0,
            in_s[2] * ((nx % fx) as f64 + fx as f64 - 1.0) / 2.0,
        ];
        let dir = image.direction();
        let in_o = image.origin();
        let mut out_origin = *in_o;
        for i in 0..3 {
            let mut acc = in_o[i];
            for (j, &d) in delta.iter().enumerate() {
                acc += dir[(i, j)] * d;
            }
            out_origin[i] = acc;
        }

        Ok(rebuild_with_metadata(
            out,
            [oz, oy, ox],
            out_origin,
            out_spacing,
            *image.direction(),
            image,
        ))
    }
}

/// Integer **tile-averaging** downsample (anti-aliased display shrink).
///
/// Each output voxel is the arithmetic mean of all input voxels in its tile,
/// with `ceil(N/f)` output size (trailing partial tiles are averaged over the
/// voxels that exist). This is NOT ITK `Shrink` (which subsamples — see
/// [`ShrinkImageFilter`]) nor ITK `BinShrink` (which uses `floor(N/f)` full
/// bins — see `BinShrinkImageFilter`); it is a display-oriented anti-alias
/// downsample. Output spacing is `in_spacing[d] · f[d]`; origin is unchanged.
#[derive(Debug, Clone)]
pub struct TileMeanShrinkFilter {
    /// Downsampling factors per axis `\[fz, fy, fx\]`. All must be ≥ 1. Default \[1,1,1\].
    pub shrink_factors: [usize; 3],
}

impl TileMeanShrinkFilter {
    /// Construct with the given per-axis shrink factors (0 is treated as 1).
    pub fn new(shrink_factors: [usize; 3]) -> Self {
        Self { shrink_factors }
    }
}

impl Default for TileMeanShrinkFilter {
    fn default() -> Self {
        Self::new([1, 1, 1])
    }
}

impl TileMeanShrinkFilter {
    /// Apply the tile-averaging shrink to a 3-D image.
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> anyhow::Result<Image<B, 3>> {
        let (vals_vec, _) = extract_vec_infallible(image);
        let (out, out_shape) = self.tile_mean(&vals_vec, image.shape());

        Ok(rebuild_with_metadata(
            out,
            out_shape,
            *image.origin(),
            self.scaled_spacing(image.spacing()),
            *image.direction(),
            image,
        ))
    }

    /// Apply the tile-averaging shrink to a Coeus-native image.
    pub fn apply_native<B>(
        &self,
        image: &ritk_image::native::Image<f32, B, 3>,
        backend: &B) -> anyhow::Result<ritk_image::native::Image<f32, B, 3>>
    where
        B: coeus_core::ComputeBackend,
        B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    {
        let (values, shape) = self.tile_mean(image.data_slice()?, image.shape());
        ritk_image::native::Image::from_flat_on(
            values,
            shape,
            *image.origin(),
            self.scaled_spacing(image.spacing()),
            *image.direction(),
            backend,
        )
    }

    fn tile_mean(&self, values: &[f32], [nz, ny, nx]: [usize; 3]) -> (Vec<f32>, [usize; 3]) {
        let [fz, fy, fx] = self.factors();
        let shape = [nz.div_ceil(fz), ny.div_ceil(fy), nx.div_ceil(fx)];
        let [oz, oy, ox] = shape;
        let mut output = vec![0.0; oz * oy * ox];
        for iz in 0..oz {
            for iy in 0..oy {
                for ix in 0..ox {
                    let mut sum = 0.0f64;
                    let mut count = 0u64;
                    for kz in iz * fz..((iz + 1) * fz).min(nz) {
                        for ky in iy * fy..((iy + 1) * fy).min(ny) {
                            for kx in ix * fx..((ix + 1) * fx).min(nx) {
                                sum += values[kz * ny * nx + ky * nx + kx] as f64;
                                count += 1;
                            }
                        }
                    }
                    output[iz * oy * ox + iy * ox + ix] = (sum / count as f64) as f32;
                }
            }
        }
        (output, shape)
    }

    fn factors(&self) -> [usize; 3] {
        self.shrink_factors.map(|factor| factor.max(1))
    }

    fn scaled_spacing(&self, spacing: &Spacing<3>) -> Spacing<3> {
        let factors = self.factors();
        Spacing::new(std::array::from_fn(|axis| {
            spacing[axis] * factors[axis] as f64
        }))
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "tests_shrink.rs"]
mod tests_shrink;
