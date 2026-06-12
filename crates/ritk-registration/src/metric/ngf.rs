//! Normalized Gradient Fields (NGF) metric for multi-modal registration.
//!
//! # Theorem: edge-orientation similarity (Haber & Modersitzki 2006)
//!
//! Cross-modal pairs (CT↔MRI) lack a functional intensity relationship, so
//! intensity metrics (MI, NCC) can be weak — e.g. a near-uniform CT brain
//! interior gives almost no mutual-information signal, and a rotation about the
//! centroid barely perturbs the joint histogram. NGF instead aligns the
//! **orientation** of image gradients, which co-locate across modalities even
//! where intensities do not (a skull/ventricle boundary is an edge in *both* CT
//! and MRI). For fixed `F` and moving `M` resampled onto the fixed grid,
//!
//! ```text
//! NGF(F, M) = (1/N) · Σ_x  (∇F·∇M)² / ((|∇F|² + η_F²)(|∇M|² + η_M²))
//! ```
//!
//! Each term is `1` when the gradients are parallel **or anti-parallel** (so a
//! bright-CT / dark-MR edge still scores `1` — the squared dot product is
//! sign-invariant) and `0` where either side is flat. `η` is the edge-noise
//! scale (the mean masked gradient magnitude, per Haber & Modersitzki), which
//! suppresses flat-region noise. `NGF ∈ [0, 1]`; higher is better aligned, so the
//! metric returns `−NGF` as a minimization loss.
//!
//! This is a **gradient-free** metric (the gradients are spatial image gradients,
//! not autodiff gradients of the transform): it returns a scalar for the
//! derivative-free optimizers (CMA-ES, coordinate descent) that cross-modal rigid
//! registration uses, where intensity-MI hill-climbing from identity is unreliable.
//! Pre-masking the images (e.g. to a brain mask) focuses NGF on the shared rigid
//! structure, since flat masked-out regions contribute ~0.

use super::trait_::Metric;
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};
use ritk_image::grid;
use ritk_image::Image;
use ritk_interpolation::{Interpolator, LinearInterpolator};
use ritk_transform::Transform;

/// Normalized Gradient Fields metric (Haber & Modersitzki 2006).
///
/// Returns `−NGF ∈ [−1, 0]` as a loss to be minimized. Robust for cross-modal
/// (CT↔MRI) alignment where intensity MI/NCC are weak. See the [module docs](self).
pub struct NormalizedGradientField {
    interpolator: LinearInterpolator,
}

impl NormalizedGradientField {
    /// Create a new NGF metric (linear interpolation of the moving image).
    #[must_use]
    pub fn new() -> Self {
        Self {
            interpolator: LinearInterpolator::new(),
        }
    }
}

impl Default for NormalizedGradientField {
    fn default() -> Self {
        Self::new()
    }
}

impl<B: Backend, const D: usize> Metric<B, D> for NormalizedGradientField {
    fn forward(
        &self,
        fixed: &Image<B, D>,
        moving: &Image<B, D>,
        transform: &impl Transform<B, D>,
    ) -> Tensor<B, 1> {
        let device = fixed.data().device();
        let ngf = self.ngf_value(fixed, moving, transform, None);
        // −NGF as a minimization loss.
        Tensor::<B, 1>::from_data(TensorData::new(vec![-ngf], [1]), &device)
    }

    fn name(&self) -> &'static str {
        "NormalizedGradientField"
    }
}

impl NormalizedGradientField {
    /// Resample `moving` onto the `fixed` grid through `transform`, then return
    /// `NGF ∈ [0, 1]` over the `true` voxels of `mask` (or all if `None`). The
    /// masked path is used by the cross-modal rigid registration; the unmasked
    /// path backs [`Metric::forward`].
    pub(crate) fn ngf_value<B: Backend, const D: usize>(
        &self,
        fixed: &Image<B, D>,
        moving: &Image<B, D>,
        transform: &impl Transform<B, D>,
        mask: Option<&[bool]>,
    ) -> f32 {
        let device = fixed.data().device();
        let shape = fixed.shape();
        let n: usize = shape.iter().product();

        // Resample the moving image onto the fixed grid (NCC idiom): fixed voxel →
        // world → transform → moving index → interpolate.
        let fixed_indices = grid::generate_grid(shape, &device);
        let fixed_points = fixed.index_to_world_tensor(fixed_indices);
        let moving_points = transform.transform_points(fixed_points);
        let moving_indices = moving.world_to_index_tensor(moving_points);
        let m_tensor = self.interpolator.interpolate(moving.data(), moving_indices);

        // Materialize both volumes on the fixed grid (gradients need neighbours;
        // the metric is gradient-free so reading to host is fine).
        let f: Vec<f32> = fixed
            .data()
            .clone()
            .reshape([n])
            .into_data()
            .to_vec()
            .expect("fixed image to f32 host vec");
        let m: Vec<f32> = m_tensor
            .into_data()
            .to_vec()
            .expect("resampled moving image to f32 host vec");

        ngf_scalar(&f, &m, &shape, mask)
    }
}

/// `NGF ∈ [0, 1]` of two co-gridded volumes `f`, `m` of (row-major) `shape`,
/// averaged over the `true` voxels of `mask` (or all voxels if `mask` is `None`).
///
/// Gradients always use real neighbours (the image is *not* zeroed outside the
/// mask), so the mask only restricts which voxels are *counted* — no artificial
/// mask-boundary edge is introduced. Masking to the brain+skull region is what
/// makes cross-modal NGF lock onto the shared rigid anatomy instead of the
/// scalp/scanner-bed/FOV edges. See the [module docs](self).
fn ngf_scalar<const D: usize>(f: &[f32], m: &[f32], shape: &[usize; D], mask: Option<&[bool]>) -> f32 {
    let n = f.len();
    if n == 0 || m.len() != n {
        return 0.0;
    }
    let included = |flat: usize| mask.is_none_or(|mk| mk[flat]);
    // Row-major (C-order) strides: last axis is fastest.
    let mut stride = [1usize; D];
    for a in (0..D.saturating_sub(1)).rev() {
        stride[a] = stride[a + 1] * shape[a + 1];
    }

    // Pass 1: η_F², η_M² = (mean masked gradient magnitude)² (Haber & Modersitzki).
    let (mut sum_f, mut sum_m, mut count) = (0.0_f64, 0.0_f64, 0.0_f64);
    let (mut gf, mut gm) = ([0.0_f32; D], [0.0_f32; D]);
    for flat in 0..n {
        if !included(flat) {
            continue;
        }
        grad_at(f, flat, shape, &stride, &mut gf);
        grad_at(m, flat, shape, &stride, &mut gm);
        sum_f += f64::from(magnitude(&gf));
        sum_m += f64::from(magnitude(&gm));
        count += 1.0;
    }
    if count < 1.0 {
        return 0.0;
    }
    let eta_f2 = ((sum_f / count).powi(2)).max(1e-12) as f32;
    let eta_m2 = ((sum_m / count).powi(2)).max(1e-12) as f32;

    // Pass 2: mean squared normalized gradient dot product over the mask.
    let mut acc = 0.0_f64;
    for flat in 0..n {
        if !included(flat) {
            continue;
        }
        grad_at(f, flat, shape, &stride, &mut gf);
        grad_at(m, flat, shape, &stride, &mut gm);
        let dot: f32 = (0..D).map(|a| gf[a] * gm[a]).sum();
        let na2: f32 = (0..D).map(|a| gf[a] * gf[a]).sum::<f32>() + eta_f2;
        let nb2: f32 = (0..D).map(|a| gm[a] * gm[a]).sum::<f32>() + eta_m2;
        acc += f64::from((dot * dot) / (na2 * nb2));
    }
    (acc / count) as f32
}

/// Central-difference spatial gradient of `data` at flat index `flat`, written
/// into `out` (one component per axis). Borders use a one-sided difference.
fn grad_at<const D: usize>(
    data: &[f32],
    flat: usize,
    shape: &[usize; D],
    stride: &[usize; D],
    out: &mut [f32; D],
) {
    for a in 0..D {
        let idx_a = (flat / stride[a]) % shape[a];
        let has_lo = idx_a > 0;
        let has_hi = idx_a + 1 < shape[a];
        let lo = if has_lo { data[flat - stride[a]] } else { data[flat] };
        let hi = if has_hi { data[flat + stride[a]] } else { data[flat] };
        let denom = (usize::from(has_lo) + usize::from(has_hi)).max(1) as f32;
        out[a] = (hi - lo) / denom;
    }
}

fn magnitude<const D: usize>(g: &[f32; D]) -> f32 {
    (0..D).map(|a| g[a] * g[a]).sum::<f32>().sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Shape, TensorData};
    use burn_ndarray::NdArray;
    use ritk_spatial::{Direction, Point, Spacing};
    use ritk_transform::TranslationTransform;

    type B = NdArray<f32>;

    fn image2d(data: Vec<f32>, shape: [usize; 2]) -> Image<B, 2> {
        let device = Default::default();
        let tensor = Tensor::from_data(TensorData::new(data, Shape::new(shape)), &device);
        Image::new(
            tensor,
            Point::new([0.0, 0.0]),
            Spacing::new([1.0, 1.0]),
            Direction::identity(),
        )
    }

    fn vertical_edge(w: usize, h: usize, at: usize, sign: f32) -> Vec<f32> {
        let mut v = vec![0.0f32; w * h];
        for y in 0..h {
            for x in 0..w {
                v[y * w + x] = if x < at { 0.0 } else { sign };
            }
        }
        v
    }

    /// Cross-modal sign invariance: a co-located edge with OPPOSITE contrast
    /// (bright→dark vs dark→bright) scores exactly the same as an identical-
    /// contrast edge — the squared gradient dot product makes a bright-CT /
    /// dark-MR boundary register just like a same-sign one.
    #[test]
    fn ngf_is_sign_invariant() {
        let (w, h) = (8usize, 8usize);
        let f = vertical_edge(w, h, 4, 1.0);
        let same = ngf_scalar(&f, &f, &[h, w], None);
        let opposite = ngf_scalar(&f, &vertical_edge(w, h, 4, -1.0), &[h, w], None);
        assert!(same > 0.0, "self-NGF should be positive, got {same}");
        assert!(
            (same - opposite).abs() < 1e-4,
            "opposite contrast must score equal: same {same} vs opposite {opposite}"
        );
    }

    /// NGF of perpendicular edges (uncorrelated orientation) is well below that of
    /// aligned edges — the property that lets NGF recover a rotation that intensity
    /// MI cannot.
    #[test]
    fn aligned_beats_perpendicular() {
        let (w, h) = (8usize, 8usize);
        let vert = vertical_edge(w, h, 4, 1.0); // gradient in x
        let mut horiz = vec![0.0f32; w * h]; // gradient in y
        for y in 0..h {
            for x in 0..w {
                horiz[y * w + x] = if y < 4 { 0.0 } else { 1.0 };
            }
        }
        let aligned = ngf_scalar(&vert, &vert, &[h, w], None);
        let perpendicular = ngf_scalar(&vert, &horiz, &[h, w], None);
        assert!(
            aligned > perpendicular + 0.1,
            "aligned {aligned} should exceed perpendicular {perpendicular}"
        );
    }

    /// End-to-end through the `Metric` trait: registering the moving edge onto the
    /// fixed edge (identity) gives a lower loss than a translation that pulls the
    /// edges apart. (Grid column 0 is x, so the x-shift moves the vertical edge.)
    #[test]
    fn metric_loss_lower_when_aligned() {
        let (w, h) = (16usize, 16usize);
        let img = image2d(vertical_edge(w, h, 8, 1.0), [h, w]);
        let metric = NormalizedGradientField::new();
        let device = Default::default();
        let loss = |dx: f32| {
            let t = TranslationTransform::<B, 2>::new(Tensor::from_data(
                TensorData::new(vec![dx, 0.0_f32], [2]),
                &device,
            ));
            metric
                .forward(&img, &img, &t)
                .into_data()
                .to_vec::<f32>()
                .unwrap()[0]
        };
        let aligned = loss(0.0);
        let shifted = loss(4.0);
        assert!(aligned < 0.0, "aligned loss should be negative, got {aligned}");
        assert!(
            aligned < shifted,
            "aligned loss {aligned} should be below shifted {shifted}"
        );
    }
}
