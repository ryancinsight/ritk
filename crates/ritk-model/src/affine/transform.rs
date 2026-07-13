//! Affine spatial transformer, Coeus-native.
//!
//! Warps a volume by a predicted `3×4` affine matrix. A normalized homogeneous
//! sampling grid is transformed by the affine matrix and consumed by the
//! differentiable `coeus_autograd::grid_sample_3d` warp
//! (PyTorch `grid_sample` semantics: `align_corners = true`, zero padding,
//! trilinear interpolation). Gradients flow to the affine matrix through the
//! grid, which is exactly the signal affine-registration optimization needs.
//!
//! # Convention
//! - `theta`: `[B, 3, 4]` (or `[B, 12]` flattened, row-major). Row `i` maps the
//!   homogeneous input coordinate to normalized output axis `i`, ordered
//!   `(z, y, x)` → `(D, H, W)`.
//! - Sampling coordinates stay normalized to `[-1, 1]`; `grid_sample_3d`
//!   performs the pixel de-normalization internally.

use coeus_autograd::{flip, grid_sample_3d, matmul, permute, reshape, Var};
use coeus_core::{Backend, CpuAddressableStorage, CpuAddressableStorageMut};
use coeus_ops::BackendOps;
use coeus_tensor::Tensor;

/// Differentiable affine warp (spatial transformer network head).
#[derive(Debug, Default, Clone, Copy)]
pub struct AffineTransform;

impl AffineTransform {
    /// Construct the (stateless) affine transformer.
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Warp `image` by the affine parameters `theta`.
    ///
    /// # Arguments
    /// * `image` — input volume `[B, C, D, H, W]`.
    /// * `theta` — affine parameters `[B, 3, 4]` or `[B, 12]`.
    ///
    /// # Returns
    /// The warped volume `[B, C, D, H, W]`.
    ///
    /// # Panics
    /// Panics if `image` is not rank-5 or `theta`'s element count is not
    /// `B * 12`.
    pub fn forward<B>(&self, image: &Var<f32, B>, theta: &Var<f32, B>) -> Var<f32, B>
    where
        B: Backend + BackendOps<f32> + Default,
        B::DeviceBuffer<f32>: CpuAddressableStorage<f32> + CpuAddressableStorageMut<f32>,
    {
        let shape = image.tensor.shape();
        assert_eq!(
            shape.len(),
            5,
            "AffineTransform expects a rank-5 [B,C,D,H,W] image"
        );
        let (b, d, h, w) = (shape[0], shape[2], shape[3], shape[4]);

        // theta → [B, 3, 4]
        let theta = reshape(theta, [b, 3, 4]);

        // Constant normalized homogeneous grid [B, 4, N], rows (z, y, x, 1).
        let grid = self.normalized_homogeneous_grid::<B>(b, d, h, w);

        // Apply the affine map: [B,3,4] × [B,4,N] → [B,3,N] (rows z, y, x).
        let warped = matmul(&theta, &grid);
        let warped = reshape(&warped, [b, 3, d, h, w]);

        // Rearrange to the grid_sample layout [B, D, H, W, 3] with last-dim
        // order (x, y, z). permute puts channels last as (z, y, x); flip
        // reverses to (x, y, z) which grid_sample maps to (W, H, D).
        let grid = permute(&warped, &[0, 2, 3, 4, 1]);
        let grid = flip(&grid, 4);

        grid_sample_3d(image, &grid)
    }

    /// Build the constant normalized homogeneous coordinate grid `[B, 4, N]`.
    ///
    /// Row 0 = `z` normalized to `[-1, 1]`, row 1 = `y`, row 2 = `x`, row 3 = 1.
    /// Voxel index `n = z·(H·W) + y·W + x` matches the row-major reshape to
    /// `[B, 3, D, H, W]` applied to the transformed grid. A singleton extent
    /// maps to coordinate `0` (its normalized center) instead of dividing by
    /// zero.
    fn normalized_homogeneous_grid<B>(&self, b: usize, d: usize, h: usize, w: usize) -> Var<f32, B>
    where
        B: Backend + BackendOps<f32> + Default,
    {
        let n = d * h * w;
        let norm = |i: usize, extent: usize| -> f32 {
            if extent > 1 {
                (i as f32) * 2.0 / ((extent - 1) as f32) - 1.0
            } else {
                0.0
            }
        };

        let mut data = vec![0.0f32; b * 4 * n];
        // Fill one batch, then replicate across the batch dimension.
        let batch_stride = 4 * n;
        for z in 0..d {
            let zc = norm(z, d);
            for y in 0..h {
                let yc = norm(y, h);
                for x in 0..w {
                    let xc = norm(x, w);
                    let idx = z * (h * w) + y * w + x;
                    data[idx] = zc; // row 0
                    data[n + idx] = yc; // row 1
                    data[2 * n + idx] = xc; // row 2
                    data[3 * n + idx] = 1.0; // row 3 (homogeneous)
                }
            }
        }
        for batch in 1..b {
            let (head, tail) = data.split_at_mut(batch * batch_stride);
            tail[..batch_stride].copy_from_slice(&head[..batch_stride]);
        }

        Var::new(Tensor::from_slice_on([b, 4, n], &data, &B::default()), false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use coeus_autograd::sum;
    use coeus_core::SequentialBackend;

    fn ramp_image(shape: [usize; 5]) -> Vec<f32> {
        let n: usize = shape.iter().product();
        (0..n).map(|i| ((i % 13) as f32) / 13.0).collect()
    }

    /// Smooth (globally linear) intensity field over voxel coordinates.
    ///
    /// Trilinear interpolation of a linear field is exact, so the warped output
    /// is a smooth function of `theta` — a well-conditioned finite-difference
    /// oracle, unlike the sawtooth [`ramp_image`].
    fn linear_image(shape: [usize; 5]) -> Vec<f32> {
        let (d, h, w) = (shape[2], shape[3], shape[4]);
        let mut data = vec![0.0f32; shape.iter().product()];
        let denom = |e: usize| if e > 1 { (e - 1) as f32 } else { 1.0 };
        for z in 0..d {
            for y in 0..h {
                for x in 0..w {
                    let idx = z * (h * w) + y * w + x;
                    data[idx] = 0.3 * (z as f32) / denom(d)
                        + 0.5 * (y as f32) / denom(h)
                        + 0.2 * (x as f32) / denom(w);
                }
            }
        }
        data
    }

    #[test]
    fn forward_preserves_shape() {
        let stn = AffineTransform::new();
        let shape = [1usize, 1, 8, 8, 8];
        let image = Var::new(
            Tensor::from_slice_on(shape, &ramp_image(shape), &SequentialBackend),
            false,
        );
        let theta = Var::new(
            Tensor::from_slice_on(
                [1, 12],
                &[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                &SequentialBackend,
            ),
            false,
        );
        let out = stn.forward(&image, &theta);
        assert_eq!(out.tensor.shape(), &[1, 1, 8, 8, 8]);
    }

    #[test]
    fn identity_transform_reproduces_input() {
        // align_corners=true grid-sample under the identity affine samples each
        // voxel at its own center, so the warp is the exact identity.
        let stn = AffineTransform::new();
        let shape = [1usize, 1, 6, 6, 6];
        let img = ramp_image(shape);
        let image = Var::new(
            Tensor::from_slice_on(shape, &img, &SequentialBackend),
            false,
        );
        let theta = Var::new(
            Tensor::from_slice_on(
                [1, 12],
                &[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                &SequentialBackend,
            ),
            false,
        );
        let out = stn.forward(&image, &theta);
        let got = out.tensor.as_slice();
        for (i, (&a, &b)) in img.iter().zip(got.iter()).enumerate() {
            assert!(
                (a - b).abs() <= 1e-5,
                "identity warp changed voxel {i}: in={a}, out={b}"
            );
        }
    }

    /// Finite-difference gradient check on `theta`.
    ///
    /// Central differences of `d(sum(warp))/d(theta)` versus the autograd
    /// gradient through `matmul → permute → flip → grid_sample_3d`.
    ///
    /// Tolerance: central differences are `O(h²)`; `grid_sample` is piecewise-
    /// trilinear, so its coordinate gradient is smooth only *inside* a
    /// trilinear cell. The base affine below scales sampling to the off-grid
    /// interior (`|coord| ≲ 0.6`), keeping every sample away from both the
    /// zero-padding boundary and the voxel-center kinks where the gradient is
    /// one-sided. On that smooth region, with `h = 2⁻⁸` and the exactly-linear
    /// image, FD is accurate to `~10⁻³`; a bound of `3e-2·(1+|g|)` covers
    /// truncation and f32 rounding without masking a real defect.
    #[test]
    fn finite_difference_gradient_wrt_theta() {
        let stn = AffineTransform::new();
        let shape = [1usize, 1, 8, 8, 8];
        let image = Var::new(
            Tensor::from_slice_on(shape, &linear_image(shape), &SequentialBackend),
            false,
        );
        // Scale-0.5 affine with off-grid translations: samples land strictly
        // interior and never on a voxel center, so the warp is locally smooth.
        let base = vec![
            0.5, 0.0, 0.0, 0.111, 0.0, 0.5, 0.0, -0.137, 0.0, 0.0, 0.5, 0.081,
        ];
        let theta = Var::new(
            Tensor::from_slice_on([1, 12], &base, &SequentialBackend),
            true,
        );
        let out = stn.forward(&image, &theta);
        let loss = sum(&out);
        loss.backward();
        let grad = theta.grad().expect("theta gradient present");
        let grad = grad.as_slice();

        let h = 1.0f32 / 256.0;
        let mut max_analytic = 0.0f32;
        for idx in 0..12 {
            let mut plus = base.clone();
            let mut minus = base.clone();
            plus[idx] += h;
            minus[idx] -= h;
            let fp = sum(&stn.forward(
                &image,
                &Var::new(Tensor::from_slice_on([1, 12], &plus, &SequentialBackend), false),
            ));
            let fm = sum(&stn.forward(
                &image,
                &Var::new(Tensor::from_slice_on([1, 12], &minus, &SequentialBackend), false),
            ));
            let fd = (fp.tensor.as_slice()[0] - fm.tensor.as_slice()[0]) / (2.0 * h);
            let analytic = grad[idx];
            max_analytic = max_analytic.max(analytic.abs());
            let diff = (fd - analytic).abs();
            let tol = 3e-2 * (1.0 + analytic.abs());
            assert!(
                diff <= tol,
                "theta grad mismatch at {idx}: fd={fd}, autograd={analytic}, |Δ|={diff} > {tol}"
            );
        }
        assert!(
            max_analytic > 1e-6,
            "gradient check is vacuous — all theta gradients are ~0 ({max_analytic})"
        );
    }
}
