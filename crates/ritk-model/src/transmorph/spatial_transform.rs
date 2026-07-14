//! Spatial transformer (warp), Coeus-native.
//!
//! Warps a `[B, C, D, H, W]` volume by a dense voxel-space displacement field
//! `[B, 3, D, H, W]` (channel order `(d, h, w)`), routing the resampling through
//! the differentiable [`coeus_autograd::grid_sample_3d`] op (PyTorch
//! `grid_sample` semantics: `align_corners = true`, zero padding, trilinear
//! interpolation). Gradients flow to the displacement field through the sampling
//! grid, which is the signal deformable-registration optimization requires.

use coeus_autograd::{add, cat, grid_sample_3d, reshape, scalar_mul, slice, Var};
use coeus_core::{Backend, CpuAddressableStorage, CpuAddressableStorageMut};
use coeus_ops::BackendOps;
use coeus_tensor::Tensor;

/// Stateless differentiable warp (spatial transformer network).
#[derive(Debug, Default, Clone, Copy)]
pub struct SpatialTransformer;

impl SpatialTransformer {
    /// Construct the (stateless) spatial transformer.
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Warp `image` `[B, C, D, H, W]` by voxel displacement `flow`
    /// `[B, 3, D, H, W]`, channel order `(d, h, w)`.
    ///
    /// # Panics
    /// Panics if `image` is not rank-5 or `flow`'s shape is inconsistent with it.
    pub fn forward<B>(&self, image: &Var<f32, B>, flow: &Var<f32, B>) -> Var<f32, B>
    where
        B: Backend + BackendOps<f32> + Default,
        B::DeviceBuffer<f32>: CpuAddressableStorage<f32> + CpuAddressableStorageMut<f32>,
    {
        let sh = image.tensor.shape();
        assert_eq!(sh.len(), 5, "SpatialTransformer expects a rank-5 image");
        let (b, d, h, w) = (sh[0], sh[2], sh[3], sh[4]);

        // Per-axis voxel→normalized scale (align_corners): 2/(extent-1), or 0 for
        // a singleton extent (its normalized center).
        let scale = |extent: usize| -> f32 {
            if extent > 1 {
                2.0 / (extent - 1) as f32
            } else {
                0.0
            }
        };
        let (sd, sh_, sw) = (scale(d), scale(h), scale(w));

        // Displacement components as [B, D, H, W, 1] (channel order d, h, w).
        let component = |ch: usize| {
            let c = slice(flow, &[(0, b), (ch, ch + 1), (0, d), (0, h), (0, w)]);
            reshape(&c, [b, d, h, w, 1])
        };
        let disp_d = scalar_mul(&component(0), sd);
        let disp_h = scalar_mul(&component(1), sh_);
        let disp_w = scalar_mul(&component(2), sw);

        // Constant normalized voxel-center grids [1, D, H, W, 1] per axis.
        let base = |axis_len: usize, axis: usize, s: f32| -> Var<f32, B> {
            let mut data = vec![0.0f32; d * h * w];
            let center = |i: usize, e: usize| if e > 1 { i as f32 * s - 1.0 } else { 0.0 };
            for z in 0..d {
                for y in 0..h {
                    for x in 0..w {
                        let coord = [z, y, x][axis];
                        data[z * (h * w) + y * w + x] = center(coord, axis_len);
                    }
                }
            }
            Var::new(
                Tensor::from_slice_on([1, d, h, w, 1], &data, &B::default()),
                false,
            )
        };
        let base_d = base(d, 0, sd);
        let base_h = base(h, 1, sh_);
        let base_w = base(w, 2, sw);

        // grid_sample last-dim order is (x, y, z) → (W, H, D).
        let grid_x = add(&base_w, &disp_w);
        let grid_y = add(&base_h, &disp_h);
        let grid_z = add(&base_d, &disp_d);
        let grid = cat(&[&grid_x, &grid_y, &grid_z], 4);

        grid_sample_3d(image, &grid)
    }
}
