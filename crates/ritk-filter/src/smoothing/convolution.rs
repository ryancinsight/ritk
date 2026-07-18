//! Spatial (sliding-window) convolution filter.
//!
//! Implements ITK's `ConvolutionImageFilter` (`sitk.Convolve`): direct
//! sliding-window convolution with zero-flux Neumann boundary conditions.
//!
//! # Algorithm
//!
//! For each output voxel the kernel is centred on the corresponding input
//! voxel; out-of-bounds positions are satisfied by clamping to the nearest
//! edge voxel (zero-flux Neumann boundary). The inner accumulation is
//! performed in `f64` to reduce catastrophic cancellation before the result
//! is narrowed back to `f32`.
//!
//! # Complexity
//!
//! O(Dz Â· Dy Â· Dx Â· Kz Â· Ky Â· Kx) â€” a separable implementation would be
//! O(N Â· (Kz + Ky + Kx)), but only separable kernels admit that optimisation;
//! the general case requires this full product.

use anyhow::{bail, Result};
use ritk_image::tensor::Backend;
use ritk_image::Image;
use ritk_tensor_ops::{extract_vec, rebuild};

/// Direct spatial (sliding-window) convolution of a 3-D image with a kernel,
/// using zero-flux Neumann (edge-clamp) boundary conditions.
///
/// # Mathematical contract
///
/// For a 3-D image `I` with shape `[Dz, Dy, Dx]` and kernel `K` with shape
/// `[Kz, Ky, Kx]`:
///
/// ```text
/// output[z, y, x] =
///     Î£_{kz=0}^{Kz-1}  Î£_{ky=0}^{Ky-1}  Î£_{kx=0}^{Kx-1}
///         K[kz, ky, kx]
///         Â· I[clamp(z + kz âˆ’ Kz/2, 0, Dz-1),
///              clamp(y + ky âˆ’ Ky/2, 0, Dy-1),
///              clamp(x + kx âˆ’ Kx/2, 0, Dx-1)]
/// ```
///
/// where `Kz/2`, `Ky/2`, `Kx/2` are floor-division half-extents that centre
/// the kernel on each output voxel.
///
/// # Boundary conditions
///
/// Zero-flux Neumann: out-of-bounds source coordinates are clamped to the
/// nearest valid voxel, matching ITK's `ZeroFluxNeumannBoundaryCondition`.
///
/// # Normalisation
///
/// The caller is responsible for normalising the kernel (e.g. dividing by its
/// element sum for a blurring kernel). This matches `sitk.Convolve` behaviour.
///
/// # ITK Parity
///
/// `ConvolutionImageFilter` (`sitk.Convolve`).
#[derive(Debug, Clone)]
pub struct SpatialConvolutionFilter {
    /// Flat kernel values in row-major `[Kz, Ky, Kx]` order.
    kernel: Vec<f32>,
    /// Kernel shape `[Kz, Ky, Kx]`.
    kernel_dims: [usize; 3],
}

impl SpatialConvolutionFilter {
    /// Construct a `SpatialConvolutionFilter` from a flat kernel buffer and its shape.
    ///
    /// # Arguments
    /// - `kernel`      â€” Kernel voxels in row-major `[Kz, Ky, Kx]` order.
    /// - `kernel_dims` â€” Shape `[Kz, Ky, Kx]`. Must satisfy
    ///   `kernel.len() == Kz * Ky * Kx`.
    ///
    /// # Errors
    /// Returns `Err` when `kernel.len() != kernel_dims[0] * kernel_dims[1] * kernel_dims[2]`.
    pub fn new(kernel: Vec<f32>, kernel_dims: [usize; 3]) -> Result<Self> {
        let expected = kernel_dims[0] * kernel_dims[1] * kernel_dims[2];
        if kernel.len() != expected {
            bail!(
                "SpatialConvolutionFilter: kernel buffer length {} \
                 does not match shape {:?} (expected {})",
                kernel.len(),
                kernel_dims,
                expected,
            );
        }
        Ok(Self {
            kernel,
            kernel_dims,
        })
    }

    /// Apply the spatial convolution to a 3-D image.
    ///
    /// Iterates over every output voxel, accumulates the weighted kernel sum in
    /// `f64` to reduce rounding error, then narrows to `f32`. Spatial metadata
    /// (origin, spacing, direction) is preserved exactly.
    ///
    /// # Errors
    /// Returns `Err` when the tensor data cannot be extracted as `f32`.
    #[allow(clippy::needless_range_loop)]
    pub fn apply<B: Backend>(&self, image: &Image<f32, B, 3>) -> Result<Image<f32, B, 3>> {
        let (vals, dims) = extract_vec(image)?;
        let [dz, dy, dx] = dims;
        let [kz, ky, kx] = self.kernel_dims;

        let hz = (kz / 2) as isize;
        let hy = (ky / 2) as isize;
        let hx = (kx / 2) as isize;

        let nzi = dz as isize;
        let nyi = dy as isize;
        let nxi = dx as isize;

        // Precompute clamped indices for all dimensions to avoid clamp, casting,
        // and allocation overhead inside the hot parallel loop.
        let mut sz_indices = Vec::with_capacity(dz * kz);
        for z in 0..dz {
            let zi = z as isize;
            for ikz in 0..kz {
                sz_indices.push((zi + ikz as isize - hz).clamp(0, nzi - 1) as usize);
            }
        }

        let mut sy_indices = Vec::with_capacity(dy * ky);
        for y in 0..dy {
            let yi = y as isize;
            for iky in 0..ky {
                sy_indices.push((yi + iky as isize - hy).clamp(0, nyi - 1) as usize);
            }
        }

        let mut sx_indices = Vec::with_capacity(dx * kx);
        for x in 0..dx {
            let xi = x as isize;
            for ikx in 0..kx {
                sx_indices.push((xi + ikx as isize - hx).clamp(0, nxi - 1) as usize);
            }
        }

        let chunk_size = dy * dx;
        let mut out = vec![0.0_f32; dz * chunk_size];

        let kernel_ref = &self.kernel;
        let vals_ref = &vals;
        let sz_indices_ref = &sz_indices;
        let sy_indices_ref = &sy_indices;
        let sx_indices_ref = &sx_indices;

        moirai::for_each_chunk_mut_enumerated_with::<moirai::Adaptive, _, _>(
            &mut out,
            chunk_size,
            |z, slice| {
                let sz_slice = &sz_indices_ref[z * kz..(z + 1) * kz];
                for y in 0..dy {
                    let sy_slice = &sy_indices_ref[y * ky..(y + 1) * ky];
                    for x in 0..dx {
                        let sx_slice = &sx_indices_ref[x * kx..(x + 1) * kx];
                        let mut sum = 0.0_f64;
                        for (ikz, &sz) in sz_slice.iter().enumerate() {
                            let sz_offset = sz * dy * dx;
                            let k_offset_z = ikz * ky * kx;
                            for (iky, &sy) in sy_slice.iter().enumerate() {
                                let sy_offset = sz_offset + sy * dx;
                                let k_offset_y = k_offset_z + iky * kx;
                                for ikx in 0..kx {
                                    let sx = sx_slice[ikx];
                                    let k_val = kernel_ref[k_offset_y + ikx] as f64;
                                    sum += k_val * vals_ref[sy_offset + sx] as f64;
                                }
                            }
                        }
                        slice[y * dx + x] = sum as f32;
                    }
                }
            },
        );

        Ok(rebuild(out, dims, image))
    }
    /// Coeus-native counterpart to the legacy application method.
    pub fn apply_native<B>(
        &self,
        image: &ritk_image::native::Image<f32, B, 3>,
        backend: &B,
    ) -> anyhow::Result<ritk_image::native::Image<f32, B, 3>>
    where
        B: coeus_core::ComputeBackend,
        B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    {
        let (vals, dims) = ritk_tensor_ops::native::extract_image_vec(image)?;
        let [dz, dy, dx] = dims;
        let [kz, ky, kx] = self.kernel_dims;

        let hz = (kz / 2) as isize;
        let hy = (ky / 2) as isize;
        let hx = (kx / 2) as isize;

        let nzi = dz as isize;
        let nyi = dy as isize;
        let nxi = dx as isize;

        // Precompute clamped indices for all dimensions to avoid clamp, casting,
        // and allocation overhead inside the hot parallel loop.
        let mut sz_indices = Vec::with_capacity(dz * kz);
        for z in 0..dz {
            let zi = z as isize;
            for ikz in 0..kz {
                sz_indices.push((zi + ikz as isize - hz).clamp(0, nzi - 1) as usize);
            }
        }

        let mut sy_indices = Vec::with_capacity(dy * ky);
        for y in 0..dy {
            let yi = y as isize;
            for iky in 0..ky {
                sy_indices.push((yi + iky as isize - hy).clamp(0, nyi - 1) as usize);
            }
        }

        let mut sx_indices = Vec::with_capacity(dx * kx);
        for x in 0..dx {
            let xi = x as isize;
            for ikx in 0..kx {
                sx_indices.push((xi + ikx as isize - hx).clamp(0, nxi - 1) as usize);
            }
        }

        let chunk_size = dy * dx;
        let mut out = vec![0.0_f32; dz * chunk_size];

        let kernel_ref = &self.kernel;
        let vals_ref = &vals;
        let sz_indices_ref = &sz_indices;
        let sy_indices_ref = &sy_indices;
        let sx_indices_ref = &sx_indices;

        moirai::for_each_chunk_mut_enumerated_with::<moirai::Adaptive, _, _>(
            &mut out,
            chunk_size,
            |z, slice| {
                let sz_slice = &sz_indices_ref[z * kz..(z + 1) * kz];
                for y in 0..dy {
                    let sy_slice = &sy_indices_ref[y * ky..(y + 1) * ky];
                    for x in 0..dx {
                        let sx_slice = &sx_indices_ref[x * kx..(x + 1) * kx];
                        let mut sum = 0.0_f64;
                        for (ikz, &sz) in sz_slice.iter().enumerate() {
                            let sz_offset = sz * dy * dx;
                            let k_offset_z = ikz * ky * kx;
                            for (iky, &sy) in sy_slice.iter().enumerate() {
                                let sy_offset = sz_offset + sy * dx;
                                let k_offset_y = k_offset_z + iky * kx;
                                for ikx in 0..kx {
                                    let sx = sx_slice[ikx];
                                    let k_val = kernel_ref[k_offset_y + ikx] as f64;
                                    sum += k_val * vals_ref[sy_offset + sx] as f64;
                                }
                            }
                        }
                        slice[y * dx + x] = sum as f32;
                    }
                }
            },
        );

        crate::native_support::rebuild_image(out, dims, image, backend)
    }
}
