//! DICOM axis reorientation (`itk::DICOMOrientImageFilter` / `sitk.DICOMOrient`).
//!
//! # Mathematical Specification
//!
//! Relabels the image axes so that each output index axis points toward a target
//! anatomical direction, given by a three-letter DICOM orientation code (e.g.
//! `"LPS"`, `"RAI"`). The relabeling is a signed axis permutation applied
//! consistently to the voxel data, spacing, origin, and direction matrix, leaving
//! the physical object unchanged — only its index parameterization changes.
//!
//! Anatomical letters (LPS-positive world frame, world axes `x, y, z`):
//!
//! ```text
//! L = +x   R = −x   P = +y   A = −y   S = +z   I = −z
//! ```
//!
//! For each output image axis `j ∈ {x, y, z}` the code letter fixes a target unit
//! world vector `v_j`. The input tensor axis `a` whose physical direction
//! (column `a` of the core direction matrix) is most parallel to `±v_j` is chosen,
//! with sign `σ = sign(D_in.col(a) · v_j)`; that axis becomes output axis `j`,
//! reversed when `σ < 0`. Spacing follows the permutation, the output direction
//! column is set to `v_j`, and the origin is recomputed as the world position of
//! the output corner voxel:
//!
//! ```text
//! origin_out = origin_in + Σ_a D_in.col(a) · spacing_in[a] · c_a
//! ```
//!
//! where `c_a` is the input index of the output `(0,0,0)` voxel along tensor axis
//! `a` (`dim_a − 1` if that axis is reversed, else `0`). Float-exact to
//! `sitk.DICOMOrient` (axis-aligned input directions).

use anyhow::{bail, Result};
use ritk_image::tensor::Backend;
use ritk_image::Image;
use ritk_spatial::{Direction, Point, Spacing};
use ritk_tensor_ops::{extract_vec_infallible, rebuild_with_metadata};

/// Reorient a 3-D image to a target DICOM orientation code.
#[derive(Debug, Clone)]
pub struct OrientImageFilter {
    /// Per image-axis `(x, y, z)` target: `(world_axis, sign)`.
    /// `world_axis ∈ {0=x, 1=y, 2=z}`, `sign ∈ {+1.0, −1.0}`.
    target: [(usize, f64); 3],
}

impl OrientImageFilter {
    /// Build from a three-letter DICOM code (`"LPS"`, `"RAI"`, `"PIR"`, …).
    ///
    /// Returns `Err` if the code is malformed (not three letters, an unknown
    /// letter, or a repeated anatomical axis).
    pub fn from_code(code: &str) -> Result<Self> {
        let letters: Vec<char> = code.chars().collect();
        if letters.len() != 3 {
            bail!("orientation code must be exactly 3 letters, got {code:?}");
        }
        let mut target = [(0usize, 1.0f64); 3];
        let mut axes_seen = [false; 3];
        for (j, &ch) in letters.iter().enumerate() {
            let (axis, sign) = match ch.to_ascii_uppercase() {
                'L' => (0, 1.0),
                'R' => (0, -1.0),
                'P' => (1, 1.0),
                'A' => (1, -1.0),
                'S' => (2, 1.0),
                'I' => (2, -1.0),
                other => bail!("invalid orientation letter {other:?} in {code:?}"),
            };
            if axes_seen[axis] {
                bail!("orientation code {code:?} repeats an anatomical axis");
            }
            axes_seen[axis] = true;
            target[j] = (axis, sign);
        }
        Ok(Self { target })
    }

    /// Apply the reorientation.
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> Result<Image<B, 3>> {
        let (vals, dims) = extract_vec_infallible(image);
        let dir = image.direction();
        let sp_in = image.spacing();
        let org_in = image.origin();

        // For each output tensor axis o (0=z, 1=y, 2=x), the corresponding image
        // axis is j = 2 - o; the target letter at j gives the world unit vector.
        // Find the input tensor axis most parallel to it.
        let mut perm = [0usize; 3]; // perm[o] = input tensor axis
        let mut flip = [false; 3]; // reverse that axis?
        let mut out_dir = Direction::zeros();
        for o in 0..3 {
            let j = 2 - o;
            let (world_axis, sign) = self.target[j];
            let mut v = [0.0f64; 3];
            v[world_axis] = sign;

            let (mut best_a, mut best_abs, mut best_sigma) = (0usize, -1.0f64, 1.0f64);
            for a in 0..3 {
                let dot = (0..3).map(|c| dir[(c, a)] * v[c]).sum::<f64>();
                if dot.abs() > best_abs {
                    best_abs = dot.abs();
                    best_a = a;
                    best_sigma = if dot >= 0.0 { 1.0 } else { -1.0 };
                }
            }
            perm[o] = best_a;
            flip[o] = best_sigma < 0.0;
            // Output tensor-axis o points toward the target world vector v.
            for (c, &vc) in v.iter().enumerate() {
                out_dir[(c, o)] = vc;
            }
        }

        // Permutation must be a bijection (axis-aligned input guarantees this).
        let mut seen = [false; 3];
        for &a in &perm {
            if seen[a] {
                bail!("input direction is not axis-aligned; cannot DICOM-orient");
            }
            seen[a] = true;
        }

        let [nz, ny, nx] = dims;
        let in_dims = [nz, ny, nx];
        let out_dims = [in_dims[perm[0]], in_dims[perm[1]], in_dims[perm[2]]];
        let (onz, ony, onx) = (out_dims[0], out_dims[1], out_dims[2]);

        let in_stride = [ny * nx, nx, 1usize];
        let mut out = vec![0.0f32; onz * ony * onx];
        let mut out_idx = [0usize; 3];
        for oz in 0..onz {
            out_idx[0] = oz;
            for oy in 0..ony {
                out_idx[1] = oy;
                for ox in 0..onx {
                    out_idx[2] = ox;
                    // Map output tensor coords → input tensor coords.
                    let mut in_lin = 0usize;
                    for o in 0..3 {
                        let a = perm[o];
                        let i = if flip[o] {
                            in_dims[a] - 1 - out_idx[o]
                        } else {
                            out_idx[o]
                        };
                        in_lin += i * in_stride[a];
                    }
                    let dst = (oz * ony + oy) * onx + ox;
                    out[dst] = vals[in_lin];
                }
            }
        }

        // Spacing follows the permutation (core spacing is tensor [z, y, x]).
        let sp_out = Spacing::new([sp_in[perm[0]], sp_in[perm[1]], sp_in[perm[2]]]);

        // Origin = world position of the output (0,0,0) voxel. Its input index
        // along tensor axis a = perm[o] is (dim_a − 1) if reversed, else 0.
        let mut corner = [0usize; 3];
        for o in 0..3 {
            corner[perm[o]] = if flip[o] { in_dims[perm[o]] - 1 } else { 0 };
        }
        let mut org_out = [0.0f64; 3];
        for c in 0..3 {
            let mut w = org_in[c];
            for a in 0..3 {
                w += dir[(c, a)] * sp_in[a] * corner[a] as f64;
            }
            org_out[c] = w;
        }

        Ok(rebuild_with_metadata(
            out,
            out_dims,
            Point::new(org_out),
            sp_out,
            out_dir,
            image,
        ))
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
        let dir = image.direction();
        let sp_in = image.spacing();
        let org_in = image.origin();

        // For each output tensor axis o (0=z, 1=y, 2=x), the corresponding image
        // axis is j = 2 - o; the target letter at j gives the world unit vector.
        // Find the input tensor axis most parallel to it.
        let mut perm = [0usize; 3]; // perm[o] = input tensor axis
        let mut flip = [false; 3]; // reverse that axis?
        let mut out_dir = Direction::zeros();
        for o in 0..3 {
            let j = 2 - o;
            let (world_axis, sign) = self.target[j];
            let mut v = [0.0f64; 3];
            v[world_axis] = sign;

            let (mut best_a, mut best_abs, mut best_sigma) = (0usize, -1.0f64, 1.0f64);
            for a in 0..3 {
                let dot = (0..3).map(|c| dir[(c, a)] * v[c]).sum::<f64>();
                if dot.abs() > best_abs {
                    best_abs = dot.abs();
                    best_a = a;
                    best_sigma = if dot >= 0.0 { 1.0 } else { -1.0 };
                }
            }
            perm[o] = best_a;
            flip[o] = best_sigma < 0.0;
            // Output tensor-axis o points toward the target world vector v.
            for (c, &vc) in v.iter().enumerate() {
                out_dir[(c, o)] = vc;
            }
        }

        // Permutation must be a bijection (axis-aligned input guarantees this).
        let mut seen = [false; 3];
        for &a in &perm {
            if seen[a] {
                bail!("input direction is not axis-aligned; cannot DICOM-orient");
            }
            seen[a] = true;
        }

        let [nz, ny, nx] = dims;
        let in_dims = [nz, ny, nx];
        let out_dims = [in_dims[perm[0]], in_dims[perm[1]], in_dims[perm[2]]];
        let (onz, ony, onx) = (out_dims[0], out_dims[1], out_dims[2]);

        let in_stride = [ny * nx, nx, 1usize];
        let mut out = vec![0.0f32; onz * ony * onx];
        let mut out_idx = [0usize; 3];
        for oz in 0..onz {
            out_idx[0] = oz;
            for oy in 0..ony {
                out_idx[1] = oy;
                for ox in 0..onx {
                    out_idx[2] = ox;
                    // Map output tensor coords → input tensor coords.
                    let mut in_lin = 0usize;
                    for o in 0..3 {
                        let a = perm[o];
                        let i = if flip[o] {
                            in_dims[a] - 1 - out_idx[o]
                        } else {
                            out_idx[o]
                        };
                        in_lin += i * in_stride[a];
                    }
                    let dst = (oz * ony + oy) * onx + ox;
                    out[dst] = vals[in_lin];
                }
            }
        }

        // Spacing follows the permutation (core spacing is tensor [z, y, x]).
        let sp_out = Spacing::new([sp_in[perm[0]], sp_in[perm[1]], sp_in[perm[2]]]);

        // Origin = world position of the output (0,0,0) voxel. Its input index
        // along tensor axis a = perm[o] is (dim_a − 1) if reversed, else 0.
        let mut corner = [0usize; 3];
        for o in 0..3 {
            corner[perm[o]] = if flip[o] { in_dims[perm[o]] - 1 } else { 0 };
        }
        let mut org_out = [0.0f64; 3];
        for c in 0..3 {
            let mut w = org_in[c];
            for a in 0..3 {
                w += dir[(c, a)] * sp_in[a] * corner[a] as f64;
            }
            org_out[c] = w;
        }

        crate::native_support::rebuild_with_metadata(
            out,
            out_dims,
            Point::new(org_out),
            sp_out,
            out_dir,
            image,
            backend,
        )
    }
}

#[cfg(test)]
#[path = "tests_orient.rs"]
mod tests_orient;
