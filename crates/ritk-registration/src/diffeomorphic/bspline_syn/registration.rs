use std::collections::VecDeque;

use crate::deformable_field_ops::{
    compute_gradient_into, gaussian_smooth_with_scratch, normalize_forces_into,
    scaling_and_squaring_into, warp_image_into, VelocityField,
};
use crate::error::RegistrationError;

use super::buffers::BSplineSyNBuffers;
use super::primitives::{accumulate_to_cp_into, cp_count, cp_laplacian_into, evaluate_dense_into};
use super::BSplineSyNConfig;
use super::BSplineSyNRegistration;
use super::BSplineSyNResult;
use crate::diffeomorphic::local_cc::{cc_forces_into, mean_local_cc};

impl BSplineSyNRegistration {
    /// Create a registration instance with the given configuration.
    pub fn new(config: BSplineSyNConfig) -> Self {
        Self { config }
    }

    /// Register `moving` to `fixed` using BSplineSyN with local CC metric.
    ///
    /// # Arguments
    /// - `fixed` — reference image, flat `[f32]` in Z-major order.
    /// - `moving` — moving image, same shape as `fixed`.
    /// - `dims` — `[nz, ny, nx]`.
    /// - `spacing` — physical voxel spacing `[sz, sy, sx]`.
    ///
    /// # Errors
    /// Returns [`RegistrationError`] on dimension mismatch or invalid config.
    pub fn register(
        &self,
        fixed: &[f32],
        moving: &[f32],
        dims: [usize; 3],
        spacing: [f64; 3],
    ) -> Result<BSplineSyNResult, RegistrationError> {
        let [nz, ny, nx] = dims;
        let n = nz * ny * nx;

        if fixed.len() != n {
            return Err(RegistrationError::DimensionMismatch(format!(
                "fixed length {} != dims product {}",
                fixed.len(),
                n
            )));
        }
        if moving.len() != n {
            return Err(RegistrationError::DimensionMismatch(format!(
                "moving length {} != dims product {}",
                moving.len(),
                n
            )));
        }
        for d in 0..3 {
            if self.config.control_spacing[d] == 0 {
                return Err(RegistrationError::InvalidConfiguration(format!(
                    "control_spacing[{}] must be >= 1",
                    d
                )));
            }
        }

        let cs = self.config.control_spacing;
        let cp_d = [
            cp_count(nz, cs[0]),
            cp_count(ny, cs[1]),
            cp_count(nx, cs[2]),
        ];
        let cp_n = cp_d[0] * cp_d[1] * cp_d[2];

        let mut buf = BSplineSyNBuffers::new(n, cp_n);

        let mut cc_history: VecDeque<f64> = VecDeque::new();
        let mut final_cc = 0.0_f64;
        let mut iter = 0usize;

        let r = self.config.cc_window_radius;
        let rw = self.config.regularization_weight as f32;
        let sigma = self.config.sigma_smooth;

        for it in 0..self.config.max_iterations {
            iter = it + 1;

            // Evaluate dense velocity fields from control points (zero alloc)
            evaluate_dense_into(&buf.cp1z, cp_d, dims, cs, &mut buf.v1z);
            evaluate_dense_into(&buf.cp1y, cp_d, dims, cs, &mut buf.v1y);
            evaluate_dense_into(&buf.cp1x, cp_d, dims, cs, &mut buf.v1x);
            evaluate_dense_into(&buf.cp2z, cp_d, dims, cs, &mut buf.v2z);
            evaluate_dense_into(&buf.cp2y, cp_d, dims, cs, &mut buf.v2y);
            evaluate_dense_into(&buf.cp2x, cp_d, dims, cs, &mut buf.v2x);

            // exp(v) via scaling-and-squaring (zero alloc, shared scratch)
            scaling_and_squaring_into(
                &buf.v1z,
                &buf.v1y,
                &buf.v1x,
                dims,
                self.config.n_squarings,
                &mut buf.phi1_z,
                &mut buf.phi1_y,
                &mut buf.phi1_x,
                &mut buf.scratch_ss_z,
                &mut buf.scratch_ss_y,
                &mut buf.scratch_ss_x,
            );
            scaling_and_squaring_into(
                &buf.v2z,
                &buf.v2y,
                &buf.v2x,
                dims,
                self.config.n_squarings,
                &mut buf.phi2_z,
                &mut buf.phi2_y,
                &mut buf.phi2_x,
                &mut buf.scratch_ss_z,
                &mut buf.scratch_ss_y,
                &mut buf.scratch_ss_x,
            );

            // Warp images (zero alloc)
            warp_image_into(
                fixed,
                dims,
                &buf.phi1_z,
                &buf.phi1_y,
                &buf.phi1_x,
                &mut buf.i_w,
            );
            warp_image_into(
                moving,
                dims,
                &buf.phi2_z,
                &buf.phi2_y,
                &buf.phi2_x,
                &mut buf.j_w,
            );

            // Compute gradients (zero alloc)
            compute_gradient_into(
                &buf.i_w,
                dims,
                spacing,
                &mut buf.gi_z,
                &mut buf.gi_y,
                &mut buf.gi_x,
            );
            compute_gradient_into(
                &buf.j_w,
                dims,
                spacing,
                &mut buf.gj_z,
                &mut buf.gj_y,
                &mut buf.gj_x,
            );

            // CC forces (zero alloc)
            cc_forces_into(
                &buf.i_w,
                &buf.j_w,
                &buf.gi_z,
                &buf.gi_y,
                &buf.gi_x,
                dims,
                r,
                &mut buf.u1z,
                &mut buf.u1y,
                &mut buf.u1x,
            );
            cc_forces_into(
                &buf.j_w,
                &buf.i_w,
                &buf.gj_z,
                &buf.gj_y,
                &buf.gj_x,
                dims,
                r,
                &mut buf.u2z,
                &mut buf.u2y,
                &mut buf.u2x,
            );

            // Normalise forces so max|u₁| = max|u₂| = gradient_step
            normalize_forces_into(
                &mut buf.u1z,
                &mut buf.u1y,
                &mut buf.u1x,
                &mut buf.u2z,
                &mut buf.u2y,
                &mut buf.u2x,
                self.config.gradient_step,
            );

            // Gaussian smooth forces (zero alloc, shared scratch)
            if sigma > 0.0 {
                gaussian_smooth_with_scratch(&mut buf.u1z, dims, sigma, &mut buf.smooth_tmp);
                gaussian_smooth_with_scratch(&mut buf.u1y, dims, sigma, &mut buf.smooth_tmp);
                gaussian_smooth_with_scratch(&mut buf.u1x, dims, sigma, &mut buf.smooth_tmp);
                gaussian_smooth_with_scratch(&mut buf.u2z, dims, sigma, &mut buf.smooth_tmp);
                gaussian_smooth_with_scratch(&mut buf.u2y, dims, sigma, &mut buf.smooth_tmp);
                gaussian_smooth_with_scratch(&mut buf.u2x, dims, sigma, &mut buf.smooth_tmp);
            }

            // Accumulate forces into CP-space (zero alloc, shared accum/weight)
            accumulate_to_cp_into(
                &buf.u1z,
                dims,
                cp_d,
                cs,
                &mut buf.cp_accum,
                &mut buf.cp_weight,
                &mut buf.d1z,
            );
            accumulate_to_cp_into(
                &buf.u1y,
                dims,
                cp_d,
                cs,
                &mut buf.cp_accum,
                &mut buf.cp_weight,
                &mut buf.d1y,
            );
            accumulate_to_cp_into(
                &buf.u1x,
                dims,
                cp_d,
                cs,
                &mut buf.cp_accum,
                &mut buf.cp_weight,
                &mut buf.d1x,
            );
            accumulate_to_cp_into(
                &buf.u2z,
                dims,
                cp_d,
                cs,
                &mut buf.cp_accum,
                &mut buf.cp_weight,
                &mut buf.d2z,
            );
            accumulate_to_cp_into(
                &buf.u2y,
                dims,
                cp_d,
                cs,
                &mut buf.cp_accum,
                &mut buf.cp_weight,
                &mut buf.d2y,
            );
            accumulate_to_cp_into(
                &buf.u2x,
                dims,
                cp_d,
                cs,
                &mut buf.cp_accum,
                &mut buf.cp_weight,
                &mut buf.d2x,
            );

            // CP Laplacian regularisation (zero alloc)
            cp_laplacian_into(&buf.cp1z, cp_d, &mut buf.l1z);
            cp_laplacian_into(&buf.cp1y, cp_d, &mut buf.l1y);
            cp_laplacian_into(&buf.cp1x, cp_d, &mut buf.l1x);
            cp_laplacian_into(&buf.cp2z, cp_d, &mut buf.l2z);
            cp_laplacian_into(&buf.cp2y, cp_d, &mut buf.l2y);
            cp_laplacian_into(&buf.cp2x, cp_d, &mut buf.l2x);

            // Update control points
            for i in 0..cp_n {
                buf.cp1z[i] += buf.d1z[i] + rw * buf.l1z[i];
                buf.cp1y[i] += buf.d1y[i] + rw * buf.l1y[i];
                buf.cp1x[i] += buf.d1x[i] + rw * buf.l1x[i];
                buf.cp2z[i] += buf.d2z[i] + rw * buf.l2z[i];
                buf.cp2y[i] += buf.d2y[i] + rw * buf.l2y[i];
                buf.cp2x[i] += buf.d2x[i] + rw * buf.l2x[i];
            }

            final_cc = mean_local_cc(&buf.i_w, &buf.j_w, dims, r);
            cc_history.push_back(final_cc);
            if cc_history.len() > self.config.convergence_window {
                cc_history.pop_front();
            }
            if cc_history.len() == self.config.convergence_window {
                let mu = cc_history.iter().sum::<f64>() / cc_history.len() as f64;
                let var = cc_history.iter().map(|&v| (v - mu).powi(2)).sum::<f64>()
                    / cc_history.len() as f64;
                if var < self.config.convergence_threshold {
                    break;
                }
            }
        }

        // ── Final dense fields and warps (reusing pre-allocated scratch) ──
        evaluate_dense_into(&buf.cp1z, cp_d, dims, cs, &mut buf.v1z);
        evaluate_dense_into(&buf.cp1y, cp_d, dims, cs, &mut buf.v1y);
        evaluate_dense_into(&buf.cp1x, cp_d, dims, cs, &mut buf.v1x);
        evaluate_dense_into(&buf.cp2z, cp_d, dims, cs, &mut buf.v2z);
        evaluate_dense_into(&buf.cp2y, cp_d, dims, cs, &mut buf.v2y);
        evaluate_dense_into(&buf.cp2x, cp_d, dims, cs, &mut buf.v2x);

        scaling_and_squaring_into(
            &buf.v1z,
            &buf.v1y,
            &buf.v1x,
            dims,
            self.config.n_squarings,
            &mut buf.phi1_z,
            &mut buf.phi1_y,
            &mut buf.phi1_x,
            &mut buf.scratch_ss_z,
            &mut buf.scratch_ss_y,
            &mut buf.scratch_ss_x,
        );
        scaling_and_squaring_into(
            &buf.v2z,
            &buf.v2y,
            &buf.v2x,
            dims,
            self.config.n_squarings,
            &mut buf.phi2_z,
            &mut buf.phi2_y,
            &mut buf.phi2_x,
            &mut buf.scratch_ss_z,
            &mut buf.scratch_ss_y,
            &mut buf.scratch_ss_x,
        );

        warp_image_into(
            fixed,
            dims,
            &buf.phi1_z,
            &buf.phi1_y,
            &buf.phi1_x,
            &mut buf.i_w,
        );
        warp_image_into(
            moving,
            dims,
            &buf.phi2_z,
            &buf.phi2_y,
            &buf.phi2_x,
            &mut buf.j_w,
        );

        Ok(BSplineSyNResult {
            forward_field: VelocityField::new(buf.v1z, buf.v1y, buf.v1x),
            inverse_field: VelocityField::new(buf.v2z, buf.v2y, buf.v2x),
            warped_fixed: buf.i_w,
            warped_moving: buf.j_w,
            final_cc,
            num_iterations: iter,
        })
    }
}
