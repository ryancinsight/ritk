//! Multi-resolution Demons deformable image registration.
//!
//! Coarse-to-fine pyramid: at scale factor k = 2^l (level l):
//! 1. Gaussian pre-smooth with sigma = 0.5*k to prevent aliasing.
//! 2. Subsample with stride k -> coarse shape [nz/k, ny/k, nx/k].
//! 3. Run Demons for max(10, base_iterations / k) iterations.
//! 4. Upsample displacement via trilinear interpolation, scale by k.
//! 5. Warm-start the next finer level with the upsampled field.
//!
//! # References
//! - Thirion (1998), Med. Image Anal. 2(3):243-260.
//! - Vercauteren et al. (2009), NeuroImage 45(S1):S61-S72.

mod resample;

use super::config::{DemonsConfig, DemonsResult, DemonsVariant};
use super::diffeomorphic::DiffeomorphicDemonsRegistration;
use super::thirion::ThirionDemonsRegistration;
use crate::error::RegistrationError;
use resample::{downsample, upsample_displacement};

/// Configuration for multi-resolution Demons registration.
#[derive(Debug, Clone)]
pub struct MultiResDemonsConfig {
    /// Base configuration passed to each pyramid level.
    /// `max_iterations` is divided by the level shrink factor (min 10).
    pub base_config: DemonsConfig,
    /// Number of pyramid levels (>= 1). Level 0 = full resolution.
    /// Default: 3. With 3 levels, factors are [4, 2, 1].
    pub levels: usize,
    /// Demons variant used at each pyramid level.
    /// Default: [`DemonsVariant::Classic`] (Thirion Demons).
    pub variant: DemonsVariant,
    /// Number of scaling-and-squaring steps (only when variant=Diffeomorphic).
    pub n_squarings: usize,
}

impl Default for MultiResDemonsConfig {
    fn default() -> Self {
        Self {
            base_config: DemonsConfig::default(),
            levels: 3,
            variant: DemonsVariant::default(),
            n_squarings: 6,
        }
    }
}

impl MultiResDemonsConfig {
    /// Returns `true` if the variant is [`DemonsVariant::Diffeomorphic`].
    pub fn is_diffeomorphic(&self) -> bool {
        self.variant.is_diffeomorphic()
    }
}

/// Multi-resolution Demons registration (coarse-to-fine pyramid).
pub struct MultiResDemonsRegistration {
    pub config: MultiResDemonsConfig,
}

impl MultiResDemonsRegistration {
    pub fn new(config: MultiResDemonsConfig) -> Self {
        Self { config }
    }
    pub fn register(
        &self,
        fixed: &[f32],
        moving: &[f32],
        dims: [usize; 3],
        spacing: [f32; 3],
    ) -> Result<DemonsResult, RegistrationError> {
        let levels = self.config.levels.max(1);
        let mut prev_result: Option<DemonsResult> = None;
        let mut prev_cdims: [usize; 3] = [0; 3];

        for l in (0..levels).rev() {
            let factor = 1usize << l; // 2^l

            let (fixed_c, cdims) = downsample(fixed, dims, factor);
            let (moving_c, _) = downsample(moving, dims, factor);
            let coarse_spacing: [f64; 3] = [
                spacing[0] as f64 * factor as f64,
                spacing[1] as f64 * factor as f64,
                spacing[2] as f64 * factor as f64,
            ];
            let [ncz, ncy, ncx] = cdims;
            let cn = ncz * ncy * ncx;

            // Warm start: upsample displacement from previous coarser level.
            let (init_z, init_y, init_x) = if let Some(ref prev) = prev_result {
                let scale_z = cdims[0] as f32 / prev_cdims[0] as f32;
                let scale_y = cdims[1] as f32 / prev_cdims[1] as f32;
                let scale_x = cdims[2] as f32 / prev_cdims[2] as f32;
                let uz = upsample_displacement(&prev.disp_z, prev_cdims, cdims, scale_z);
                let uy = upsample_displacement(&prev.disp_y, prev_cdims, cdims, scale_y);
                let ux = upsample_displacement(&prev.disp_x, prev_cdims, cdims, scale_x);
                (uz, uy, ux)
            } else {
                (vec![0.0f32; cn], vec![0.0f32; cn], vec![0.0f32; cn])
            };

            // Pre-warp moving with init displacement to inject warm start.
            let warmed_moving = crate::deformable_field_ops::warp_image(
                &moving_c,
                cdims.into(),
                &init_z,
                &init_y,
                &init_x,
                crate::deformable_field_ops::WarpInterpolation::Trilinear,
            );

            // Reduce iterations proportionally to scale factor, minimum 10.
            let level_iters = (self.config.base_config.max_iterations / factor).max(10);
            let level_config = DemonsConfig {
                max_iterations: level_iters,
                ..self.config.base_config.clone()
            };

            // Run Demons on (fixed_c, warmed_moving).
            let level_result = match self.config.variant {
                DemonsVariant::Diffeomorphic => DiffeomorphicDemonsRegistration::with_squarings(
                    level_config,
                    self.config.n_squarings,
                )
                .register(&fixed_c, &warmed_moving, cdims, coarse_spacing)?,
                DemonsVariant::Classic => ThirionDemonsRegistration::new(level_config).register(
                    &fixed_c,
                    &warmed_moving,
                    cdims,
                    coarse_spacing,
                )?,
            };

            // First-order displacement composition: d_total = d_init + d_level.
            let total_z: Vec<f32> = init_z
                .iter()
                .zip(level_result.disp_z.iter())
                .map(|(a, b)| a + b)
                .collect();
            let total_y: Vec<f32> = init_y
                .iter()
                .zip(level_result.disp_y.iter())
                .map(|(a, b)| a + b)
                .collect();
            let total_x: Vec<f32> = init_x
                .iter()
                .zip(level_result.disp_x.iter())
                .map(|(a, b)| a + b)
                .collect();

            prev_result = Some(DemonsResult {
                warped: Vec::with_capacity(cn),
                disp_z: total_z,
                disp_y: total_y,
                disp_x: total_x,
                vel_z: None,
                vel_y: None,
                vel_x: None,
                final_mse: level_result.final_mse,
                num_iterations: level_result.num_iterations,
            });
            prev_cdims = cdims;
        }
        // levels >= 1 guarantees prev_result is Some.
        let final_result =
            prev_result.expect("at least one resolution level must succeed in multires demons");

        // l=0 ran at factor=1: displacement is already at full resolution.
        // Compute warped image and final MSE against the original moving image.
        let warped = crate::deformable_field_ops::warp_image(
            moving,
            dims.into(),
            &final_result.disp_z,
            &final_result.disp_y,
            &final_result.disp_x,
            crate::deformable_field_ops::WarpInterpolation::Trilinear,
        );
        let final_mse = crate::deformable_field_ops::compute_mse_streaming(
            fixed,
            moving,
            dims.into(),
            &final_result.disp_z,
            &final_result.disp_y,
            &final_result.disp_x,
        );

        Ok(DemonsResult {
            warped,
            disp_z: final_result.disp_z,
            disp_y: final_result.disp_y,
            disp_x: final_result.disp_x,
            vel_z: None,
            vel_y: None,
            vel_x: None,
            final_mse,
            num_iterations: final_result.num_iterations,
        })
    }
}

#[cfg(test)]
mod tests_multires;
