//! Isolated-connected region growing.
//!
//! # Mathematical Specification
//!
//! Ports `itk::IsolatedConnectedImageFilter`. Given two seeds, it binary-searches
//! the threshold that **just separates** them: the largest upper threshold
//! (`find_upper_threshold`) — or smallest lower threshold — for which a
//! connected-threshold region grown from `seed1` does **not** reach `seed2`.
//!
//! For `find_upper_threshold` (ITK default), with fixed band floor `lower`:
//!
//! ```text
//! lo = lower,  hi = upper,  guess = hi
//! while lo + tol < guess:
//!   region = flood(seed1, band = [lower, guess])
//!   if seed2 ∈ region:  hi = guess        # too permissive, lower the ceiling
//!   else:               lo = guess        # still separated, raise the floor
//!   guess = (hi + lo) / 2
//! isolated = lo
//! output = flood(seed1, [lower, isolated]) · replace_value
//! ```
//!
//! `find_upper_threshold = false` mirrors this, searching the lower threshold in
//! the band `[guess, upper]` and taking `isolated = hi`. The flood is ritk's
//! [`connected_threshold()`] (face connectivity), bit-exact to
//! `sitk.ConnectedThreshold`, so the separating threshold — and the binary output
//! — is bit-exact to `sitk.IsolatedConnected`.

use burn::tensor::backend::Backend;
use ritk_core::spatial::VoxelIndex;
use ritk_image::Image;
use ritk_tensor_ops::{extract_vec_infallible, rebuild};

use super::connected_threshold;

/// Isolated-connected region-growing filter (`itk::IsolatedConnectedImageFilter`).
#[derive(Debug, Clone, Copy)]
pub struct IsolatedConnectedFilter {
    /// Seed inside the region to keep, `[z, y, x]`.
    pub seed1: [usize; 3],
    /// Seed to isolate (must end up excluded), `[z, y, x]`.
    pub seed2: [usize; 3],
    /// Fixed band floor (`find_upper_threshold`) / search floor. ITK `Lower`.
    pub lower: f32,
    /// Fixed band ceiling (`!find_upper_threshold`) / search ceiling. ITK `Upper`.
    pub upper: f32,
    /// Value written to the grown region. ITK default `1`.
    pub replace_value: f32,
    /// Bisection stop tolerance. ITK default `1.0`.
    pub isolated_value_tolerance: f64,
    /// Search the upper threshold (ITK default `true`) vs the lower.
    pub find_upper_threshold: bool,
}

impl Default for IsolatedConnectedFilter {
    fn default() -> Self {
        Self {
            seed1: [0, 0, 0],
            seed2: [0, 0, 0],
            lower: 0.0,
            upper: 1.0,
            replace_value: 1.0,
            isolated_value_tolerance: 1.0,
            find_upper_threshold: true,
        }
    }
}

impl IsolatedConnectedFilter {
    /// Grow the region from `seed1` at the separating threshold.
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> Image<B, 3> {
        let [_, ny, nx] = image.shape();
        let s2 = (self.seed2[0] * ny + self.seed2[1]) * nx + self.seed2[2];
        let tol = self.isolated_value_tolerance;
        let seed1 = VoxelIndex::new(self.seed1[0], self.seed1[1], self.seed1[2]);

        // Flood from seed1 over [lo, hi]; returns whether seed2 was reached.
        let reaches_seed2 = |lo: f32, hi: f32| -> bool {
            let region = connected_threshold(image, seed1, lo, hi);
            extract_vec_infallible(&region).0[s2] != 0.0
        };

        let isolated = if self.find_upper_threshold {
            let mut lo = self.lower as f64;
            let mut hi = self.upper as f64;
            let mut guess = hi;
            while lo + tol < guess {
                if reaches_seed2(self.lower, guess as f32) {
                    hi = guess;
                } else {
                    lo = guess;
                }
                guess = (hi + lo) / 2.0;
            }
            lo as f32
        } else {
            let mut lo = self.lower as f64;
            let mut hi = self.upper as f64;
            let mut guess = lo;
            while guess < hi - tol {
                if reaches_seed2(guess as f32, self.upper) {
                    lo = guess;
                } else {
                    hi = guess;
                }
                guess = (hi + lo) / 2.0;
            }
            hi as f32
        };

        let (band_lo, band_hi) = if self.find_upper_threshold {
            (self.lower, isolated)
        } else {
            (isolated, self.upper)
        };
        let region = connected_threshold(image, seed1, band_lo, band_hi);
        let (mask, dims) = extract_vec_infallible(&region);
        let rv = self.replace_value;
        let out: Vec<f32> = mask
            .iter()
            .map(|&v| if v != 0.0 { rv } else { 0.0 })
            .collect();
        rebuild(out, dims, image)
    }
}

#[cfg(test)]
#[path = "tests_isolated_connected.rs"]
mod tests_isolated_connected;
