//! Threshold that maximizes the number of connected components.
//!
//! # Mathematical Specification
//!
//! Ports `itk::ThresholdMaximumConnectedComponentsImageFilter`. It binary-
//! searches the lower threshold `T` that maximizes the number of connected
//! components (of size ≥ `minimum_object_size`) in the binary mask
//! `T ≤ I ≤ upper_boundary`, then outputs that mask (`inside_value` /
//! `outside_value`).
//!
//! The search follows ITK exactly, in integer pixel arithmetic:
//!
//! ```text
//! lo = min I,  hi = min(max I, upper_boundary)
//! mid = (hi − lo) / 2
//! while hi − lo > 2:
//!   midL = lo + (mid − lo)/2,  midR = hi − (hi − mid)/2
//!   if count(midR) > count(midL):  lo = mid; mid = midR
//!   else:                          hi = mid; mid = midL
//! T = mid
//! ```
//!
//! where `count(t)` is the number of connected components (face connectivity,
//! ITK default `FullyConnected = false`) of size ≥ `minimum_object_size` in the
//! mask thresholded at `t`. Because ritk's [`connected_components`] is bit-exact
//! to `sitk.ConnectedComponent`, the chosen threshold — and thus the binary
//! output — is bit-exact to `sitk.ThresholdMaximumConnectedComponents`.

use std::collections::HashMap;

use burn::tensor::backend::Backend;
use ritk_image::Image;
use ritk_tensor_ops::{extract_vec_infallible, rebuild};

use super::{connected_components, Connectivity};

/// Threshold-maximum-connected-components filter
/// (`itk::ThresholdMaximumConnectedComponentsImageFilter`).
#[derive(Debug, Clone, Copy)]
pub struct ThresholdMaximumConnectedComponentsFilter {
    /// Minimum component size (in pixels) counted as an object. ITK default `0`.
    pub minimum_object_size: usize,
    /// Upper threshold bound; `None` uses the image maximum (ITK default).
    pub upper_boundary: Option<i64>,
    /// Value written inside the selected threshold band. ITK default `1`.
    pub inside_value: f32,
    /// Value written outside. ITK default `0`.
    pub outside_value: f32,
}

impl Default for ThresholdMaximumConnectedComponentsFilter {
    fn default() -> Self {
        Self {
            minimum_object_size: 0,
            upper_boundary: None,
            inside_value: 1.0,
            outside_value: 0.0,
        }
    }
}

impl ThresholdMaximumConnectedComponentsFilter {
    /// Construct with a minimum object size (other fields default).
    pub fn new(minimum_object_size: usize) -> Self {
        Self {
            minimum_object_size,
            ..Self::default()
        }
    }

    /// Find the component-maximizing threshold and return the binary mask.
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> Image<B, 3> {
        let (vals, dims) = extract_vec_infallible(image);

        let mut img_min = i64::MAX;
        let mut img_max = i64::MIN;
        for &v in &vals {
            let iv = v.round() as i64;
            img_min = img_min.min(iv);
            img_max = img_max.max(iv);
        }
        // The threshold band ceiling is fixed at `m_UpperBoundary` (clamped to
        // the image max, since no pixel exceeds it). The bisection's `lo`/`hi`
        // search bounds both move; the mask ceiling stays `mask_upper`.
        let mask_upper = self.upper_boundary.map_or(img_max, |ub| img_max.min(ub));

        // count(t): connected components of size ≥ minimum_object_size in the
        // mask `t ≤ I ≤ mask_upper` (face connectivity).
        let count = |t: i64| -> usize {
            let mask: Vec<f32> = vals
                .iter()
                .map(|&v| {
                    let iv = v.round() as i64;
                    f32::from(iv >= t && iv <= mask_upper)
                })
                .collect();
            let mask_img = rebuild(mask, dims, image);
            let (labels_img, _) = connected_components(&mask_img, Connectivity::Six);
            let (labels, _) = extract_vec_infallible(&labels_img);
            let mut hist: HashMap<i64, usize> = HashMap::new();
            for &l in &labels {
                if l != 0.0 {
                    *hist.entry(l.round() as i64).or_insert(0) += 1;
                }
            }
            hist.values()
                .filter(|&&c| c >= self.minimum_object_size)
                .count()
        };

        let mut lo = img_min;
        let mut hi = mask_upper;
        let mut mid = (hi - lo) / 2;
        while hi - lo > 2 {
            let midl = lo + (mid - lo) / 2;
            let midr = hi - (hi - mid) / 2;
            if count(midr) > count(midl) {
                lo = mid;
                mid = midr;
            } else {
                hi = mid;
                mid = midl;
            }
        }
        let threshold = mid;

        let (inside, outside) = (self.inside_value, self.outside_value);
        let out: Vec<f32> = vals
            .iter()
            .map(|&v| {
                let iv = v.round() as i64;
                if iv >= threshold && iv <= mask_upper {
                    inside
                } else {
                    outside
                }
            })
            .collect();
        rebuild(out, dims, image)
    }
}

#[cfg(test)]
#[path = "tests_threshold_max_cc.rs"]
mod tests_threshold_max_cc;
