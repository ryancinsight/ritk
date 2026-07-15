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

use ritk_image::tensor::Backend;
use ritk_image::Image;
use ritk_tensor_ops::{extract_vec_infallible, rebuild};

use super::{connected_components_values, Connectivity};

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
        rebuild(
            threshold_max_cc_values(
                &vals,
                dims,
                self.minimum_object_size,
                self.upper_boundary,
                self.inside_value,
                self.outside_value,
            ),
            dims,
            image,
        )
    }

    /// Find the component-maximizing threshold and return a Coeus-native mask.
    ///
    /// # Errors
    ///
    /// Returns an error when the image tensor is not host-addressable/contiguous
    /// or the native output image cannot be constructed.
    pub fn apply_native<B>(
        &self,
        image: &ritk_image::native::Image<f32, B, 3>,
        backend: &B,
    ) -> anyhow::Result<ritk_image::native::Image<f32, B, 3>>
    where
        B: coeus_core::ComputeBackend,
        B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    {
        crate::native_support::map_flat_image(image, backend, |vals, dims| {
            threshold_max_cc_values(
                vals,
                dims,
                self.minimum_object_size,
                self.upper_boundary,
                self.inside_value,
                self.outside_value,
            )
        })
    }
}

fn threshold_max_cc_values(
    vals: &[f32],
    dims: [usize; 3],
    minimum_object_size: usize,
    upper_boundary: Option<i64>,
    inside_value: f32,
    outside_value: f32,
) -> Vec<f32> {
    let mut img_min = i64::MAX;
    let mut img_max = i64::MIN;
    for &v in vals {
        let iv = v.round() as i64;
        img_min = img_min.min(iv);
        img_max = img_max.max(iv);
    }
    let mask_upper = upper_boundary.map_or(img_max, |ub| img_max.min(ub));

    let count = |t: i64| -> usize {
        let mask: Vec<f32> = vals
            .iter()
            .map(|&v| {
                let iv = v.round() as i64;
                f32::from(iv >= t && iv <= mask_upper)
            })
            .collect();
        let (labels, _) = connected_components_values(&mask, dims, Connectivity::Six, 0.0);
        let mut hist: HashMap<i64, usize> = HashMap::new();
        for &label in &labels {
            if label != 0.0 {
                *hist.entry(label.round() as i64).or_insert(0) += 1;
            }
        }
        hist.values()
            .filter(|&&count| count >= minimum_object_size)
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

    vals.iter()
        .map(|&v| {
            let iv = v.round() as i64;
            if iv >= threshold && iv <= mask_upper {
                inside_value
            } else {
                outside_value
            }
        })
        .collect()
}

#[cfg(test)]
#[path = "tests_threshold_max_cc.rs"]
mod tests_threshold_max_cc;
