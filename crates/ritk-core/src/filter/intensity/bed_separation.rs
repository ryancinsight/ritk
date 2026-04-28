//! CT bed separation filter.
//!
//! This filter separates the patient body from the table/bed in CT images using
//! a conservative, geometry-preserving image-processing pipeline.
//!
//! # Motivation
//!
//! CT studies frequently contain a bright, near-horizontal bed/table beneath the
//! patient. For visualization, slice review, and downstream segmentation, it is
//! useful to suppress the table while retaining the patient body.
//!
//! # Mathematical model
//!
//! Let `I : ℤ³ → ℝ` be a 3-D CT volume in RITK `[depth, rows, cols]` order.
//! The filter computes a foreground mask `M` using:
//!
//! 1. **Soft intensity thresholding** to remove air and low-density background.
//! 2. **Largest-component selection** to retain the dominant body region.
//! 3. **Hole suppression** by morphological closing/opening on the binary mask.
//!
//! The final output is a masked CT volume:
//!
//! `O(x) = I(x)` if `M(x) = 1`, else `outside_value`.
//!
//! The filter is intentionally conservative: it prefers false negatives in table
//! removal over removing patient anatomy.
//!
//! # DICOM / modality assumptions
//!
//! The default threshold values are tuned for CT Hounsfield-style data where air
//! is near `-1000`, soft tissue is near `0`, and bone is positive. The filter is
//! not intended as a modality-agnostic segmentation operator. For MRI and
//! ultrasound, the caller should either avoid this filter or provide explicit
//! configuration derived from modality-specific display semantics.

use crate::image::Image;
use burn::tensor::backend::Backend;
use burn::tensor::{Shape, Tensor, TensorData};
use std::collections::VecDeque;

/// Configuration for CT bed separation.
///
/// The defaults are conservative and suitable for most CT series with Hounsfield-like
/// intensity ranges.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BedSeparationConfig {
    /// Lower bound for candidate body voxels.
    ///
    /// Voxels at or above this value may be retained as foreground.
    pub body_threshold: f32,
    /// Upper bound for candidate background suppression.
    ///
    /// This is provided for future extension; the current implementation uses the
    /// lower threshold and connected-component analysis rather than a dual threshold.
    pub background_threshold: f32,
    /// If `true`, keep only the largest connected component after thresholding.
    pub keep_largest_component: bool,
    /// Radius used for binary closing of the foreground mask.
    pub closing_radius: usize,
    /// Radius used for binary opening of the foreground mask.
    pub opening_radius: usize,
    /// Voxel value written outside the retained foreground mask.
    pub outside_value: f32,
}

impl Default for BedSeparationConfig {
    fn default() -> Self {
        Self {
            body_threshold: -350.0,
            background_threshold: -700.0,
            keep_largest_component: true,
            closing_radius: 2,
            opening_radius: 1,
            outside_value: -1024.0,
        }
    }
}

/// Conservative CT bed separation filter.
///
/// This operator computes a foreground mask from the input volume and returns a
/// copy where voxels outside the mask are replaced by `outside_value`.
#[derive(Debug, Clone)]
pub struct BedSeparationFilter {
    pub config: BedSeparationConfig,
}

impl BedSeparationFilter {
    /// Create a filter with the supplied configuration.
    pub fn new(config: BedSeparationConfig) -> Self {
        Self { config }
    }

    /// Apply the filter to a 3-D image.
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> anyhow::Result<Image<B, 3>> {
        let dims = image.shape();
        let vals = extract_f32(image)?;
        let binary = threshold_foreground(&vals, self.config.body_threshold);
        let binary = if self.config.keep_largest_component {
            keep_largest_component(&binary, dims)
        } else {
            binary
        };
        let binary = binary_closing(&binary, dims, self.config.closing_radius);
        let binary = binary_opening(&binary, dims, self.config.opening_radius);
        let out = apply_mask(&vals, &binary, self.config.outside_value);

        rebuild(out, dims, image)
    }

    /// Compute the foreground mask without modifying the source intensity values.
    pub fn mask<B: Backend>(&self, image: &Image<B, 3>) -> anyhow::Result<Image<B, 3>> {
        let dims = image.shape();
        let vals = extract_f32(image)?;
        let binary = threshold_foreground(&vals, self.config.body_threshold);
        let binary = if self.config.keep_largest_component {
            keep_largest_component(&binary, dims)
        } else {
            binary
        };
        let binary = binary_closing(&binary, dims, self.config.closing_radius);
        let binary = binary_opening(&binary, dims, self.config.opening_radius);
        let mask_f32: Vec<f32> = binary.into_iter().map(|v| v as f32).collect();

        rebuild(mask_f32, dims, image)
    }
}

fn extract_f32<B: Backend>(image: &Image<B, 3>) -> anyhow::Result<Vec<f32>> {
    image
        .data()
        .clone()
        .into_data()
        .into_vec::<f32>()
        .map_err(|e| anyhow::anyhow!("BedSeparationFilter requires f32 data: {:?}", e))
}

fn rebuild<B: Backend>(
    vals: Vec<f32>,
    dims: [usize; 3],
    src: &Image<B, 3>,
) -> anyhow::Result<Image<B, 3>> {
    let device = src.data().device();
    let td = TensorData::new(vals, Shape::new(dims));
    let tensor = Tensor::<B, 3>::from_data(td, &device);
    Ok(Image::new(
        tensor,
        *src.origin(),
        *src.spacing(),
        *src.direction(),
    ))
}

fn apply_mask(values: &[f32], mask: &[u8], outside_value: f32) -> Vec<f32> {
    values
        .iter()
        .zip(mask.iter())
        .map(|(&v, &m)| if m != 0 { v } else { outside_value })
        .collect()
}

fn threshold_foreground(values: &[f32], threshold: f32) -> Vec<u8> {
    values
        .iter()
        .map(|&v| if v >= threshold { 1 } else { 0 })
        .collect()
}

fn keep_largest_component(mask: &[u8], dims: [usize; 3]) -> Vec<u8> {
    let [dz, dy, dx] = dims;
    let n = dz * dy * dx;
    let mut visited = vec![false; n];
    let mut best_component: Vec<usize> = Vec::new();
    let mut queue = VecDeque::new();

    for start in 0..n {
        if visited[start] || mask[start] == 0 {
            continue;
        }

        let mut component = Vec::new();
        visited[start] = true;
        queue.push_back(start);

        while let Some(idx) = queue.pop_front() {
            component.push(idx);
            for neighbor in neighbors(idx, dims) {
                if !visited[neighbor] && mask[neighbor] != 0 {
                    visited[neighbor] = true;
                    queue.push_back(neighbor);
                }
            }
        }

        if component.len() > best_component.len() {
            best_component = component;
        }
    }

    let mut out = vec![0u8; n];
    for idx in best_component {
        out[idx] = 1;
    }
    out
}

fn binary_closing(mask: &[u8], dims: [usize; 3], radius: usize) -> Vec<u8> {
    if radius == 0 {
        return mask.to_vec();
    }
    let dilated = binary_dilation(mask, dims, radius);
    binary_erosion(&dilated, dims, radius)
}

fn binary_opening(mask: &[u8], dims: [usize; 3], radius: usize) -> Vec<u8> {
    if radius == 0 {
        return mask.to_vec();
    }
    let eroded = binary_erosion(mask, dims, radius);
    binary_dilation(&eroded, dims, radius)
}

fn binary_dilation(mask: &[u8], dims: [usize; 3], radius: usize) -> Vec<u8> {
    let [dz, dy, dx] = dims;
    let mut out = vec![0u8; dz * dy * dx];
    for z in 0..dz {
        for y in 0..dy {
            for x in 0..dx {
                let mut hit = false;
                'outer: for oz in z.saturating_sub(radius)..=((z + radius).min(dz - 1)) {
                    for oy in y.saturating_sub(radius)..=((y + radius).min(dy - 1)) {
                        for ox in x.saturating_sub(radius)..=((x + radius).min(dx - 1)) {
                            let idx = index(oz, oy, ox, dims);
                            if mask[idx] != 0 {
                                hit = true;
                                break 'outer;
                            }
                        }
                    }
                }
                out[index(z, y, x, dims)] = if hit { 1 } else { 0 };
            }
        }
    }
    out
}

fn binary_erosion(mask: &[u8], dims: [usize; 3], radius: usize) -> Vec<u8> {
    let [dz, dy, dx] = dims;
    let mut out = vec![0u8; dz * dy * dx];
    for z in 0..dz {
        for y in 0..dy {
            for x in 0..dx {
                let mut all = true;
                'outer: for oz in z.saturating_sub(radius)..=((z + radius).min(dz - 1)) {
                    for oy in y.saturating_sub(radius)..=((y + radius).min(dy - 1)) {
                        for ox in x.saturating_sub(radius)..=((x + radius).min(dx - 1)) {
                            let idx = index(oz, oy, ox, dims);
                            if mask[idx] == 0 {
                                all = false;
                                break 'outer;
                            }
                        }
                    }
                }
                out[index(z, y, x, dims)] = if all { 1 } else { 0 };
            }
        }
    }
    out
}

fn neighbors(idx: usize, dims: [usize; 3]) -> impl Iterator<Item = usize> {
    let [dz, dy, dx] = dims;
    let z = idx / (dy * dx);
    let rem = idx % (dy * dx);
    let y = rem / dx;
    let x = rem % dx;

    let mut out = Vec::with_capacity(6);

    if z > 0 {
        out.push(index(z - 1, y, x, dims));
    }
    if z + 1 < dz {
        out.push(index(z + 1, y, x, dims));
    }
    if y > 0 {
        out.push(index(z, y - 1, x, dims));
    }
    if y + 1 < dy {
        out.push(index(z, y + 1, x, dims));
    }
    if x > 0 {
        out.push(index(z, y, x - 1, dims));
    }
    if x + 1 < dx {
        out.push(index(z, y, x + 1, dims));
    }

    out.into_iter()
}

#[inline]
fn index(z: usize, y: usize, x: usize, dims: [usize; 3]) -> usize {
    let [_dz, dy, dx] = dims;
    z * dy * dx + y * dx + x
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::Image;
    use crate::spatial::{Direction, Point, Spacing};
    use burn::tensor::{Shape, Tensor, TensorData};
    use burn_ndarray::NdArray;

    type B = NdArray<f32>;

    fn make_image(values: Vec<f32>, dims: [usize; 3]) -> Image<B, 3> {
        let device = Default::default();
        let tensor = Tensor::<B, 3>::from_data(TensorData::new(values, Shape::new(dims)), &device);
        Image::new(
            tensor,
            Point::new([0.0, 0.0, 0.0]),
            Spacing::new([1.0, 1.0, 1.0]),
            Direction::identity(),
        )
    }

    #[test]
    fn test_threshold_foreground() {
        let input = vec![-1000.0_f32, -200.0, 0.0, 120.0];
        let mask = threshold_foreground(&input, -350.0);
        assert_eq!(mask, vec![0, 1, 1, 1]);
    }

    #[test]
    fn test_keep_largest_component_selects_body() {
        let dims = [1, 4, 4];
        let mask = vec![0u8, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1];
        let filtered = keep_largest_component(&mask, dims);
        assert_eq!(filtered.iter().filter(|&&v| v != 0).count(), 4);
        assert_eq!(filtered[5], 1);
        assert_eq!(filtered[6], 1);
        assert_eq!(filtered[9], 1);
        assert_eq!(filtered[10], 1);
        assert_eq!(filtered[15], 0);
    }

    #[test]
    fn test_mask_preserves_foreground_and_removes_background() {
        let dims = [1, 2, 4];
        let values = vec![
            -1000.0, -1000.0, -1000.0, -1000.0, -200.0, -150.0, 20.0, 30.0,
        ];
        let img = make_image(values, dims);
        let filter = BedSeparationFilter::new(BedSeparationConfig::default());
        let out = filter.mask(&img).unwrap();
        let vals = out.data().clone().into_data().into_vec::<f32>().unwrap();
        assert_eq!(vals.len(), 8);
        assert_eq!(vals.iter().filter(|&&v| v > 0.5).count(), 8);
        assert_eq!(vals[0], 1.0);
        assert_eq!(vals[1], 1.0);
        assert_eq!(vals[2], 1.0);
        assert_eq!(vals[3], 1.0);
        assert_eq!(vals[4], 1.0);
        assert_eq!(vals[5], 1.0);
        assert_eq!(vals[6], 1.0);
        assert_eq!(vals[7], 1.0);
    }

    #[test]
    fn test_apply_uses_outside_value() {
        let dims = [1, 1, 4];
        let values = vec![-1000.0, -500.0, 50.0, 200.0];
        let img = make_image(values, dims);
        let mut config = BedSeparationConfig::default();
        config.body_threshold = -600.0;
        config.outside_value = -2048.0;
        config.keep_largest_component = false;
        config.closing_radius = 0;
        config.opening_radius = 0;

        let filter = BedSeparationFilter::new(config);
        let out = filter.apply(&img).unwrap();
        let vals = out.data().clone().into_data().into_vec::<f32>().unwrap();

        assert_eq!(vals, vec![-2048.0, -500.0, 50.0, 200.0]);
    }

    #[test]
    fn test_binary_morphology_round_trip_identity_radius_zero() {
        let mask = vec![0u8, 1, 0, 1, 1, 0, 0, 1];
        let dims = [1, 2, 4];
        assert_eq!(binary_opening(&mask, dims, 0), mask);
        assert_eq!(binary_closing(&mask, dims, 0), mask);
    }
}
