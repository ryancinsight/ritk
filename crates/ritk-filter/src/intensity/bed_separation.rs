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

use coeus_core::{ComputeBackend, CpuAddressableStorage};
use ritk_image::tensor::Backend;
use ritk_image::Image;
use ritk_tensor_ops::{extract_vec, rebuild};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Component retention policy for CT bed separation.
///
/// Selects which connected components are retained after intensity thresholding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum ComponentPolicy {
    /// Keep only the largest connected component (body) and discard all others.
    #[default]
    LargestOnly,
    /// Retain all components that pass the threshold.
    All,
}

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
    /// Component retention policy after thresholding.
    pub component_policy: ComponentPolicy,
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
            component_policy: ComponentPolicy::LargestOnly,
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
        let (vals, _) = extract_vec(image)?;
        Ok(rebuild(self.apply_values(&vals, dims), dims, image))
    }

    /// Apply bed separation to a Coeus-native image.
    pub fn apply_native<B>(
        &self,
        image: &ritk_image::native::Image<f32, B, 3>,
        backend: &B,
    ) -> anyhow::Result<ritk_image::native::Image<f32, B, 3>>
    where
        B: ComputeBackend,
        B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
    {
        ritk_image::native::Image::from_flat_on(
            self.apply_values(image.data_slice()?, image.shape()),
            image.shape(),
            *image.origin(),
            *image.spacing(),
            *image.direction(),
            backend,
        )
    }

    fn apply_values(&self, vals: &[f32], dims: [usize; 3]) -> Vec<f32> {
        let binary = threshold_foreground(vals, self.config.body_threshold);
        let binary = if self.config.component_policy == ComponentPolicy::LargestOnly {
            keep_largest_component(&binary, dims)
        } else {
            binary
        };
        let binary = binary_closing(&binary, dims, self.config.closing_radius);
        let binary = binary_opening(&binary, dims, self.config.opening_radius);
        apply_mask(vals, &binary, self.config.outside_value)
    }

    /// Compute the foreground mask without modifying the source intensity values.
    pub fn mask<B: Backend>(&self, image: &Image<B, 3>) -> anyhow::Result<Image<B, 3>> {
        let dims = image.shape();
        let (vals, _) = extract_vec(image)?;
        let binary = threshold_foreground(&vals, self.config.body_threshold);
        let binary = if self.config.component_policy == ComponentPolicy::LargestOnly {
            keep_largest_component(&binary, dims)
        } else {
            binary
        };
        let binary = binary_closing(&binary, dims, self.config.closing_radius);
        let binary = binary_opening(&binary, dims, self.config.opening_radius);
        let mask: Vec<f32> = binary.into_iter().map(|v| v as f32).collect();

        Ok(rebuild(mask, dims, image))
    }
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
    let mut best_component: Vec<usize> = Vec::with_capacity(n / 4);
    let mut queue = VecDeque::with_capacity(n / 16);

    for start in 0..n {
        if visited[start] || mask[start] == 0 {
            continue;
        }

        let mut component = Vec::with_capacity(n / 4);
        visited[start] = true;
        queue.push_back(start);

        while let Some(idx) = queue.pop_front() {
            component.push(idx);
            let (nbrs, nbr_count) = neighbors(idx, dims);
            for &neighbor in &nbrs[..nbr_count] {
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

#[inline]
fn neighbors(idx: usize, dims: [usize; 3]) -> ([usize; 6], usize) {
    let [dz, dy, dx] = dims;
    let z = idx / (dy * dx);
    let rem = idx % (dy * dx);
    let y = rem / dx;
    let x = rem % dx;
    let mut out = [0usize; 6];
    let mut count = 0;
    if z > 0 {
        out[count] = index(z - 1, y, x, dims);
        count += 1;
    }
    if z + 1 < dz {
        out[count] = index(z + 1, y, x, dims);
        count += 1;
    }
    if y > 0 {
        out[count] = index(z, y - 1, x, dims);
        count += 1;
    }
    if y + 1 < dy {
        out[count] = index(z, y + 1, x, dims);
        count += 1;
    }
    if x > 0 {
        out[count] = index(z, y, x - 1, dims);
        count += 1;
    }
    if x + 1 < dx {
        out[count] = index(z, y, x + 1, dims);
        count += 1;
    }
    (out, count)
}

#[inline]
fn index(z: usize, y: usize, x: usize, dims: [usize; 3]) -> usize {
    let [_dz, dy, dx] = dims;
    z * dy * dx + y * dx + x
}

#[cfg(test)]
#[path = "tests_bed_separation.rs"]
mod tests;
