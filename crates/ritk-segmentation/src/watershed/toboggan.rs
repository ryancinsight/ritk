//! Toboggan watershed labeling, matching `itk::TobogganImageFilter`.
//!
//! # Mathematical Specification
//!
//! Each voxel slides along a face-connected steepest-descent path to a local
//! minimum; voxels reaching the same minimum share a label.  Concretely, in
//! raster order, every still-unlabeled voxel begins a slide that repeatedly steps
//! to its strictly-smallest face neighbour until either (a) no strictly-smaller
//! neighbour exists (a local minimum, whose equal-or-lower plateau is then
//! flood-filled), or (b) the path enters an already-labeled region (whose label
//! the whole path adopts).  New minima receive consecutive labels starting at 2,
//! assigned in order of discovery.
//!
//! This is a deterministic, order-exact port: the neighbour scan order
//! (`+x, −x, +y, −y, +z, −z`), the strict-`<` slide rule, the LIFO plateau
//! flood-fill (`value ≤ seed`), and the raster discovery order all reproduce ITK
//! bit-for-bit.  Labels are emitted as `f32` (ritk's scalar convention).
//!
//! Face connectivity and the `z = 1` degenerate axis are handled naturally: an
//! out-of-image neighbour is simply skipped, so a 2-D (`z = 1`) volume reproduces
//! SimpleITK's 2-D `Toboggan` exactly.

use ritk_image::tensor::Backend;
use ritk_image::Image;
use ritk_tensor_ops::{extract_vec, rebuild};

/// Per-voxel processing state of the output buffer during labeling.
const UNLABELED: u32 = 0;
const IN_PROGRESS: u32 = 1;

/// Toboggan steepest-descent watershed filter.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct TobogganFilter;

impl TobogganFilter {
    /// Create a Toboggan filter.
    pub fn new() -> Self {
        Self
    }

    /// Label the steepest-descent basins of a legacy image.
    ///
    /// # Errors
    ///
    /// Returns an error for inaccessible storage, zero/overflowing shape,
    /// storage-cardinality mismatch, non-finite relief, or a volume whose
    /// possible labels exceed exact `f32` representation.
    pub fn apply<B: Backend>(&self, image: &Image<f32, B, 3>) -> anyhow::Result<Image<f32, B, 3>> {
        let (input, dimensions) = extract_vec(image)?;
        validate_relief(&input, dimensions)?;
        Ok(rebuild(
            toboggan_values(&input, dimensions),
            dimensions,
            image,
        ))
    }

    /// Label the steepest-descent basins of a Coeus-native image.
    ///
    /// # Errors
    ///
    /// Returns the validation errors documented by [`Self::apply`], or a
    /// backend storage/output construction error.
    pub fn apply_native<B>(
        &self,
        image: &ritk_image::Image<f32, B, 3>,
        backend: &B,
    ) -> anyhow::Result<ritk_image::Image<f32, B, 3>>
    where
        B: coeus_core::ComputeBackend,
        B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    {
        let values = image.data_slice()?;
        validate_relief(values, image.shape())?;
        crate::native_output::from_values(image, toboggan_values(values, image.shape()), backend)
    }
}

const MAX_SAMPLE_COUNT: usize = (1usize << f32::MANTISSA_DIGITS) - 1;

fn validate_relief(values: &[f32], dimensions: [usize; 3]) -> anyhow::Result<()> {
    anyhow::ensure!(
        dimensions.iter().all(|&extent| extent > 0),
        "Toboggan requires nonzero dimensions, got {dimensions:?}"
    );
    let expected = dimensions
        .iter()
        .try_fold(1usize, |count, &extent| count.checked_mul(extent))
        .ok_or_else(|| anyhow::anyhow!("Toboggan shape product overflows usize: {dimensions:?}"))?;
    anyhow::ensure!(
        expected <= MAX_SAMPLE_COUNT,
        "Toboggan supports at most {MAX_SAMPLE_COUNT} samples for exact f32 labels starting at 2, got {expected}"
    );
    anyhow::ensure!(
        values.len() == expected,
        "Toboggan shape {dimensions:?} requires {expected} samples, got {}",
        values.len()
    );
    if let Some((index, value)) = values
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        anyhow::bail!("Toboggan relief at flat index {index} must be finite, got {value}");
    }
    Ok(())
}

/// Canonical flat Toboggan core after [`validate_relief`].
fn toboggan_values(input: &[f32], dims: [usize; 3]) -> Vec<f32> {
    let [zn, yn, xn] = dims;
    let strides = [(yn * xn) as isize, xn as isize, 1isize];
    let n = input.len();

    // Output classes (0 = unlabeled, 1 = in-progress, ≥2 = final label).
    let mut class = vec![UNLABELED; n];
    let mut current_label: u32 = 2;
    let mut visited = Vec::new();
    let mut open = Vec::new();

    // Slide axes in ITK Dimension order (0 = x = ritk innermost axis 2, then y,
    // z); each is probed t = +1 then −1, giving the order +x,−x,+y,−y,+z,−z.
    let axes: [usize; 3] = [2, 1, 0];
    // Decode a flat index to [z, y, x].
    let coord = |i: usize| -> [isize; 3] {
        let z = i / (yn * xn);
        let rem = i % (yn * xn);
        [z as isize, (rem / xn) as isize, (rem % xn) as isize]
    };
    let in_bounds = |c: [isize; 3]| -> bool {
        c[0] >= 0
            && c[0] < zn as isize
            && c[1] >= 0
            && c[1] < yn as isize
            && c[2] >= 0
            && c[2] < xn as isize
    };
    let flat = |c: [isize; 3]| -> usize {
        (c[0] * strides[0] + c[1] * strides[1] + c[2] * strides[2]) as usize
    };

    for start in 0..n {
        if class[start] != UNLABELED {
            continue;
        }

        visited.clear();
        visited.push(start);
        let mut cur = start;
        // Running minimum value along the path (carries across steps, matching
        // ITK's never-reset `MinimumNeighborValue`).
        let mut min_val = input[start];
        // Assigned on every path out of the slide loop below (before each break).
        let mut min_class: u32;
        let mut label_for_region = current_label;

        // ── Steepest-descent slide ──────────────────────────────────────────
        loop {
            class[cur] = IN_PROGRESS;
            let cc = coord(cur);
            let mut min_idx = cur;
            // Check the face neighbours in ITK order +x,−x,+y,−y,+z,−z; the
            // first strictly-smaller neighbour at the global running minimum wins.
            for &axis in &axes {
                for &t in &[1isize, -1] {
                    let mut nc = cc;
                    nc[axis] += t;
                    if !in_bounds(nc) {
                        continue;
                    }
                    let ni = flat(nc);
                    if class[ni] == IN_PROGRESS {
                        continue;
                    }
                    let nv = input[ni];
                    if nv < min_val {
                        min_val = nv;
                        min_idx = ni;
                    }
                }
            }
            let mut found = false;
            if min_idx != cur {
                visited.push(min_idx);
                cur = min_idx;
            } else {
                found = true; // local minimum: no strictly-smaller neighbour
            }
            // Class of the pixel we landed on (the new `cur`); >1 means we slid
            // into an already-labeled region and adopt it.
            min_class = class[cur];
            if min_class > IN_PROGRESS {
                found = true;
            }
            if found {
                break;
            }
        }

        // ── Plateau flood-fill at a local minimum ───────────────────────────
        if min_class == IN_PROGRESS {
            open.clear();
            open.push(cur);
            while let Some(seed) = open.pop() {
                visited.push(seed);
                let sv = input[seed];
                let sc = coord(seed);
                for &axis in &axes {
                    for &t in &[-1isize, 1] {
                        let mut nc = sc;
                        nc[axis] += t;
                        if !in_bounds(nc) {
                            continue;
                        }
                        let ni = flat(nc);
                        if input[ni] <= sv {
                            let nclass = class[ni];
                            if nclass == UNLABELED {
                                open.push(ni);
                                class[ni] = IN_PROGRESS;
                            }
                            if nclass > IN_PROGRESS {
                                min_class = nclass;
                            }
                        }
                    }
                }
            }
        }

        if min_class == IN_PROGRESS {
            label_for_region = current_label;
            current_label += 1;
        } else if min_class > IN_PROGRESS {
            label_for_region = min_class;
        }

        for &idx in &visited {
            class[idx] = label_for_region;
        }
    }

    class.into_iter().map(|label| label as f32).collect()
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "tests_toboggan.rs"]
mod tests;
