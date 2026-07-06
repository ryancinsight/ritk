//! Merge several label images into one, matching `itk::MergeLabelMapFilter`.
//!
//! # Mathematical Specification
//!
//! Each input label image is interpreted as a sparse `LabelMap`: every distinct
//! non-zero value `L` defines one label object whose support is the set of voxels
//! holding `L`.  The output starts as a copy of input 0 and the remaining inputs
//! are folded in according to one of four methods (ITK `ChoiceMethodEnum`):
//!
//! - **Keep** (0): a later object retains its label if that label is unused in
//!   the accumulated output; otherwise it is deferred and re-added with the next
//!   free label (`max_label + 1`).  Faithful to ITK, the deferred queue persists
//!   across inputs.
//! - **Aggregate** (1): objects sharing a label are unioned; new labels are kept.
//! - **Pack** (2): every object from every input is appended and relabeled
//!   `1, 2, 3, …` in (input order, ascending-label) sequence.
//! - **Strict** (3): like Keep, but a label collision is an error rather than a
//!   relabel.
//!
//! The output image is the rasterization of the accumulated objects painted in
//! ascending final-label order: on overlap, the higher final label wins (exactly
//! `itk::LabelMapToLabelImageFilter`).
//!
//! # ITK / SimpleITK parity
//! Bit-exact to `sitk.LabelMapToLabel(sitk.MergeLabelMap([…], method))` for all
//! four methods, including the persistent-deferred-queue behavior with ≥3 inputs.

use ritk_image::tensor::Backend;
use ritk_image::Image;
use ritk_tensor_ops::{extract_vec_infallible, rebuild};
use std::collections::BTreeMap;

/// Conflict-resolution strategy for [`merge_label_maps`].
///
/// Discriminants match `itk::MergeLabelMapFilter::ChoiceMethodEnum` and
/// `SimpleITK.MergeLabelMap`'s integer `method` argument.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MergeLabelMethod {
    /// Keep a later object's label when free; otherwise reassign the next label.
    Keep,
    /// Union objects that share a label.
    Aggregate,
    /// Append all objects and relabel `1, 2, 3, …`.
    Pack,
    /// Keep labels; a collision is a hard error.
    Strict,
}

/// Failure modes of [`merge_label_maps`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MergeLabelError {
    /// Fewer than one input image was supplied.
    NoInputs,
    /// Inputs do not share identical voxel dimensions.
    ShapeMismatch,
    /// `Strict` method encountered a label already present in the output.
    StrictConflict {
        /// The colliding label value.
        label: u32,
        /// Index (1-based among the merged inputs) where the collision arose.
        input: usize,
    },
}

impl std::fmt::Display for MergeLabelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoInputs => write!(f, "merge_label_maps requires at least one input"),
            Self::ShapeMismatch => write!(f, "merge_label_maps inputs differ in dimensions"),
            Self::StrictConflict { label, input } => {
                write!(
                    f,
                    "Strict merge: label {label} from input {input} is already in use"
                )
            }
        }
    }
}

impl std::error::Error for MergeLabelError {}

/// Collect distinct non-zero labels of a flat label slice, each with its voxel
/// indices, ordered ascending by label (an `ImageType::ConstIterator` walks
/// label objects in ascending-label order).
fn distinct_objects(flat: &[f32]) -> BTreeMap<u32, Vec<usize>> {
    let mut objs: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
    for (idx, &v) in flat.iter().enumerate() {
        let label = v.round().max(0.0) as u32;
        if label != 0 {
            objs.entry(label).or_default().push(idx);
        }
    }
    objs
}

/// Paint accumulated objects into a dense buffer in ascending final-label order,
/// so the higher label wins on overlap (`LabelMapToLabelImageFilter`).
fn rasterize(objects: &[(u32, Vec<usize>)], n: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; n];
    let mut ordered: Vec<&(u32, Vec<usize>)> = objects.iter().collect();
    ordered.sort_by_key(|o| o.0);
    for (label, pixels) in ordered {
        let fv = *label as f32;
        for &p in pixels {
            out[p] = fv;
        }
    }
    out
}

/// Merge `images` into a single label image under the chosen [`MergeLabelMethod`].
///
/// # Preconditions
/// - `images` is non-empty and all entries share identical dimensions.
/// - Voxel values are non-negative integer labels (0 = background).
///
/// # Postcondition
/// - Output equals `sitk.LabelMapToLabel(sitk.MergeLabelMap(images, method))`.
/// - Spatial metadata is inherited from `images[0]`.
pub fn merge_label_maps<B: Backend>(
    images: &[&Image<B, 3>],
    method: MergeLabelMethod,
) -> Result<Image<B, 3>, MergeLabelError> {
    let first = *images.first().ok_or(MergeLabelError::NoInputs)?;
    let (flat0, dims) = extract_vec_infallible(first);
    let n = flat0.len();

    // Extract every input's flat slice up front; verify matching shape.
    let mut flats: Vec<Vec<f32>> = Vec::with_capacity(images.len());
    flats.push(flat0);
    for img in &images[1..] {
        let (f, d) = extract_vec_infallible(*img);
        if d != dims {
            return Err(MergeLabelError::ShapeMismatch);
        }
        flats.push(f);
    }

    let out_flat = match method {
        MergeLabelMethod::Pack => {
            // Append every object from every input, relabel 1, 2, 3, ….
            let mut next: u32 = 0;
            let mut objects: Vec<(u32, Vec<usize>)> = Vec::new();
            for flat in &flats {
                for (_label, px) in distinct_objects(flat) {
                    next += 1;
                    objects.push((next, px));
                }
            }
            rasterize(&objects, n)
        }
        MergeLabelMethod::Aggregate => {
            let mut map: BTreeMap<u32, Vec<usize>> = distinct_objects(&flats[0]);
            for flat in &flats[1..] {
                for (label, px) in distinct_objects(flat) {
                    map.entry(label).or_default().extend(px);
                }
            }
            let objects: Vec<(u32, Vec<usize>)> = map.into_iter().collect();
            rasterize(&objects, n)
        }
        MergeLabelMethod::Keep | MergeLabelMethod::Strict => {
            let strict = method == MergeLabelMethod::Strict;
            let mut map: BTreeMap<u32, Vec<usize>> = distinct_objects(&flats[0]);
            // Deferred queue persists across inputs, exactly as in ITK.
            let mut deferred: Vec<Vec<usize>> = Vec::new();
            for (i, flat) in flats[1..].iter().enumerate() {
                for (label, px) in distinct_objects(flat) {
                    match map.entry(label) {
                        std::collections::btree_map::Entry::Vacant(e) => {
                            e.insert(px);
                        }
                        std::collections::btree_map::Entry::Occupied(_) if strict => {
                            return Err(MergeLabelError::StrictConflict {
                                label,
                                input: i + 1,
                            });
                        }
                        std::collections::btree_map::Entry::Occupied(_) => deferred.push(px),
                    }
                }
                // Re-add the deferred objects with fresh labels (max + 1 each).
                for px in &deferred {
                    let new_label = map.keys().next_back().copied().unwrap_or(0) + 1;
                    map.insert(new_label, px.clone());
                }
            }
            let objects: Vec<(u32, Vec<usize>)> = map.into_iter().collect();
            rasterize(&objects, n)
        }
    };

    Ok(rebuild(out_flat, dims, first))
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "tests_merge_label_map.rs"]
mod tests;
