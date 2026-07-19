//! Backend-independent composite transform serialization.
//!
//! # Design
//!
//! The [`Transform<B, D>`](crate::Transform) trait is generic over a Burn
//! tensor backend `B`, which makes direct serialization of live transform objects
//! backend-dependent.  This module provides a **parameter-only** representation
//! ([`TransformDescription`]) that captures each supported transform type as plain
//! `f64` vectors, and a **composite container** ([`CompositeTransform`]) that holds
//! an ordered sequence of such descriptions.
//!
//! Both types derive `serde::Serialize` and `serde::Deserialize`, enabling
//! round-trip JSON persistence that is independent of any particular Coeus backend.
//!
//! # Supported transform types
//!
//! | Variant              | Parameters                                               |
//! |----------------------|----------------------------------------------------------|
//! | `Translation`        | offset vector \[D\]                                      |
//! | `Rigid`              | rotation matrix D×D (row-major) + translation \[D\]      |
//! | `Affine`             | homogeneous matrix (D+1)×(D+1) (row-major)               |
//! | `DisplacementField`  | grid dims, origin, spacing, per-component displacement   |
//! | `BSpline`            | grid dims, origin, spacing, per-component ctrl-pt displ. |
//!
//! # Composition order
//!
//! Transforms are applied **left-to-right** (first-to-last):
//! a point `p` is mapped through `T[0]`, then `T[1]`, … , then `T[n-1]`.
//! Mathematically this corresponds to `T_{n-1} ∘ … ∘ T_1 ∘ T_0`.
//!
//! # Dependencies
//!
//! This module requires `serde_json = "1.0"` in `[dependencies]` of
//! `crates/ritk-core/Cargo.toml`.  The crate already depends on `serde`.

use serde::{Deserialize, Serialize};
use std::fs;
use std::io;
use std::path::Path;

// ---------------------------------------------------------------------------
// TransformDescription
// ---------------------------------------------------------------------------

/// Backend-independent description of a single transform's parameters.
///
/// Each variant stores its parameters as flat `f64` vectors in row-major order
/// so that no Burn tensor types appear in the serialized representation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TransformDescription {
    /// Translation: offset vector of length D.
    Translation {
        /// Offset per spatial dimension.  Length must equal the composite's
        /// `dimensionality`.
        offset: Vec<f64>,
    },

    /// Rigid: rotation matrix (D×D, row-major) + translation vector (length D).
    ///
    /// The rotation matrix is stored as `D * D` elements in row-major order.
    Rigid {
        /// D×D rotation matrix, row-major.
        rotation: Vec<f64>,
        /// Translation vector of length D.
        translation: Vec<f64>,
    },

    /// Affine: full homogeneous matrix ((D+1)×(D+1), row-major).
    Affine {
        /// (D+1)×(D+1) homogeneous matrix, row-major.
        matrix: Vec<f64>,
    },

    /// Displacement field: dense voxel displacement vectors on a regular grid.
    DisplacementField {
        /// Grid dimensions (e.g. `[nz, ny, nx]` for 3-D).
        dims: Vec<usize>,
        /// Physical-space origin of the grid.
        origin: Vec<f64>,
        /// Physical-space spacing between voxels.
        spacing: Vec<f64>,
        /// Displacement components.  One inner `Vec` per spatial dimension,
        /// each of length `prod(dims)`.  For 3-D: `[dz, dy, dx]`.
        components: Vec<Vec<f64>>,
    },

    /// B-spline: control-point displacements on a regular lattice.
    BSpline {
        /// Control-point grid dimensions.
        grid_dims: Vec<usize>,
        /// Physical-space origin of the control grid.
        grid_origin: Vec<f64>,
        /// Physical-space spacing of control points.
        grid_spacing: Vec<f64>,
        /// Control-point displacement components.  One inner `Vec` per
        /// spatial dimension, each of length `prod(grid_dims)`.
        components: Vec<Vec<f64>>,
    },
}

impl TransformDescription {
    /// Return the spatial dimensionality implied by the variant's data.
    ///
    /// Returns `None` when the data is internally inconsistent (e.g. a `Rigid`
    /// whose rotation length is not a perfect square matching the translation
    /// length).
    pub fn implied_dimensionality(&self) -> Option<usize> {
        match self {
            Self::Translation { offset } => Some(offset.len()),
            Self::Rigid {
                rotation,
                translation,
            } => {
                let d = translation.len();
                if d > 0 && rotation.len() == d * d {
                    Some(d)
                } else {
                    None
                }
            }
            Self::Affine { matrix } => {
                // (D+1)^2 elements → solve for D.
                let n = matrix.len();
                let side = (n as f64).sqrt() as usize;
                if side >= 2 && side * side == n {
                    Some(side - 1)
                } else {
                    None
                }
            }
            Self::DisplacementField { components, .. } => Some(components.len()),
            Self::BSpline { components, .. } => Some(components.len()),
        }
    }
}

// ---------------------------------------------------------------------------
// CompositeTransform
// ---------------------------------------------------------------------------

/// Ordered composite transform: applied first-to-last.
///
/// `point → T[0] → T[1] → … → T[n-1]`
///
/// # Invariants
///
/// * `dimensionality ∈ {2, 3}` (enforced by convention, not by the type system,
///   to allow forward-compatible payloads).
/// * Every element of `transforms` must have parameter lengths consistent with
///   `dimensionality`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CompositeTransform {
    /// Spatial dimensionality (2 or 3).
    pub dimensionality: usize,

    /// Human-readable description (optional metadata).
    #[serde(default)]
    pub description: String,

    /// Ordered list of transforms.
    pub transforms: Vec<TransformDescription>,
}

impl CompositeTransform {
    /// Create a new empty composite transform for the given dimensionality.
    pub fn new(dimensionality: usize) -> Self {
        Self {
            dimensionality,
            description: String::new(),
            transforms: Vec::new(),
        }
    }

    /// Append a transform description to the chain.
    pub fn push(&mut self, transform: TransformDescription) {
        self.transforms.push(transform);
    }

    /// Number of transforms in the chain.
    pub fn len(&self) -> usize {
        self.transforms.len()
    }

    /// Whether the chain is empty.
    pub fn is_empty(&self) -> bool {
        self.transforms.is_empty()
    }

    /// Serialize to a pretty-printed JSON string.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Deserialize from a JSON string.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    /// Write to a JSON file at `path`.
    ///
    /// Creates or truncates the file.  Parent directories must already exist.
    pub fn save_json<P: AsRef<Path>>(&self, path: P) -> Result<(), io::Error> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        fs::write(path, json)
    }

    /// Read from a JSON file at `path`.
    pub fn load_json<P: AsRef<Path>>(path: P) -> Result<Self, io::Error> {
        let contents = fs::read_to_string(path)?;
        serde_json::from_str(&contents).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
    }

    /// Validate that every transform's implied dimensionality matches
    /// `self.dimensionality`.
    ///
    /// Returns the indices of any mismatched transforms.  An empty `Vec`
    /// indicates a fully consistent composite.
    pub fn validate_dimensionality(&self) -> Vec<usize> {
        self.transforms
            .iter()
            .enumerate()
            .filter_map(|(i, t)| match t.implied_dimensionality() {
                Some(d) if d == self.dimensionality => None,
                _ => Some(i),
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "tests_composite_io.rs"]
mod tests_composite_io;
