//! Backend-independent composite transform serialization.
//!
//! # Design
//!
//! The [`Transform<B, D>`](super::trait_::Transform) trait is generic over a Burn
//! tensor backend `B`, which makes direct serialization of live transform objects
//! backend-dependent.  This module provides a **parameter-only** representation
//! ([`TransformDescription`]) that captures each supported transform type as plain
//! `f64` vectors, and a **composite container** ([`CompositeTransform`]) that holds
//! an ordered sequence of such descriptions.
//!
//! Both types derive `serde::Serialize` and `serde::Deserialize`, enabling
//! round-trip JSON persistence that is independent of any particular Burn backend.
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
mod tests {
    use super::{CompositeTransform, TransformDescription};

    // -- helpers ----------------------------------------------------------

    /// 3-D identity matrix in row-major order (4×4 homogeneous).
    fn identity_4x4() -> Vec<f64> {
        vec![
            1.0, 0.0, 0.0, 0.0, // row 0
            0.0, 1.0, 0.0, 0.0, // row 1
            0.0, 0.0, 1.0, 0.0, // row 2
            0.0, 0.0, 0.0, 1.0, // row 3
        ]
    }

    /// 3-D rotation matrix: 90° about Z axis (row-major, 3×3).
    ///
    /// R_z(π/2) = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    ///
    /// Analytically: cos(π/2) = 0, sin(π/2) = 1.
    fn rotation_z_90_3x3() -> Vec<f64> {
        vec![
            0.0, -1.0, 0.0, // row 0
            1.0, 0.0, 0.0, // row 1
            0.0, 0.0, 1.0, // row 2
        ]
    }

    // -- round-trip serialization ----------------------------------------

    #[test]
    fn roundtrip_translation() {
        let desc = TransformDescription::Translation {
            offset: vec![1.5, -2.25, 3.0],
        };
        let json = serde_json::to_string(&desc).expect("serialize");
        let back: TransformDescription = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(desc, back);
        // Value inspection: ensure offset survived.
        if let TransformDescription::Translation { offset } = &back {
            assert_eq!(offset.len(), 3);
            assert_eq!(offset[0], 1.5);
            assert_eq!(offset[1], -2.25);
            assert_eq!(offset[2], 3.0);
        } else {
            panic!("expected Translation variant");
        }
    }

    #[test]
    fn roundtrip_rigid() {
        let desc = TransformDescription::Rigid {
            rotation: rotation_z_90_3x3(),
            translation: vec![10.0, 20.0, 30.0],
        };
        let json = serde_json::to_string(&desc).expect("serialize");
        let back: TransformDescription = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(desc, back);

        if let TransformDescription::Rigid {
            rotation,
            translation,
        } = &back
        {
            assert_eq!(rotation.len(), 9);
            assert_eq!(translation.len(), 3);
            // Verify R[0][1] = -1 (second element).
            assert_eq!(rotation[1], -1.0);
            assert_eq!(translation[2], 30.0);
        } else {
            panic!("expected Rigid variant");
        }
    }

    #[test]
    fn roundtrip_affine() {
        let desc = TransformDescription::Affine {
            matrix: identity_4x4(),
        };
        let json = serde_json::to_string_pretty(&desc).expect("serialize");
        let back: TransformDescription = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(desc, back);

        if let TransformDescription::Affine { matrix } = &back {
            assert_eq!(matrix.len(), 16);
            // Diagonal elements are 1.
            assert_eq!(matrix[0], 1.0);
            assert_eq!(matrix[5], 1.0);
            assert_eq!(matrix[10], 1.0);
            assert_eq!(matrix[15], 1.0);
            // Off-diagonal sample is 0.
            assert_eq!(matrix[1], 0.0);
        } else {
            panic!("expected Affine variant");
        }
    }

    #[test]
    fn roundtrip_displacement_field() {
        // 2×2×2 displacement field in 3-D.
        // Total voxels: 8.  Three component vectors of length 8.
        let dims = vec![2, 2, 2];
        let n: usize = dims.iter().product();
        assert_eq!(n, 8);

        // Analytically simple: constant displacement of (0.5, -0.5, 1.0).
        let dz = vec![0.5_f64; n];
        let dy = vec![-0.5_f64; n];
        let dx = vec![1.0_f64; n];

        let desc = TransformDescription::DisplacementField {
            dims: dims.clone(),
            origin: vec![0.0, 0.0, 0.0],
            spacing: vec![1.0, 1.0, 1.0],
            components: vec![dz.clone(), dy.clone(), dx.clone()],
        };

        let json = serde_json::to_string(&desc).expect("serialize");
        let back: TransformDescription = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(desc, back);

        if let TransformDescription::DisplacementField {
            dims: d,
            origin,
            spacing,
            components,
        } = &back
        {
            assert_eq!(d, &dims);
            assert_eq!(origin.len(), 3);
            assert_eq!(spacing.len(), 3);
            assert_eq!(components.len(), 3);
            assert_eq!(components[0].len(), n);
            assert_eq!(components[0][0], 0.5);
            assert_eq!(components[1][0], -0.5);
            assert_eq!(components[2][0], 1.0);
        } else {
            panic!("expected DisplacementField variant");
        }
    }

    #[test]
    fn roundtrip_bspline() {
        let grid_dims = vec![3, 3, 3];
        let n: usize = grid_dims.iter().product();
        assert_eq!(n, 27);

        // Linearly increasing control-point displacement per axis.
        let comp_z: Vec<f64> = (0..n).map(|i| i as f64 * 0.1).collect();
        let comp_y: Vec<f64> = (0..n).map(|i| -(i as f64) * 0.05).collect();
        let comp_x: Vec<f64> = (0..n).map(|i| i as f64 * 0.2).collect();

        let desc = TransformDescription::BSpline {
            grid_dims: grid_dims.clone(),
            grid_origin: vec![-1.0, -1.0, -1.0],
            grid_spacing: vec![1.0, 1.0, 1.0],
            components: vec![comp_z.clone(), comp_y.clone(), comp_x.clone()],
        };

        let json = serde_json::to_string(&desc).expect("serialize");
        let back: TransformDescription = serde_json::from_str(&json).expect("deserialize");
        // Approximate comparison: some f64 values (e.g. 1.4, 1.9) are not
        // exactly representable in IEEE 754 and may differ by 1 ULP after
        // JSON text round-trip.
        if let (
            TransformDescription::BSpline {
                components: c_orig, ..
            },
            TransformDescription::BSpline {
                components: c_back, ..
            },
        ) = (&desc, &back)
        {
            assert_eq!(c_orig.len(), c_back.len());
            for (orig_vec, back_vec) in c_orig.iter().zip(c_back.iter()) {
                assert_eq!(orig_vec.len(), back_vec.len());
                for (a, b) in orig_vec.iter().zip(back_vec.iter()) {
                    assert!(
                        (a - b).abs() < 1e-15,
                        "component mismatch: {a} vs {b}, diff = {}",
                        (a - b).abs()
                    );
                }
            }
        } else {
            panic!("expected both to be BSpline variants");
        }

        if let TransformDescription::BSpline {
            grid_dims: gd,
            grid_origin,
            grid_spacing,
            components,
        } = &back
        {
            assert_eq!(gd, &grid_dims);
            assert_eq!(grid_origin, &[-1.0, -1.0, -1.0]);
            assert_eq!(grid_spacing, &[1.0, 1.0, 1.0]);
            assert_eq!(components.len(), 3);
            assert_eq!(components[0].len(), 27);
            // Spot-check: element 5 of Z component = 5 * 0.1 = 0.5.
            assert_eq!(components[0][5], 0.5);
            // Element 10 of Y component = -(10 * 0.05) = -0.5.
            assert_eq!(components[1][10], -0.5);
        } else {
            panic!("expected BSpline variant");
        }
    }

    // -- composite construction ------------------------------------------

    #[test]
    fn composite_construction_and_accessors() {
        let mut ct = CompositeTransform::new(3);
        assert_eq!(ct.dimensionality, 3);
        assert!(ct.is_empty());
        assert_eq!(ct.len(), 0);

        ct.push(TransformDescription::Translation {
            offset: vec![1.0, 0.0, 0.0],
        });
        assert_eq!(ct.len(), 1);
        assert!(!ct.is_empty());

        ct.push(TransformDescription::Rigid {
            rotation: rotation_z_90_3x3(),
            translation: vec![0.0, 0.0, 0.0],
        });
        assert_eq!(ct.len(), 2);

        ct.push(TransformDescription::Affine {
            matrix: identity_4x4(),
        });
        assert_eq!(ct.len(), 3);

        // Validate dimensionality consistency.
        let mismatches = ct.validate_dimensionality();
        assert!(
            mismatches.is_empty(),
            "expected no dimensionality mismatches, got indices: {:?}",
            mismatches
        );
    }

    #[test]
    fn composite_roundtrip_json_string() {
        let mut ct = CompositeTransform::new(3);
        ct.description = "test composite: translate then rotate".into();
        ct.push(TransformDescription::Translation {
            offset: vec![5.0, -3.0, 0.0],
        });
        ct.push(TransformDescription::Rigid {
            rotation: rotation_z_90_3x3(),
            translation: vec![0.0, 0.0, 0.0],
        });

        let json = ct.to_json().expect("to_json");
        let back = CompositeTransform::from_json(&json).expect("from_json");

        assert_eq!(ct, back);
        assert_eq!(back.dimensionality, 3);
        assert_eq!(back.description, "test composite: translate then rotate");
        assert_eq!(back.len(), 2);

        // Inspect first transform value.
        if let TransformDescription::Translation { offset } = &back.transforms[0] {
            assert_eq!(offset, &[5.0, -3.0, 0.0]);
        } else {
            panic!("expected Translation at index 0");
        }
    }

    #[test]
    fn composite_default_description_absent_in_json() {
        // When `description` is absent from the JSON payload, it must default
        // to an empty string via `#[serde(default)]`.
        let json = r#"{"dimensionality":2,"transforms":[]}"#;
        let ct = CompositeTransform::from_json(json).expect("from_json");
        assert_eq!(ct.dimensionality, 2);
        assert_eq!(ct.description, "");
        assert!(ct.is_empty());
    }

    // -- file I/O --------------------------------------------------------

    #[test]
    fn composite_file_io_roundtrip() {
        let dir = tempfile::tempdir().expect("create tempdir");
        let file_path = dir.path().join("composite.json");

        let mut ct = CompositeTransform::new(3);
        ct.description = "file I/O test".into();
        ct.push(TransformDescription::Translation {
            offset: vec![1.0, 2.0, 3.0],
        });
        ct.push(TransformDescription::Affine {
            matrix: identity_4x4(),
        });

        ct.save_json(&file_path).expect("save_json");

        // Verify the file exists and has non-zero length.
        let metadata = std::fs::metadata(&file_path).expect("file metadata");
        assert!(metadata.len() > 0);

        let loaded = CompositeTransform::load_json(&file_path).expect("load_json");
        assert_eq!(ct, loaded);

        // Value-level check on loaded data.
        assert_eq!(loaded.dimensionality, 3);
        assert_eq!(loaded.description, "file I/O test");
        assert_eq!(loaded.len(), 2);
        if let TransformDescription::Translation { offset } = &loaded.transforms[0] {
            assert_eq!(offset, &[1.0, 2.0, 3.0]);
        } else {
            panic!("expected Translation at index 0");
        }
        if let TransformDescription::Affine { matrix } = &loaded.transforms[1] {
            assert_eq!(matrix.len(), 16);
            assert_eq!(matrix[0], 1.0);
            assert_eq!(matrix[15], 1.0);
        } else {
            panic!("expected Affine at index 1");
        }
    }

    #[test]
    fn load_json_nonexistent_file_returns_io_error() {
        let result = CompositeTransform::load_json("/nonexistent/path/composite.json");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::NotFound);
    }

    // -- invalid JSON ----------------------------------------------------

    #[test]
    fn from_json_invalid_json_returns_error() {
        let bad_json = r#"{"dimensionality": 3, "transforms": [{"Translation": {}}]}"#;
        let result = CompositeTransform::from_json(bad_json);
        assert!(result.is_err());
    }

    #[test]
    fn from_json_truncated_payload_returns_error() {
        let truncated = r#"{"dimensionality": 3, "transf"#;
        let result = CompositeTransform::from_json(truncated);
        assert!(result.is_err());
    }

    #[test]
    fn from_json_wrong_type_returns_error() {
        // `dimensionality` must be an integer, not a string.
        let bad = r#"{"dimensionality": "three", "transforms": []}"#;
        let result = CompositeTransform::from_json(bad);
        assert!(result.is_err());
    }

    // -- dimensionality validation ---------------------------------------

    #[test]
    fn validate_detects_mismatched_dimensionality() {
        let mut ct = CompositeTransform::new(3);
        // Correct: 3-D translation.
        ct.push(TransformDescription::Translation {
            offset: vec![1.0, 2.0, 3.0],
        });
        // Wrong: 2-D translation in a 3-D composite.
        ct.push(TransformDescription::Translation {
            offset: vec![1.0, 2.0],
        });

        let bad = ct.validate_dimensionality();
        assert_eq!(bad, vec![1]);
    }

    #[test]
    fn implied_dimensionality_rigid_consistency() {
        let good = TransformDescription::Rigid {
            rotation: rotation_z_90_3x3(),
            translation: vec![0.0, 0.0, 0.0],
        };
        assert_eq!(good.implied_dimensionality(), Some(3));

        // Inconsistent: 9-element rotation but 2-element translation.
        let bad = TransformDescription::Rigid {
            rotation: rotation_z_90_3x3(),
            translation: vec![0.0, 0.0],
        };
        assert_eq!(bad.implied_dimensionality(), None);
    }

    #[test]
    fn implied_dimensionality_affine() {
        // 4×4 = 16 elements → D = 3.
        assert_eq!(
            TransformDescription::Affine {
                matrix: identity_4x4()
            }
            .implied_dimensionality(),
            Some(3)
        );
        // 3×3 = 9 elements → D = 2.
        let id_3x3 = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        assert_eq!(
            TransformDescription::Affine { matrix: id_3x3 }.implied_dimensionality(),
            Some(2)
        );
        // Non-square element count → None.
        assert_eq!(
            TransformDescription::Affine {
                matrix: vec![1.0; 7]
            }
            .implied_dimensionality(),
            None
        );
    }

    // -- composite with all variant types --------------------------------

    #[test]
    fn composite_all_variants_roundtrip() {
        let dims = vec![2, 2, 2];
        let n: usize = dims.iter().product();

        let mut ct = CompositeTransform::new(3);
        ct.description = "all-variants composite".into();

        ct.push(TransformDescription::Translation {
            offset: vec![1.0, 2.0, 3.0],
        });
        ct.push(TransformDescription::Rigid {
            rotation: rotation_z_90_3x3(),
            translation: vec![0.0, 5.0, 0.0],
        });
        ct.push(TransformDescription::Affine {
            matrix: identity_4x4(),
        });
        ct.push(TransformDescription::DisplacementField {
            dims: dims.clone(),
            origin: vec![0.0, 0.0, 0.0],
            spacing: vec![1.0, 1.0, 1.0],
            components: vec![vec![0.0; n], vec![0.0; n], vec![0.0; n]],
        });
        ct.push(TransformDescription::BSpline {
            grid_dims: vec![3, 3, 3],
            grid_origin: vec![-1.0, -1.0, -1.0],
            grid_spacing: vec![1.0, 1.0, 1.0],
            components: vec![vec![0.0; 27], vec![0.0; 27], vec![0.0; 27]],
        });

        assert_eq!(ct.len(), 5);
        assert!(ct.validate_dimensionality().is_empty());

        let json = ct.to_json().expect("to_json");
        let back = CompositeTransform::from_json(&json).expect("from_json");
        assert_eq!(ct, back);
        assert_eq!(back.len(), 5);

        // Spot-check: second transform is Rigid with translation[1] = 5.0.
        if let TransformDescription::Rigid { translation, .. } = &back.transforms[1] {
            assert_eq!(translation[1], 5.0);
        } else {
            panic!("expected Rigid at index 1");
        }
    }
}
