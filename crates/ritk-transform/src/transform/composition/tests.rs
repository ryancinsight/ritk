//! Tests for composite_io
//! Extracted to keep the 500-line structural limit.
use super::io::{CompositeTransform, TransformDescription};

// -- helpers ----------------------------------------------------------

/// 3-D identity matrix in row-major order (4Ã—4 homogeneous).
fn identity_4x4() -> Vec<f64> {
    vec![
        1.0, 0.0, 0.0, 0.0, // row 0
        0.0, 1.0, 0.0, 0.0, // row 1
        0.0, 0.0, 1.0, 0.0, // row 2
        0.0, 0.0, 0.0, 1.0, // row 3
    ]
}

/// 3-D rotation matrix: 90Â° about Z axis (row-major, 3Ã—3).
///
/// R_z(Ï€/2) = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
///
/// Analytically: cos(Ï€/2) = 0, sin(Ï€/2) = 1.
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
    // 2Ã—2Ã—2 displacement field in 3-D.
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
    // 4Ã—4 = 16 elements â†’ D = 3.
    assert_eq!(
        TransformDescription::Affine {
            matrix: identity_4x4()
        }
        .implied_dimensionality(),
        Some(3)
    );
    // 3Ã—3 = 9 elements â†’ D = 2.
    let id_3x3 = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    assert_eq!(
        TransformDescription::Affine { matrix: id_3x3 }.implied_dimensionality(),
        Some(2)
    );
    // Non-square element count â†’ None.
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
