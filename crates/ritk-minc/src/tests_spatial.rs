use super::*;

#[test]
fn default_direction_cosines_canonical_axes() {
    assert_eq!(default_direction_cosines("xspace"), [1.0, 0.0, 0.0]);
    assert_eq!(default_direction_cosines("yspace"), [0.0, 1.0, 0.0]);
    assert_eq!(default_direction_cosines("zspace"), [0.0, 0.0, 1.0]);
    assert_eq!(default_direction_cosines("tspace"), [1.0, 0.0, 0.0]);
}

#[test]
fn order_dimensions_by_dimorder_zyx() {
    let dims = vec![
        MincDimension {
            name: "xspace".to_string(),
            start: 0.0,
            step: 1.0,
            length: 64,
            direction_cosines: [1.0, 0.0, 0.0],
        },
        MincDimension {
            name: "yspace".to_string(),
            start: 0.0,
            step: 1.0,
            length: 80,
            direction_cosines: [0.0, 1.0, 0.0],
        },
        MincDimension {
            name: "zspace".to_string(),
            start: 0.0,
            step: 1.0,
            length: 48,
            direction_cosines: [0.0, 0.0, 1.0],
        },
    ];
    let dimorder = vec![
        "zspace".to_string(),
        "yspace".to_string(),
        "xspace".to_string(),
    ];
    let ordered =
        order_dimensions_by_dimorder(&dims, &dimorder).expect("infallible: validated precondition");
    assert_eq!(ordered[0].name, "zspace");
    assert_eq!(ordered[0].length, 48);
    assert_eq!(ordered[1].name, "yspace");
    assert_eq!(ordered[1].length, 80);
    assert_eq!(ordered[2].name, "xspace");
    assert_eq!(ordered[2].length, 64);
}

#[test]
fn order_dimensions_missing_dim_errors() {
    let dims = vec![MincDimension {
        name: "xspace".to_string(),
        start: 0.0,
        step: 1.0,
        length: 64,
        direction_cosines: [1.0, 0.0, 0.0],
    }];
    let dimorder = vec![
        "zspace".to_string(),
        "yspace".to_string(),
        "xspace".to_string(),
    ];
    assert!(order_dimensions_by_dimorder(&dims, &dimorder).is_err());
}

#[test]
fn build_spatial_metadata_positive_steps() {
    let dims = vec![
        MincDimension {
            name: "zspace".to_string(),
            start: -10.0,
            step: 2.0,
            length: 20,
            direction_cosines: [0.0, 0.0, 1.0],
        },
        MincDimension {
            name: "yspace".to_string(),
            start: -20.0,
            step: 1.5,
            length: 30,
            direction_cosines: [0.0, 1.0, 0.0],
        },
        MincDimension {
            name: "xspace".to_string(),
            start: -15.0,
            step: 1.0,
            length: 40,
            direction_cosines: [1.0, 0.0, 0.0],
        },
    ];
    let (origin, spacing, _direction) = build_spatial_metadata(&dims);
    assert!((origin[0] - (-10.0)).abs() < 1e-10);
    assert!((origin[1] - (-20.0)).abs() < 1e-10);
    assert!((origin[2] - (-15.0)).abs() < 1e-10);
    assert!((spacing[0] - 2.0).abs() < 1e-10);
    assert!((spacing[1] - 1.5).abs() < 1e-10);
    assert!((spacing[2] - 1.0).abs() < 1e-10);
}

#[test]
fn build_spatial_metadata_negative_step_negates_cosines() {
    let dims = vec![
        MincDimension {
            name: "zspace".to_string(),
            start: 10.0,
            step: -2.0,
            length: 20,
            direction_cosines: [0.0, 0.0, 1.0],
        },
        MincDimension {
            name: "yspace".to_string(),
            start: 0.0,
            step: 1.0,
            length: 30,
            direction_cosines: [0.0, 1.0, 0.0],
        },
        MincDimension {
            name: "xspace".to_string(),
            start: 0.0,
            step: 1.0,
            length: 40,
            direction_cosines: [1.0, 0.0, 0.0],
        },
    ];
    let (origin, spacing, direction) = build_spatial_metadata(&dims);
    assert!((spacing[0] - 2.0).abs() < 1e-10);
    // Column 0 of direction = negated zspace cosines.
    assert!((direction.0[(0, 0)] - 0.0).abs() < 1e-10);
    assert!((direction.0[(1, 0)] - 0.0).abs() < 1e-10);
    assert!((direction.0[(2, 0)] - (-1.0)).abs() < 1e-10);
    assert!((origin[0] - 10.0).abs() < 1e-10);
}
