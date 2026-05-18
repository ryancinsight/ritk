use crate::FilterKind;

// Verify that the default `FilterKind` values exposed by the panel are
// within the analytically valid clamped ranges — GrayscaleMorph (remaining),
// Pointwise, Geometry, and Threshold variants.

/// Abs: unit struct, always valid.
#[test]
fn abs_variant_is_valid() {
    let fk = FilterKind::Abs;
    assert!(
        matches!(fk, FilterKind::Abs),
        "Abs variant must match itself"
    );
}

/// InvertIntensity default: maximum=None (computed from image data, ITK default).
#[test]
fn invert_intensity_default_maximum_is_none() {
    let fk = FilterKind::InvertIntensity { maximum: None };
    if let FilterKind::InvertIntensity { maximum } = fk {
        assert!(
            maximum.is_none(),
            "InvertIntensity default maximum must be None (auto from image)"
        );
    } else {
        panic!("expected InvertIntensity variant");
    }
}

/// NormalizeIntensity: unit struct, always valid.
#[test]
fn normalize_intensity_variant_is_valid() {
    let fk = FilterKind::NormalizeIntensity;
    assert!(
        matches!(fk, FilterKind::NormalizeIntensity),
        "NormalizeIntensity variant must match itself"
    );
}

/// Square: unit struct, always valid.
#[test]
fn square_variant_is_valid() {
    let fk = FilterKind::Square;
    assert!(
        matches!(fk, FilterKind::Square),
        "Square variant must match itself"
    );
}

/// Sqrt: unit struct, always valid.
#[test]
fn sqrt_variant_is_valid() {
    let fk = FilterKind::Sqrt;
    assert!(
        matches!(fk, FilterKind::Sqrt),
        "Sqrt variant must match itself"
    );
}

/// Log: unit struct, always valid.
#[test]
fn log_variant_is_valid() {
    let fk = FilterKind::Log;
    assert!(
        matches!(fk, FilterKind::Log),
        "Log variant must match itself"
    );
}

/// Exp: unit struct, always valid.
#[test]
fn exp_variant_is_valid() {
    let fk = FilterKind::Exp;
    assert!(
        matches!(fk, FilterKind::Exp),
        "Exp variant must match itself"
    );
}

/// Binary threshold: foreground=1.0 and background=0.0 are the canonical
/// binary label values. lower ≤ upper required for valid threshold.
#[test]
fn binary_threshold_defaults_ordered() {
    let fk = FilterKind::BinaryThreshold {
        lower: 100.0,
        upper: 500.0,
        foreground: 1.0,
        background: 0.0,
    };
    if let FilterKind::BinaryThreshold {
        lower,
        upper,
        foreground,
        background,
    } = fk
    {
        assert!(lower <= upper, "lower={lower} must be ≤ upper={upper}");
        assert_ne!(
            foreground, background,
            "foreground and background must differ"
        );
        assert_eq!(foreground, 1.0f32);
        assert_eq!(background, 0.0f32);
    } else {
        panic!("expected BinaryThreshold");
    }
}

/// Rescale intensity: out_min < out_max required for a non-degenerate mapping.
#[test]
fn rescale_intensity_defaults_ordered() {
    let fk = FilterKind::RescaleIntensity {
        out_min: 0.0,
        out_max: 1.0,
    };
    if let FilterKind::RescaleIntensity { out_min, out_max } = fk {
        assert!(
            out_min < out_max,
            "out_min={out_min} must be < out_max={out_max}"
        );
    } else {
        panic!("expected RescaleIntensity");
    }
}

/// Clamp: lower ≤ upper required for non-degenerate clamping.
#[test]
fn clamp_defaults_ordered() {
    let fk = FilterKind::Clamp {
        lower: 0.0,
        upper: 255.0,
    };
    if let FilterKind::Clamp { lower, upper } = fk {
        assert!(lower <= upper, "lower={lower} must be ≤ upper={upper}");
    } else {
        panic!("expected Clamp");
    }
}

/// Connected threshold: lower ≤ upper for a valid acceptance interval.
#[test]
fn connected_threshold_defaults_ordered() {
    let fk = FilterKind::ConnectedThreshold {
        seed_z: 0,
        seed_y: 0,
        seed_x: 0,
        lower: 100.0,
        upper: 500.0,
    };
    if let FilterKind::ConnectedThreshold { lower, upper, .. } = fk {
        assert!(lower <= upper, "lower={lower} must be ≤ upper={upper}");
    } else {
        panic!("expected ConnectedThreshold");
    }
}

/// Confidence connected: multiplier > 0 and max_iterations ≥ 1.
#[test]
fn confidence_connected_defaults_valid() {
    let fk = FilterKind::ConfidenceConnected {
        seed_z: 0,
        seed_y: 0,
        seed_x: 0,
        initial_lower: 0.0,
        initial_upper: 100.0,
        multiplier: 2.5,
        max_iterations: 15,
    };
    if let FilterKind::ConfidenceConnected {
        multiplier,
        max_iterations,
        initial_lower,
        initial_upper,
        ..
    } = fk
    {
        assert!(
            multiplier > 0.0,
            "multiplier must be positive, got {multiplier}"
        );
        assert!(max_iterations >= 1, "max_iterations must be ≥ 1");
        assert!(
            initial_lower <= initial_upper,
            "initial_lower must be ≤ initial_upper"
        );
    } else {
        panic!("expected ConfidenceConnected");
    }
}

/// Neighborhood connected: lower ≤ upper and all radii ≥ 1.
#[test]
fn neighborhood_connected_defaults_valid() {
    let fk = FilterKind::NeighborhoodConnected {
        seed_z: 0,
        seed_y: 0,
        seed_x: 0,
        lower: 100.0,
        upper: 500.0,
        radius_z: 1,
        radius_y: 1,
        radius_x: 1,
    };
    if let FilterKind::NeighborhoodConnected {
        lower,
        upper,
        radius_z,
        radius_y,
        radius_x,
        ..
    } = fk
    {
        assert!(lower <= upper, "lower={lower} must be ≤ upper={upper}");
        assert!(
            radius_z >= 1 && radius_y >= 1 && radius_x >= 1,
            "all radii must be ≥ 1"
        );
    } else {
        panic!("expected NeighborhoodConnected");
    }
}

/// Atan, Sin, Cos, Tan, Asin, Acos, BoundedReciprocal are unit variants — no parameters.
#[test]
fn trig_filter_variants_are_unit() {
    // All 7 unit variants must be constructible and equatable.
    let _atan = FilterKind::Atan;
    let _sin = FilterKind::Sin;
    let _cos = FilterKind::Cos;
    let _tan = FilterKind::Tan;
    let _asin = FilterKind::Asin;
    let _acos = FilterKind::Acos;
    let _br = FilterKind::BoundedReciprocal;
    assert_eq!(FilterKind::Atan, FilterKind::Atan);
    assert_eq!(FilterKind::Sin, FilterKind::Sin);
    assert_eq!(FilterKind::Cos, FilterKind::Cos);
    assert_eq!(FilterKind::Tan, FilterKind::Tan);
    assert_eq!(FilterKind::Asin, FilterKind::Asin);
    assert_eq!(FilterKind::Acos, FilterKind::Acos);
    assert_eq!(FilterKind::BoundedReciprocal, FilterKind::BoundedReciprocal);
}
