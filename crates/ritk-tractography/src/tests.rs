use super::*;

fn horizontal(_: &Point<3>) -> Option<Vector<3>> {
    Some(Vector::new([1.0, 0.0, 0.0]))
}

#[test]
fn configuration_rejects_every_invalid_partition() {
    for step in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(matches!(
            TractographyConfig::new(step, 10, 60.0, TrackingDirection::Forward),
            Err(TractographyError::InvalidStepSize { .. })
        ));
    }
    assert!(matches!(
        TractographyConfig::new(1.0, 0, 60.0, TrackingDirection::Forward),
        Err(TractographyError::InvalidMaxSteps { .. })
    ));
    for angle in [-1.0, 181.0, f64::NAN, f64::INFINITY] {
        assert!(matches!(
            TractographyConfig::new(1.0, 10, angle, TrackingDirection::Forward),
            Err(TractographyError::InvalidTurnLimit { .. })
        ));
    }
}

#[test]
fn forward_step_limit_has_exact_geometry_and_reason() -> Result<(), TractographyError> {
    let config = TractographyConfig::new(1.0, 5, 60.0, TrackingDirection::Forward)?;
    let result = euler_tractography(&[Point::new([0.0, 0.0, 0.0])], config, horizontal)?;
    let line = &result.streamlines()[0];
    assert_eq!(line.geometry().len(), 6);
    assert_eq!(line.geometry().points()[0].x, 0.0);
    assert_eq!(line.geometry().points()[5].x, 5.0);
    assert_eq!(line.forward_termination(), TerminationReason::StepLimit);
    Ok(())
}

#[test]
fn boundary_point_is_not_appended() -> Result<(), TractographyError> {
    let field = |point: &Point<3>| (point[0] <= 5.0).then_some(Vector::new([1.0, 0.0, 0.0]));
    let config = TractographyConfig::new(1.0, 10, 60.0, TrackingDirection::Forward)?;
    let result = euler_tractography(&[Point::new([3.0, 0.0, 0.0])], config, field)?;
    let line = &result.streamlines()[0];
    assert_eq!(line.geometry().len(), 3);
    assert_eq!(line.geometry().points()[2].x, 5.0);
    assert_eq!(line.forward_termination(), TerminationReason::FieldBoundary);
    Ok(())
}

#[test]
fn bidirectional_join_contains_seed_once() -> Result<(), TractographyError> {
    let config = TractographyConfig::new(1.0, 3, 60.0, TrackingDirection::Bidirectional)?;
    let result = euler_tractography(&[Point::new([2.0, 0.0, 0.0])], config, horizontal)?;
    let points = result.streamlines()[0].geometry().points();
    assert_eq!(points.len(), 7);
    assert_eq!(points[0].x, -1.0);
    assert_eq!(points[3].x, 2.0);
    assert_eq!(points[6].x, 5.0);
    assert_eq!(
        result.streamlines()[0].backward_termination(),
        Some(TerminationReason::StepLimit)
    );
    Ok(())
}

#[test]
fn sharp_turn_stops_at_last_valid_point() -> Result<(), TractographyError> {
    let field = |point: &Point<3>| {
        if point[0] < 1.0 {
            Some(Vector::new([1.0, 0.0, 0.0]))
        } else {
            Some(Vector::new([0.0, 1.0, 0.0]))
        }
    };
    let config = TractographyConfig::new(1.0, 5, 45.0, TrackingDirection::Forward)?;
    let result = euler_tractography(&[Point::new([0.0, 0.0, 0.0])], config, field)?;
    let line = &result.streamlines()[0];
    assert_eq!(line.geometry().len(), 2);
    assert_eq!(line.forward_termination(), TerminationReason::TurningAngle);
    Ok(())
}

#[test]
fn invalid_field_direction_is_an_error_not_a_dropped_line() -> Result<(), TractographyError> {
    let config = TractographyConfig::new(1.0, 5, 45.0, TrackingDirection::Forward)?;
    let error = euler_tractography(&[Point::new([0.0, 0.0, 0.0])], config, |_| {
        Some(Vector::new([2.0, 0.0, 0.0]))
    })
    .expect_err("non-unit direction");
    assert!(matches!(
        error,
        TractographyError::InvalidDirection {
            seed_index: 0,
            step_index: 0,
            ..
        }
    ));
    Ok(())
}

#[test]
fn empty_and_untrackable_seed_sets_are_value_exact() -> Result<(), TractographyError> {
    let config = TractographyConfig::default();
    let empty = euler_tractography(&[], config, horizontal)?;
    assert_eq!(empty.seeds_attempted(), 0);
    assert_eq!(empty.streamlines_generated(), 0);
    let absent = euler_tractography(&[Point::new([0.0, 0.0, 0.0])], config, |_| None)?;
    assert_eq!(absent.seeds_attempted(), 1);
    assert_eq!(absent.streamlines_generated(), 0);
    Ok(())
}

#[test]
fn non_finite_seed_fails_before_field_callback() {
    let callback_called = std::cell::Cell::new(false);
    let error = euler_tractography(
        &[Point::new([f64::NAN, 0.0, 0.0])],
        TractographyConfig::default(),
        |_| {
            callback_called.set(true);
            None
        },
    )
    .expect_err("non-finite seed");
    assert!(!callback_called.get());
    assert!(matches!(
        error,
        TractographyError::NonFinitePoint {
            seed_index: 0,
            step_index: 0,
            ..
        }
    ));
}

#[test]
fn dti_pev_field_produces_straight_streamline() -> Result<(), TractographyError> {
    // Construct a minimal DTI-like PEV directly — the dti_pev_direction_field
    // function takes a DiffusionTensor reference, so we build a synthetic
    // tensor with a known PEV via the public DTI estimation path.
    use ritk_diffusion::dti::{DtiConfig, estimate_dti};
    use ritk_diffusion_scheme::{DiffusionWeighting, GradientDirection, GradientFrame, GradientScheme};

    // Build a single-fibre scheme: 2 b0 + 12 diverse directions.
    let mut entries = Vec::new();
    let b0 = DiffusionWeighting::from_seconds_per_square_millimeter(0.0).unwrap();
    let zero = Vector::new([0.0, 0.0, 0.0]);
    entries.push(GradientDirection::new(b0, zero).unwrap());
    entries.push(GradientDirection::new(b0, zero).unwrap());
    let b1000 =
        DiffusionWeighting::from_seconds_per_square_millimeter(1000.0).unwrap();
    // Well-conditioned directions — enough to identify all 6 D elements.
    let r2 = 2.0_f64.sqrt() / 2.0;
    let r3 = 3.0_f64.sqrt() / 3.0;
    for dir in [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [r2, r2, 0.0],
        [r2, -r2, 0.0],
        [r2, 0.0, r2],
        [0.0, r2, r2],
        [-r2, 0.0, r2],
        [r3, r3, r3],
        [-r3, r3, r3],
        [r3, -r3, r3],
        [r3, r3, -r3],
    ] {
        entries.push(
            GradientDirection::new(b1000, Vector::new(dir)).unwrap(),
        );
    }
    let scheme = GradientScheme::new(entries, GradientFrame::ImageAxis).unwrap();

    // Generate signals from a fibre along +x with AD=1.7e-3, RD=0.3e-3.
    let s0 = 1000.0;
    let d: [f64; 6] = [0.0017, 0.0003, 0.0003, 0.0, 0.0, 0.0];
    let signals: Vec<f64> = scheme
        .directions()
        .iter()
        .map(|entry| {
            let b = entry.weighting().seconds_per_square_millimeter();
            if b == 0.0 {
                return s0;
            }
            let [gx, gy, gz] = entry.direction().to_array();
            let adc = d[0] * gx * gx + d[1] * gy * gy + d[2] * gz * gz;
            s0 * (-b * adc).exp()
        })
        .collect();

    let tensor = estimate_dti(&scheme, &signals, DtiConfig::default())
        .expect("DTI fit");

    // PEV should align with +x.
    let pev = tensor.principal_eigenvector();
    assert!(pev[0].abs() > 0.99, "PEV x={:.4}", pev[0]);

    let field = dti_pev_direction_field(&tensor);
    let config = TractographyConfig::new(1.0, 5, 60.0, TrackingDirection::Forward)?;
    let result = euler_tractography(&[Point::new([0.0, 0.0, 0.0])], config, field)?;

    assert_eq!(result.streamlines_generated(), 1);
    let line = &result.streamlines()[0];
    // Streamline should follow +x (or -x, depending on PEV sign).
    let dx = (line.geometry().points()[5].x - line.geometry().points()[0].x).abs();
    assert!((dx - 5.0).abs() < 0.01, "streamline length mismatch, dx={dx:.4}");
    Ok(())
}

#[test]
fn dti_pev_field_is_skipped_for_untrackable_seed() -> Result<(), TractographyError> {
    // A degenerate (isotropic, FA ~= 0) DTI result produces no streamlines:
    // the direction field returns None everywhere because the PEV is not
    // trackable for a near-isotropic tensor.
    use ritk_diffusion::dti::{DtiConfig, estimate_dti};
    use ritk_diffusion_scheme::{DiffusionWeighting, GradientDirection, GradientFrame, GradientScheme};

    let mut entries = Vec::new();
    let b0 = DiffusionWeighting::from_seconds_per_square_millimeter(0.0).unwrap();
    let zero = Vector::new([0.0, 0.0, 0.0]);
    entries.push(GradientDirection::new(b0, zero).unwrap());
    entries.push(GradientDirection::new(b0, zero).unwrap());
    let b1000 =
        DiffusionWeighting::from_seconds_per_square_millimeter(1000.0).unwrap();
    // Well-conditioned directions for a full-rank design matrix.
    let r2 = 2.0_f64.sqrt() / 2.0;
    for dir in [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [r2, r2, 0.0],
        [r2, 0.0, r2],
        [0.0, r2, r2],
    ] {
        entries.push(
            GradientDirection::new(b1000, Vector::new(dir)).unwrap(),
        );
    }
    let scheme = GradientScheme::new(entries, GradientFrame::ImageAxis).unwrap();

    // Isotropic tensor: all eigenvalues equal.
    let s0 = 1000.0;
    let d: [f64; 6] = [0.0007, 0.0007, 0.0007, 0.0, 0.0, 0.0];
    let signals: Vec<f64> = scheme
        .directions()
        .iter()
        .map(|entry| {
            let b = entry.weighting().seconds_per_square_millimeter();
            if b == 0.0 {
                return s0;
            }
            s0 * (-b * 0.0007).exp()
        })
        .collect();

    let tensor = estimate_dti(&scheme, &signals, DtiConfig::default())
        .expect("DTI fit");
    assert!(tensor.fa() < 1e-3, "isotropic FA should be ~0");

    let field = dti_pev_direction_field(&tensor);
    let config = TractographyConfig::default();
    let result =
        euler_tractography(&[Point::new([0.0, 0.0, 0.0])], config, field)?;
    assert_eq!(
        result.streamlines_generated(),
        0,
        "degenerate PEV should produce no streamlines"
    );
    Ok(())
}

// ── Whole-brain tractography via FodVolume ───────────────────────────────

#[test]
fn fod_volume_field_tracks_through_homogeneous_z_fibre() -> Result<(), TractographyError> {
    use ritk_diffusion::csd::{
        CsdConfig, CsdError, FodVolume, ResponseFunction, estimate_fod,
    };
    use ritk_diffusion_scheme::{
        DiffusionWeighting, GradientDirection, GradientFrame, GradientScheme,
    };

    // Build a single-voxel CSD result for a z-aligned fibre, then replicate
    // across a 2×1×2 volume so the streamline can travel in z.
    let mut entries = Vec::new();
    let b0 = DiffusionWeighting::from_seconds_per_square_millimeter(0.0).unwrap();
    let zero = Vector::new([0.0, 0.0, 0.0]);
    entries.push(GradientDirection::new(b0, zero).unwrap());
    let b3000 =
        DiffusionWeighting::from_seconds_per_square_millimeter(3_000.0).unwrap();
    let golden_angle = std::f64::consts::PI * (3.0 - 5.0_f64.sqrt());
    for i in 0..60 {
        let z = 1.0 - 2.0 * (i as f64 + 0.5) / 60.0;
        let radius = (1.0 - z * z).sqrt();
        let phi = golden_angle * i as f64;
        entries.push(
            GradientDirection::new(
                b3000,
                Vector::new([radius * phi.cos(), radius * phi.sin(), z]),
            )
            .unwrap(),
        );
    }
    let scheme = GradientScheme::new(entries, GradientFrame::Lps).unwrap();

    // Single-fibre signal along +z.
    let signals: Vec<f64> = scheme
        .directions()
        .iter()
        .map(|entry| {
            let b = entry.weighting().seconds_per_square_millimeter();
            if b == 0.0 {
                return 1.0;
            }
            let [gx, gy, gz] = entry.direction().to_array();
            let adc = 0.0003 + (0.0017 - 0.0003) * gz * gz;
            (-b * adc).exp()
        })
        .collect();

    let response =
        ResponseFunction::from_tensor(3_000.0, 0.0017, 0.0003, 8)
            .map_err(|e: CsdError| TractographyError::InvalidDirection {
                seed_index: 0,
                step_index: 0,
                reason: e.to_string(),
            })?;
    let config = CsdConfig::new(
        8,
        DiffusionWeighting::from_seconds_per_square_millimeter(50.0).unwrap(),
        Default::default(),
    )
    .map_err(|e: CsdError| TractographyError::InvalidDirection {
        seed_index: 0,
        step_index: 0,
        reason: e.to_string(),
    })?;
    let fod = estimate_fod(&scheme, &signals, &response, &config)
        .map_err(|e: CsdError| TractographyError::InvalidDirection {
            seed_index: 0,
            step_index: 0,
            reason: e.to_string(),
        })?;

    // Replicate across 2×1×2 volume: nx=2, ny=1, nz=2 so z-extent is 4 mm
    // with 2 mm spacing.
    let nc = fod.coefficients().len();
    let shape = [2usize, 1, 2];
    let n_voxels = shape.iter().product::<usize>();
    let mut flat = Vec::with_capacity(n_voxels * nc);
    for _ in 0..n_voxels {
        flat.extend_from_slice(fod.coefficients());
    }
    let basis = apollo_sht::RealSphericalHarmonicBasis::new(8)
        .map_err(|e| TractographyError::InvalidDirection {
            seed_index: 0,
            step_index: 0,
            reason: e.to_string(),
        })?;
    let volume = FodVolume::new(
        flat.into_boxed_slice(),
        shape,
        [2.0, 2.0, 2.0],
        [0.0, 0.0, 0.0],
        basis,
        GradientFrame::Lps,
    )
    .map_err(|e: CsdError| TractographyError::InvalidDirection {
        seed_index: 0,
        step_index: 0,
        reason: e.to_string(),
    })?;

    let field = fod_volume_direction_field(&volume);
    let config = TractographyConfig::new(0.5, 4, 60.0, TrackingDirection::Forward)?;
    let seed = Point::new([1.0, 0.0, 1.0]); // centre of the volume in x/z
    let result = euler_tractography(&[seed], config, field)?;

    assert_eq!(result.streamlines_generated(), 1);
    let line = &result.streamlines()[0];
    // The streamline should travel along ±z.
    let points = line.geometry().points();
    assert!(points.len() >= 3, "streamline too short: {} points", points.len());
    let dz = (points.last().unwrap().z - points.first().unwrap().z).abs();
    assert!(dz > 0.5, "streamline should move in z, dz={dz:.4}");
    // Lateral drift should be negligible.
    for p in points.iter() {
        let dx = (p.x - seed[0]).abs();
        let dy = (p.y - seed[1]).abs();
        assert!(dx < 0.2, "x drifted to {}", p.x);
        assert!(dy < 0.1, "y drifted to {}", p.y);
    }
    Ok(())
}

// ── Whole-brain tractography via NoddiVolume ────────────────────────────

#[test]
fn noddi_volume_field_tracks_through_homogeneous_z_fibre() -> Result<(), TractographyError> {
    use ritk_diffusion::noddi::{NoddiConfig, NoddiVolume, estimate_noddi};
    use ritk_diffusion_scheme::{
        DiffusionWeighting, GradientDirection, GradientFrame, GradientScheme,
    };

    // Build a single-voxel NODDI result for a z-aligned fibre, then replicate
    // across a 2×1×2 volume so the streamline can travel in z.
    let mut entries = Vec::new();
    let b0 = DiffusionWeighting::from_seconds_per_square_millimeter(0.0).unwrap();
    let zero = Vector::new([0.0, 0.0, 0.0]);
    entries.push(GradientDirection::new(b0, zero).unwrap());
    entries.push(GradientDirection::new(b0, zero).unwrap());
    entries.push(GradientDirection::new(b0, zero).unwrap());
    entries.push(GradientDirection::new(b0, zero).unwrap());
    let b3000 =
        DiffusionWeighting::from_seconds_per_square_millimeter(3_000.0).unwrap();
    let golden_angle = std::f64::consts::PI * (3.0 - 5.0_f64.sqrt());
    for i in 0..30 {
        let z = 1.0 - 2.0 * (i as f64 + 0.5) / 30.0;
        let radius = (1.0 - z * z).sqrt();
        let phi = golden_angle * i as f64;
        entries.push(
            GradientDirection::new(
                b3000,
                Vector::new([radius * phi.cos(), radius * phi.sin(), z]),
            )
            .unwrap(),
        );
    }
    let scheme = GradientScheme::new(entries, GradientFrame::ImageAxis).unwrap();

    // Single-fibre signal along +z (strongly anisotropic, low ODI).
    let s0 = 1000.0;
    let signals: Vec<f64> = scheme
        .directions()
        .iter()
        .map(|entry| {
            let b = entry.weighting().seconds_per_square_millimeter();
            if b == 0.0 {
                return s0;
            }
            let [gx, gy, gz] = entry.direction().to_array();
            let adc = 0.0003 + (0.0017 - 0.0003) * gz * gz;
            s0 * (-b * adc).exp()
        })
        .collect();

    let fit = estimate_noddi(&scheme, &signals, &NoddiConfig::default())
        .map_err(|e| TractographyError::InvalidDirection {
            seed_index: 0,
            step_index: 0,
            reason: e.to_string(),
        })?;
    assert!(fit.converged());
    let dir = fit.principal_direction();
    assert!(dir[2].abs() > 0.96, "expected z-aligned NODDI direction, got {dir:?}");

    // Replicate across 2×1×2 volume: nx=2, ny=1, nz=2 so z-extent is 4 mm
    // with 2 mm spacing.
    let shape = [2usize, 1, 2];
    let n_voxels = shape.iter().product::<usize>();
    let mut flat = Vec::with_capacity(n_voxels * 3);
    for _ in 0..n_voxels {
        flat.extend_from_slice(&dir);
    }
    let volume = NoddiVolume::new(
        flat.into_boxed_slice(),
        shape,
        [2.0, 2.0, 2.0],
        [0.0, 0.0, 0.0],
        GradientFrame::ImageAxis,
    )
    .map_err(|e| TractographyError::InvalidDirection {
        seed_index: 0,
        step_index: 0,
        reason: e.to_string(),
    })?;

    let field = noddi_direction_field(&volume);
    let config = TractographyConfig::new(0.5, 4, 60.0, TrackingDirection::Forward)?;
    let seed = Point::new([1.0, 0.0, 1.0]); // centre of the volume in x/z
    let result = euler_tractography(&[seed], config, field)?;

    assert_eq!(result.streamlines_generated(), 1);
    let line = &result.streamlines()[0];
    let points = line.geometry().points();
    assert!(points.len() >= 3, "streamline too short: {} points", points.len());
    let dz = (points.last().unwrap().z - points.first().unwrap().z).abs();
    assert!(dz > 0.5, "streamline should move in z, dz={dz:.4}");
    // Lateral drift should be negligible.
    for p in points.iter() {
        let dx = (p.x - seed[0]).abs();
        let dy = (p.y - seed[1]).abs();
        assert!(dx < 0.2, "x drifted to {}", p.x);
        assert!(dy < 0.1, "y drifted to {}", p.y);
    }
    Ok(())
}
