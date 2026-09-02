#![expect(clippy::unwrap_used, reason = "ratchet RITK-UNWRAP-1")]
use ritk_diffusion::maps::{DiffusionMapsConfig, DtiVolume, fit_diffusion_maps};
use ritk_diffusion_scheme::{DiffusionWeighting, GradientDirection, GradientFrame, GradientScheme};
use ritk_spatial::{Point, Vector};

use super::*;

fn horizontal(_: &Point<3>) -> Option<Vector<3>> {
    Some(Vector::new([1.0, 0.0, 0.0]))
}

fn image_axis_scheme() -> GradientScheme {
    let count = 30_usize;
    let mut entries = Vec::with_capacity(count + 1);
    entries.push(
        GradientDirection::new(
            DiffusionWeighting::from_seconds_per_square_millimeter(0.0)
                .expect("finite b0 weighting"),
            Vector::new([0.0, 0.0, 0.0]),
        )
        .expect("valid b0 direction"),
    );
    let golden_angle = std::f64::consts::PI * (3.0 - 5.0_f64.sqrt());
    for index in 0..count {
        #[expect(
            clippy::cast_precision_loss,
            reason = "the synthetic scheme has only thirty directions"
        )]
        let z = 1.0 - 2.0 * (index as f64 + 0.5) / count as f64;
        let radius = (1.0 - z * z).sqrt();
        #[expect(
            clippy::cast_precision_loss,
            reason = "the synthetic scheme has only thirty directions"
        )]
        let phi = golden_angle * index as f64;
        entries.push(
            GradientDirection::new(
                DiffusionWeighting::from_seconds_per_square_millimeter(1_000.0)
                    .expect("finite diffusion weighting"),
                Vector::new([radius * phi.cos(), radius * phi.sin(), z]),
            )
            .expect("unit Fibonacci direction"),
        );
    }
    GradientScheme::new(entries, GradientFrame::ImageAxis).expect("valid image-axis scheme")
}

fn dti_signal(scheme: &GradientScheme, tensor: [f64; 6], baseline: f64) -> Vec<f64> {
    let [dxx, dyy, dzz, dxy, dxz, dyz] = tensor;
    scheme
        .directions()
        .iter()
        .map(|entry| {
            let b = entry.weighting().seconds_per_square_millimeter();
            if b == 0.0 {
                return baseline;
            }
            let [gx, gy, gz] = entry.direction().to_array();
            let q = dxx * gx * gx
                + dyy * gy * gy
                + dzz * gz * gz
                + 2.0 * dxy * gx * gy
                + 2.0 * dxz * gx * gz
                + 2.0 * dyz * gy * gz;
            baseline * (-b * q).exp()
        })
        .collect()
}

fn dti_volume(tensors: &[[f64; 6]], shape: [usize; 3], floor: f64) -> DtiVolume {
    let scheme = image_axis_scheme();
    let per_voxel: Vec<Vec<f64>> = tensors
        .iter()
        .map(|tensor| dti_signal(&scheme, *tensor, 1_000.0))
        .collect();
    let volumes: Vec<Vec<f64>> = (0..scheme.len())
        .map(|acquisition| per_voxel.iter().map(|voxel| voxel[acquisition]).collect())
        .collect();
    let borrowed: Vec<&[f64]> = volumes.iter().map(Vec::as_slice).collect();
    let maps = fit_diffusion_maps(
        &scheme,
        &borrowed,
        &DiffusionMapsConfig {
            background_fraction: 0.0,
            ..DiffusionMapsConfig::default()
        },
    )
    .expect("synthetic DTI series fits");
    DtiVolume::new(maps, shape, floor).expect("shape matches synthetic maps")
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
    use ritk_diffusion_scheme::{
        DiffusionWeighting, GradientDirection, GradientFrame, GradientScheme,
    };

    // Build a single-fibre scheme: 2 b0 + 12 diverse directions.
    let mut entries = Vec::new();
    let b0 = DiffusionWeighting::from_seconds_per_square_millimeter(0.0).unwrap();
    let zero = Vector::new([0.0, 0.0, 0.0]);
    entries.push(GradientDirection::new(b0, zero).unwrap());
    entries.push(GradientDirection::new(b0, zero).unwrap());
    let b1000 = DiffusionWeighting::from_seconds_per_square_millimeter(1000.0).unwrap();
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
        entries.push(GradientDirection::new(b1000, Vector::new(dir)).unwrap());
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

    let tensor = estimate_dti(&scheme, &signals, DtiConfig::default()).expect("DTI fit");

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
    assert!(
        (dx - 5.0).abs() < 0.01,
        "streamline length mismatch, dx={dx:.4}"
    );
    Ok(())
}

#[test]
fn dti_pev_field_is_skipped_for_untrackable_seed() -> Result<(), TractographyError> {
    // A degenerate (isotropic, FA ~= 0) DTI result produces no streamlines:
    // the direction field returns None everywhere because the PEV is not
    // trackable for a near-isotropic tensor.
    use ritk_diffusion::dti::{DtiConfig, estimate_dti};
    use ritk_diffusion_scheme::{
        DiffusionWeighting, GradientDirection, GradientFrame, GradientScheme,
    };

    let mut entries = Vec::new();
    let b0 = DiffusionWeighting::from_seconds_per_square_millimeter(0.0).unwrap();
    let zero = Vector::new([0.0, 0.0, 0.0]);
    entries.push(GradientDirection::new(b0, zero).unwrap());
    entries.push(GradientDirection::new(b0, zero).unwrap());
    let b1000 = DiffusionWeighting::from_seconds_per_square_millimeter(1000.0).unwrap();
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
        entries.push(GradientDirection::new(b1000, Vector::new(dir)).unwrap());
    }
    let scheme = GradientScheme::new(entries, GradientFrame::ImageAxis).unwrap();

    // Isotropic tensor: all eigenvalues equal.
    let s0 = 1000.0;
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

    let tensor = estimate_dti(&scheme, &signals, DtiConfig::default()).expect("DTI fit");
    assert!(tensor.fa() < 1e-3, "isotropic FA should be ~0");

    let field = dti_pev_direction_field(&tensor);
    let config = TractographyConfig::default();
    let result = euler_tractography(&[Point::new([0.0, 0.0, 0.0])], config, field)?;
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
    use ritk_diffusion::csd::{CsdConfig, CsdError, FodVolume, ResponseFunction, estimate_fod};
    use ritk_diffusion_scheme::{
        DiffusionWeighting, GradientDirection, GradientFrame, GradientScheme,
    };

    // Build a single-voxel CSD result for a z-aligned fibre, then replicate
    // across a 2×1×2 volume so the streamline can travel in z.
    let mut entries = Vec::new();
    let b0 = DiffusionWeighting::from_seconds_per_square_millimeter(0.0).unwrap();
    let zero = Vector::new([0.0, 0.0, 0.0]);
    entries.push(GradientDirection::new(b0, zero).unwrap());
    let b3000 = DiffusionWeighting::from_seconds_per_square_millimeter(3_000.0).unwrap();
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
            let [_, _, gz] = entry.direction().to_array();
            let adc = 0.0003 + (0.0017 - 0.0003) * gz * gz;
            (-b * adc).exp()
        })
        .collect();

    let response =
        ResponseFunction::from_tensor(3_000.0, 0.0017, 0.0003, 8).map_err(|e: CsdError| {
            TractographyError::InvalidDirection {
                seed_index: 0,
                step_index: 0,
                reason: e.to_string(),
            }
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
    let fod = estimate_fod(&scheme, &signals, &response, &config).map_err(|e: CsdError| {
        TractographyError::InvalidDirection {
            seed_index: 0,
            step_index: 0,
            reason: e.to_string(),
        }
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
    let basis = apollo_sht::RealSphericalHarmonicBasis::new(8).map_err(|e| {
        TractographyError::InvalidDirection {
            seed_index: 0,
            step_index: 0,
            reason: e.to_string(),
        }
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
    assert!(
        points.len() >= 3,
        "streamline too short: {} points",
        points.len()
    );
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
    let b3000 = DiffusionWeighting::from_seconds_per_square_millimeter(3_000.0).unwrap();
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
            let [_, _, gz] = entry.direction().to_array();
            let adc = 0.0003 + (0.0017 - 0.0003) * gz * gz;
            s0 * (-b * adc).exp()
        })
        .collect();

    let fit = estimate_noddi(&scheme, &signals, &NoddiConfig::default()).map_err(|e| {
        TractographyError::InvalidDirection {
            seed_index: 0,
            step_index: 0,
            reason: e.to_string(),
        }
    })?;
    assert!(fit.converged());
    let dir = fit.principal_direction();
    assert!(
        dir[2].abs() > 0.96,
        "expected z-aligned NODDI direction, got {dir:?}"
    );

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
    assert!(
        points.len() >= 3,
        "streamline too short: {} points",
        points.len()
    );
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

// ── Per-point scalar export ──────────────────────────────────────────────

#[test]
fn to_trk_with_scalars_round_trips_fa_and_md() -> Result<(), TractographyError> {
    use ritk_trk::TrkTractogram;

    let config = TractographyConfig::new(1.0, 3, 60.0, TrackingDirection::Forward)?;
    let seeds = &[Point::new([0.0, 0.0, 0.0])];
    let result = euler_tractography(seeds, config, horizontal)?;
    assert_eq!(result.streamlines_generated(), 1);

    // Per-point scalars: FA=0.8, MD=0.0007 at each of 4 points, 2 scalars each.
    let n_points = result.streamlines()[0].geometry().len();
    assert_eq!(n_points, 4);
    let mut scalars = Vec::new();
    let mut flat = Vec::with_capacity(n_points * 2);
    for _ in 0..n_points {
        flat.push(0.8f32);
        flat.push(0.0007f32);
    }
    scalars.push(flat.into_boxed_slice());

    let trk = result.to_trk_with_scalars([64, 64, 30], [2.0, 2.0, 2.0], &["FA", "MD"], scalars);

    assert_eq!(trk.header.n_scalars, 2);
    let name_str = std::str::from_utf8(&trk.header.scalar_name)
        .unwrap()
        .trim_matches('\0');
    assert_eq!(name_str, "FA MD");
    assert_eq!(trk.scalars.len(), 1);
    assert_eq!(trk.scalars[0].len(), n_points * 2);

    // Round-trip: write + read.
    let mut buf = Vec::new();
    trk.write(&mut buf).expect("write .trk with scalars");
    let read_back = TrkTractogram::read(&mut buf.as_slice()).expect("read .trk with scalars");
    assert_eq!(read_back.header.n_scalars, 2);
    assert_eq!(read_back.scalars.len(), 1);
    assert_eq!(read_back.scalars[0].len(), n_points * 2);

    // Verify scalar values survived round-trip.
    for (orig, recovered) in trk.scalars[0].iter().zip(read_back.scalars[0].iter()) {
        assert!((orig - recovered).abs() < 1e-6);
    }
    Ok(())
}

#[test]
fn to_trx_with_dpv_stores_fa_data() -> Result<(), TractographyError> {
    use std::collections::HashMap;

    let config = TractographyConfig::new(1.0, 2, 60.0, TrackingDirection::Forward)?;
    let seeds = &[Point::new([0.0, 0.0, 0.0])];
    let result = euler_tractography(seeds, config, horizontal)?;

    // Encode per-vertex FA as f32 bytes.
    let n_points = result.streamlines()[0].geometry().len();
    assert_eq!(n_points, 3);
    let fa_values: Vec<f32> = vec![0.8, 0.75, 0.7];
    let fa_bytes: Vec<u8> = fa_values.iter().flat_map(|v| v.to_le_bytes()).collect();

    let mut dpv_data: HashMap<String, Vec<u8>> = HashMap::new();
    dpv_data.insert("FA".into(), fa_bytes);

    let mut trx = result.to_trx_with_dpv(dpv_data);
    trx.header.dpv.insert(
        "FA".into(),
        ritk_trx::TrxArrayDef {
            dtype: "float32".into(),
            n_components: 1,
        },
    );

    assert_eq!(trx.dpv_data.len(), 1);
    assert!(trx.header.dpv.contains_key("FA"));

    // to_raw preserves DPV.
    let (_hdr, _pos, _off, dpv) = trx.to_raw().expect("encode .trx with dpv");
    assert_eq!(dpv.len(), 1);
    assert!(dpv.contains_key("FA"));
    Ok(())
}

// ── .trk export round-trip ───────────────────────────────────────────────

#[test]
fn to_trk_header_with_non_identity_affine() -> Result<(), TractographyError> {
    use ritk_trk::TrkTractogram;

    let config = TractographyConfig::new(1.0, 3, 60.0, TrackingDirection::Forward)?;
    let seeds = &[Point::new([0.0, 0.0, 0.0])];
    let result = euler_tractography(seeds, config, horizontal)?;
    assert_eq!(result.streamlines_generated(), 1);

    let affine: [[f32; 4]; 4] = [
        [2.0, 0.0, 0.0, 10.0],
        [0.0, 2.0, 0.0, 20.0],
        [0.0, 0.0, 1.0, 30.0],
        [0.0, 0.0, 0.0, 1.0],
    ];

    let trk = result.to_trk_header([64, 64, 30], [2.0, 2.0, 2.0], Some(affine));
    assert_eq!(trk.header.vox_to_ras, affine);
    assert_eq!(trk.header.dim, [64, 64, 30]);
    assert_eq!(trk.header.voxel_size, [2.0, 2.0, 2.0]);

    // Identity-via-None should still produce identity.
    let default_trk = result.to_trk_header([64, 64, 30], [2.0, 2.0, 2.0], None);
    assert_eq!(
        default_trk.header.vox_to_ras,
        ritk_trk::TrkHeader::default().vox_to_ras
    );

    // Round-trip: write with custom affine, read back.
    let mut buf = Vec::new();
    trk.write(&mut buf).expect("write .trk");
    let read_back = TrkTractogram::read(&mut buf.as_slice()).expect("read .trk");
    assert_eq!(read_back.header.vox_to_ras, affine);
    Ok(())
}

#[test]
fn to_trk_round_trips_through_writer_and_reader() -> Result<(), TractographyError> {
    use ritk_trk::{TrkHeader, TrkTractogram};

    let config = TractographyConfig::new(1.0, 5, 60.0, TrackingDirection::Forward)?;
    let seeds = &[Point::new([0.0, 0.0, 0.0]), Point::new([0.0, 1.0, 0.0])];
    let result = euler_tractography(seeds, config, horizontal)?;
    assert_eq!(result.streamlines_generated(), 2);

    let trk = result.to_trk([128, 128, 60], [2.0, 2.0, 2.0]);
    assert_eq!(trk.header.n_count, 2);
    assert_eq!(trk.header.dim, [128, 128, 60]);
    assert_eq!(trk.header.voxel_size, [2.0, 2.0, 2.0]);
    assert_eq!(trk.streamlines.len(), 2);
    assert_eq!(trk.header.vox_to_ras, TrkHeader::default().vox_to_ras);

    let mut buf = Vec::new();
    trk.write(&mut buf).expect("write .trk");

    let read_back = TrkTractogram::read(&mut buf.as_slice()).expect("read .trk");
    assert_eq!(read_back.streamlines.len(), 2);
    for (original, recovered) in result
        .streamlines()
        .iter()
        .zip(read_back.streamlines.iter())
    {
        let orig_pts = original.geometry().points();
        let rec_pts = recovered.points();
        assert_eq!(orig_pts.len(), rec_pts.len());
        for (p1, p2) in orig_pts.iter().zip(rec_pts.iter()) {
            assert!((p1.x - p2.x).abs() < 1e-3);
            assert!((p1.y - p2.y).abs() < 1e-3);
            assert!((p1.z - p2.z).abs() < 1e-3);
        }
    }
    Ok(())
}

#[test]
fn to_tck_header_preserves_custom_fields() -> Result<(), TractographyError> {
    use ritk_tck::TckTractogram;

    let config = TractographyConfig::new(1.0, 3, 60.0, TrackingDirection::Forward)?;
    let result = euler_tractography(&[Point::new([0.0, 0.0, 0.0])], config, horizontal)?;
    assert_eq!(result.streamlines_generated(), 1);

    let transform: [[f64; 4]; 4] = [
        [2.0, 0.0, 0.0, 10.0],
        [0.0, 2.0, 0.0, 20.0],
        [0.0, 0.0, 1.0, 30.0],
        [0.0, 0.0, 0.0, 1.0],
    ];

    let tck = result.to_tck_header(
        Some("3.0.4".into()),
        Some("RITK Euler tractography".into()),
        Some(transform),
    );

    assert_eq!(tck.header.mrtrix_version.as_deref(), Some("3.0.4"));
    assert_eq!(
        tck.header.comments.as_deref(),
        Some("RITK Euler tractography")
    );
    assert_eq!(tck.header.transform, Some(transform));
    assert_eq!(tck.header.datatype, ritk_tck::TckDatatype::Float32LE);
    assert_eq!(tck.streamlines.len(), 1);

    // Round-trip: write with custom header, read back, verify fields survive.
    let mut buf = Vec::new();
    tck.write(&mut buf).expect("write .tck");
    let read_back = TckTractogram::read(buf.as_slice()).expect("read .tck");
    assert_eq!(read_back.header.mrtrix_version.as_deref(), Some("3.0.4"));
    assert_eq!(
        read_back.header.comments.as_deref(),
        Some("RITK Euler tractography")
    );
    assert_eq!(read_back.header.transform, Some(transform));
    Ok(())
}

#[test]
fn to_tck_header_partial_none_leaves_defaults() -> Result<(), TractographyError> {
    let config = TractographyConfig::new(1.0, 2, 60.0, TrackingDirection::Forward)?;
    let result = euler_tractography(&[Point::new([0.0, 0.0, 0.0])], config, horizontal)?;

    // Only set comments; version and transform should stay None.
    let tck = result.to_tck_header(None, Some("partial test".into()), None);
    assert_eq!(tck.header.mrtrix_version, None);
    assert_eq!(tck.header.comments.as_deref(), Some("partial test"));
    assert_eq!(tck.header.transform, None);
    Ok(())
}

#[test]
fn to_tck_round_trips_through_writer_and_reader() -> Result<(), TractographyError> {
    use ritk_tck::TckTractogram;

    let config = TractographyConfig::new(1.0, 5, 60.0, TrackingDirection::Forward)?;
    let seeds = &[Point::new([0.0, 0.0, 0.0]), Point::new([0.0, 2.0, 0.0])];
    let result = euler_tractography(seeds, config, horizontal)?;
    assert_eq!(result.streamlines_generated(), 2);

    let tck = result.to_tck();
    assert_eq!(tck.header.datatype, ritk_tck::TckDatatype::Float32LE);
    assert_eq!(tck.streamlines.len(), 2);

    let mut buf = Vec::new();
    tck.write(&mut buf).expect("write .tck");

    let read_back = TckTractogram::read(buf.as_slice()).expect("read .tck");
    assert_eq!(read_back.streamlines.len(), 2);
    for (original, recovered) in result
        .streamlines()
        .iter()
        .zip(read_back.streamlines.iter())
    {
        let orig_pts = original.geometry().points();
        let rec_pts = recovered.points();
        assert_eq!(orig_pts.len(), rec_pts.len());
        for (p1, p2) in orig_pts.iter().zip(rec_pts.iter()) {
            assert!((p1.x - p2.x).abs() < 1e-4);
            assert!((p1.y - p2.y).abs() < 1e-4);
            assert!((p1.z - p2.z).abs() < 1e-4);
        }
    }
    Ok(())
}

// ── Cross-codec differential test ────────────────────────────────────────

#[test]
fn trk_and_tck_recover_identical_coordinates() -> Result<(), TractographyError> {
    use ritk_tck::TckTractogram;
    use ritk_trk::TrkTractogram;
    use ritk_trx::TrxTractogram;

    let config = TractographyConfig::new(0.5, 10, 60.0, TrackingDirection::Forward)?;
    let seeds = &[
        Point::new([0.0, 0.0, 0.0]),
        Point::new([1.0, 0.0, 0.0]),
        Point::new([0.0, 2.0, 0.0]),
    ];
    let result = euler_tractography(seeds, config, horizontal)?;
    assert_eq!(result.streamlines_generated(), 3);

    // Export to all three formats.
    let trk = result.to_trk([128, 128, 60], [2.0, 2.0, 2.0]);
    let tck = result.to_tck();
    let trx = result.to_trx();

    // Write all, read all back.
    let mut trk_buf = Vec::new();
    trk.write(&mut trk_buf).expect("write .trk");
    let trk_back = TrkTractogram::read(&mut trk_buf.as_slice()).expect("read .trk");

    let mut tck_buf = Vec::new();
    tck.write(&mut tck_buf).expect("write .tck");
    let tck_back = TckTractogram::read(tck_buf.as_slice()).expect("read .tck");

    let (trx_hdr, trx_pos, trx_off, _dpv) = trx.to_raw().expect("encode .trx");
    let trx_back = TrxTractogram::from_raw(&trx_hdr, &trx_pos, &trx_off).expect("read .trx");

    assert_eq!(trk_back.streamlines.len(), tck_back.streamlines.len());
    assert_eq!(trk_back.streamlines.len(), trx_back.streamlines.len());

    // Cross-validate all three pairwise.
    for (trk_poly, tck_poly) in trk_back.streamlines.iter().zip(tck_back.streamlines.iter()) {
        assert_eq!(trk_poly.len(), tck_poly.len());
        for (p_trk, p_tck) in trk_poly.points().iter().zip(tck_poly.points().iter()) {
            assert!((p_trk.x - p_tck.x).abs() < 1e-4);
            assert!((p_trk.y - p_tck.y).abs() < 1e-4);
            assert!((p_trk.z - p_tck.z).abs() < 1e-4);
        }
    }
    for (trk_poly, trx_poly) in trk_back.streamlines.iter().zip(trx_back.streamlines.iter()) {
        assert_eq!(trk_poly.len(), trx_poly.len());
        for (p_trk, p_trx) in trk_poly.points().iter().zip(trx_poly.points().iter()) {
            assert!((p_trk.x - p_trx.x).abs() < 1e-4);
            assert!((p_trk.y - p_trx.y).abs() < 1e-4);
            assert!((p_trk.z - p_trx.z).abs() < 1e-4);
        }
    }
    Ok(())
}

// ── Cross-codec with non-identity affine ─────────────────────────────────

#[test]
fn cross_codec_non_identity_affine_recovers_same_physical_coordinates()
-> Result<(), TractographyError> {
    use ritk_tck::TckTractogram;
    use ritk_trk::TrkTractogram;

    // Non-identity vox_to_ras: scale by 2 in x/y and translate [10, 20, 30].
    // The .tck header carries this as its `transform`; the .trk header uses
    // it as `vox_to_ras`.  The writer converts physical→voxel internally
    // for .trk, so we store the same physical coords in both tractograms.
    let vox_to_ras_f32: [[f32; 4]; 4] = [
        [2.0, 0.0, 0.0, 10.0],
        [0.0, 2.0, 0.0, 20.0],
        [0.0, 0.0, 1.0, 30.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    let vox_to_ras_f64: [[f64; 4]; 4] = [
        [2.0, 0.0, 0.0, 10.0],
        [0.0, 2.0, 0.0, 20.0],
        [0.0, 0.0, 1.0, 30.0],
        [0.0, 0.0, 0.0, 1.0],
    ];

    // Run Euler tractography to get physical coordinates.
    let config = TractographyConfig::new(1.0, 5, 60.0, TrackingDirection::Forward)?;
    let seeds = &[Point::new([0.0, 0.0, 0.0]), Point::new([0.0, 1.0, 0.0])];
    let result = euler_tractography(seeds, config, horizontal)?;
    assert_eq!(result.streamlines_generated(), 2);

    // Both tractograms store the same physical coordinates.  The .trk
    // writer applies `invert_affine(&vox_to_ras)` internally; we do not
    // pre-convert.
    let streamlines: Vec<_> = result
        .streamlines()
        .iter()
        .map(|s| s.geometry().clone())
        .collect();

    let trk_header = ritk_trk::TrkHeader {
        dim: [128, 128, 60],
        voxel_size: [2.0, 2.0, 2.0],
        vox_to_ras: vox_to_ras_f32,
        n_count: 2,
        ..ritk_trk::TrkHeader::default()
    };
    let trk = TrkTractogram {
        header: trk_header,
        streamlines: streamlines.clone(),
        scalars: vec![],
        properties: vec![],
    };

    let tck_header = ritk_tck::TckHeader {
        transform: Some(vox_to_ras_f64),
        ..ritk_tck::TckHeader::default()
    };
    let tck = TckTractogram {
        header: tck_header,
        streamlines,
    };

    // ── Write both, read both back ─────────────────────────────────────
    let mut trk_buf = Vec::new();
    trk.write(&mut trk_buf).expect("write .trk");
    let trk_back = TrkTractogram::read(&mut trk_buf.as_slice()).expect("read .trk");

    let mut tck_buf = Vec::new();
    tck.write(&mut tck_buf).expect("write .tck");
    let tck_back = TckTractogram::read(tck_buf.as_slice()).expect("read .tck");

    assert_eq!(trk_back.streamlines.len(), tck_back.streamlines.len());

    // Both paths go through f32 encoding (phys→f32 voxel→phys for .trk;
    // f64→f32→f64 for .tck).  The .trk path additionally loses precision
    // from the affine multiply, so use a slightly looser tolerance.
    for (trk_poly, tck_poly) in trk_back.streamlines.iter().zip(tck_back.streamlines.iter()) {
        assert_eq!(trk_poly.len(), tck_poly.len());
        for (p_trk, p_tck) in trk_poly.points().iter().zip(tck_poly.points().iter()) {
            assert!(
                (p_trk.x - p_tck.x).abs() < 2e-4,
                "x mismatch: trk={:.6} tck={:.6}",
                p_trk.x,
                p_tck.x
            );
            assert!(
                (p_trk.y - p_tck.y).abs() < 2e-4,
                "y mismatch: trk={:.6} tck={:.6}",
                p_trk.y,
                p_tck.y
            );
            assert!(
                (p_trk.z - p_tck.z).abs() < 2e-4,
                "z mismatch: trk={:.6} tck={:.6}",
                p_trk.z,
                p_tck.z
            );
        }
    }
    Ok(())
}

#[test]
fn map_points_moves_geometry_without_changing_terminations() -> Result<(), TractographyError> {
    // The CLI tracks in voxel indices and writes in physical millimetres, so
    // this is the conversion every export depends on. A termination reason
    // describes why tracking stopped, which no change of coordinates affects --
    // carrying it over unchanged is part of the contract.
    let config = TractographyConfig::new(0.5, 10, 60.0, TrackingDirection::Forward)?;
    let seeds = &[Point::new([0.0, 0.0, 0.0]), Point::new([1.0, 0.0, 0.0])];
    let result = euler_tractography(seeds, config, horizontal)?;
    assert_eq!(result.streamlines_generated(), 2);

    // An anisotropic scale plus an offset: a uniform scale would not catch an
    // axis being transposed, and a pure scale would not catch a dropped origin.
    let scale = [2.0, 3.0, 4.0];
    let offset = [10.0, -5.0, 0.5];
    let moved = result.map_points(|point| {
        let [x, y, z] = point.to_array();
        Point::new([
            x * scale[0] + offset[0],
            y * scale[1] + offset[1],
            z * scale[2] + offset[2],
        ])
    })?;

    assert_eq!(moved.streamlines().len(), result.streamlines().len());
    assert_eq!(moved.seeds_attempted(), result.seeds_attempted());

    for (before, after) in result.streamlines().iter().zip(moved.streamlines()) {
        assert_eq!(
            before.forward_termination(),
            after.forward_termination(),
            "a change of coordinates cannot change why tracking stopped"
        );
        assert_eq!(before.backward_termination(), after.backward_termination());

        assert_eq!(before.geometry().len(), after.geometry().len());
        for (p, q) in before
            .geometry()
            .points()
            .iter()
            .zip(after.geometry().points())
        {
            assert!((q.x - (p.x * scale[0] + offset[0])).abs() < 1e-9);
            assert!((q.y - (p.y * scale[1] + offset[1])).abs() < 1e-9);
            assert!((q.z - (p.z * scale[2] + offset[2])).abs() < 1e-9);
        }
    }

    // The original is untouched: exports may be produced in several frames from
    // one tracking run.
    assert!((result.streamlines()[0].geometry().points()[0].x - 0.0).abs() < 1e-9);
    Ok(())
}

#[test]
fn dti_seed_selection_is_inclusive_and_evenly_strided() -> Result<(), TractographyError> {
    let tensor = [1.7e-3, 3.0e-4, 3.0e-4, 0.0, 0.0, 0.0];
    let volume = dti_volume(&[tensor; 8], [2, 2, 2], 0.2);
    let seeds = dti_volume_seed_points(&volume, 0.25, 4)?;

    assert_eq!(seeds.len(), 4);
    let indices: Vec<[f64; 3]> = seeds.iter().map(Point::to_array).collect();
    assert_eq!(
        indices,
        vec![
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
        ]
    );
    Ok(())
}

#[test]
fn dti_seed_selection_excludes_unfitted_voxels_at_zero_threshold() -> Result<(), TractographyError>
{
    let tensor = [1.7e-3, 3.0e-4, 3.0e-4, 0.0, 0.0, 0.0];
    let scheme = image_axis_scheme();
    let fitted = dti_signal(&scheme, tensor, 1_000.0);
    let background = dti_signal(&scheme, tensor, 1.0);
    let volumes: Vec<Vec<f64>> = (0..scheme.len())
        .map(|acquisition| vec![fitted[acquisition], background[acquisition]])
        .collect();
    let borrowed: Vec<&[f64]> = volumes.iter().map(Vec::as_slice).collect();
    let maps = fit_diffusion_maps(&scheme, &borrowed, &DiffusionMapsConfig::default())
        .expect("synthetic series fits");
    assert_eq!(maps.mask(), [true, false]);
    let volume = DtiVolume::new(maps, [2, 1, 1], 0.0).expect("shape matches maps");

    let seeds = dti_volume_seed_points(&volume, 0.0, 0)?;
    assert_eq!(seeds.len(), 1);
    assert_eq!(seeds[0].to_array(), [0.0, 0.0, 0.0]);
    Ok(())
}

#[test]
fn dti_seed_mask_filters_candidates_in_dti_grid_order() -> Result<(), TractographyError> {
    let tensor = [1.7e-3, 3.0e-4, 3.0e-4, 0.0, 0.0, 0.0];
    let volume = dti_volume(&[tensor; 8], [2, 2, 2], 0.2);

    let unmasked = dti_volume_seed_points(&volume, 0.25, 0)?;
    let all_true = [true; 8];
    let selected = dti_volume_seed_points_with_mask(&volume, 0.25, 0, Some(&all_true))?;
    assert_eq!(
        selected.iter().map(Point::to_array).collect::<Vec<_>>(),
        unmasked.iter().map(Point::to_array).collect::<Vec<_>>()
    );

    let all_false = [false; 8];
    assert!(
        dti_volume_seed_points_with_mask(&volume, 0.25, 0, Some(&all_false))?.is_empty(),
        "an all-false region must select no seeds"
    );

    let sparse = [false, true, false, true, false, false, false, true];
    let selected = dti_volume_seed_points_with_mask(&volume, 0.25, 0, Some(&sparse))?;
    assert_eq!(
        selected.iter().map(Point::to_array).collect::<Vec<_>>(),
        vec![[0.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 1.0],]
    );
    Ok(())
}

#[test]
fn dti_seed_mask_restricts_tracking_before_integration() {
    let tensor = [3.0e-4, 3.0e-4, 1.7e-3, 0.0, 0.0, 0.0];
    let volume = dti_volume(&[tensor; 4], [4, 1, 1], 0.2);
    let tracking = TractographyConfig::new(1.0, 2, 60.0, TrackingDirection::Forward)
        .expect("valid tracking policy");
    let config = DtiTractographyConfig::new(0.25, 0, tracking).expect("valid DTI policy");

    let sparse = [true, false, true, false];
    let result = dti_volume_tractography_with_mask(&volume, config, Some(&sparse))
        .expect("selected voxels track");
    assert_eq!(result.seeds_attempted(), 2);
    assert_eq!(result.streamlines_generated(), 2);

    let all_false = [false; 4];
    let error = dti_volume_tractography_with_mask(&volume, config, Some(&all_false))
        .expect_err("an empty seed region must be reported");
    assert!(matches!(
        error,
        TractographyError::NoSeeds {
            threshold: 0.25,
            maximum: 0.0
        }
    ));
}

#[test]
fn dti_seed_mask_rejects_wrong_length_before_selection() {
    let tensor = [1.7e-3, 3.0e-4, 3.0e-4, 0.0, 0.0, 0.0];
    let volume = dti_volume(&[tensor; 2], [2, 1, 1], 0.2);

    let error = dti_volume_seed_points_with_mask(&volume, 0.25, 0, Some(&[true]))
        .expect_err("one flag cannot describe two DTI voxels");
    assert!(matches!(
        error,
        TractographyError::InvalidSeedMaskLength {
            actual: 1,
            expected: 2
        }
    ));
}

#[test]
fn dti_seed_mask_accepts_empty_volume_and_empty_mask() -> Result<(), TractographyError> {
    let volume = dti_volume(&[], [0, 1, 1], 0.2);
    let seeds = dti_volume_seed_points_with_mask(&volume, 0.0, 0, Some(&[]))?;
    assert!(seeds.is_empty());
    Ok(())
}

#[test]
fn dti_volume_tractography_tracks_selected_seeds_and_reports_empty_selection() {
    let tensor = [3.0e-4, 3.0e-4, 1.7e-3, 0.0, 0.0, 0.0];
    let volume = dti_volume(&[tensor; 3], [3, 1, 1], 0.2);
    let tracking = TractographyConfig::new(1.0, 2, 60.0, TrackingDirection::Bidirectional)
        .expect("valid tracking policy");
    let config = DtiTractographyConfig::new(0.25, 0, tracking).expect("valid DTI policy");
    let result = dti_volume_tractography(&volume, config).expect("selected voxels track");

    assert_eq!(result.seeds_attempted(), 3);
    assert_eq!(result.streamlines_generated(), 3);
    assert!(result.streamlines().iter().all(|line| {
        line.forward_termination() == TerminationReason::FieldBoundary
            || line.forward_termination() == TerminationReason::StepLimit
    }));

    let no_seeds = DtiTractographyConfig::new(1.0, 0, tracking).expect("valid threshold");
    let error = dti_volume_tractography(&volume, no_seeds).expect_err("FA is below one");
    assert!(matches!(
        error,
        TractographyError::NoSeeds {
            threshold: 1.0,
            maximum
        } if maximum > 0.25
    ));
}

#[test]
fn dti_seed_configuration_rejects_non_fractional_thresholds() {
    let tracking = TractographyConfig::new(1.0, 1, 60.0, TrackingDirection::Forward)
        .expect("valid tracking policy");
    for value in [-f64::EPSILON, 1.0 + f64::EPSILON, f64::NAN, f64::INFINITY] {
        assert!(matches!(
            DtiTractographyConfig::new(value, 1, tracking),
            Err(TractographyError::InvalidSeedAnisotropy { .. })
        ));
    }
}
