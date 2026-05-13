use super::*;

// ── Positive: Connected-threshold grows sphere region ─────────────────────

/// Seeding at the centre of the sphere must grow exactly the 7 high-intensity
/// voxels (centre + 6 face-adjacent neighbours).
#[test]
fn test_segment_connected_threshold_grows_sphere_from_centre_seed() {
    let dir = tempdir().unwrap();
    let input = dir.path().join("input.nii");
    let output = dir.path().join("mask.nii");

    ritk_io::write_nifti(&input, &make_sphere_image()).unwrap();

    run(SegmentArgs {
        input: input.clone(),
        output: output.clone(),
        method: "connected-threshold".to_string(),
        classes: 3,
        lower: Some(100.0),
        upper: Some(255.0),
        seed: Some("2,2,2".to_string()),
        multiplier: 2.5,
        ..Default::default()
    })
    .unwrap();

    assert!(output.exists(), "output mask must be created");
    let mask = ritk_io::read_nifti::<Backend, _>(&output, &Default::default()).unwrap();
    let foreground = count_foreground(&mask);
    assert_eq!(
        foreground, 7,
        "connected-threshold from centre seed must grow exactly 7 sphere voxels"
    );
}

// ── Positive: Connected-threshold output is strictly binary ───────────────

#[test]
fn test_segment_connected_threshold_output_is_strictly_binary() {
    let dir = tempdir().unwrap();
    let input = dir.path().join("input.mha");
    let output = dir.path().join("mask.mha");

    ritk_io::write_metaimage(&input, &make_sphere_image()).unwrap();

    run(SegmentArgs {
        input: input.clone(),
        output: output.clone(),
        method: "connected-threshold".to_string(),
        classes: 3,
        lower: Some(100.0),
        upper: Some(255.0),
        seed: Some("2,2,2".to_string()),
        multiplier: 2.5,
        ..Default::default()
    })
    .unwrap();

    let mask = ritk_io::read_metaimage::<Backend, _>(&output, &Default::default()).unwrap();
    let td = mask.data().clone().into_data();
    let values = td.as_slice::<f32>().unwrap();
    for &v in values {
        assert!(
            v == 0.0 || v == 1.0,
            "connected-threshold output must be strictly binary, got {v}"
        );
    }
}

// ── Negative: connected-threshold missing --lower ─────────────────────────

#[test]
fn test_segment_connected_threshold_missing_lower_returns_error() {
    let dir = tempdir().unwrap();
    let input = dir.path().join("input.nii");
    let output = dir.path().join("out.nii");
    ritk_io::write_nifti(&input, &make_sphere_image()).unwrap();

    let result = run(SegmentArgs {
        input,
        output,
        method: "connected-threshold".to_string(),
        classes: 3,
        lower: None,
        upper: Some(255.0),
        seed: Some("2,2,2".to_string()),
        multiplier: 2.5,
        ..Default::default()
    });

    assert!(result.is_err(), "missing --lower must yield an error");
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("--lower is required"),
        "error must name the missing argument, got: {msg}"
    );
}

// ── Negative: connected-threshold missing --upper ─────────────────────────

#[test]
fn test_segment_connected_threshold_missing_upper_returns_error() {
    let dir = tempdir().unwrap();
    let input = dir.path().join("input.nii");
    let output = dir.path().join("out.nii");
    ritk_io::write_nifti(&input, &make_sphere_image()).unwrap();

    let result = run(SegmentArgs {
        input,
        output,
        method: "connected-threshold".to_string(),
        classes: 3,
        lower: Some(100.0),
        upper: None,
        seed: Some("2,2,2".to_string()),
        multiplier: 2.5,
        ..Default::default()
    });

    assert!(result.is_err(), "missing --upper must yield an error");
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("--upper is required"),
        "error must name the missing argument, got: {msg}"
    );
}

// ── Negative: connected-threshold missing --seed ──────────────────────────

#[test]
fn test_segment_connected_threshold_missing_seed_returns_error() {
    let dir = tempdir().unwrap();
    let input = dir.path().join("input.nii");
    let output = dir.path().join("out.nii");
    ritk_io::write_nifti(&input, &make_sphere_image()).unwrap();

    let result = run(SegmentArgs {
        input,
        output,
        method: "connected-threshold".to_string(),
        classes: 3,
        lower: Some(100.0),
        upper: Some(255.0),
        seed: None,
        multiplier: 2.5,
        ..Default::default()
    });

    assert!(result.is_err(), "missing --seed must yield an error");
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("--seed is required"),
        "error must name the missing argument, got: {msg}"
    );
}

// ── Negative: connected-threshold lower > upper ───────────────────────────

#[test]
fn test_segment_connected_threshold_lower_gt_upper_returns_error() {
    let dir = tempdir().unwrap();
    let input = dir.path().join("input.nii");
    let output = dir.path().join("out.nii");
    ritk_io::write_nifti(&input, &make_sphere_image()).unwrap();

    let result = run(SegmentArgs {
        input,
        output,
        method: "connected-threshold".to_string(),
        classes: 3,
        lower: Some(255.0),
        upper: Some(100.0),
        seed: Some("2,2,2".to_string()),
        multiplier: 2.5,
        ..Default::default()
    });

    assert!(result.is_err(), "lower > upper must yield an error");
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("must be \u{2264}")
            || msg.contains("must be <=")
            || msg.contains('\u{2264}'),
        "error must explain the bound constraint, got: {msg}"
    );
}

// ── Negative: out-of-bounds seed returns error ────────────────────────────

#[test]
fn test_segment_connected_threshold_out_of_bounds_seed_returns_error() {
    let dir = tempdir().unwrap();
    let input = dir.path().join("input.nii");
    let output = dir.path().join("out.nii");
    ritk_io::write_nifti(&input, &make_sphere_image()).unwrap();

    let result = run(SegmentArgs {
        input,
        output,
        method: "connected-threshold".to_string(),
        classes: 3,
        lower: Some(100.0),
        upper: Some(255.0),
        seed: Some("99,99,99".to_string()),
        multiplier: 2.5,
        ..Default::default()
    });

    assert!(result.is_err(), "out-of-bounds seed must yield an error");
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("out of bounds"),
        "error must explain the bounds problem, got: {msg}"
    );
}

// ── Negative: malformed seed string returns error ─────────────────────────

#[test]
fn test_segment_malformed_seed_returns_error() {
    let dir = tempdir().unwrap();
    let input = dir.path().join("input.nii");
    let output = dir.path().join("out.nii");
    ritk_io::write_nifti(&input, &make_sphere_image()).unwrap();

    let result = run(SegmentArgs {
        input,
        output,
        method: "connected-threshold".to_string(),
        classes: 3,
        lower: Some(100.0),
        upper: Some(255.0),
        seed: Some("2,2".to_string()),
        multiplier: 2.5,
        ..Default::default()
    });

    assert!(result.is_err(), "malformed seed must yield an error");
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("Z,Y,X"),
        "error must explain the expected format, got: {msg}"
    );
}

// -- confidence-connected: positive ---------------------------------------

#[test]
fn test_segment_confidence_connected_grows_region() {
    let dir = tempdir().unwrap();
    let input = dir.path().join("input.nii");
    let output = dir.path().join("mask.nii");
    ritk_io::write_nifti(&input, &make_sphere_image()).unwrap();

    let mut args = default_args(input.clone(), output.clone(), "confidence-connected");
    args.lower = Some(150.0);
    args.upper = Some(250.0);
    args.seed = Some("2,2,2".to_string());
    run(args).unwrap();

    assert!(output.exists(), "output mask must be created");
    let mask = ritk_io::read_nifti::<Backend, _>(&output, &Default::default()).unwrap();
    assert_eq!(mask.shape(), [5, 5, 5], "shape must match input");
    let n_fg = count_foreground(&mask);
    assert!(n_fg > 0, "must find at least one foreground voxel, got 0");
}

#[test]
fn test_segment_confidence_connected_output_is_binary() {
    let dir = tempdir().unwrap();
    let input = dir.path().join("input.nii");
    let output = dir.path().join("mask.nii");
    ritk_io::write_nifti(&input, &make_sphere_image()).unwrap();

    let mut args = default_args(input.clone(), output.clone(), "confidence-connected");
    args.lower = Some(150.0);
    args.upper = Some(250.0);
    args.seed = Some("2,2,2".to_string());
    run(args).unwrap();

    let mask = ritk_io::read_nifti::<Backend, _>(&output, &Default::default()).unwrap();
    let td = mask.data().clone().into_data();
    let vals = td.as_slice::<f32>().unwrap();
    for &v in vals {
        assert!(
            v == 0.0 || v == 1.0,
            "all voxels must be 0.0 or 1.0, found {v}"
        );
    }
}

#[test]
fn test_segment_confidence_connected_missing_lower_returns_error() {
    let dir = tempdir().unwrap();
    let input = dir.path().join("input.nii");
    let output = dir.path().join("mask.nii");
    ritk_io::write_nifti(&input, &make_sphere_image()).unwrap();
    let mut args = default_args(input, output, "confidence-connected");
    args.upper = Some(1.5);
    args.seed = Some("2,2,2".to_string());
    let result = run(args);
    assert!(result.is_err(), "--lower missing must produce an error");
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("--lower"),
        "error must mention --lower, got: {msg}"
    );
}

#[test]
fn test_segment_confidence_connected_missing_upper_returns_error() {
    let dir = tempdir().unwrap();
    let input = dir.path().join("input.nii");
    let output = dir.path().join("mask.nii");
    ritk_io::write_nifti(&input, &make_sphere_image()).unwrap();
    let mut args = default_args(input, output, "confidence-connected");
    args.lower = Some(0.5);
    args.seed = Some("2,2,2".to_string());
    let result = run(args);
    assert!(result.is_err(), "--upper missing must produce an error");
}

#[test]
fn test_segment_confidence_connected_missing_seed_returns_error() {
    let dir = tempdir().unwrap();
    let input = dir.path().join("input.nii");
    let output = dir.path().join("mask.nii");
    ritk_io::write_nifti(&input, &make_sphere_image()).unwrap();
    let mut args = default_args(input, output, "confidence-connected");
    args.lower = Some(0.5);
    args.upper = Some(1.5);
    let result = run(args);
    assert!(result.is_err(), "--seed missing must produce an error");
}

// -- neighborhood-connected: positive -------------------------------------

#[test]
fn test_segment_neighborhood_connected_grows_region() {
    let dir = tempdir().unwrap();
    let input = dir.path().join("input.nii");
    let output = dir.path().join("mask.nii");
    ritk_io::write_nifti(&input, &make_sphere_image()).unwrap();

    let mut args = default_args(input.clone(), output.clone(), "neighborhood-connected");
    args.lower = Some(0.0);
    args.upper = Some(250.0);
    args.seed = Some("2,2,2".to_string());
    run(args).unwrap();

    assert!(output.exists(), "output mask must be created");
    let mask = ritk_io::read_nifti::<Backend, _>(&output, &Default::default()).unwrap();
    assert_eq!(mask.shape(), [5, 5, 5]);
    let n_fg = count_foreground(&mask);
    assert!(n_fg > 0, "must find foreground voxels, got 0");
}

#[test]
fn test_segment_neighborhood_connected_output_is_binary() {
    let dir = tempdir().unwrap();
    let input = dir.path().join("input.nii");
    let output = dir.path().join("mask.nii");
    ritk_io::write_nifti(&input, &make_sphere_image()).unwrap();

    let mut args = default_args(input.clone(), output.clone(), "neighborhood-connected");
    args.lower = Some(0.0);
    args.upper = Some(250.0);
    args.seed = Some("2,2,2".to_string());
    run(args).unwrap();

    let mask = ritk_io::read_nifti::<Backend, _>(&output, &Default::default()).unwrap();
    let td = mask.data().clone().into_data();
    let vals = td.as_slice::<f32>().unwrap();
    for &v in vals {
        assert!(
            v == 0.0 || v == 1.0,
            "all voxels must be 0.0 or 1.0, found {v}"
        );
    }
}
