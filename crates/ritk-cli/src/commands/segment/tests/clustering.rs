use super::*;

// ── Helper: binary image with specified components ────────────────────────────

fn make_binary_image_with_components(
    dims: [usize; 3],
    components: &[(usize, usize, usize, usize, usize, usize)],
) -> Image<Backend, 3> {
    let device: <Backend as BurnBackend>::Device = Default::default();
    let [nz, ny, nx] = dims;
    let n = nz * ny * nx;
    let mut vals = vec![0.0_f32; n];
    for &(z0, y0, x0, z1, y1, x1) in components {
        for iz in z0..z1 {
            for iy in y0..y1 {
                for ix in x0..x1 {
                    vals[iz * ny * nx + iy * nx + ix] = 1.0;
                }
            }
        }
    }
    let td = TensorData::new(vals, Shape::new(dims));
    let tensor = Tensor::<Backend, 3>::from_data(td, &device);
    Image::new(
        tensor,
        Point::new([0.0; 3]),
        Spacing::new([1.0; 3]),
        Direction::identity(),
    )
}

// ── Positive: K-Means creates output with cluster labels ──────────────────

#[test]
fn test_segment_kmeans_creates_output_with_valid_labels() {
    let dir = tempdir().unwrap();
    let input = dir.path().join("input.nii");
    let output = dir.path().join("labels.nii");

    ritk_io::write_nifti(&input, &make_trimodal_image()).unwrap();

    run(default_args(
        input.clone(),
        output.clone(),
        SegmentMethod::Kmeans,
    ))
    .unwrap();

    assert!(output.exists(), "kmeans output must be created");
    let labels = ritk_io::read_nifti::<Backend, _>(&output, &Default::default()).unwrap();
    assert_eq!(labels.shape(), [6, 6, 6]);
    let vals: Vec<f32> = labels.data_slice().into_owned();
    for &v in &vals {
        assert!(
            (0.0..3.0 + 0.5).contains(&v),
            "kmeans label {v} must be in [0, 2]"
        );
    }
    let mut unique = vals.clone();
    unique.sort_by(|a, b| a.partial_cmp(b).unwrap());
    unique.dedup_by(|a, b| (*a - *b).abs() < 0.01);
    assert_eq!(
        unique.len(),
        3,
        "trimodal image with k=3 must produce exactly 3 distinct labels, got {:?}",
        unique
    );
}

#[test]
fn test_segment_kmeans_max_iterations_param_accepted() {
    let dir = tempdir().unwrap();
    let input = dir.path().join("in.nii.gz");
    let output = dir.path().join("out.nii.gz");
    ritk_io::write_nifti(&input, &make_bimodal_image()).unwrap();
    let args = SegmentArgs {
        input,
        output: output.clone(),
        method: SegmentMethod::Kmeans,
        classes: 2,
        kmeans_max_iterations: Some(50),
        ..Default::default()
    };
    run(args).unwrap();
    assert!(
        output.exists(),
        "kmeans output must be created when kmeans_max_iterations is set"
    );
}

#[test]
fn test_segment_kmeans_seed_produces_deterministic_output() {
    let dir = tempdir().unwrap();
    let img = make_bimodal_image();
    let inp1 = dir.path().join("in1.nii.gz");
    let inp2 = dir.path().join("in2.nii.gz");
    let out1 = dir.path().join("out1.nii.gz");
    let out2 = dir.path().join("out2.nii.gz");
    ritk_io::write_nifti(&inp1, &img).unwrap();
    ritk_io::write_nifti(&inp2, &img).unwrap();
    let make_args = |input: std::path::PathBuf, output: std::path::PathBuf| SegmentArgs {
        input,
        output,
        method: SegmentMethod::Kmeans,
        classes: 2,
        kmeans_seed: Some(7),
        ..Default::default()
    };
    run(make_args(inp1, out1.clone())).unwrap();
    run(make_args(inp2, out2.clone())).unwrap();
    let read_vals = |p: &std::path::Path| -> Vec<f32> {
        let im: Image<Backend, 3> = ritk_io::read_nifti(p, &Default::default()).unwrap();
        im.data_slice().into_owned()
    };
    assert_eq!(
        read_vals(&out1),
        read_vals(&out2),
        "same seed must produce identical output"
    );
}

#[test]
fn test_segment_kmeans_tolerance_param_accepted() {
    let dir = tempdir().unwrap();
    let input = dir.path().join("in.nii.gz");
    let output = dir.path().join("out.nii.gz");
    ritk_io::write_nifti(&input, &make_bimodal_image()).unwrap();
    let args = SegmentArgs {
        input,
        output: output.clone(),
        method: SegmentMethod::Kmeans,
        classes: 2,
        kmeans_tolerance: Some(1e-4),
        ..Default::default()
    };
    run(args).unwrap();
    assert!(
        output.exists(),
        "kmeans output must be created when kmeans_tolerance is set"
    );
}

#[test]
fn native_kmeans_cli_matches_canonical_legacy_output_exactly() {
    let dir = tempdir().unwrap();
    let input = dir.path().join("input.nii");
    let output = dir.path().join("labels.nii");
    let image = make_trimodal_image();
    ritk_io::write_nifti(&input, &image).unwrap();
    let args = SegmentArgs {
        input,
        output: output.clone(),
        method: SegmentMethod::Kmeans,
        classes: 3,
        kmeans_max_iterations: Some(20),
        kmeans_tolerance: Some(0.0),
        kmeans_seed: Some(7),
        ..Default::default()
    };
    run(args).unwrap();

    let expected = ritk_segmentation::KMeansSegmentation::new(3)
        .unwrap()
        .with_max_iterations(20)
        .unwrap()
        .with_tolerance(0.0)
        .unwrap()
        .with_seed(7)
        .apply(&image)
        .unwrap();
    let actual = ritk_io::read_nifti::<Backend, _>(&output, &Default::default()).unwrap();
    assert_eq!(actual.data_slice(), expected.data_slice());
    assert_eq!(actual.origin(), image.origin());
    assert_eq!(actual.spacing(), image.spacing());
    assert_eq!(actual.direction(), image.direction());
}

#[test]
fn native_kmeans_cli_rejects_invalid_configuration_and_nonnative_input() {
    let dir = tempdir().unwrap();
    let valid_input = dir.path().join("input.nii");
    ritk_io::write_nifti(&valid_input, &make_bimodal_image()).unwrap();
    let invalid_k = run(SegmentArgs {
        input: valid_input,
        output: dir.path().join("output.nii"),
        method: SegmentMethod::Kmeans,
        classes: 0,
        ..Default::default()
    })
    .unwrap_err();
    assert_eq!(invalid_k.to_string(), "k must be at least 1, got 0");

    let output = dir.path().join("output.nii");
    let format_error = run(default_args(
        dir.path().join("input.vtk"),
        output.clone(),
        SegmentMethod::Kmeans,
    ))
    .unwrap_err();
    assert_eq!(
        format_error.to_string(),
        "kmeans requires native input/output formats"
    );
    assert!(!output.exists());
}

// ── Positive: Distance transform creates output ───────────────────────────

#[test]
fn test_segment_distance_transform_creates_output() {
    let dir = tempdir().unwrap();
    let input = dir.path().join("input.nii");
    let output = dir.path().join("dt.nii");

    ritk_io::write_nifti(&input, &make_binary_image()).unwrap();

    run(default_args(
        input.clone(),
        output.clone(),
        SegmentMethod::DistanceTransform,
    ))
    .unwrap();

    assert!(output.exists(), "distance-transform output must be created");
    let dt = ritk_io::read_nifti::<Backend, _>(&output, &Default::default()).unwrap();
    assert_eq!(dt.shape(), [4, 4, 4], "shape must be preserved");
    dt.with_data_slice(|vals| {
        for &v in vals {
            assert!(
                v >= 0.0,
                "distance transform values must be non-negative, got {v}"
            );
        }
        let has_positive = vals.iter().any(|&v| v > 0.0);
        assert!(
            has_positive,
            "distance-transform must produce at least one positive value for a non-trivial mask"
        );
    });
}

#[test]
fn test_segment_distance_transform_background_is_zero() {
    let dir = tempdir().unwrap();
    let input = dir.path().join("input.nii");
    let output = dir.path().join("dt.nii");

    let device: <Backend as BurnBackend>::Device = Default::default();
    let values = vec![0.0_f32; 27];
    let td = TensorData::new(values, Shape::new([3, 3, 3]));
    let tensor = Tensor::<Backend, 3>::from_data(td, &device);
    let img = Image::new(
        tensor,
        Point::new([0.0; 3]),
        Spacing::new([1.0; 3]),
        Direction::identity(),
    );
    ritk_io::write_nifti(&input, &img).unwrap();

    run(default_args(
        input.clone(),
        output.clone(),
        SegmentMethod::DistanceTransform,
    ))
    .unwrap();

    let dt = ritk_io::read_nifti::<Backend, _>(&output, &Default::default()).unwrap();
    dt.with_data_slice(|vals| {
        for &v in vals {
            assert_eq!(
                v, 0.0,
                "all-background image must have EDT=0 everywhere, got {v}"
            );
        }
    });
}

#[test]
fn native_distance_transform_cli_preserves_exact_physical_values_and_geometry() {
    let dir = tempdir().unwrap();
    let input = dir.path().join("input.nii");
    let output = dir.path().join("distance.nii");
    let device: <Backend as BurnBackend>::Device = Default::default();
    let tensor = Tensor::<Backend, 3>::from_data(
        TensorData::new(vec![1.0, 0.0, 0.0, 0.0], Shape::new([1, 1, 4])),
        &device,
    );
    let image = Image::new(
        tensor,
        Point::new([2.0, 3.0, 5.0]),
        Spacing::new([2.0, 3.0, 4.0]),
        Direction::identity(),
    );
    ritk_io::write_nifti(&input, &image).unwrap();
    run(default_args(
        input,
        output.clone(),
        SegmentMethod::DistanceTransform,
    ))
    .unwrap();
    let actual = ritk_io::read_nifti::<Backend, _>(&output, &Default::default()).unwrap();
    assert_eq!(actual.data_slice().as_ref(), &[0.0, 4.0, 8.0, 12.0]);
    assert_eq!(actual.origin(), image.origin());
    assert_eq!(actual.spacing(), image.spacing());
    assert_eq!(actual.direction(), image.direction());
}

// ── Fill-holes tests ──────────────────────────────────────────────────────

#[test]
fn test_segment_fill_holes_fills_enclosed_cavity() {
    let dir = tempdir().unwrap();
    let input = dir.path().join("input.nii");
    let output = dir.path().join("filled.nii");
    let device: <Backend as BurnBackend>::Device = Default::default();
    let (nz, ny, nx) = (7usize, 7usize, 7usize);
    let n = nz * ny * nx;
    let mut vals = vec![0.0_f32; n];
    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let d2 = ((iz as i32 - 3).pow(2) + (iy as i32 - 3).pow(2) + (ix as i32 - 3).pow(2))
                    as f32;
                if (4.0..=9.0).contains(&d2) {
                    vals[iz * ny * nx + iy * nx + ix] = 1.0;
                }
            }
        }
    }
    let td = TensorData::new(vals, Shape::new([nz, ny, nx]));
    let tensor = Tensor::<Backend, 3>::from_data(td, &device);
    let hollow_sphere = Image::new(
        tensor,
        Point::new([0.0; 3]),
        Spacing::new([1.0; 3]),
        Direction::identity(),
    );
    ritk_io::write_nifti(&input, &hollow_sphere).unwrap();
    run(default_args(
        input.clone(),
        output.clone(),
        SegmentMethod::FillHoles,
    ))
    .unwrap();
    let result = ritk_io::read_nifti::<Backend, _>(&output, &Default::default()).unwrap();
    result.with_data_slice(|out_vals| {
        for iz in 0..nz {
            for iy in 0..ny {
                for ix in 0..nx {
                    let d2 = ((iz as i32 - 3).pow(2)
                        + (iy as i32 - 3).pow(2)
                        + (ix as i32 - 3).pow(2)) as f32;
                    if d2 < 4.0 {
                        assert_eq!(
                            out_vals[iz * ny * nx + iy * nx + ix],
                            1.0,
                            "interior voxel ({},{},{}) at d2={} must be filled",
                            iz,
                            iy,
                            ix,
                            d2
                        );
                    }
                }
            }
        }
    });
}

// ── Morphological gradient tests ─────────────────────────────────────────

#[test]
fn test_segment_morphological_gradient_extracts_boundary() {
    let dir = tempdir().unwrap();
    let input = dir.path().join("input.nii");
    let output = dir.path().join("gradient.nii");
    ritk_io::write_nifti(&input, &make_binary_sphere_image()).unwrap();
    run(default_args(
        input.clone(),
        output.clone(),
        SegmentMethod::MorphologicalGradient,
    ))
    .unwrap();
    let result = ritk_io::read_nifti::<Backend, _>(&output, &Default::default()).unwrap();
    result.with_data_slice(|vals| {
        assert_eq!(vals.len(), 125);
        assert!(
            vals.contains(&1.0),
            "morphological gradient must contain boundary voxels"
        );
        assert!(
            vals.iter().all(|&v| v == 0.0 || v == 1.0),
            "morphological gradient must be binary"
        );
    });
}

// ── Skeletonization tests ─────────────────────────────────────────────────

#[test]
fn test_segment_skeletonization_creates_output() {
    let dir = tempdir().unwrap();
    let input = dir.path().join("input.nii");
    let output = dir.path().join("skeleton.nii");
    ritk_io::write_nifti(&input, &make_binary_sphere_image()).unwrap();

    run(default_args(
        input.clone(),
        output.clone(),
        SegmentMethod::Skeletonization,
    ))
    .unwrap();

    assert!(output.exists(), "skeleton output must be created");
    let skel = ritk_io::read_nifti::<Backend, _>(&output, &Default::default()).unwrap();
    assert_eq!(skel.shape(), [5, 5, 5], "skeleton shape must match input");
}

#[test]
fn test_segment_skeletonization_strictly_binary() {
    let dir = tempdir().unwrap();
    let input = dir.path().join("input.nii");
    let output = dir.path().join("skeleton.nii");
    ritk_io::write_nifti(&input, &make_binary_sphere_image()).unwrap();
    run(default_args(
        input.clone(),
        output.clone(),
        SegmentMethod::Skeletonization,
    ))
    .unwrap();
    let skel = ritk_io::read_nifti::<Backend, _>(&output, &Default::default()).unwrap();
    skel.with_data_slice(|vals| {
        for &v in vals {
            assert!(
                v == 0.0 || v == 1.0,
                "skeleton voxels must be 0.0 or 1.0, found {v}"
            );
        }
    });
}

#[test]
fn native_postprocessing_cli_matches_legacy_exactly() {
    use ritk_segmentation::{
        BinaryFillHoles, MorphologicalGradient, MorphologicalOperation, Skeletonization,
    };

    let dir = tempdir().unwrap();
    let input = dir.path().join("input.nii");
    let image = make_binary_sphere_image();
    ritk_io::write_nifti(&input, &image).unwrap();
    let cases = [
        (
            SegmentMethod::FillHoles,
            BinaryFillHoles.apply(&image).data_slice().to_vec(),
        ),
        (
            SegmentMethod::MorphologicalGradient,
            MorphologicalGradient::new(1)
                .apply(&image)
                .data_slice()
                .to_vec(),
        ),
        (
            SegmentMethod::Skeletonization,
            Skeletonization::new().apply(&image).data_slice().to_vec(),
        ),
    ];
    for (index, (method, expected)) in cases.into_iter().enumerate() {
        let output = dir.path().join(format!("output-{index}.nii"));
        run(default_args(input.clone(), output.clone(), method)).unwrap();
        let actual = ritk_io::read_nifti::<Backend, _>(&output, &Default::default()).unwrap();
        assert_eq!(actual.data_slice(), expected);
        assert_eq!(actual.origin(), image.origin());
        assert_eq!(actual.spacing(), image.spacing());
        assert_eq!(actual.direction(), image.direction());
    }
}

// ── Connected-components tests ────────────────────────────────────────────

#[test]
fn test_segment_connected_components_creates_output_with_correct_shape() {
    let dir = tempdir().unwrap();
    let input = dir.path().join("mask.nii");
    let output = dir.path().join("labels.nii");

    let image =
        make_binary_image_with_components([8, 8, 8], &[(2, 2, 2, 4, 4, 4), (5, 5, 5, 7, 7, 7)]);

    ritk_io::write_nifti(&input, &image).unwrap();

    let args = default_args(
        input.clone(),
        output.clone(),
        SegmentMethod::ConnectedComponents,
    );
    let result = run(args);
    assert!(result.is_ok(), "connected-components should succeed");

    let labels = ritk_io::read_nifti::<Backend, _>(&output, &Default::default()).unwrap();
    assert_eq!(labels.shape(), [8, 8, 8], "output shape must match input");
}

#[test]
fn test_segment_connected_components_output_labels_are_valid() {
    let dir = tempdir().unwrap();
    let input = dir.path().join("mask.nii");
    let output = dir.path().join("labels.nii");

    let image =
        make_binary_image_with_components([6, 6, 6], &[(1, 1, 1, 3, 3, 3), (4, 4, 4, 6, 6, 6)]);

    ritk_io::write_nifti(&input, &image).unwrap();

    let args = default_args(
        input.clone(),
        output.clone(),
        SegmentMethod::ConnectedComponents,
    );
    run(args).unwrap();

    let labels = ritk_io::read_nifti::<Backend, _>(&output, &Default::default()).unwrap();
    labels.with_data_slice(|slice| {
        for &v in slice {
            assert!(
                v == 0.0 || v == 1.0 || v == 2.0,
                "label must be 0, 1, or 2, got {}",
                v
            );
        }
        let has_label_1 = slice.contains(&1.0);
        let has_label_2 = slice.contains(&2.0);
        assert!(has_label_1, "label 1 must be present");
        assert!(has_label_2, "label 2 must be present");
    });
}

#[test]
fn test_segment_connected_components_connectivity_26() {
    let dir = tempdir().unwrap();
    let input = dir.path().join("mask.nii");
    let output = dir.path().join("labels.nii");

    let image =
        make_binary_image_with_components([4, 4, 4], &[(1, 1, 1, 2, 2, 2), (2, 2, 2, 3, 3, 3)]);

    ritk_io::write_nifti(&input, &image).unwrap();

    let mut args = default_args(
        input.clone(),
        output.clone(),
        SegmentMethod::ConnectedComponents,
    );
    args.connectivity = 26;
    run(args).unwrap();

    let labels = ritk_io::read_nifti::<Backend, _>(&output, &Default::default()).unwrap();
    labels.with_data_slice(|slice| {
        let has_label_1 = slice.contains(&1.0);
        assert!(
            has_label_1,
            "component must be labeled with 26-connectivity"
        );
    });
}
