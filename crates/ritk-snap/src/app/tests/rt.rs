//! RT dose/plan loading and viewer-core filter dispatch tests.

use super::*;
use arrayvec::ArrayString;

// ── SnapApp RT plan/dose tests ──────────────────────────────────────────────

#[test]
fn load_rt_plan_file_sets_plan_summary_state() {
    let mut app = SnapApp::default();
    let tmp = tempfile::tempdir().expect("tempdir");
    let path = tmp.path().join("plan.dcm");
    let plan = ritk_io::RtPlanInfo {
        sop_instance_uid: ArrayString::new(),
        rt_plan_label: "PLAN_A".to_owned(),
        rt_plan_name: "Plan A".to_owned(),
        rt_plan_description: "Synthetic plan".to_owned(),
        plan_intent: ArrayString::from("CURATIVE").unwrap(),
        beams: vec![ritk_io::RtBeamInfo {
            beam_number: 1,
            beam_name: "BEAM_1".to_owned(),
            beam_description: "Beam one".to_owned(),
            radiation_type: ArrayString::from("PHOTON").unwrap(),
            treatment_delivery_type: ArrayString::from("TREATMENT").unwrap(),
            n_control_points: 2,
        }],
        fraction_groups: vec![ritk_io::RtFractionGroup {
            fraction_group_number: 1,
            n_fractions_planned: 30,
            referenced_beam_numbers: vec![1],
        }],
    };
    ritk_io::write_rt_plan(&path, &plan).expect("write rt plan");
    app.load_rt_plan_file(path.clone());
    let loaded = app.rt_plan.as_ref().expect("rt plan loaded");
    assert_eq!(loaded.rt_plan_label, "PLAN_A");
    assert_eq!(loaded.beams.len(), 1);
    assert_eq!(loaded.fraction_groups.len(), 1);
    assert!(
        app.status_message.contains("Loaded RT-PLAN PLAN_A"),
        "status: {}",
        app.status_message
    );
}

#[test]
fn rt_dose_plan_link_status_reports_linked_uid() {
    let mut app = SnapApp::default();
    app.rt_plan = Some(ritk_io::RtPlanInfo {
        sop_instance_uid: ArrayString::from("2.25.9001").unwrap(),
        rt_plan_label: "PLAN_LINK".to_owned(),
        rt_plan_name: String::new(),
        rt_plan_description: String::new(),
        plan_intent: ArrayString::new(),
        beams: vec![],
        fraction_groups: vec![],
    });
    app.rt_dose = Some(ritk_io::RtDoseGrid {
        rows: 1,
        cols: 1,
        n_frames: 1,
        dose_type: ArrayString::from("PHYSICAL").unwrap(),
        dose_summation_type: ArrayString::from("PLAN").unwrap(),
        dose_grid_scaling: 1.0,
        frame_offsets: vec![0.0],
        dose_gy: vec![1.0],
        image_position: None,
        image_orientation: None,
        pixel_spacing: None,
        referenced_rt_plan_sop_instance_uid: Some(ArrayString::from("2.25.9001").unwrap()),
    });
    let msg = app.rt_dose_plan_link_status().expect("link status present");
    assert!(
        msg.contains("linked to loaded RT-PLAN UID 2.25.9001"),
        "{msg}"
    );
}

// ── ViewerCore filter dispatch tests (from lib.rs) ──────────────────────────
//
// These tests exercise ViewerCore, Study, Image, and FilterKind — types
// defined in the crate root (lib.rs). They use `burn_ndarray::NdArray<f32>`
// as the backend.

use burn::tensor::{Shape, Tensor, TensorData};
use ritk_core::image::Image;
use ritk_core::spatial::{Direction, Point, Spacing};
use ritk_filter::{BedSeparationConfig, ComponentPolicy};

type Backend = burn_ndarray::NdArray<f32>;

/// Analytically constructed 2×4×4 volume:
/// - First 16 voxels (slice 0, all rows/cols): -1000.0 HU (air)
/// - Next 16 voxels (slice 1, all rows/cols): 0.0 HU (soft tissue)
///
/// BedSeparationConfig:
/// body_threshold = -500.0 → voxels < -500 are background
/// background_threshold = -700.0 (unused by current impl; kept for API completeness)
/// keep_largest_component = false (avoids connected-component overhead)
/// closing_radius = 0, opening_radius = 0 (identity morphology)
/// outside_value = -2048.0
///
/// Expected outcome:
/// All 16 air voxels (-1000 < -500) → replaced by -2048.0
/// All 16 tissue voxels (0 ≥ -500) → unchanged at 0.0
/// Shape is preserved: [2, 4, 4]
#[test]
fn test_filter_kind_bed_separation_dispatch_replaces_study_image() {
    let device: <Backend as burn::tensor::backend::Backend>::Device = Default::default();
    // Analytically: first 16 = -1000.0 (air, below threshold), next 16 = 0.0 (tissue).
    let mut vals = vec![-1000.0_f32; 16];
    vals.extend_from_slice(&[0.0_f32; 16]);
    assert_eq!(vals.len(), 32, "2×4×4 = 32 voxels");
    let tensor =
        Tensor::<Backend, 3>::from_data(TensorData::new(vals, Shape::new([2, 4, 4])), &device);
    let image = Image::new(
        tensor,
        Point::new([0.0, 0.0, 0.0]),
        Spacing::new([1.0, 1.0, 1.0]),
        Direction::identity(),
    );
    let mut core = crate::ViewerCore::<Backend, 3>::new();
    core.load_study(crate::Study::new(image));
    let config = BedSeparationConfig {
        body_threshold: -500.0,
        background_threshold: -700.0,
        component_policy: ComponentPolicy::All,
        closing_radius: 0,
        opening_radius: 0,
        outside_value: -2048.0,
    };
    let result = core.apply_filter(&crate::FilterKind::BedSeparation(config));
    // Event must be Status.
    let event = result.expect("apply_filter must succeed");
    assert!(
        matches!(event, crate::ViewerEvent::Status { .. }),
        "expected ViewerEvent::Status, got {:?}",
        event
    );
    // Study must still be present and shape must be preserved.
    let study = core.study().expect("study must be present after filter");
    assert_eq!(
        study.image.shape(),
        [2, 4, 4],
        "shape must be preserved by BedSeparationFilter"
    );
    // Extract voxel values to validate filter computation.
    let out_vals: Vec<f32> = study
        .image
        .data()
        .clone()
        .into_data()
        .as_slice::<f32>()
        .expect("must extract f32 slice")
        .to_vec();
    // At least one voxel must equal outside_value (-2048.0): the air voxels.
    let has_outside = out_vals.iter().any(|&v| (v - (-2048.0_f32)).abs() < 1e-6);
    assert!(
        has_outside,
        "at least one voxel must equal outside_value=-2048.0 after BedSeparation; got {:?}",
        &out_vals[..4]
    );
    // All tissue voxels (originally 0.0) must remain at 0.0.
    for (i, &v) in out_vals[16..].iter().enumerate() {
        assert!(
            (v - 0.0_f32).abs() < 1e-6,
            "tissue voxel at index {} expected 0.0, got {}",
            16 + i,
            v
        );
    }
}

/// CPR dispatch via `ViewerCore::apply_filter` produces a 1×H×W image
/// from the 2-D CPR output.
///
/// Input: 4×4×4 identity-spacing volume with all voxels = 100.0.
/// Path: single linear segment from (0,0,0) to (3,3,3) in physical coords.
///
/// Expected: filter succeeds, study image shape becomes [1, num_cross, num_path].
#[test]
fn test_filter_kind_cpr_dispatch_reshapes_2d_to_3d() {
    let device: <Backend as burn::tensor::backend::Backend>::Device = Default::default();
    let vals = vec![100.0_f32; 4 * 4 * 4];
    let tensor =
        Tensor::<Backend, 3>::from_data(TensorData::new(vals, Shape::new([4, 4, 4])), &device);
    let image = Image::new(
        tensor,
        Point::new([0.0, 0.0, 0.0]),
        Spacing::new([1.0, 1.0, 1.0]),
        Direction::identity(),
    );
    let mut core = crate::ViewerCore::<Backend, 3>::new();
    core.load_study(crate::Study::new(image));

    let filter_kind = crate::FilterKind::Cpr {
        control_points: vec![[0.0, 0.0, 0.0], [3.0, 3.0, 3.0]],
        num_path_samples: 16,
        cross_section_half_width: 2.0,
        num_cross_samples: 8,
    };
    let result = core.apply_filter(&filter_kind);
    let event = result.expect("CPR apply_filter must succeed");
    assert!(
        matches!(event, crate::ViewerEvent::Status { .. }),
        "expected ViewerEvent::Status, got {:?}",
        event
    );
    let study = core.study().expect("study must be present after CPR");
    let shape = study.image.shape();
    assert_eq!(
        shape[0], 1,
        "CPR output must be single-slice (Z=1), got Z={}",
        shape[0]
    );
    assert_eq!(shape[1], 8, "num_cross_samples must be 8, got {}", shape[1]);
    assert_eq!(
        shape[2], 16,
        "num_path_samples must be 16, got {}",
        shape[2]
    );
}

/// An empty `ViewerCore` (no study loaded) must return a `Status` event
/// whose message contains "no study", not an error.
///
/// Precondition: `self.study` is `None`.
/// Postcondition: `Ok(ViewerEvent::Status { message })` where `message`
/// contains the substring "no study".
#[test]
fn test_filter_kind_no_study_returns_status_message() {
    let mut core = crate::ViewerCore::<Backend, 3>::new();
    let result = core.apply_filter(&crate::FilterKind::Gaussian { sigma: 1.0 });
    let event = result.expect("apply_filter on empty core must return Ok");
    match event {
        crate::ViewerEvent::Status { message } => {
            assert!(
                message.contains("no study"),
                "status message must contain 'no study'; got: {:?}",
                message
            );
        }
        other => panic!(
            "expected ViewerEvent::Status for empty core, got {:?}",
            other
        ),
    }
}

#[test]
fn test_geometry_summary_from_dicom() {
    let meta = ritk_io::DicomReadMetadata {
        series_instance_uid: None,
        study_instance_uid: None,
        frame_of_reference_uid: None,
        series_description: None,
        modality: Some(ArrayString::from("CT").unwrap()),
        patient_id: None,
        patient_name: None,
        study_date: None,
        series_date: None,
        series_time: None,
        dimensions: [3, 4, 5],
        spacing: [0.8, 0.8, 1.5],
        origin: [10.0, 20.0, -30.0],
        direction: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        bits_allocated: Some(16),
        bits_stored: Some(16),
        high_bit: Some(15),
        photometric_interpretation: Some(ArrayString::from("MONOCHROME2").unwrap()),
        slices: Vec::new(),
        private_tags: std::collections::HashMap::new(),
        preservation: Default::default(),
        patient_weight_kg: None,
        decay_correction: None,
        radionuclide_total_dose_bq: None,
        radiopharmaceutical_start_time: None,
        radionuclide_half_life_s: None,
    };
    let summary = crate::GeometrySummary::from_dicom(&meta);
    assert_eq!(summary.dimensions, [3, 4, 5]);
    assert_eq!(summary.spacing, [0.8, 0.8, 1.5]);
    assert_eq!(summary.origin, [10.0, 20.0, -30.0]);
}

#[test]
fn test_viewer_status_new() {
    let status = crate::ViewerStatus::new("loaded");
    assert_eq!(status.message, "loaded");
}

#[test]
fn test_viewer_core_load_study_emits_event() {
    let device: <Backend as burn::tensor::backend::Backend>::Device = Default::default();
    let tensor = Tensor::<Backend, 3>::from_data(
        TensorData::new(vec![1.0f32; 8], Shape::new([2, 2, 2])),
        &device,
    );
    let image = Image::new(
        tensor,
        Point::new([0.0, 0.0, 0.0]),
        Spacing::new([1.0, 1.0, 1.0]),
        Direction::identity(),
    );
    let study = crate::Study::new(image);
    let mut core = crate::ViewerCore::<Backend, 3>::new();
    let event = core.load_study(study);
    match event {
        crate::ViewerEvent::StudyLoaded { depth, rows, cols } => {
            assert_eq!(depth, 2);
            assert_eq!(rows, 2);
            assert_eq!(cols, 2);
        }
        other => panic!("unexpected event: {:?}", other),
    }
}
