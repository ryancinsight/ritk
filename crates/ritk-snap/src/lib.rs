#![allow(
    clippy::too_many_arguments,
    clippy::field_reassign_with_default, // stylistic; test code patterns
)]

//! `ritk-snap` viewer core.
//!
//! This crate defines the viewer domain model and backend abstraction for
//! DICOM and other medical image studies. It does not perform I/O itself;
//! loading is delegated to `ritk-io` or another data source.
//!
//! The design goal is to keep the viewer frontend/backend split explicit:
//! - core state and navigation live here,
//! - rendering and presentation live behind a backend trait,
//! - DICOM/volume loading remains in `ritk-io`.
//!
//! This crate is intended to support multiple presentation targets, including
//! native desktop and web-backed shells, without duplicating viewer logic.
//!
//! Geometry handling is modality-aware at the summary layer:
//! - CT summaries may be derived from DICOM spatial metadata or loaded image geometry.
//! - MRI summaries preserve the same affine contract but do not assume CT-specific table/bed semantics.
//! - Ultrasound summaries must respect acquisition-specific orientation metadata and may not use CT-only display heuristics.

pub mod app;
pub mod dicom;
pub mod filter;
pub mod geometry;
pub mod label;
pub mod launch;
pub mod loaded_volume;
pub mod pacs;
pub mod render;
pub mod session;
pub mod tools;
pub mod ui;
pub mod viewer;

// Re-export flat API surface so downstream crates don't need path changes.
pub use filter::{BedSeparationConfigSerde, FilterKind};
pub use geometry::{GeometrySummary, ModalityDisplay, ViewerResult, ViewerStatus};
#[cfg(target_arch = "wasm32")]
pub use launch::start_web;
pub use launch::{run_app, run_app_with_options, AppLaunchOptions};
pub use loaded_volume::LoadedVolume;
pub use viewer::{DefaultBackend, Study, ViewerBackend, ViewerCore, ViewerEvent, ViewerState};

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Shape, Tensor, TensorData};
    use ritk_core::image::Image;
    use ritk_core::spatial::{Direction, Point, Spacing};

    type Backend = burn_ndarray::NdArray<f32>;

    #[test]
    fn test_viewer_state_default() {
        let state = ViewerState::default();
        assert_eq!(state.slice_index, 0);
        assert_eq!(state.window_center, None);
        assert_eq!(state.window_width, None);
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
        let study = Study::new(image);
        let mut core = ViewerCore::<Backend, 3>::new();
        let event = core.load_study(study);
        match event {
            ViewerEvent::StudyLoaded { depth, rows, cols } => {
                assert_eq!(depth, 2);
                assert_eq!(rows, 2);
                assert_eq!(cols, 2);
            }
            other => panic!("unexpected event: {:?}", other),
        }
    }

    #[test]
    fn test_geometry_summary_from_dicom() {
        let meta = ritk_io::DicomReadMetadata {
            series_instance_uid: None,
            study_instance_uid: None,
            frame_of_reference_uid: None,
            series_description: None,
            modality: Some("CT".to_string()),
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
            photometric_interpretation: Some("MONOCHROME2".to_string()),
            slices: Vec::new(),
            private_tags: std::collections::HashMap::new(),
            preservation: Default::default(),
            patient_weight_kg: None,
            decay_correction: None,
            radionuclide_total_dose_bq: None,
            radiopharmaceutical_start_time: None,
            radionuclide_half_life_s: None,
        };
        let summary = GeometrySummary::from_dicom(&meta);
        assert_eq!(summary.dimensions, [3, 4, 5]);
        assert_eq!(summary.spacing, [0.8, 0.8, 1.5]);
        assert_eq!(summary.origin, [10.0, 20.0, -30.0]);
    }

    #[test]
    fn test_viewer_status_new() {
        let status = ViewerStatus::new("loaded");
        assert_eq!(status.message, "loaded");
    }

    #[test]
    fn test_app_launch_options_default_has_no_initial_path() {
        let options = AppLaunchOptions::default();
        assert_eq!(
            options.initial_path, None,
            "default launch options must not queue a startup load"
        );
    }

    // ── New tests ─────────────────────────────────────────────────────────────

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
        let mut core = ViewerCore::<Backend, 3>::new();
        core.load_study(Study::new(image));

        let config = ritk_core::filter::BedSeparationConfig {
            body_threshold: -500.0,
            background_threshold: -700.0,
            keep_largest_component: false,
            closing_radius: 0,
            opening_radius: 0,
            outside_value: -2048.0,
        };

        let result = core.apply_filter(&FilterKind::BedSeparation(config));
        // Event must be Status.
        let event = result.expect("apply_filter must succeed");
        assert!(
            matches!(event, ViewerEvent::Status { .. }),
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

    /// An empty `ViewerCore` (no study loaded) must return a `Status` event
    /// whose message contains "no study", not an error.
    ///
    /// Precondition: `self.study` is `None`.
    /// Postcondition: `Ok(ViewerEvent::Status { message })` where `message`
    /// contains the substring "no study".
    #[test]
    fn test_filter_kind_no_study_returns_status_message() {
        let mut core = ViewerCore::<Backend, 3>::new();
        let result = core.apply_filter(&FilterKind::Gaussian { sigma: 1.0 });
        let event = result.expect("apply_filter on empty core must return Ok");
        match event {
            ViewerEvent::Status { message } => {
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

    /// Verify `ModalityDisplay::for_modality` against the analytically derived
    /// standard clinical window parameters documented in the struct-level rustdoc.
    ///
    /// CT lung window: centre = -400, width = 1500 → range [-1150, 350] HU
    /// MR brain window: centre = 600, width = 1200 → typical soft-tissue
    /// US 8-bit range: centre = 128, width = 256 → [0, 255]
    /// None / unknown: centre = 128, width = 256 → conservative default
    #[test]
    fn test_modality_display_ct_window_parameters() {
        let ct = ModalityDisplay::for_modality(Some("CT"));
        assert_eq!(
            ct.window_center, -400.0,
            "CT window_center must be -400.0 (standard lung window)"
        );
        assert_eq!(
            ct.window_width, 1500.0,
            "CT window_width must be 1500.0 (standard lung window)"
        );
        assert_eq!(ct.modality, "CT");

        let mr = ModalityDisplay::for_modality(Some("MR"));
        assert_eq!(
            mr.window_center, 600.0,
            "MR window_center must be 600.0 (typical brain window)"
        );
        assert_eq!(
            mr.window_width, 1200.0,
            "MR window_width must be 1200.0 (typical brain window)"
        );
        assert_eq!(mr.modality, "MR");

        let us = ModalityDisplay::for_modality(Some("US"));
        assert_eq!(
            us.window_center, 128.0,
            "US window_center must be 128.0 (8-bit acoustic range midpoint)"
        );
        assert_eq!(
            us.window_width, 256.0,
            "US window_width must be 256.0 (full 8-bit range)"
        );
        assert_eq!(us.modality, "US");

        let unknown = ModalityDisplay::for_modality(None);
        assert_eq!(
            unknown.window_center, 128.0,
            "None modality must fall back to default centre 128.0"
        );
        assert_eq!(
            unknown.window_width, 256.0,
            "None modality must fall back to default width 256.0"
        );
        assert_eq!(unknown.modality, "");
    }
}
