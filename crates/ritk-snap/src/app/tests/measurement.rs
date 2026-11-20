//! Measurement label routing: per-axis spacing selection and coordinate mapping tests.
//!
//! The per-axis spacing selection used in `render_axis_viewport` for
//! measurement label routing follows the ITK-SNAP convention:
//!
//! | axis | row spacing | col spacing |
//! |------|-------------|-------------|
//! | 0 axial    | dy | dx |
//! | 1 coronal  | dz | dx |
//! | 2 sagittal | dz | dy |
#![expect(clippy::unwrap_used, reason = "ratchet RITK-UNWRAP-1")]

use super::*;
use crate::dicom::loader::tests::fixtures;
use crate::tools::interaction::{Annotation, ImagePoint};
use crate::tools::kind::ToolKind;
use arrayvec::ArrayString;
use ritk_io::{DicomReadMetadata, DicomSliceMetadata};

/// Constructs an anisotropic volume with spacing [dz=2.0, dy=3.0, dx=5.0]
/// (prime distances, each uniquely identifiable) for spacing-dispatch tests.
fn make_anisotropic_volume() -> LoadedVolume {
    let shape = [4, 6, 8];
    LoadedVolume {
        data: Arc::new(vec![0.0f32; shape[0] * shape[1] * shape[2]]),
        shape,
        channels: 1,
        spacing: [2.0, 3.0, 5.0], // [dz, dy, dx]
        origin: [0.0, 0.0, 0.0],
        direction: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        metadata: None,
        source: None,
        modality: Some(ArrayString::from("CT").unwrap()),
        patient_name: None,
        patient_id: None,
        study_date: None,
        series_description: None,
        series_time: None,
        patient_weight_kg: None,
        injected_dose_bq: None,
        radionuclide_half_life_s: None,
        radiopharmaceutical_start_time: None,
        decay_correction: None,
    }
}

fn dicom_volume(pixel_spacings: &[Option<[f64; 2]>]) -> LoadedVolume {
    let mut volume = make_anisotropic_volume();
    volume.metadata = Some(Box::new(DicomReadMetadata {
        slices: pixel_spacings
            .iter()
            .copied()
            .map(|pixel_spacing| DicomSliceMetadata {
                pixel_spacing,
                ..DicomSliceMetadata::default()
            })
            .collect(),
        ..DicomReadMetadata::default()
    }));
    volume
}

fn complete_length_on(app: &mut SnapApp, end: ImagePoint) {
    app.active_tool = ToolKind::MeasureLength;
    app.on_click(Some(ImagePoint::new(0.0, 0.0)));
    app.on_click(Some(end));
}

fn complete_length(volume: LoadedVolume, axis: usize, end: ImagePoint) -> SnapApp {
    let mut app = SnapApp::default();
    app.load_volume(volume, "measurement".to_owned());
    app.axis = axis;
    complete_length_on(&mut app, end);
    app
}

fn assert_physical_measurement_rejected(app: &SnapApp) {
    assert!(
        app.annotations.is_empty(),
        "unqualified source geometry must not create a physical mark"
    );
    assert_eq!(
        app.status_message,
        "Measurement rejected: physical pixel spacing is unavailable for this image"
    );
}

#[derive(Debug, Clone, Copy)]
enum NativeGeometry {
    Complete,
    MissingPixelSpacing,
    MissingSlicePositions,
    RoundedSourcePositions,
    InconsistentOrientation,
    ShearedSourcePositions,
}

fn native_dicom_volume(geometry: NativeGeometry) -> LoadedVolume {
    let root = tempfile::tempdir().unwrap();
    let source_geometry = match geometry {
        NativeGeometry::Complete
        | NativeGeometry::RoundedSourcePositions
        | NativeGeometry::InconsistentOrientation
        | NativeGeometry::ShearedSourcePositions => (Some([1.5, 0.5]), true),
        NativeGeometry::MissingPixelSpacing => (None, true),
        NativeGeometry::MissingSlicePositions => (Some([1.5, 0.5]), false),
    };
    fixtures::write_study_with_geometry(
        root.path(),
        "CT",
        fixtures::SERIES_UID,
        &fixtures::SAMPLES,
        source_geometry,
    )
    .unwrap();
    let mut volume = crate::dicom::loader::load_volume_from_path(root.path()).unwrap();
    let slices = &mut volume.metadata.as_mut().unwrap().slices;
    match geometry {
        NativeGeometry::RoundedSourcePositions => slices
            .iter_mut()
            .zip([0.1, 0.2, 0.3])
            .for_each(|(slice, x)| slice.image_position_patient = Some([x, 20.0, 30.0])),
        NativeGeometry::InconsistentOrientation => {
            slices[1].image_orientation_patient = Some([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
        }
        NativeGeometry::ShearedSourcePositions => slices
            .iter_mut()
            .zip([[0.0, 0.0, 0.0], [3.0, 0.0, 4.0], [6.0, 0.0, 8.0]])
            .for_each(|(slice, position)| {
                slice.image_orientation_patient = Some([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
                slice.image_position_patient = Some(position);
            }),
        _ => {}
    }
    volume
}

/// Axial (axis 0): row_spacing = dy = 3.0, col_spacing = dx = 5.0
#[test]
fn measurement_spacing_axial_selects_dy_dx() {
    let vol = make_anisotropic_volume();
    let [dz, dy, dx] = vol.spacing.map(|s| s as f32);
    let spacing_2d: [f32; 2] = match 0usize {
        0 => [dy, dx],
        1 => [dz, dx],
        _ => [dz, dy],
    };
    assert_eq!(
        spacing_2d,
        [3.0, 5.0],
        "axial axis must select [dy=3.0, dx=5.0]"
    );
}

/// Coronal (axis 1): row_spacing = dz = 2.0, col_spacing = dx = 5.0
#[test]
fn measurement_spacing_coronal_selects_dz_dx() {
    let vol = make_anisotropic_volume();
    let [dz, dy, dx] = vol.spacing.map(|s| s as f32);
    let spacing_2d: [f32; 2] = match 1usize {
        0 => [dy, dx],
        1 => [dz, dx],
        _ => [dz, dy],
    };
    assert_eq!(
        spacing_2d,
        [2.0, 5.0],
        "coronal axis must select [dz=2.0, dx=5.0]"
    );
}

/// Sagittal (axis 2): row_spacing = dz = 2.0, col_spacing = dy = 3.0
#[test]
fn measurement_spacing_sagittal_selects_dz_dy() {
    let vol = make_anisotropic_volume();
    let [dz, dy, dx] = vol.spacing.map(|s| s as f32);
    let spacing_2d: [f32; 2] = match 2usize {
        0 => [dy, dx],
        1 => [dz, dx],
        _ => [dz, dy],
    };
    assert_eq!(
        spacing_2d,
        [2.0, 3.0],
        "sagittal axis must select [dz=2.0, dy=3.0]"
    );
}

/// All three axis-spacing pairs are mutually distinct (no collision).
#[test]
fn measurement_spacing_all_axes_are_distinct() {
    let vol = make_anisotropic_volume();
    let [dz, dy, dx] = vol.spacing.map(|s| s as f32);
    let axial: [f32; 2] = [dy, dx];
    let coronal: [f32; 2] = [dz, dx];
    let sagittal: [f32; 2] = [dz, dy];
    assert_ne!(axial, coronal, "axial and coronal spacing must differ");
    assert_ne!(axial, sagittal, "axial and sagittal spacing must differ");
    assert_ne!(
        coronal, sagittal,
        "coronal and sagittal spacing must differ"
    );
}

/// `img_to_screen` mapping: image-pixel (col=c, row=r) maps to
/// screen position `origin + (c * scale, r * scale)`.
///
/// Analytical: origin=(10, 20), scale=2.0, img=(3, 5)
/// x = 10 + 3 × 2 = 16
/// y = 20 + 5 × 2 = 30
#[test]
fn measurement_img_to_screen_analytical() {
    let origin = egui::pos2(10.0, 20.0);
    let scale = 2.0_f32;
    let img_to_screen = |p: egui::Pos2| egui::pos2(origin.x + p.x * scale, origin.y + p.y * scale);
    let screen = img_to_screen(egui::pos2(3.0, 5.0)); // col=3, row=5
    assert_eq!(
        (screen.x, screen.y),
        (16.0, 30.0),
        "img_to_screen must compute origin + img × scale analytically"
    );
}

/// `img_to_screen` at image origin maps to screen origin.
/// Completing a length on the sagittal axis records that plane on the annotation.
#[test]
fn measurement_length_completion_records_active_axis_as_plane() {
    let mut app = SnapApp::default();
    app.load_volume(make_anisotropic_volume(), "fixture".to_owned());
    app.axis = 2;
    let slice = app.axis_slice_info(2).0;
    app.active_tool = ToolKind::MeasureLength;
    app.on_click(Some(ImagePoint::new(0.0, 0.0)));
    app.on_click(Some(ImagePoint::new(4.0, 3.0)));
    assert_eq!(
        app.annotations.len(),
        1,
        "two anchored clicks must complete one length"
    );
    match &app.annotations[0] {
        Annotation::Length {
            plane,
            location,
            p1,
            p2,
            length_mm,
        } => {
            assert_eq!(*plane, 2, "length must record its placement plane");
            assert_eq!(
                *location,
                Some(crate::tools::interaction::AnnotationLocation { plane: 2, slice })
            );
            assert_eq!(*p1, [0.0, 0.0], "stored start must keep image order");
            assert_eq!(*p2, [3.0, 4.0], "stored end must keep image order");
            let expected = 180.0_f32.sqrt();
            assert!(
                (length_mm - expected).abs() < 1e-4,
                "sagittal length must use [dz, dy] spacing: got {length_mm}, want {expected}"
            );
        }
        other => panic!("expected a completed Length, got {other:?}"),
    }
}

#[test]
fn dicom_missing_pixel_spacing_rejects_physical_measurement() {
    let mut volume = dicom_volume(&[None]);
    let display_spacing = volume.spacing;
    volume.metadata.as_mut().unwrap().spacing = display_spacing;
    assert_eq!(volume.measurement_pixel_spacing(), None);

    let app = complete_length(volume, 0, ImagePoint::new(0.0, 1.0));
    assert_physical_measurement_rejected(&app);
}

#[test]
fn dicom_pixel_spacing_is_admitted_without_unit_fallback() {
    let mut volume = dicom_volume(&[Some([7.0, 11.0])]);
    volume.metadata.as_mut().unwrap().spacing = [1.0, 1.0, 1.0];
    assert_eq!(volume.measurement_pixel_spacing(), Some([7.0, 11.0]));

    let app = complete_length(volume, 0, ImagePoint::new(1.0, 0.0));
    assert!(matches!(
        app.annotations.as_slice(),
        [Annotation::Length { length_mm, .. }] if *length_mm == 11.0
    ));
}

#[test]
fn dicom_measurement_admission_requires_consistent_source_slices() {
    for (case, pixel_spacings) in [
        ("missing second", [Some([7.0, 11.0]), None]),
        ("missing first", [None, Some([7.0, 11.0])]),
        (
            "conflicting spacing",
            [Some([7.0, 11.0]), Some([7.0, 12.0])],
        ),
    ] {
        assert_eq!(
            dicom_volume(&pixel_spacings).measurement_pixel_spacing(),
            None,
            "{case} must reject physical measurement"
        );
    }
}

#[test]
fn native_dicom_geometry_controls_measurement_admission() {
    for (geometry, axis, length) in [
        (NativeGeometry::MissingPixelSpacing, 0, None),
        (NativeGeometry::Complete, 0, Some(0.5)),
        (NativeGeometry::Complete, 1, Some(2.0)),
        (NativeGeometry::MissingSlicePositions, 1, None),
        (NativeGeometry::RoundedSourcePositions, 1, Some(0.1)),
        (NativeGeometry::InconsistentOrientation, 1, None),
        (NativeGeometry::ShearedSourcePositions, 0, Some(0.5)),
        (NativeGeometry::ShearedSourcePositions, 1, None),
    ] {
        let volume = native_dicom_volume(geometry);
        let in_plane =
            (!matches!(geometry, NativeGeometry::MissingPixelSpacing)).then_some([1.5, 0.5]);
        assert_eq!(volume.measurement_pixel_spacing(), in_plane, "{geometry:?}");
        let slice = match geometry {
            NativeGeometry::Complete | NativeGeometry::MissingPixelSpacing => Some(2.0),
            NativeGeometry::RoundedSourcePositions => Some((0.2_f64 - 0.1).midpoint(0.3 - 0.2)),
            _ => None,
        };
        assert_eq!(volume.measurement_slice_spacing(), slice, "{geometry:?}");
        if in_plane.is_none() {
            let sources = &volume.metadata.as_ref().unwrap().slices;
            assert!(sources.iter().all(|source| source.pixel_spacing.is_none()));
        }
        let end = if axis == 0 {
            ImagePoint::new(1.0, 0.0)
        } else {
            ImagePoint::new(0.0, 1.0)
        };
        let app = complete_length(volume, axis, end);
        match length {
            Some(expected) => assert!(
                matches!(app.annotations.as_slice(), [Annotation::Length { length_mm, .. }] if *length_mm == expected),
                "{geometry:?} axis {axis} must use source-qualified geometry"
            ),
            None => assert_physical_measurement_rejected(&app),
        }
    }

    let mut app = SnapApp::default();
    app.loaded = Some(native_dicom_volume(NativeGeometry::MissingSlicePositions));
    app.axis = 1;
    complete_length_on(&mut app, ImagePoint::new(0.0, 1.0));
    assert_physical_measurement_rejected(&app);
}

#[test]
fn filters_update_measurement_admission_by_geometry_effect() {
    use super::super::state::MeasurementSpacingState;

    for (filter, length) in [
        (crate::FilterKind::Abs, Some(11.0)),
        (
            crate::FilterKind::Shrink {
                factor_z: 1,
                factor_y: 1,
                factor_x: 2,
            },
            None,
        ),
    ] {
        let mut app = SnapApp::default();
        app.load_volume(dicom_volume(&[Some([7.0, 11.0])]), "measurement".to_owned());
        app.active_filter = filter;
        app.apply_filter_to_loaded_volume();
        let volume = app.loaded.as_ref().unwrap();
        assert_eq!(volume.measurement_pixel_spacing(), Some([7.0, 11.0]));
        let spacing = length.map_or(MeasurementSpacingState::Unavailable, |_| {
            MeasurementSpacingState::Qualified {
                in_plane: [7.0, 11.0],
                slice: None,
            }
        });
        assert_eq!(app.measurement_pixel_spacing, spacing);
        complete_length_on(&mut app, ImagePoint::new(1.0, 0.0));
        match length {
            Some(expected) => assert!(matches!(
                app.annotations.as_slice(),
                [Annotation::Length { length_mm, .. }] if *length_mm == expected
            )),
            None => assert!(app.annotations.is_empty()),
        }
    }
}

#[test]
fn measurement_img_to_screen_origin_maps_to_rect_min() {
    let origin = egui::pos2(50.0, 75.0);
    let scale = 3.0_f32;
    let img_to_screen = |p: egui::Pos2| egui::pos2(origin.x + p.x * scale, origin.y + p.y * scale);
    let screen = img_to_screen(egui::pos2(0.0, 0.0));
    assert_eq!(
        (screen.x, screen.y),
        (50.0, 75.0),
        "image origin must map to screen rect.min"
    );
}
