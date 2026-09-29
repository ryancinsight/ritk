//! Event mapping through the physical oblique reslice plane.

use super::super::test_volume;
use crate::app::{ObliqueViewport, SnapApp};
use crate::geometry::PatientPointMm;
use crate::presentation::{PointerButton, PresentationEvent};
use crate::render::{ResliceInterpolation, ResliceOrientation, ReslicePlane};
use crate::tools::interaction::{Annotation, ToolState};
use crate::tools::kind::ToolKind;
use crate::ui::LinkedCursor;
use std::sync::Arc;

fn oblique_plane(volume: &crate::LoadedVolume) -> ReslicePlane {
    ReslicePlane::centered_oblique(
        volume,
        [4.0; 3],
        ResliceOrientation::default(),
        ResliceInterpolation::Linear,
    )
    .expect("centered oblique plane lies inside test volume")
}

fn viewport(plane: &ReslicePlane) -> ObliqueViewport {
    let dimensions = plane.dimensions();
    let dimensions = [
        u32::try_from(dimensions[0]).expect("test width fits u32"),
        u32::try_from(dimensions[1]).expect("test height fits u32"),
    ];
    ObliqueViewport::try_new(
        [0.0; 2],
        [1.0; 2],
        dimensions,
        [0.0, 0.0, f64::from(dimensions[0]), f64::from(dimensions[1])],
    )
    .expect("test plane has a non-empty viewport")
}

fn click(app: &mut SnapApp, viewport: &ObliqueViewport, plane: &ReslicePlane, pixel: [f64; 2]) {
    let position = [pixel[0] + 0.5, pixel[1] + 0.5];
    app.apply_oblique_presentation_events(
        &[
            PresentationEvent::PointerDown {
                x: position[0],
                y: position[1],
                button: PointerButton::Left,
            },
            PresentationEvent::PointerUp {
                x: position[0],
                y: position[1],
                button: PointerButton::Left,
            },
        ],
        viewport,
        plane,
    )
    .expect("oblique pointer click is valid");
}

#[test]
fn oblique_click_updates_linked_cursor_and_patient_length_from_render_plane() {
    let mut volume = test_volume([9, 9, 9]);
    volume.spacing = [2.0, 3.0, 4.0];
    volume.data = Arc::new(
        (0..9)
            .flat_map(|z| {
                (0..9).flat_map(move |row| {
                    (0..9).map(move |column| {
                        f32::from(
                            u16::try_from(z * 100 + row * 10 + column)
                                .expect("test intensity fits u16"),
                        )
                    })
                })
            })
            .collect(),
    );
    let plane = oblique_plane(&volume);
    let viewport = viewport(&plane);
    let mut app = SnapApp::default();
    app.loaded = Some(volume);
    app.linked_cursor = Some(LinkedCursor::centered([9; 3]));
    app.active_tool = ToolKind::MeasureLength;

    click(&mut app, &viewport, &plane, [1.0, 1.0]);

    assert_eq!(app.linked_cursor.expect("linked cursor").voxel(), [4, 1, 1]);
    assert_eq!(app.pointer_intensity, 411.0);
    let ToolState::PatientLength1 { p1 } = app.tool_state else {
        panic!("expected patient length anchor, got {:?}", app.tool_state);
    };
    assert_eq!(
        p1.coordinates(),
        PatientPointMm::try_new([8.0, 3.0, 4.0])
            .expect("finite point")
            .coordinates()
    );

    click(&mut app, &viewport, &plane, [2.0, 2.0]);

    let [Annotation::PatientLength(length)] = app.annotations.as_slice() else {
        panic!("oblique measurement must persist patient-space endpoints");
    };
    assert_eq!(length.start_mm().coordinates(), [8.0, 3.0, 4.0]);
    assert_eq!(length.end_mm().coordinates(), [8.0, 6.0, 8.0]);
    assert_eq!(length.length_mm(), 5.0);
    assert_eq!(app.linked_cursor.expect("linked cursor").voxel(), [4, 2, 2]);
    assert_eq!(app.pointer_intensity, 422.0);
    assert_eq!(
        app.axis, 0,
        "oblique clicks must not change the active MPR axis"
    );
}

#[test]
fn oblique_click_outside_rendered_image_does_not_clamp_or_create_measurement() {
    let volume = test_volume([9, 9, 9]);
    let plane = oblique_plane(&volume);
    let viewport = viewport(&plane);
    let mut app = SnapApp::default();
    app.loaded = Some(volume);
    app.active_tool = ToolKind::MeasureLength;
    app.pointer_intensity = 17.0;

    click(&mut app, &viewport, &plane, [9.0, 4.0]);

    assert!(matches!(app.tool_state, ToolState::Idle));
    assert!(app.annotations.is_empty());
    assert_eq!(app.pointer_intensity, 0.0);
}

#[test]
fn oblique_plane_rejects_a_loaded_volume_with_different_geometry() {
    let original = test_volume([9, 9, 9]);
    let plane = oblique_plane(&original);
    let viewport = viewport(&plane);
    let mut app = SnapApp::default();
    app.loaded = Some(test_volume([8, 8, 8]));
    app.pointer_intensity = 17.0;

    let result = app.apply_oblique_presentation_events(
        &[
            PresentationEvent::PointerDown {
                x: 4.5,
                y: 4.5,
                button: PointerButton::Left,
            },
            PresentationEvent::PointerUp {
                x: 4.5,
                y: 4.5,
                button: PointerButton::Left,
            },
        ],
        &viewport,
        &plane,
    );

    assert!(matches!(
        result,
        Err(crate::app::action_adapter::ViewerInputError::Reslice(_))
    ));
    assert_eq!(app.pointer_intensity, 17.0);
}

#[test]
fn unsupported_oblique_measurements_report_rejection_without_fabricating_data() {
    let volume = test_volume([9, 9, 9]);
    let plane = oblique_plane(&volume);
    let viewport = viewport(&plane);
    let mut app = SnapApp::default();
    app.loaded = Some(volume);
    app.active_tool = ToolKind::PointHu;

    click(&mut app, &viewport, &plane, [1.0, 1.0]);

    assert!(app.annotations.is_empty());
    assert!(app
        .status_message
        .contains("not supported in the oblique view"));
}
