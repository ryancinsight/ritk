use super::measurement_overlay::{axial_panel, composed};
use super::pointer_capture::{down, up};
use super::support::{control_center, session};
use super::*;
use crate::geometry::affine::AffineTransform;
use crate::geometry::PatientPointMm;
use crate::tools::interaction::{Annotation, AnnotationLocation, PatientLength};
use metis_platform::native::MouseButton;
use metis_ui_lang::DisplayCommand;

const IDENTITY_DIRECTION: [f64; 9] = [
    1.0, 0.0, 0.0, //
    0.0, 1.0, 0.0, //
    0.0, 0.0, 1.0,
];

fn sorted_texts(session: &NativeViewerSession, affine: AffineTransform) -> Vec<String> {
    let mut texts = composed(session, Some(affine))
        .into_iter()
        .filter_map(|command| match command {
            DisplayCommand::DrawText { text, .. } => Some(text.to_string()),
            _ => None,
        })
        .collect::<Vec<_>>();
    texts.sort_unstable();
    texts
}

fn sorted_expected(values: &[&str]) -> Vec<String> {
    let mut values = values
        .iter()
        .map(|value| (*value).to_owned())
        .collect::<Vec<_>>();
    values.sort_unstable();
    values
}

fn calibration_clicks(session: &NativeViewerSession) -> [WindowEvent; 4] {
    use super::super::window_controls::Menu;

    let (menu_x, menu_y) = control_center(session, None, WindowAction::OpenMenu(Menu::Tools));
    let (row_x, row_y) =
        control_center(session, Some(Menu::Tools), WindowAction::LengthCalibration);
    let button = MouseButton::Left;
    [
        down(menu_x, menu_y, button),
        up(menu_x, menu_y, button),
        down(row_x, row_y, button),
        up(row_x, row_y, button),
    ]
}

fn key(virtual_key: u32) -> WindowEvent {
    WindowEvent::KeyDown {
        virtual_key,
        repeated: false,
        modifiers: ModifierState::NONE,
    }
}

fn annotations(location: AnnotationLocation) -> Vec<Annotation> {
    let patient_length = PatientLength::try_new(
        PatientPointMm::try_new([0.0, 0.0, 0.0]).expect("finite patient point"),
        PatientPointMm::try_new([0.0, 3.0, 4.0]).expect("finite patient point"),
    )
    .expect("3-4-5 patient segment");
    vec![
        Annotation::Length {
            plane: location.plane,
            location: Some(location),
            p1: [0.0, 0.0],
            p2: [1.0, 1.0],
            length_mm: 5.0,
        },
        Annotation::PatientLength {
            plane: location.plane,
            location: Some(location),
            length: patient_length,
        },
        Annotation::Angle {
            location: Some(location),
            p1: [0.0, 0.0],
            p2: [0.5, 0.5],
            p3: [0.0, 1.0],
            angle_deg: 30.0,
        },
        Annotation::Deviation {
            location: Some(location),
            start: [0.0, 0.0],
            end: [1.0, 1.0],
            reference: [0.0, 1.0],
            distance_mm: 4.0,
            angle_deg: 30.0,
        },
        Annotation::RoiRect {
            location: Some(location),
            top_left: [0.0, 0.0],
            bottom_right: [1.0, 1.0],
            mean: 11.0,
            std_dev: 2.0,
            min: 8.0,
            max: 14.0,
            area_mm2: 6.0,
        },
        Annotation::RoiEllipse {
            location: Some(location),
            center: [0.5, 0.5],
            radii: [0.5, 0.5],
            mean: 21.0,
            std_dev: 3.0,
            min: 17.0,
            max: 25.0,
            area_mm2: 7.0,
        },
        Annotation::HuPoint {
            location: Some(location),
            pos: [0.5, 0.5],
            value: 42.0,
        },
        Annotation::ClosedPolygon {
            location: Some(location),
            points: vec![[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]],
            area_mm2: 8.0,
            perimeter_mm: 9.0,
        },
        Annotation::OpenPolygon {
            location: Some(location),
            points: vec![[0.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            length_mm: 10.0,
        },
    ]
}

#[test]
fn manual_calibration_scales_image_measurements_only() {
    let (mut session, _root) = session();
    session.app.axis = 0;
    let (_, slice_count) = session.app.axis_slice_info(0);
    let (pane, ..) = axial_panel(&session);
    let slice = session.views[pane].slice_index;
    let other_slice = (slice + 1) % slice_count;
    let location = AnnotationLocation { plane: 0, slice };
    let other_location = AnnotationLocation {
        plane: 0,
        slice: other_slice,
    };
    session.app.annotations.extend(annotations(location));
    session.app.annotations.extend([
        Annotation::Length {
            plane: 0,
            location: Some(other_location),
            p1: [0.0, 0.0],
            p2: [1.0, 1.0],
            length_mm: 7.0,
        },
        Annotation::RoiRect {
            location: Some(other_location),
            top_left: [0.0, 0.0],
            bottom_right: [1.0, 1.0],
            mean: 31.0,
            std_dev: 4.0,
            min: 27.0,
            max: 35.0,
            area_mm2: 3.0,
        },
    ]);
    let raw = session.app.annotations.clone();
    let affine =
        AffineTransform::from_parts([0.0, 0.0, 0.0], IDENTITY_DIRECTION, [10.0, 10.0, 10.0])
            .expect("finite patient affine");
    let original = sorted_expected(&[
        "10.0 mm",
        "30.0 deg",
        "4.0 mm 30.0 deg",
        "42",
        "5.0 mm",
        "5.0 mm",
        "8.0 mm2  9.0 mm",
        "Mean 11.0 SD 2.0 6.0 mm2",
        "Mean 21.0 SD 3.0 7.0 mm2",
    ]);
    session
        .refresh_frame()
        .expect("render the original measurement labels");
    assert_eq!(sorted_texts(&session, affine), original);

    let mut events = calibration_clicks(&session).to_vec();
    events.extend([
        WindowEvent::TextInput { character: '0' },
        key(0x0d),
        key(0x08),
        WindowEvent::TextInput { character: '1' },
        WindowEvent::TextInput { character: '0' },
        key(0x0d),
    ]);
    session
        .handle_events(&events)
        .expect("native calibration dialog accepts 10 mm");
    let calibrated = sorted_texts(&session, affine);
    let deviation = calibrated
        .iter()
        .find(|label| label.ends_with("mm 30.0 deg"))
        .expect("deviation annotation emits its distance and angle");
    assert_eq!(deviation, "8.0 mm 30.0 deg");
    assert_eq!(
        calibrated,
        sorted_expected(&[
            "10.0 mm",
            "20.0 mm",
            "30.0 deg",
            "32.0 mm2  18.0 mm",
            "42",
            "5.0 mm",
            "8.0 mm 30.0 deg",
            "Mean 11.0 SD 2.0 24.0 mm2",
            "Mean 21.0 SD 3.0 28.0 mm2",
        ])
    );
    assert_eq!(session.app.annotations, raw);

    session.app.set_slice_for_axis(0, other_slice);
    session
        .refresh_frame()
        .expect("compose the uncalibrated image");
    assert_eq!(
        sorted_texts(&session, affine),
        sorted_expected(&["7.0 mm", "Mean 31.0 SD 4.0 3.0 mm2"])
    );

    session.app.set_slice_for_axis(0, slice);
    session
        .refresh_frame()
        .expect("compose the calibrated image");
    session
        .handle_events(&calibration_clicks(&session))
        .expect("reset image calibration through Tools");
    assert_eq!(sorted_texts(&session, affine), original);
    assert_eq!(session.app.annotations, raw);
}
