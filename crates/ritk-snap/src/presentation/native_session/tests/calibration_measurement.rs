use super::measurement_overlay::{axial_panel, composed, draw_texts};
use super::support::{control_center, session};
use super::*;
use crate::app::state::MeasurementSpacingState;
use crate::tools::interaction::ImagePoint;
use metis_platform::native::MouseButton;

#[test]
fn native_length_calibration_scales_labels_and_resets() {
    use super::super::window_controls::Menu;
    let (mut session, _root) = session();
    session.app.axis = 0;
    let (_, slice_count) = session.app.axis_slice_info(0);
    let (pane, ..) = axial_panel(&session);
    let slice = session.views[pane].slice_index;
    let other_slice = (slice + 1) % slice_count;
    let volume = session.app.loaded.as_mut().expect("fixture volume");
    volume.spacing = [1.0, 4.0, 3.0];
    volume
        .metadata
        .as_mut()
        .expect("fixture DICOM metadata")
        .slices
        .iter_mut()
        .for_each(|slice| slice.pixel_spacing = Some([4.0, 3.0]));
    session.app.measurement_pixel_spacing = MeasurementSpacingState::Qualified {
        in_plane: [4.0, 3.0],
        slice: Some(1.0),
    };
    let measure = |app: &mut SnapApp, end| {
        app.on_drag_start(
            crate::tools::kind::ToolKind::MeasureLength,
            Some(ImagePoint::new(0.0, 0.0)),
        );
        app.on_drag_end(Some(end));
    };
    for (index, end) in [
        (slice, ImagePoint::new(1.0, 0.0)),
        (other_slice, ImagePoint::new(1.0, 1.0)),
        (slice, ImagePoint::new(1.0, 1.0)),
    ] {
        session.app.set_slice_for_axis(0, index);
        measure(&mut session.app, end);
    }
    let raw = session.app.annotations.clone();
    let click_row = |session: &NativeViewerSession| {
        let (x, y) = control_center(session, None, WindowAction::OpenMenu(Menu::Tools));
        let (rx, ry) = control_center(session, Some(Menu::Tools), WindowAction::LengthCalibration);
        let button = MouseButton::Left;
        [
            WindowEvent::PointerDown { x, y, button },
            WindowEvent::PointerUp { x, y, button },
            WindowEvent::PointerDown {
                x: rx,
                y: ry,
                button,
            },
            WindowEvent::PointerUp {
                x: rx,
                y: ry,
                button,
            },
        ]
    };
    let key = |virtual_key| WindowEvent::KeyDown {
        virtual_key,
        repeated: false,
        modifiers: ModifierState::NONE,
    };
    assert_eq!(draw_texts(&composed(&session, None)), ["3.0 mm", "5.0 mm"]);
    let mut calibrate = click_row(&session).to_vec();
    calibrate.extend([
        WindowEvent::TextInput { character: '0' },
        key(0x0d),
        key(0x08),
        WindowEvent::TextInput { character: '1' },
        WindowEvent::TextInput { character: '0' },
        key(0x0d),
    ]);
    session
        .handle_events(&calibrate)
        .expect("calibration dialog owns input in the menu event batch");
    assert_eq!(draw_texts(&composed(&session, None)), ["6.0 mm", "10.0 mm"]);
    assert_eq!(session.app.annotations, raw);
    let reference = session.app.length_calibration_reference();
    session.app.set_slice_for_axis(0, other_slice);
    assert!(!session.app.set_length_calibration(reference, Some(20.0)));
    session.refresh_frame().expect("compose other image");
    assert_eq!(draw_texts(&composed(&session, None)), ["5.0 mm"]);
    session.app.set_slice_for_axis(0, slice);
    session.refresh_frame().expect("compose calibrated image");
    session
        .handle_events(&click_row(&session))
        .expect("reset calibration");
    assert_eq!(draw_texts(&composed(&session, None)), ["3.0 mm", "5.0 mm"]);
}
