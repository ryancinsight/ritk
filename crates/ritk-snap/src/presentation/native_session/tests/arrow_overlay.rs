use super::super::layout::{append_measurement_preview, OverlayCommands};
use super::measurement_overlay::{axial_panel, composed, draw_lines};
use super::support::{control_center, session};
use super::*;
use crate::tools::interaction::{Annotation, AnnotationLocation, ToolState};
use crate::tools::kind::ToolKind;
use crate::ui::{RotationSteps, ViewTransform};
use metis_platform::native::MouseButton;

#[test]
fn native_arrow_preserves_direction_location_and_input_ownership() {
    let (mut session, _root) = session();
    session.app.view_transform = ViewTransform {
        flip_h: true,
        flip_v: false,
        rotation: RotationSteps::Ninety,
    };
    session.refresh_frame().expect("compose transformed image");
    let key = |virtual_key| WindowEvent::KeyDown {
        virtual_key,
        repeated: false,
        modifiers: ModifierState::NONE,
    };
    let (pane, ..) = axial_panel(&session);
    let (x, y) = session.viewports[pane].center();
    let button = MouseButton::Left;
    let down = |x, y| WindowEvent::PointerDown { x, y, button };
    let up = |x, y| WindowEvent::PointerUp { x, y, button };
    let movement = WindowEvent::PointerMove { x: x + 12, y };
    let menu = super::super::window_controls::Menu::Tools;
    let (mx, my) = control_center(&session, None, WindowAction::OpenMenu(menu));
    let initial_tool = session.app.active_tool;
    session
        .handle_events(&[down(mx, my), up(mx, my), key(0x59)])
        .expect("Tools owns Y");
    assert_eq!(session.app.active_tool, initial_tool);
    session
        .handle_events(&[key(0x1b), key(0x59)])
        .expect("select Arrow after closing Tools");
    assert_eq!(session.app.active_tool, ToolKind::Arrow);
    for cancel in [key(0x1b), WindowEvent::PointerCaptureLost] {
        session
            .handle_events(&[down(x, y), movement.clone(), cancel, up(x + 12, y)])
            .expect("cancel arrow in its drag batch");
        assert_eq!(session.app.tool_state, ToolState::Idle);
        assert_eq!(session.app.annotations, []);
    }
    session
        .handle_events(&[down(x, y), movement])
        .expect("start arrow");
    let view = &session.views[pane];
    let location = Some(AnnotationLocation {
        plane: 0,
        slice: view.slice_index,
    });
    let adjacent = (view.slice_index + 1) % view.slice_count;
    let mut overlay = OverlayCommands::with_capacity(3).expect("reserve arrow overlay");
    append_measurement_preview(
        &mut overlay,
        view,
        &session.viewports[pane],
        &session.app.tool_state,
        session.app.pointer.drag_measurement_location,
    )
    .expect("draw preview");
    let preview = draw_lines(overlay.into_display_list().commands());
    assert_eq!(preview.len(), 3);
    assert_eq!(preview[0], ((x, y), (x + 12, y)));
    assert_eq!([preview[1].0, preview[2].0], [preview[0].1; 2]);

    session
        .handle_events(&[up(x + 12, y)])
        .expect("complete arrow");
    let [Annotation::Arrow {
        location: source,
        tail,
        head,
    }] = session.app.annotations.as_slice()
    else {
        panic!("released drag stores one arrow");
    };
    // Rightward motion decreases the source row under this transform.
    assert_eq!(*source, location);
    assert_eq!(tail[1], head[1]);
    assert!(tail[0] > head[0]);
    assert_eq!(draw_lines(&composed(&session, None)), preview);
    session.app.set_slice_for_axis(0, adjacent);
    session.refresh_frame().expect("compose adjacent image");
    assert_eq!(draw_lines(&composed(&session, None)), []);
}
