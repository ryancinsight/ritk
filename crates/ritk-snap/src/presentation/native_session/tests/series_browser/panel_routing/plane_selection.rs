//! A menu selection changes the rendered slice, scrollbar axis and only the intended grid pane.
use super::held_gestures::{down, show_two_panels, up};
use super::*;
use crate::presentation::native_session::window_controls::{Menu, WindowAction};
use crate::ui::AnatomicalPlane;

#[test]
fn native_plane_menu_reformats_the_clicked_comparison_panel() {
    let (mut viewer, _root) = session();
    let study = four_series_study();
    show_two_panels(&mut viewer, study.path());
    let before = viewer.views[viewer.app.axis].frame().clone();
    let (x, y) = viewer.viewports[1].center();
    let left = MouseButton::Left;
    viewer
        .handle_events(&[down(x, y, left), up(x, y, left)])
        .expect("focus second panel");
    let (x, y) = viewer.viewports[0].center();
    viewer
        .handle_events(&[WindowEvent::PointerMove { x, y }])
        .expect("cross the first pane");
    for plane in [
        AnatomicalPlane::Sagittal,
        AnatomicalPlane::Coronal,
        AnatomicalPlane::Axial,
    ] {
        for (menu, action) in [
            (None, WindowAction::OpenMenu(Menu::View)),
            (Some(Menu::View), WindowAction::SetPlane(plane)),
        ] {
            let (x, y) = crate::presentation::native_session::tests::support::control_center(
                &viewer, menu, action,
            );
            viewer
                .handle_events(&[down(x, y, left), up(x, y, left)])
                .expect("choose plane through native menu");
        }
        let second = &viewer.compare_panels[0];
        let axis = second.app.axis_for_plane(plane);
        assert_eq!(viewer.active_panel, 1);
        assert_eq!(viewer.viewports[1].axis(), axis);
        assert_eq!(second.view.plane_name, plane.label());
        assert_eq!(second.view.slice_count, second.app.axis_slice_info(axis).1);
        assert_eq!(viewer.views[viewer.app.axis].frame(), &before);
        let mut scratch = Default::default();
        let expected = crate::presentation::native_session::frame::render_orthogonal_views(
            &second.app,
            &mut scratch,
        )
        .expect("render reference planes");
        assert_eq!(second.view.frame(), expected[axis].frame());
    }
}
