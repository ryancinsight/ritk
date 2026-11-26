//! Exercise the Tools menu action and background publication through native events.
use super::super::layout::{PanelGrid, WorkspaceLayout};
use super::super::window_controls::Menu;
use super::support::{control_center, session_with_volume};
use super::*;
use crate::app::tests::test_volume;
use metis_platform::native::MouseButton;
use std::time::{Duration, Instant};

fn click_control(session: &mut NativeViewerSession, menu: Option<Menu>, action: WindowAction) {
    let (x, y) = control_center(session, menu, action);
    let button = MouseButton::Left;
    session
        .handle_events(&[
            WindowEvent::PointerDown { x, y, button },
            WindowEvent::PointerUp { x, y, button },
        ])
        .expect("native menu click");
}

#[test]
fn rigid_registration_from_tools_opens_a_separate_preview() {
    let mut fixed = test_volume([8; 3]);
    fixed.data = Arc::new((0..512).map(|n| ((n * n + n / 7) % 31) as f32).collect());
    let mut moving = fixed.clone();
    moving.origin = [20., 30., 40.];
    moving.data = Arc::new(fixed.data.iter().map(|v| 100. - v).collect());
    let expected = Arc::clone(&moving.data);
    let (mut session, _dir) = session_with_volume(fixed);
    let register = WindowAction::RigidRegistration;
    click_control(&mut session, None, WindowAction::OpenMenu(Menu::Tools));
    click_control(&mut session, Some(Menu::Tools), register);
    assert!(session.app.status_message.contains("F4"));
    session
        .set_workspace_layout(WorkspaceLayout::Panels(
            PanelGrid::new(2, 1).expect("two panels"),
        ))
        .expect("layout");
    session.compare_panels[0]
        .replace(moving, 1, "moving".into())
        .expect("second series");
    click_control(&mut session, None, WindowAction::OpenMenu(Menu::Tools));
    click_control(&mut session, Some(Menu::Tools), register);
    let until = Instant::now() + Duration::from_secs(30);
    while session.registration.is_some() {
        assert!(Instant::now() < until, "worker did not finish");
        session.handle_events(&[]).expect("idle update");
        std::thread::yield_now();
    }
    assert_eq!(
        session.active_panel,
        2,
        "{}",
        session.active_app().status_message
    );
    let preview = &session.compare_panels[1];
    assert_eq!(preview.series_index, None);
    assert!(session
        .series_panel_label(2)
        .expect("panel label")
        .contains("Rigid registration"));
    assert_eq!(
        preview
            .app
            .loaded
            .as_ref()
            .expect("derived data")
            .data
            .as_slice(),
        expected.as_slice()
    );
    assert!(Arc::ptr_eq(
        &session.compare_panels[0]
            .app
            .loaded
            .as_ref()
            .expect("original moving volume")
            .data,
        &expected
    ));
}

#[test]
fn closing_an_input_discards_the_pending_fit() {
    let mut fixed = test_volume([8; 3]);
    fixed.data = Arc::new((0..512).map(|n| (n % 31) as f32).collect());
    let mut moving = fixed.clone();
    moving.data = Arc::new(fixed.data.iter().map(|v| v + 1.).collect());
    let (mut session, _root) = session_with_volume(fixed);
    let grid = PanelGrid::new(2, 1).expect("two panes");
    session
        .set_workspace_layout(WorkspaceLayout::Panels(grid))
        .expect("comparison");
    session.compare_panels[0]
        .replace(moving, 1, "moving".into())
        .expect("second series");
    session.start_registration().expect("start fit");
    session.compare_panels[0]
        .clear()
        .expect("close moving input");
    assert!(session.poll_registration());
    assert!(session.registration.is_none());
    assert!(session.compare_panels[0].app.loaded.is_none());
    assert_eq!(session.compare_panels.len(), 1);
}
