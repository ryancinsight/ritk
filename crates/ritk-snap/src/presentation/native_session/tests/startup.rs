//! Empty desktop startup and subsequent opening of a study in the same session.

use super::super::startup::create_session;
use super::support::control_center;
use super::*;

#[test]
fn pathless_launch_stays_open_until_the_user_adds_a_study() {
    let root = tempfile::tempdir().expect("study folder");
    fixtures::write_study(root.path(), "CT", fixtures::SERIES_UID, &fixtures::SAMPLES)
        .expect("write a DICOM series");

    for mode in [
        NativePresentationSelection::Fixed(NativePresentationMode::Orthogonal),
        NativePresentationSelection::Responsive,
    ] {
        let mut viewer = create_session(None, None, false, mode, false, None, Arc::default())
            .expect("start without an image set");
        assert!(!viewer.app.presentation_snapshot().loaded());
        assert!(viewer.study_root.is_none());
        assert!(viewer.series_browser.is_none());
        assert!(matches!(
            viewer.handle_events(&[]).expect("idle empty window"),
            NativeFlow::Continue { .. }
        ));
        control_center(&viewer, None, WindowAction::OpenStudy);
        let pane = &viewer.viewports[0];
        assert!(pane.scrollbar.is_none());
        let x = i32::try_from(pane.panel_x + pane.panel_width / 2).expect("pane center");
        let y = i32::try_from(pane.panel_y + pane.panel_height - 2).expect("pane footer");
        assert_eq!(
            viewer.framebuffer.get_pixel(x, y),
            metis_platform::Color::BLACK
        );

        viewer
            .open_study_path(root.path())
            .expect("open first study");
        viewer.refresh_frame().expect("render the first study");
        assert!(viewer.app.presentation_snapshot().loaded());
        assert_eq!(viewer.study_root.as_deref(), Some(root.path()));
        assert_eq!(viewer.primary_series_index, Some(0));
        assert_eq!(viewer.views[0].frame().width(), 4);
    }
}
