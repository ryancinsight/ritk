//! Shared native session fixtures for the sibling test modules.

use super::*;
#[cfg(feature = "eframe-shell")]
use crate::presentation::PresentationFrame;
#[cfg(feature = "eframe-shell")]
use crate::LoadedVolume;

pub(super) fn session() -> (NativeViewerSession, tempfile::TempDir) {
    session_with_mode(NativePresentationMode::Orthogonal)
}

pub(super) fn control_center(
    session: &NativeViewerSession,
    open_menu: Option<crate::presentation::native_session::window_controls::Menu>,
    action: WindowAction,
) -> (i32, i32) {
    session
        .window_chrome
        .control_center(
            session.framebuffer.width(),
            session.framebuffer.height(),
            &session.app,
            session.workspace_layout,
            open_menu,
            action,
        )
        .expect("build the viewer's native control layout")
        .expect("requested control is visible")
}

pub(super) fn series_card_center(session: &NativeViewerSession, index: usize) -> (i32, i32) {
    let browser = session.series_browser.as_ref().expect("study navigator");
    session
        .window_chrome
        .series_card_center(
            session.framebuffer.width(),
            session.framebuffer.height(),
            &session.app,
            session.workspace_layout,
            browser,
            index,
        )
        .expect("build the viewer's native series-card layout")
        .expect("requested series card is visible")
}

pub(super) fn session_with_mode(
    presentation_mode: NativePresentationMode,
) -> (NativeViewerSession, tempfile::TempDir) {
    session_with_browser(NativePresentationSelection::Fixed(presentation_mode))
}

pub(super) fn session_with_responsive_mode() -> (NativeViewerSession, tempfile::TempDir) {
    session_with_browser(NativePresentationSelection::Responsive)
}

pub(super) fn session_with_browser(
    presentation_mode: NativePresentationSelection,
) -> (NativeViewerSession, tempfile::TempDir) {
    let root = tempfile::tempdir().expect("study root");
    let path = root.path().to_path_buf();
    fixtures::write_study(&path, "CT", fixtures::SERIES_UID).expect("write study");
    let mut app = SnapApp::default();
    let volume = load_volume_from_path(&path).expect("load study fixture");
    app.load_volume(volume, "fixture".to_owned());
    (
        NativeViewerSession::new_with_browser(
            app,
            Arc::new(NativeViewerObservation::default()),
            false,
            presentation_mode,
            false,
            None,
            None,
        )
        .expect("native session"),
        root,
    )
}

#[cfg(feature = "eframe-shell")]
pub(super) fn session_with_volume(
    volume: LoadedVolume,
) -> (NativeViewerSession, tempfile::TempDir) {
    let root = tempfile::tempdir().expect("study root");
    let mut app = SnapApp::default();
    app.load_volume(volume, "fixture".to_owned());
    (
        NativeViewerSession::new_with_browser(
            app,
            Arc::new(NativeViewerObservation::default()),
            false,
            NativePresentationSelection::Fixed(NativePresentationMode::Orthogonal),
            false,
            None,
            None,
        )
        .expect("native session"),
        root,
    )
}

#[cfg(feature = "eframe-shell")]
pub(super) fn expected_native_frame(
    session: &NativeViewerSession,
    axis: usize,
) -> PresentationFrame {
    let volume = session.app.loaded.as_ref().expect("loaded volume");
    let (index, _) = session.app.axis_slice_info(axis);
    PresentationFrame::from_slice(
        volume,
        axis,
        index,
        super::super::frame::window_level_for_app(&session.app),
        session.app.colormap,
    )
    .expect("expected native frame")
}
