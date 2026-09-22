//! Shared native session fixtures for the sibling test modules.

use super::*;
#[cfg(feature = "eframe-shell")]
use crate::presentation::PresentationFrame;
#[cfg(feature = "eframe-shell")]
use crate::LoadedVolume;

pub(super) fn session() -> (NativeViewerSession, tempfile::TempDir) {
    session_with_mode(NativePresentationMode::Orthogonal)
}

pub(super) fn session_with_mode(
    presentation_mode: NativePresentationMode,
) -> (NativeViewerSession, tempfile::TempDir) {
    session_with_selection(NativePresentationSelection::Fixed(presentation_mode))
}

pub(super) fn session_with_responsive_mode() -> (NativeViewerSession, tempfile::TempDir) {
    session_with_selection(NativePresentationSelection::Responsive)
}

pub(super) fn session_with_selection(
    presentation_mode: NativePresentationSelection,
) -> (NativeViewerSession, tempfile::TempDir) {
    let root = tempfile::tempdir().expect("study root");
    let path = root.path().to_path_buf();
    fixtures::write_study(&path, "CT", fixtures::SERIES_UID).expect("write study");
    let mut app = SnapApp::default();
    let volume = load_volume_from_path(&path).expect("load study fixture");
    app.load_volume(volume, "fixture".to_owned());
    (
        NativeViewerSession::new_with_selection(
            app,
            Arc::new(NativeViewerObservation::default()),
            false,
            presentation_mode,
            false,
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
        NativeViewerSession::new_with_selection(
            app,
            Arc::new(NativeViewerObservation::default()),
            false,
            NativePresentationSelection::Fixed(NativePresentationMode::Orthogonal),
            false,
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
