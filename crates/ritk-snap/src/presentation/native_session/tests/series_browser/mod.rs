//! Native study navigator and multi-series loading tests.

use super::session;
use crate::dicom::loader::{load_volume_from_series_info, scan_folder_for_series, tests::fixtures};
use crate::presentation::native_session::layout::{PanelGrid, WorkspaceLayout};
use crate::presentation::native_session::SeriesBrowser;
use metis_platform::native::{
    ModifierState, MouseButton, NativeApplication, NativeFlow, WindowEvent,
};

const SECOND_SERIES_UID: &str = "2.25.20260905002";
const THIRD_SERIES_UID: &str = "2.25.20260905004";
const FOURTH_SERIES_UID: &str = "2.25.20260905005";

fn replacement_study() -> tempfile::TempDir {
    let root = tempfile::tempdir().expect("replacement study root");
    fixtures::write_study(root.path(), "CT", fixtures::SERIES_UID).expect("write CT series");
    fixtures::write_study(root.path(), "MR", SECOND_SERIES_UID).expect("write MR series");
    root
}

fn four_series_study() -> tempfile::TempDir {
    let root = tempfile::tempdir().expect("four-series study root");
    fixtures::write_study(root.path(), "CT", fixtures::SERIES_UID).expect("write CT series");
    fixtures::write_study(root.path(), "MR", SECOND_SERIES_UID).expect("write MR series");
    fixtures::write_study(root.path(), "PT", THIRD_SERIES_UID).expect("write PET series");
    fixtures::write_study(root.path(), "US", FOURTH_SERIES_UID).expect("write ultrasound series");
    root
}

fn click_series(
    session: &mut crate::presentation::native_session::NativeViewerSession,
    index: i32,
) {
    let x = 100 + index.saturating_mul(192);
    let y = 714;
    session
        .handle_events(&[
            WindowEvent::PointerDown {
                x,
                y,
                button: MouseButton::Left,
            },
            WindowEvent::PointerUp {
                x,
                y,
                button: MouseButton::Left,
            },
        ])
        .expect("select series from the study navigator");
}

fn drag_series_to_panel(
    session: &mut crate::presentation::native_session::NativeViewerSession,
    series_index: i32,
    panel_index: usize,
) {
    let start_x = 100 + series_index.saturating_mul(192);
    let start_y = 714;
    let (end_x, end_y) = session.viewports[panel_index].center();
    session
        .handle_events(&[
            WindowEvent::PointerDown {
                x: start_x,
                y: start_y,
                button: MouseButton::Left,
            },
            WindowEvent::PointerMove { x: end_x, y: end_y },
            WindowEvent::PointerUp {
                x: end_x,
                y: end_y,
                button: MouseButton::Left,
            },
        ])
        .expect("drag the series card into the selected panel");
}

mod catalog;
mod panels;
