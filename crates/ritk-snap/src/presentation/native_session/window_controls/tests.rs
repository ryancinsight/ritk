//! Native RadiAnt-clone workspace geometry and input tests.

use super::*;
use crate::dicom::loader::{scan_folder_for_series, tests::fixtures};
use crate::presentation::native_session::layout::{PanelGrid, WorkspaceLayout};

fn browser_with_series(count: usize) -> (SeriesBrowser, tempfile::TempDir) {
    let root = tempfile::tempdir().expect("study root");
    for index in 0..count {
        fixtures::write_study(root.path(), "MR", &format!("2.25.20260905{index:04}"))
            .expect("write series");
    }
    let tree = scan_folder_for_series(root.path()).expect("scan study");
    let browser = SeriesBrowser::from_tree(&tree, None).expect("series browser");
    (browser, root)
}

#[path = "tests/menus.rs"]
mod menus;
#[path = "tests/render.rs"]
mod render;
#[path = "tests/series.rs"]
mod series;
