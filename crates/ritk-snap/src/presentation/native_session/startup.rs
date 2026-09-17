//! RITK startup discovery for the native Métis session.

use super::SeriesSelection;
use crate::app::SnapApp;
use crate::dicom::loader::{
    load_volume_from_path, load_volume_from_series_uid, scan_folder_for_series,
};
use crate::dicom::series_tree::SeriesEntryView;
use anyhow::{anyhow, Context, Result};
use std::path::Path;

pub(super) fn prepare_initial_study(
    app: &mut SnapApp,
    path: &Path,
    capture_requested: bool,
) -> Result<Option<SeriesSelection>> {
    if path.is_dir() {
        let tree = scan_folder_for_series(path)
            .with_context(|| format!("discover initial DICOM input at {}", path.display()))?;
        match tree.total_series() {
            0 => {
                let volume = load_volume_from_path(path)
                    .with_context(|| format!("open initial RITK study at {}", path.display()))?;
                app.load_volume(volume, format!("Loaded native Métis study: {}", path.display()));
                Ok(None)
            }
            1 => {
                let series = tree
                    .iter_series()
                    .next()
                    .expect("invariant: one discovered series has one entry");
                let uid = series.series_uid();
                let volume = load_volume_from_series_uid(path, uid).with_context(|| {
                    format!("open initial RITK series {} from {}", uid, path.display())
                })?;
                app.load_volume(
                    volume,
                    format!("Loaded native Métis series {}: {}", uid, path.display()),
                );
                Ok(None)
            }
            _ if capture_requested => Err(anyhow!(
                "native capture requires --series-instance-uid when '{}' contains multiple DICOM series",
                path.display()
            )),
            _ => {
                let selection = SeriesSelection::from_tree(path, &tree)?;
                app.status_message = format!(
                    "Select one of {} DICOM series before loading {}",
                    selection.len(),
                    path.display()
                );
                Ok(Some(selection))
            }
        }
    } else {
        let volume = load_volume_from_path(path)
            .with_context(|| format!("open initial RITK study at {}", path.display()))?;
        app.load_volume(
            volume,
            format!("Loaded native Métis study: {}", path.display()),
        );
        Ok(None)
    }
}
