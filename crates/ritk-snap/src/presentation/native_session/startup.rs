//! RITK startup discovery for the native Métis session.

use super::SeriesBrowser;
use crate::app::SnapApp;
use crate::dicom::loader::{
    load_volume_from_path, load_volume_from_series_info, scan_folder_for_series,
};
use anyhow::{anyhow, Context, Result};
use std::path::Path;

pub(super) fn prepare_initial_study(
    app: &mut SnapApp,
    path: &Path,
    selected_uid: Option<&str>,
    capture_requested: bool,
) -> Result<Option<SeriesBrowser>> {
    if path.is_dir() {
        let tree = scan_folder_for_series(path)
            .with_context(|| format!("discover initial DICOM input at {}", path.display()))?;
        let series_count = tree.total_series();
        if series_count == 0 {
            if selected_uid.is_some() {
                return Err(anyhow!(
                    "requested SeriesInstanceUID is absent from the selected folder"
                ));
            }
            let volume = load_volume_from_path(path)
                .with_context(|| format!("open initial RITK study at {}", path.display()))?;
            app.load_volume(volume, "Loaded DICOM study.".to_owned());
            return Ok(None);
        }

        if capture_requested && series_count > 1 && selected_uid.is_none() {
            return Err(anyhow!(
                "native capture requires --series-instance-uid when '{}' contains multiple DICOM series",
                path.display()
            ));
        }
        let browser = SeriesBrowser::from_tree(&tree, selected_uid)?;
        let choice = browser
            .choice(browser.active_index())
            .expect("invariant: a non-empty study browser has an active series");
        let volume = load_volume_from_series_info(&choice.acquisition)
            .with_context(|| "open the selected DICOM series")?;
        app.load_volume(
            volume,
            format!(
                "Loaded {} series ({} images).",
                choice.modality, choice.image_count
            ),
        );
        Ok(Some(browser))
    } else {
        if selected_uid.is_some() {
            return Err(anyhow!(
                "--series-instance-uid requires a DICOM study folder"
            ));
        }
        let volume = load_volume_from_path(path)
            .with_context(|| format!("open initial RITK study at {}", path.display()))?;
        app.load_volume(volume, "Loaded DICOM study.".to_owned());
        Ok(None)
    }
}
