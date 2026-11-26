//! RITK startup discovery for the native Métis session.

use super::{NativeViewerObservation, NativeViewerSession, SeriesBrowser};
use crate::app::SnapApp;
use crate::dicom::input_path::{classify_dicom_input_path, DicomInputPath};
use crate::dicom::loader::{load_volume_from_path, scan_folder_for_series};
use crate::launch::NativePresentationSelection;
use anyhow::{anyhow, ensure, Context, Result};
use std::path::Path;
use std::sync::Arc;

pub(super) fn create_session(
    path: Option<&Path>,
    selected_uid: Option<&str>,
    capture_requested: bool,
    presentation_mode: NativePresentationSelection,
    capture_application: bool,
    comparison_uid: Option<&str>,
    observation: Arc<NativeViewerObservation>,
) -> Result<NativeViewerSession> {
    ensure!(
        selected_uid.is_none() || path.is_some(),
        "--series-instance-uid requires a startup study folder"
    );
    let mut app = SnapApp::default();
    let browser = match path {
        Some(path) => prepare_initial_study(&mut app, path, selected_uid, capture_requested)?,
        None => None,
    };
    let mut session = NativeViewerSession::new_with_browser(
        app,
        observation,
        capture_requested,
        presentation_mode,
        capture_application,
        browser,
        comparison_uid,
    )?;
    session.study_root = path.map(Path::to_path_buf);
    Ok(session)
}

pub(super) fn prepare_initial_study(
    app: &mut SnapApp,
    path: &Path,
    selected_uid: Option<&str>,
    capture_requested: bool,
) -> Result<Option<SeriesBrowser>> {
    if selected_uid.is_some() && !path.is_dir() {
        return Err(anyhow!(
            "--series-instance-uid requires a DICOM study folder"
        ));
    }
    let input = classify_dicom_input_path(path);
    if input.dicom_root().is_some() {
        let single_instance = matches!(&input, DicomInputPath::SingleDicomFile { .. });
        let discovery_path = match &input {
            DicomInputPath::SingleDicomFile { root, .. } if root.as_os_str().is_empty() => {
                Path::new(".")
            }
            _ => path,
        };
        let tree = scan_folder_for_series(discovery_path)
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

        if capture_requested && series_count > 1 && selected_uid.is_none() && !single_instance {
            return Err(anyhow!(
                "native capture requires --series-instance-uid when '{}' contains multiple DICOM series",
                path.display()
            ));
        }
        let mut browser = SeriesBrowser::from_tree(&tree, selected_uid)?;
        if single_instance {
            let instance = path
                .canonicalize()
                .context("resolve selected DICOM instance")?;
            let index = browser
                .index_for_instance(&instance)
                .context("selected DICOM instance is not in the study catalog")?;
            browser.set_active(index);
        }
        let explicit =
            (selected_uid.is_some() || single_instance).then_some(browser.active_index());
        let (volume, status) = browser.load_initial(explicit)?;
        app.load_volume(volume, status);
        Ok(Some(browser))
    } else {
        let volume = load_volume_from_path(path)
            .with_context(|| format!("open initial RITK study at {}", path.display()))?;
        app.load_volume(volume, "Loaded DICOM study.".to_owned());
        Ok(None)
    }
}
