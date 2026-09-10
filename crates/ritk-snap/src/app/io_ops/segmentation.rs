//! Segmentation save and load, in the native and DICOM SEG forms.

use crate::app::state::SnapApp;

use tracing::{error, info};

use super::dialog::FileDialog;

impl SnapApp {
    /// Save the current label map to a NIfTI file.
    ///
    /// Requires a loaded volume (for geometry) and an initialised label editor.
    /// The dialog is a no-op when either is absent; a status message explains
    /// the missing precondition.
    pub fn save_segmentation_dialog(&mut self) {
        let (Some(vol), Some(editor)) = (self.loaded.as_ref(), self.label_editor.as_ref()) else {
            self.status_message = "Save segmentation: no volume or segmentation loaded.".to_owned();
            return;
        };

        let map = editor.current_map();

        let origin = [
            vol.origin[0] as f32,
            vol.origin[1] as f32,
            vol.origin[2] as f32,
        ];
        let spacing = [
            vol.spacing[0] as f32,
            vol.spacing[1] as f32,
            vol.spacing[2] as f32,
        ];
        let direction: [f32; 9] = std::array::from_fn(|i| vol.direction[i] as f32);

        let Some(path) = FileDialog::new()
            .set_file_name("segmentation.nii.gz")
            .add_filter("NIfTI", &["nii", "gz"][..])
            .save_file()
        else {
            return;
        };

        match ritk_io::write_nifti_labels(
            &path,
            map.as_slice(),
            map.shape.0,
            origin,
            spacing,
            direction,
        ) {
            Ok(()) => {
                self.status_message = format!("Saved segmentation to {}", path.display());
                info!("{}", self.status_message);
            }
            Err(e) => {
                self.status_message = format!("Segmentation save failed: {e:#}");
                error!("{}", self.status_message);
            }
        }
    }

    pub fn save_segmentation_dicom_seg_dialog(&mut self) {
        let (Some(vol), Some(editor)) = (self.loaded.as_ref(), self.label_editor.as_ref()) else {
            self.status_message = "Save DICOM-SEG: no volume or segmentation loaded.".to_owned();
            return;
        };

        let map = editor.current_map();
        let origin = vol.origin;
        let spacing = vol.spacing;
        let direction = vol.direction;

        let Some(path) = FileDialog::new()
            .set_file_name("segmentation.dcm")
            .add_filter("DICOM SEG", &["dcm"][..])
            .save_file()
        else {
            return;
        };

        match ritk_io::label_map_to_dicom_seg(
            map,
            origin,
            spacing,
            direction,
            ritk_io::SegEncoding::Binary,
        ) {
            Ok(seg) => match ritk_io::write_dicom_seg(&path, &seg) {
                Ok(()) => {
                    self.status_message = format!("Saved DICOM-SEG to {}", path.display());
                    info!("{}", self.status_message);
                }
                Err(e) => {
                    self.status_message = format!("DICOM-SEG write failed: {e:#}");
                    error!("{}", self.status_message);
                }
            },
            Err(e) => {
                self.status_message = format!("DICOM-SEG conversion failed: {e:#}");
                error!("{}", self.status_message);
            }
        }
    }

    /// Load a label map from a NIfTI file and replace the current segmentation.
    pub fn load_segmentation_dialog(&mut self) {
        let Some(vol) = self.loaded.as_ref() else {
            self.status_message = "Load segmentation: no volume loaded.".to_owned();
            return;
        };
        let expected_shape = vol.shape;

        let Some(path) = FileDialog::new()
            .add_filter("NIfTI", &["nii", "gz"][..])
            .pick_file()
        else {
            return;
        };

        match ritk_io::read_nifti_labels(&path) {
            Ok((labels, shape)) => {
                if shape != expected_shape {
                    self.status_message = format!(
                        "Segmentation shape {:?} does not match volume {:?}",
                        shape, expected_shape
                    );
                    error!("{}", self.status_message);
                    return;
                }
                match ritk_annotation::LabelMap::from_data(
                    shape,
                    labels,
                    crate::label::default_label_table(),
                ) {
                    Ok(map) => {
                        self.label_editor = Some(crate::label::LabelEditor::from_label_map(map));
                        self.status_message =
                            format!("Loaded segmentation from {}", path.display());
                        info!("{}", self.status_message);
                    }
                    Err(e) => {
                        self.status_message = format!("Segmentation data error: {e}");
                        error!("{}", self.status_message);
                    }
                }
            }
            Err(e) => {
                self.status_message = format!("Segmentation load failed: {e:#}");
                error!("{}", self.status_message);
            }
        }
    }

    /// Load a label map from a DICOM-SEG file and replace the current segmentation.
    ///
    /// The reconstructed shape must match the currently loaded volume.
    pub fn load_segmentation_dicom_seg_file(&mut self, path: &std::path::Path) {
        let Some(vol) = self.loaded.as_ref() else {
            self.status_message = "Load DICOM-SEG: no volume loaded.".to_owned();
            return;
        };
        let expected_shape = vol.shape;

        match ritk_io::read_dicom_seg(path) {
            Ok(seg) => match ritk_io::dicom_seg_to_label_map(&seg) {
                Ok(map) => {
                    if map.shape.0 != expected_shape {
                        self.status_message = format!(
                            "DICOM-SEG shape {:?} does not match volume {:?}",
                            map.shape, expected_shape
                        );
                        error!("{}", self.status_message);
                        return;
                    }
                    self.label_editor = Some(crate::label::LabelEditor::from_label_map(map));
                    self.status_message = format!("Loaded DICOM-SEG from {}", path.display());
                    info!("{}", self.status_message);
                }
                Err(e) => {
                    self.status_message = format!("DICOM-SEG decode failed: {e:#}");
                    error!("{}", self.status_message);
                }
            },
            Err(e) => {
                self.status_message = format!("DICOM-SEG load failed: {e:#}");
                error!("{}", self.status_message);
            }
        }
    }

    /// Load a label map from a DICOM-SEG file and replace the current segmentation.
    ///
    /// The reconstructed shape must match the currently loaded volume.
    pub fn load_segmentation_dicom_seg_dialog(&mut self) {
        let Some(path) = FileDialog::new()
            .add_filter("DICOM SEG", &["dcm", "dicom"][..])
            .pick_file()
        else {
            return;
        };
        self.load_segmentation_dicom_seg_file(&path);
    }
}
