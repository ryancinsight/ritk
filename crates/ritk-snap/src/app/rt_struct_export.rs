use super::state::SnapApp;
use tracing::{error, info};

#[cfg(not(target_arch = "wasm32"))]
use rfd::FileDialog;

impl SnapApp {
    pub(super) fn save_rt_struct_dialog(&mut self) {
        let (Some(vol), Some(editor)) = (self.loaded.as_ref(), self.label_editor.as_ref()) else {
            self.status_message =
                "Save RT-STRUCT: no volume or segmentation loaded.".to_owned();
            return;
        };

        let map = editor.current_map();

        let Some(path) = FileDialog::new()
            .set_file_name("rtstruct.dcm")
            .add_filter("DICOM RT-STRUCT", &["dcm"][..])
            .save_file()
        else {
            return;
        };

        let origin = vol.origin;
        let spacing = vol.spacing;
        let direction = vol.direction;

        match ritk_io::label_map_to_rt_struct(map, origin, spacing, direction) {
            Ok(ss) => match ritk_io::write_rt_struct(&path, &ss) {
                Ok(()) => {
                    self.status_message =
                        format!("Saved RT-STRUCT ({} ROIs) to {}", ss.rois.len(), path.display());
                    info!("{}", self.status_message);
                }
                Err(e) => {
                    self.status_message =
                        format!("RT-STRUCT write failed: {e:#}");
                    error!("{}", self.status_message);
                }
            },
            Err(e) => {
                self.status_message =
                    format!("RT-STRUCT conversion failed: {e:#}");
                error!("{}", self.status_message);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::SnapApp;
    use crate::label::LabelEditor;
    use crate::LoadedVolume;

    fn volume_with_segmentation() -> SnapApp {
        let mut app = SnapApp::default();
        let shape = [2usize, 4, 4];
        let data = vec![0.0f32; shape[0] * shape[1] * shape[2]];
        app.loaded = Some(LoadedVolume {
            data: std::sync::Arc::new(data),
            shape,
            channels: 1,
            spacing: [1.0, 1.0, 1.0],
            origin: [0.0, 0.0, 0.0],
            direction: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            metadata: None,
            source: None,
            modality: None,
            patient_name: None,
            patient_id: None,
            study_date: None,
            series_description: None,
            series_time: None,
            patient_weight_kg: None,
            injected_dose_bq: None,
            radionuclide_half_life_s: None,
            radiopharmaceutical_start_time: None,
            decay_correction: None,
        });

        let mut table = ritk_core::annotation::LabelTable::new();
        table.add_label(10, "Tumor", [255, 0, 0, 255]).unwrap();
        let mut lm = ritk_core::annotation::LabelMap::new(shape, table);
        for y in 1..3 {
            for x in 1..3 {
                lm.set_label_at([0, y, x], 10);
            }
        }
        app.label_editor = Some(LabelEditor::from_label_map(lm));
        app
    }

    #[test]
    fn save_rt_struct_no_volume_shows_status() {
        let mut app = SnapApp::default();
        app.save_rt_struct_dialog();
        assert!(app.status_message.contains("no volume or segmentation"));
    }

    #[test]
    fn save_rt_struct_with_segmentation_writes_valid_file() {
        let app = volume_with_segmentation();
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("test_rt.dcm");

        // Override the file dialog result by writing directly
        let vol = app.loaded.as_ref().unwrap();
        let editor = app.label_editor.as_ref().unwrap();
        let map = editor.current_map();
        let ss = ritk_io::label_map_to_rt_struct(
            map, vol.origin, vol.spacing, vol.direction,
        ).expect("convert");

        ritk_io::write_rt_struct(&path, &ss).expect("write");

        let loaded = ritk_io::read_rt_struct(&path).expect("read");
        assert_eq!(loaded.rois.len(), 1);
        assert_eq!(loaded.rois[0].roi_name, "Tumor");
        assert_eq!(loaded.rois[0].roi_number, 10);
        assert!(!loaded.rois[0].contours.is_empty());
        assert!(loaded.rois[0].contours[0].points.len() >= 3);
    }

    #[test]
    fn save_rt_struct_all_rois_preserved() {
        let mut app = volume_with_segmentation();
        let editor = app.label_editor.as_mut().unwrap();
        let next_id = editor.add_label("Organ", [0, 255, 0, 255]).expect("add label");
        let map = editor.current_map();
        let mut lm = map.clone();
        for y in 2..4 {
            for x in 2..4 {
                lm.set_label_at([1, y, x], next_id);
            }
        }
        *app.label_editor.as_mut().unwrap() = LabelEditor::from_label_map(lm);

        let vol = app.loaded.as_ref().unwrap();
        let editor = app.label_editor.as_ref().unwrap();
        let map = editor.current_map();
        let ss = ritk_io::label_map_to_rt_struct(
            map, vol.origin, vol.spacing, vol.direction,
        ).expect("convert");

        assert_eq!(ss.rois.len(), 2);
        let names: Vec<&str> = ss.rois.iter().map(|r| r.roi_name.as_str()).collect();
        assert!(names.contains(&"Tumor"));
        assert!(names.contains(&"Organ"));
    }
}
