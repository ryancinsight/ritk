use super::state::SnapApp;
use crate::ui::{compute_roi_dose_analytics, rt_dose_analytics::VolumeGeometry};
use tracing::{error, info};

impl SnapApp {
    #[cfg(test)]
    pub(crate) fn rt_dose_plan_link_status(&self) -> Option<String> {
        let dose = self.rt_dose.as_ref()?;
        let Some(ref_uid) = dose
            .referenced_rt_plan_sop_instance_uid
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty())
        else {
            return Some("Plan linkage: no ReferencedRTPlanSequence UID".to_owned());
        };
        match self.rt_plan.as_ref() {
            None => Some(format!(
                "Plan linkage: references UID {ref_uid} (no RT-PLAN loaded)"
            )),
            Some(plan) => {
                let loaded_uid = plan.sop_instance_uid.trim();
                if loaded_uid.is_empty() {
                    Some(format!(
                        "Plan linkage: references UID {ref_uid} (loaded RT-PLAN has empty SOP UID)"
                    ))
                } else if loaded_uid == ref_uid {
                    Some(format!(
                        "Plan linkage: linked to loaded RT-PLAN UID {ref_uid}"
                    ))
                } else {
                    Some(format!(
                        "Plan linkage: mismatch (dose references {ref_uid}, loaded plan is {loaded_uid})"
                    ))
                }
            }
        }
    }

    pub(crate) fn refresh_rt_dvh_cache(&mut self) {
        let (Some(vol), Some(rt_struct), Some(rt_dose), Some(roi_number)) = (
            self.loaded.as_ref(),
            self.rt_struct.as_ref(),
            self.rt_dose.as_ref(),
            self.rt_dvh_selected_roi,
        ) else {
            self.rt_dvh_cache = None;
            return;
        };
        self.rt_dvh_cache = compute_roi_dose_analytics(
            rt_struct,
            rt_dose,
            roi_number,
            VolumeGeometry {
                shape: vol.shape,
                origin: vol.origin,
                direction: vol.direction,
                spacing: vol.spacing,
            },
            128,
        );
    }

    pub(crate) fn load_rt_struct_file(&mut self, path: std::path::PathBuf) {
        match ritk_io::read_rt_struct(&path) {
            Ok(rt) => {
                let roi_count = rt.rois.len();
                let label = rt.structure_set_label.clone();
                self.rt_dvh_selected_roi = rt.rois.first().map(|roi| roi.roi_number);
                self.rt_struct = Some(rt);
                self.show_rt_struct_overlay = true;
                self.refresh_rt_dvh_cache();
                self.bump_visual_revision();
                self.status_message = format!(
                    "Loaded RT-STRUCT {} ({} ROIs) from {}",
                    label,
                    roi_count,
                    path.display()
                );
                info!("{}", self.status_message);
            }
            Err(e) => {
                self.status_message =
                    format!("RT-STRUCT load failed for {}: {e:#}", path.display());
                error!("{}", self.status_message);
            }
        }
    }

    pub(crate) fn load_rt_dose_file(&mut self, path: std::path::PathBuf) {
        match ritk_io::read_rt_dose(&path) {
            Ok(grid) => {
                let max_dose_gy = grid.dose_gy.iter().copied().fold(0.0_f64, f64::max);
                self.status_message = format!(
                    "Loaded RT-DOSE ({} type, {}×{}×{} grid) from {}",
                    grid.dose_type.as_dicom_str(),
                    grid.rows,
                    grid.cols,
                    grid.n_frames,
                    path.display()
                );
                info!("{}", self.status_message);
                self.rt_dose = Some(grid);
                self.rt_dose_max_gy = Some(max_dose_gy);
                self.show_rt_dose_overlay = true;
                self.refresh_rt_dvh_cache();
                self.bump_visual_revision();
            }
            Err(e) => {
                self.status_message = format!("RT-DOSE load failed for {}: {e:#}", path.display());
                error!("{}", self.status_message);
            }
        }
    }

    pub(crate) fn load_rt_plan_file(&mut self, path: std::path::PathBuf) {
        match ritk_io::read_rt_plan(&path) {
            Ok(plan) => {
                let beam_count = plan.beams.len();
                let fg_count = plan.fraction_groups.len();
                let label = plan.rt_plan_label.clone();
                self.rt_plan = Some(plan);
                self.refresh_rt_dvh_cache();
                self.status_message = format!(
                    "Loaded RT-PLAN {} ({} beams, {} fraction groups) from {}",
                    label,
                    beam_count,
                    fg_count,
                    path.display()
                );
                info!("{}", self.status_message);
            }
            Err(e) => {
                self.status_message = format!("RT-PLAN load failed for {}: {e:#}", path.display());
                error!("{}", self.status_message);
            }
        }
    }
}
