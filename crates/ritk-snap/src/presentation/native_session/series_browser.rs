//! Persistent study-series catalog for the native RITK viewer.

use anyhow::{anyhow, Result};
use ritk_io::DicomSeriesInfo;
use std::sync::Arc;

use crate::dicom::series_tree::SeriesTree;

/// One discovered series and the display metadata used by its preview card.
pub(crate) struct SeriesChoice {
    pub(crate) acquisition: Arc<DicomSeriesInfo>,
    pub(crate) description: Box<str>,
    pub(crate) modality: Box<str>,
    pub(crate) instance_count: usize,
    pub(crate) patient_number: usize,
    pub(crate) study_number: usize,
}

/// Discovered series retained while the user switches the active acquisition.
pub(crate) struct SeriesBrowser {
    choices: Box<[SeriesChoice]>,
    active_index: usize,
    first_visible: usize,
    study_count: usize,
}

impl SeriesBrowser {
    /// Build the browser from RITK's validated study tree.
    pub(crate) fn from_tree(tree: &SeriesTree<'_>, active_uid: Option<&str>) -> Result<Self> {
        let total = tree.total_series();
        if total == 0 {
            return Err(anyhow!("series browser requires a non-empty study tree"));
        }
        let mut choices = Vec::new();
        choices
            .try_reserve_exact(total)
            .map_err(|_| anyhow!("series browser allocation failed"))?;
        let mut study_count = 0_usize;
        for (patient_index, patient) in tree.patients.iter().enumerate() {
            study_count = study_count
                .checked_add(patient.studies.len())
                .ok_or_else(|| anyhow!("study count overflows usize"))?;
            for (study_index, study) in patient.studies.iter().enumerate() {
                for series in &study.series {
                    let acquisition = Arc::clone(&series.acquisition);
                    let description = acquisition.series_description.trim();
                    let description = description
                        .chars()
                        .filter(|character| !character.is_control())
                        .collect::<String>();
                    let description = if description.is_empty() {
                        format!("Series {}", choices.len().saturating_add(1))
                    } else {
                        description
                    };
                    choices.push(SeriesChoice {
                        instance_count: acquisition.file_paths.len(),
                        modality: acquisition.modality().into(),
                        description: description.into_boxed_str(),
                        patient_number: patient_index.saturating_add(1),
                        study_number: study_index.saturating_add(1),
                        acquisition,
                    });
                }
            }
        }
        if choices.len() != total {
            return Err(anyhow!(
                "study tree reported {total} series but exposed {}",
                choices.len()
            ));
        }
        let active_index = match active_uid {
            Some(uid) => choices
                .iter()
                .position(|choice| choice.acquisition.series_instance_uid() == uid)
                .ok_or_else(|| anyhow!("selected SeriesInstanceUID is absent from study"))?,
            None => 0,
        };
        let first_visible = active_index;
        Ok(Self {
            choices: choices.into_boxed_slice(),
            active_index,
            first_visible,
            study_count,
        })
    }

    pub(crate) const fn len(&self) -> usize {
        self.choices.len()
    }

    pub(crate) const fn study_count(&self) -> usize {
        self.study_count
    }

    pub(crate) const fn active_index(&self) -> usize {
        self.active_index
    }

    pub(crate) const fn first_visible(&self) -> usize {
        self.first_visible
    }

    pub(crate) fn choice(&self, index: usize) -> Option<&SeriesChoice> {
        self.choices.get(index)
    }

    pub(crate) fn index_for_uid(&self, uid: &str) -> Option<usize> {
        self.choices
            .iter()
            .position(|choice| choice.acquisition.series_instance_uid() == uid)
    }

    /// Commit a successfully loaded series as active and keep it in view.
    pub(crate) fn set_active(&mut self, index: usize) -> bool {
        if index >= self.choices.len() {
            return false;
        }
        self.active_index = index;
        self.first_visible = index;
        true
    }

    /// Scroll the visible series rows without decoding their pixel data.
    pub(crate) fn scroll_series(&mut self, delta: i32, visible_count: usize) -> bool {
        if delta == 0 || self.choices.is_empty() {
            return false;
        }
        let previous = self.first_visible;
        let visible_count = visible_count.max(1);
        let maximum = self.choices.len().saturating_sub(visible_count);
        self.first_visible = self.first_visible.min(maximum);
        let step = usize::try_from(delta.unsigned_abs())
            .expect("invariant: absolute i32 scroll step fits in usize");
        let next = if delta.is_negative() {
            self.first_visible.saturating_sub(step)
        } else {
            self.first_visible.saturating_add(step).min(maximum)
        };
        if next == previous {
            return false;
        }
        self.first_visible = next;
        true
    }
}

#[cfg(test)]
#[path = "series_browser/tests.rs"]
mod tests;
