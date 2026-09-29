//! Persistent study-series catalog for the native RITK viewer.

use anyhow::{anyhow, Result};
use arrayvec::ArrayVec;
use ritk_io::DicomSeriesInfo;
use std::sync::Arc;

use crate::app::SnapApp;
use crate::dicom::loader::load_volume_from_dicom_instance;
use crate::dicom::series_tree::SeriesTree;
use crate::presentation::{PresentationFrame, PresentationSpacing};

// Twenty grid panels plus six rail cards, each retaining at most 16 KiB RGBA.
const THUMBNAIL_CAPACITY: usize = 26;
const THUMBNAIL_EDGE: u32 = 64;

struct SeriesThumbnail {
    index: usize,
    frame: Option<PresentationFrame>,
}

/// One discovered series and the display metadata used by its preview card.
pub(crate) struct SeriesChoice {
    pub(crate) acquisition: Arc<DicomSeriesInfo>,
    pub(crate) description: Box<str>,
    pub(crate) modality: Box<str>,
    pub(crate) image_count: usize,
    pub(crate) patient_number: usize,
    pub(crate) study_number: usize,
    pub(crate) study_series_number: usize,
    pub(crate) study_series_count: usize,
}

/// Discovered series retained while the user switches the active acquisition.
pub(crate) struct SeriesBrowser {
    choices: Box<[SeriesChoice]>,
    active_index: usize,
    first_visible: usize,
    study_count: usize,
    thumbnails: ArrayVec<SeriesThumbnail, THUMBNAIL_CAPACITY>,
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
                let study_series_count = study.series.len();
                for (series_index, series) in study.series.iter().enumerate() {
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
                        image_count: acquisition.image_count(),
                        modality: acquisition.modality().into(),
                        description: description.into_boxed_str(),
                        patient_number: patient_index.saturating_add(1),
                        study_number: study_index.saturating_add(1),
                        study_series_number: series_index.saturating_add(1),
                        study_series_count,
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
            thumbnails: ArrayVec::new(),
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

    /// Restore a scroll position saved before validating a host input batch.
    pub(crate) fn restore_first_visible(&mut self, index: usize) {
        self.first_visible = index.min(self.choices.len().saturating_sub(1));
    }

    pub(crate) fn choice(&self, index: usize) -> Option<&SeriesChoice> {
        self.choices.get(index)
    }

    /// Decode a displayed row on first use, retaining only its bounded frame.
    /// Missing or unsupported instances remain unavailable until eviction.
    #[cfg(test)]
    pub(crate) fn thumbnail(&mut self, index: usize) -> Option<&PresentationFrame> {
        self.ensure_thumbnail(index).then_some(())?;
        self.cached_thumbnail(index)
    }

    /// Cache the real first-instance preview for a catalog row.
    pub(crate) fn ensure_thumbnail(&mut self, index: usize) -> bool {
        let Some(choice) = self.choices.get(index) else {
            return false;
        };
        let thumbnail =
            if let Some(position) = self.thumbnails.iter().position(|item| item.index == index) {
                self.thumbnails.remove(position)
            } else {
                if self.thumbnails.is_full() {
                    self.thumbnails.remove(0);
                }
                match render_thumbnail(choice) {
                    Ok(frame) => SeriesThumbnail {
                        index,
                        frame: Some(frame),
                    },
                    Err(_error) => {
                        tracing::warn!(
                            failure = "dicom_decode_or_presentation",
                            "series preview unavailable"
                        );
                        SeriesThumbnail { index, frame: None }
                    }
                }
            };
        self.thumbnails.push(thumbnail);
        true
    }

    /// Read a previously requested card without decoding during drawing.
    pub(crate) fn cached_thumbnail(&self, index: usize) -> Option<&PresentationFrame> {
        self.thumbnails
            .iter()
            .find(|item| item.index == index)?
            .frame
            .as_ref()
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

fn render_thumbnail(choice: &SeriesChoice) -> Result<PresentationFrame> {
    let path = choice
        .acquisition
        .file_paths
        .first()
        .ok_or_else(|| anyhow!("series preview requires an instance"))?;
    let volume = load_volume_from_dicom_instance(path)?;
    let mut app = SnapApp::default();
    app.load_volume(volume, "Loaded series preview.".into());
    let volume = app
        .loaded
        .as_ref()
        .ok_or_else(|| anyhow!("series preview requires a loaded volume"))?;
    let mut frame = PresentationFrame::from_slice(
        volume,
        0,
        0,
        super::frame::window_level_for_app(&app),
        app.colormap,
    )?;
    let longest = frame.width().max(frame.height());
    if longest <= THUMBNAIL_EDGE {
        return Ok(frame);
    }
    let scaled = |length: u32| -> Result<u32> {
        Ok(
            u32::try_from(u64::from(length) * u64::from(THUMBNAIL_EDGE) / u64::from(longest))?
                .max(1),
        )
    };
    let width = scaled(frame.width())?;
    let height = scaled(frame.height())?;
    let byte_count = usize::try_from(width * height * 4)?;
    let mut rgba = Vec::new();
    rgba.try_reserve_exact(byte_count)?;
    for row in 0..height {
        let source_row = u64::from(row) * u64::from(frame.height()) / u64::from(height);
        for column in 0..width {
            let source_column = u64::from(column) * u64::from(frame.width()) / u64::from(width);
            let offset =
                usize::try_from((source_row * u64::from(frame.width()) + source_column) * 4)?;
            let pixel = frame
                .rgba()
                .get(offset..offset + 4)
                .ok_or_else(|| anyhow!("series preview sample lies outside frame"))?;
            rgba.extend_from_slice(pixel);
        }
    }
    let [row_spacing, column_spacing] = frame.display_spacing().values();
    let spacing = PresentationSpacing::try_new(
        row_spacing * f64::from(frame.height()) / f64::from(height),
        column_spacing * f64::from(frame.width()) / f64::from(width),
    )?;
    frame.replace_rgba_storage(width, height, spacing, &mut rgba)?;
    Ok(frame)
}

#[cfg(test)]
#[path = "series_browser/tests.rs"]
mod tests;
