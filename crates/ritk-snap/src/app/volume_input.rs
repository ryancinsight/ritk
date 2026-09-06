//! Deferred volume requests preserve discovery identity until verified decode.
use crate::LoadedVolume;
use anyhow::{bail, Result};
use ritk_io::DicomSeriesInfo;
use std::{path::PathBuf, sync::Arc};

#[derive(Debug, Clone)]
pub(crate) enum VolumeInput {
    Path(PathBuf),
    Series(Arc<DicomSeriesInfo>),
}

impl VolumeInput {
    pub(crate) fn restore(source: crate::session::StudySource) -> Result<Self> {
        match source {
            crate::session::StudySource::Path(path) => Ok(Self::Path(path)),
            crate::session::StudySource::Dicom { series_uid, files } => {
                if series_uid.is_empty()
                    || series_uid.len() > 64
                    || !series_uid
                        .bytes()
                        .all(|byte| byte.is_ascii_digit() || byte == b'.')
                    || files.is_empty()
                {
                    bail!("invalid persisted DICOM acquisition identity or empty file selection");
                }
                Ok(Self::Series(Arc::new(DicomSeriesInfo::new(
                    &series_uid,
                    String::new(),
                    "",
                    String::new(),
                    files,
                ))))
            }
        }
    }

    pub(crate) fn acquisition(volume: &LoadedVolume) -> Option<Arc<DicomSeriesInfo>> {
        volume.source.as_ref()?;
        let metadata = volume.metadata.as_ref()?;
        let uid = metadata.series_instance_uid.as_deref()?;
        let mut files: Vec<_> = metadata
            .slices
            .iter()
            .map(|slice| slice.path.clone())
            .collect();
        files.sort();
        Some(Arc::new(DicomSeriesInfo::new(
            uid,
            volume.series_description.clone().unwrap_or_default(),
            volume.modality.as_deref().unwrap_or(""),
            volume.patient_id.clone().unwrap_or_default(),
            files,
        )))
    }

    pub(crate) fn load(&self) -> Result<LoadedVolume> {
        match self {
            Self::Path(path) => crate::dicom::loader::load_volume_from_path(path),
            Self::Series(info) => {
                let scanned = ritk_io::scan_dicom_files(&info.file_paths)?;
                if scanned.metadata.series_instance_uid.as_deref()
                    != Some(info.series_instance_uid())
                {
                    bail!("selected DICOM SeriesInstanceUID changed since discovery");
                }
                let mut expected: Vec<_> = info.file_paths.iter().collect();
                let mut actual: Vec<_> = scanned
                    .metadata
                    .slices
                    .iter()
                    .map(|slice| &slice.path)
                    .collect();
                expected.sort();
                actual.sort();
                if actual != expected {
                    bail!("selected DICOM acquisition membership changed since discovery");
                }
                let mut volume = crate::dicom::loader::load_volume_from_scanned_series(scanned)?;
                volume.source = info.file_paths.first().cloned();
                Ok(volume)
            }
        }
    }
}
