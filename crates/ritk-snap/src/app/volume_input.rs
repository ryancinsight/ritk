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
                if crate::dicom::loader::validate_series_uid(&series_uid).is_err()
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
            Self::Series(info) => crate::dicom::loader::load_volume_from_series_info(info),
        }
    }
}
