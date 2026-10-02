//! NIfTI volume loading into LoadedVolume.

use anyhow::{Context, Result};
use std::path::Path;
use tracing::info;

use crate::LoadedVolume;

use super::convert::volume_from_image_no_meta;

/// Load a NIfTI volume from `path` (`.nii` or `.nii.gz`) into a [`LoadedVolume`].
///
/// # Algorithm
/// 1. Calls the native [`ritk_io::read_image_native`] dispatch.
/// 2. Preserves shape, spacing, origin, and direction from the Coeus image.
/// 3. Transfers pixel ownership into the viewer volume.
///
/// NIfTI files carry no patient metadata; all optional DICOM fields are
/// left as `None`.
///
/// # Errors
/// Propagates any error returned by `ritk_io`.
pub fn load_nifti_volume<P: AsRef<Path>>(path: P) -> Result<LoadedVolume> {
    let path = path.as_ref();
    info!(path = %path.display(), "loading NIfTI volume");

    let image = ritk_io::read_image_native(path)
        .with_context(|| format!("failed to read NIfTI file '{}'", path.display()))?;
    volume_from_image_no_meta(image, path.to_path_buf())
}
