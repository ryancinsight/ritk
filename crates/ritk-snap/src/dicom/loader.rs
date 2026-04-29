//! Volume loading from DICOM folders and NIfTI files.
//!
//! # Functions
//!
//! - [`load_dicom_volume`]      — load a DICOM series folder into a [`LoadedVolume`].
//! - [`load_nifti_volume`]      — load a NIfTI `.nii` / `.nii.gz` file.
//! - [`load_volume_from_path`]  — auto-detect format and dispatch to the above.
//! - [`scan_folder_for_series`] — walk a directory tree and return a [`SeriesTree`].
//!
//! # Backend
//!
//! All tensor operations use `burn_ndarray::NdArray<f32>` (CPU, synchronous).
//! This isolates the `<B: Backend>` type parameter to this module; callers
//! receive a format-erased [`LoadedVolume`].
//!
//! # Data layout
//!
//! Loaded pixel data is stored in row-major `[depth, rows, cols]` order,
//! matching the RITK tensor convention established in `ritk-io`.
//! Spacing is `[dz, dy, dx]` mm/voxel; origin and direction follow the
//! physical coordinate system of the source format.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use burn_ndarray::NdArray;
use ritk_io::{load_dicom_series_with_metadata, read_nifti, scan_dicom_directory};
use tracing::{info, warn};
use walkdir::WalkDir;

use crate::dicom::series_tree::{SeriesEntry, SeriesTree};
use crate::LoadedVolume;

/// CPU backend alias used for all loading operations in this module.
type B = NdArray<f32>;

// ── DICOM ─────────────────────────────────────────────────────────────────────

/// Load a DICOM series from `folder` into a [`LoadedVolume`].
///
/// # Algorithm
/// 1. Calls `ritk_io::load_dicom_series_with_metadata::<NdArray<f32>>`.
/// 2. Extracts the 3-D f32 tensor (shape `[depth, rows, cols]`).
/// 3. Reads spatial metadata (spacing, origin, direction) from the image.
/// 4. Populates optional DICOM-specific fields from [`DicomReadMetadata`].
/// 5. Sets the window/level hint from the first slice's `(WindowCenter,
///    WindowWidth)` tags when present.
///
/// # Errors
/// Propagates any error returned by `ritk_io`.
pub fn load_dicom_volume<P: AsRef<Path>>(folder: P) -> Result<LoadedVolume> {
    let folder = folder.as_ref();
    info!(path = %folder.display(), "loading DICOM volume");

    let device = <B as burn::tensor::backend::Backend>::Device::default();
    let (image, meta) = load_dicom_series_with_metadata::<B, _>(folder, &device)
        .with_context(|| format!("failed to load DICOM series from '{}'", folder.display()))?;

    let shape = image.shape(); // [depth, rows, cols]
    let sp = image.spacing();
    let orig = image.origin();
    let dir = image.direction();

    let spacing = [sp[0], sp[1], sp[2]];
    let origin = [orig.0[0], orig.0[1], orig.0[2]];
    let dir_slice = dir.0.as_slice();
    let direction: [f64; 9] = [
        dir_slice[0],
        dir_slice[1],
        dir_slice[2],
        dir_slice[3],
        dir_slice[4],
        dir_slice[5],
        dir_slice[6],
        dir_slice[7],
        dir_slice[8],
    ];

    // Extract pixel data from the tensor.
    let tensor = image.into_tensor();
    let tensor_data = tensor.into_data();
    let pixels: Vec<f32> = tensor_data
        .as_slice::<f32>()
        .map_err(|e| anyhow::anyhow!("failed to extract f32 pixel data from DICOM tensor: {e:?}"))?
        .to_vec();

    let modality = meta.modality.clone();
    let patient_name = meta.patient_name.clone();
    let patient_id = meta.patient_id.clone();
    let study_date = meta.study_date.clone();
    let series_description = meta.series_description.clone();

    Ok(LoadedVolume {
        data: std::sync::Arc::new(pixels),
        shape,
        spacing,
        origin,
        direction,
        metadata: Some(Box::new(meta)),
        source: Some(folder.to_path_buf()),
        modality,
        patient_name,
        patient_id,
        study_date,
        series_description,
    })
}

// ── NIfTI ─────────────────────────────────────────────────────────────────────

/// Load a NIfTI volume from `path` (`.nii` or `.nii.gz`) into a [`LoadedVolume`].
///
/// # Algorithm
/// 1. Calls `ritk_io::read_nifti::<NdArray<f32>>`.
/// 2. Extracts shape, spacing, origin, and direction from the returned
///    [`ritk_core::image::Image`].
/// 3. Copies pixel data from the tensor into a heap `Vec<f32>`.
///
/// NIfTI files carry no patient metadata; all optional DICOM fields are
/// left as `None`.
///
/// # Errors
/// Propagates any error returned by `ritk_io`.
pub fn load_nifti_volume<P: AsRef<Path>>(path: P) -> Result<LoadedVolume> {
    let path = path.as_ref();
    info!(path = %path.display(), "loading NIfTI volume");

    let device = <B as burn::tensor::backend::Backend>::Device::default();
    let image = read_nifti::<B, _>(path, &device)
        .with_context(|| format!("failed to read NIfTI file '{}'", path.display()))?;

    let shape = image.shape(); // [depth, rows, cols] per RITK convention
    let sp = image.spacing();
    let orig = image.origin();
    let dir = image.direction();

    let spacing = [sp[0], sp[1], sp[2]];
    let origin = [orig.0[0], orig.0[1], orig.0[2]];
    let dir_slice = dir.0.as_slice();
    let direction: [f64; 9] = [
        dir_slice[0],
        dir_slice[1],
        dir_slice[2],
        dir_slice[3],
        dir_slice[4],
        dir_slice[5],
        dir_slice[6],
        dir_slice[7],
        dir_slice[8],
    ];

    let tensor = image.into_tensor();
    let tensor_data = tensor.into_data();
    let pixels: Vec<f32> = tensor_data
        .as_slice::<f32>()
        .map_err(|e| anyhow::anyhow!("failed to extract f32 pixel data from NIfTI tensor: {e:?}"))?
        .to_vec();

    Ok(LoadedVolume {
        data: std::sync::Arc::new(pixels),
        shape,
        spacing,
        origin,
        direction,
        metadata: None,
        source: Some(path.to_path_buf()),
        modality: None,
        patient_name: None,
        patient_id: None,
        study_date: None,
        series_description: None,
    })
}

// ── Auto-detect ───────────────────────────────────────────────────────────────

/// Auto-detect the volume format from `path` and load accordingly.
///
/// | Condition                                              | Dispatched to       |
/// |--------------------------------------------------------|---------------------|
/// | `path` is a directory                                  | [`load_dicom_volume`] |
/// | Extension is `.nii` or the path ends in `.nii.gz`     | [`load_nifti_volume`] |
/// | Extension is `.mha` or `.mhd`                         | MetaImage (via ritk_io) |
/// | Extension is `.nrrd`                                   | NRRD (via ritk_io)  |
/// | Extension is `.mgh` or `.mgz`                         | MGH (via ritk_io)   |
/// | No extension / unknown extension                       | [`load_dicom_volume`] |
///
/// # Errors
/// Returns an error when the format is unsupported or when the underlying
/// loader fails.
pub fn load_volume_from_path<P: AsRef<Path>>(path: P) -> Result<LoadedVolume> {
    let path = path.as_ref();

    if path.is_dir() {
        return load_dicom_volume(path);
    }

    let path_str = path.to_string_lossy().to_lowercase();

    if path_str.ends_with(".nii.gz") || path_str.ends_with(".nii") {
        return load_nifti_volume(path);
    }

    if path_str.ends_with(".mha") || path_str.ends_with(".mhd") {
        let device = <B as burn::tensor::backend::Backend>::Device::default();
        let image = ritk_io::read_metaimage::<B, _>(path, &device)
            .with_context(|| format!("failed to read MetaImage '{}'", path.display()))?;
        return volume_from_image_no_meta(image, path.to_path_buf());
    }

    if path_str.ends_with(".nrrd") || path_str.ends_with(".nhdr") {
        let device = <B as burn::tensor::backend::Backend>::Device::default();
        let image = ritk_io::read_nrrd::<B, _>(path, &device)
            .with_context(|| format!("failed to read NRRD '{}'", path.display()))?;
        return volume_from_image_no_meta(image, path.to_path_buf());
    }

    if path_str.ends_with(".mgh") || path_str.ends_with(".mgz") {
        let device = <B as burn::tensor::backend::Backend>::Device::default();
        let image = ritk_io::read_mgh::<B, _>(path, &device)
            .with_context(|| format!("failed to read MGH '{}'", path.display()))?;
        return volume_from_image_no_meta(image, path.to_path_buf());
    }

    // Fallback: treat as DICOM folder or single-file DICOM.
    load_dicom_volume(path)
}

/// Convert a generic `Image<B, 3>` (with no DICOM metadata) into a
/// [`LoadedVolume`], recording `source_path` as the origin.
fn volume_from_image_no_meta(
    image: ritk_core::image::Image<B, 3>,
    source_path: PathBuf,
) -> Result<LoadedVolume> {
    let shape = image.shape();
    let sp = image.spacing();
    let orig = image.origin();
    let dir = image.direction();

    let spacing = [sp[0], sp[1], sp[2]];
    let origin = [orig.0[0], orig.0[1], orig.0[2]];
    let dir_slice = dir.0.as_slice();
    let direction: [f64; 9] = [
        dir_slice[0],
        dir_slice[1],
        dir_slice[2],
        dir_slice[3],
        dir_slice[4],
        dir_slice[5],
        dir_slice[6],
        dir_slice[7],
        dir_slice[8],
    ];

    let tensor = image.into_tensor();
    let tensor_data = tensor.into_data();
    let pixels: Vec<f32> = tensor_data
        .as_slice::<f32>()
        .map_err(|e| anyhow::anyhow!("failed to extract f32 pixel data from image tensor: {e:?}"))?
        .to_vec();

    Ok(LoadedVolume {
        data: std::sync::Arc::new(pixels),
        shape,
        spacing,
        origin,
        direction,
        metadata: None,
        source: Some(source_path),
        modality: None,
        patient_name: None,
        patient_id: None,
        study_date: None,
        series_description: None,
    })
}

// ── Directory scan ────────────────────────────────────────────────────────────

/// Walk `folder` and its immediate subdirectories, attempting to scan each
/// directory for DICOM files.
///
/// # Algorithm
/// 1. Try `scan_dicom_directory(folder)` first; add the result when successful.
/// 2. Walk all subdirectories up to depth 5.  For each subdirectory, try
///    `scan_dicom_directory`; skip silently on failure.
/// 3. Deduplicate by folder path so multi-level discovery never double-counts.
/// 4. Build and return a [`SeriesTree`] from the collected [`SeriesEntry`] list.
///
/// This heuristic covers both flat DICOM folders and patient/study/series
/// hierarchies without requiring a DICOMDIR index file.
///
/// # Errors
/// Returns an error only when `folder` itself cannot be read as a directory.
pub fn scan_folder_for_series<P: AsRef<Path>>(folder: P) -> Result<SeriesTree> {
    let folder = folder.as_ref();
    info!(path = %folder.display(), "scanning folder for DICOM series");

    let mut entries: Vec<SeriesEntry> = Vec::new();
    let mut seen_folders: std::collections::HashSet<PathBuf> = std::collections::HashSet::new();

    // Try to scan `dir` for DICOM content and append to `entries` if not
    // already seen.
    let try_add = |dir: &Path,
                   entries: &mut Vec<SeriesEntry>,
                   seen: &mut std::collections::HashSet<PathBuf>| {
        let canonical = dir.canonicalize().unwrap_or_else(|_| dir.to_path_buf());
        if seen.contains(&canonical) {
            return;
        }
        seen.insert(canonical);
        match scan_dicom_directory(dir) {
            Ok(info) => {
                entries.push(SeriesEntry::from_dicom_series_info(info));
            }
            Err(e) => {
                warn!(path = %dir.display(), error = %e, "skipping directory (not a DICOM series)");
            }
        }
    };

    // Scan the root folder itself.
    try_add(folder, &mut entries, &mut seen_folders);

    // Walk subdirectories up to depth 5.
    for entry in WalkDir::new(folder)
        .min_depth(1)
        .max_depth(5)
        .follow_links(false)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_dir())
    {
        try_add(entry.path(), &mut entries, &mut seen_folders);
    }

    info!(
        root = %folder.display(),
        series_found = entries.len(),
        "scan complete"
    );

    Ok(SeriesTree::from_entries(entries))
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Load the OpenNeuro T1w NIfTI file when it is present on disk and verify
    /// that shape, spacing, and pixel data satisfy basic sanity invariants.
    ///
    /// Skip the test when the file does not exist (CI environments without
    /// large test data).
    ///
    /// # Mathematical specification
    /// - `shape[i] > 0` for all i (no zero-extent dimension).
    /// - `spacing[i] > 0.0` for all i (positive voxel pitch).
    /// - `pixels.len() == shape[0] * shape[1] * shape[2]`.
    #[test]
    fn test_load_nifti_volume_shape() {
        let path = std::path::Path::new(r"D:\ritk\test_data\openneuro\sub-01_T1w.nii.gz");
        if !path.exists() {
            eprintln!(
                "SKIP test_load_nifti_volume_shape: test data absent at {:?}",
                path
            );
            return;
        }

        let vol =
            load_nifti_volume(path).expect("load_nifti_volume must succeed on a valid NIfTI file");

        // Shape invariant: every dimension must be > 0.
        let [d, r, c] = vol.shape;
        assert!(d > 0, "depth dimension must be > 0, got {d}");
        assert!(r > 0, "rows dimension must be > 0, got {r}");
        assert!(c > 0, "cols dimension must be > 0, got {c}");

        // Spacing invariant: all spacing values must be positive (mm/voxel).
        for (i, &sp) in vol.spacing.iter().enumerate() {
            assert!(sp > 0.0, "spacing[{i}] must be > 0.0 mm/voxel, got {sp}");
        }

        // Pixel count must exactly match the declared shape.
        let expected_len = d * r * c;
        assert_eq!(
            vol.data.len(),
            expected_len,
            "pixel data length {actual} must equal depth×rows×cols = {expected_len}",
            actual = vol.data.len(),
        );

        // Source path must be recorded.
        assert_eq!(
            vol.source.as_deref(),
            Some(path),
            "source path must be recorded in LoadedVolume"
        );
    }

    /// Load the skull CT DICOM series when it is present and verify basic
    /// structural invariants on the returned volume.
    ///
    /// # Mathematical specification
    /// - `shape[0] > 0` (depth, i.e. number of slices, must be positive).
    /// - `shape[1] > 0` and `shape[2] > 0` (rows and cols must be positive).
    /// - `spacing[i] > 0.0` for all i.
    /// - `data.len() == shape[0] * shape[1] * shape[2]`.
    #[test]
    fn test_load_dicom_volume_shape() {
        let path = std::path::Path::new(r"D:\ritk\test_data\2_skull_ct\DICOM");
        if !path.exists() {
            eprintln!(
                "SKIP test_load_dicom_volume_shape: test data absent at {:?}",
                path
            );
            return;
        }

        let vol = load_dicom_volume(path)
            .expect("load_dicom_volume must succeed on a valid DICOM directory");

        let [depth, rows, cols] = vol.shape;
        assert!(depth > 0, "depth (num slices) must be > 0, got {depth}");
        assert!(rows > 0, "rows must be > 0, got {rows}");
        assert!(cols > 0, "cols must be > 0, got {cols}");

        for (i, &sp) in vol.spacing.iter().enumerate() {
            assert!(sp > 0.0, "spacing[{i}] must be > 0.0, got {sp}");
        }

        assert_eq!(
            vol.data.len(),
            depth * rows * cols,
            "pixel buffer length must equal depth×rows×cols"
        );
    }

    /// Load the head T2 MRI DICOM series (MRI-DIR porcine phantom, CC BY 4.0)
    /// when present and verify basic structural invariants.
    ///
    /// # Mathematical specification
    /// - `shape[0] > 0` (depth / number of slices must be positive).
    /// - `spacing[i] > 0.0` for all i.
    /// - `modality == Some("MR")`.
    /// - `data.len() == shape[0] * shape[1] * shape[2]`.
    #[test]
    fn test_load_head_mri_t2_volume_shape() {
        let path = std::path::Path::new(r"D:\ritk\test_data\2_head_mri_t2\DICOM");
        if !path.exists() {
            eprintln!(
                "SKIP test_load_head_mri_t2_volume_shape: test data absent at {:?}",
                path
            );
            return;
        }

        let vol = load_dicom_volume(path)
            .expect("load_dicom_volume must succeed on the head T2 MRI DICOM directory");

        let [depth, rows, cols] = vol.shape;
        assert!(depth > 0, "depth must be > 0, got {depth}");
        assert!(rows > 0, "rows must be > 0, got {rows}");
        assert!(cols > 0, "cols must be > 0, got {cols}");

        for (i, &sp) in vol.spacing.iter().enumerate() {
            assert!(sp > 0.0, "spacing[{i}] must be > 0.0 mm/voxel, got {sp}");
        }

        assert_eq!(
            vol.data.len(),
            depth * rows * cols,
            "pixel buffer length must equal depth×rows×cols"
        );

        // Modality must be MR.
        assert_eq!(
            vol.modality.as_deref(),
            Some("MR"),
            "modality must be MR for the T2 head series"
        );
    }

    /// `scan_folder_for_series` must return an empty [`SeriesTree`] — not an
    /// error — when the target directory contains no DICOM files.
    #[test]
    fn test_scan_folder_for_series_empty_dir() {
        let dir = tempfile::tempdir().expect("tempdir must be created");
        let tree = scan_folder_for_series(dir.path())
            .expect("scan_folder_for_series must not error on an empty directory");
        assert_eq!(
            tree.total_series(),
            0,
            "empty directory must produce an empty SeriesTree"
        );
    }
}
