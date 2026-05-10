//! Volume loading from DICOM folders and NIfTI files.
//!
//! # Functions
//!
//! - [`load_dicom_volume`]      — load a DICOM series folder into a [`LoadedVolume`].
//! - [`load_nifti_volume`]      — load a NIfTI `.nii` / `.nii.gz` file.
//! - [`load_volume_from_path`]  — auto-detect format and dispatch to the above.
//! - [`load_volume_from_bytes`] — load a pathless in-memory medical file payload.
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

use crate::dicom::input_path::classify_dicom_input_path;
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
    let requested = folder.as_ref();
    let folder = classify_dicom_input_path(requested)
        .dicom_root()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| requested.to_path_buf());
    info!(path = %folder.display(), "loading DICOM volume");

    let device = <B as burn::tensor::backend::Backend>::Device::default();
    let (image, meta) = load_dicom_series_with_metadata::<B, _>(&folder, &device)
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
        .into_vec::<f32>()
        .map_err(|e| anyhow::anyhow!("failed to extract f32 pixel data from DICOM tensor: {e:?}"))?;

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
        source: Some(folder),
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
        .into_vec::<f32>()
        .map_err(|e| anyhow::anyhow!("failed to extract f32 pixel data from NIfTI tensor: {e:?}"))?;

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

    if classify_dicom_input_path(path).dicom_root().is_some() {
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

/// Load a pathless in-memory medical payload.
///
/// Currently supports NIfTI byte payloads identified by `name_hint`
/// (`.nii` / `.nii.gz`).
pub fn load_volume_from_bytes(name_hint: &str, bytes: &[u8]) -> Result<LoadedVolume> {
    let name = name_hint.to_ascii_lowercase();
    if is_likely_dicom_bytes(name_hint, bytes) {
        return load_dicom_series_from_named_bytes(&[(name_hint.to_owned(), bytes)]);
    }

    if name.ends_with(".nii") || name.ends_with(".nii.gz") {
        let device = <B as burn::tensor::backend::Backend>::Device::default();
        let image = ritk_io::read_nifti_from_bytes::<B>(bytes, &device)
            .with_context(|| format!("failed to read dropped NIfTI bytes '{}'", name_hint))?;
        return volume_from_image_no_meta(image, PathBuf::from(name_hint));
    }

    anyhow::bail!(
        "unsupported dropped in-memory file '{}' (supported: DICOM, .nii, .nii.gz)",
        name_hint
    )
}

/// Load a DICOM series from a pathless dropped in-memory byte batch.
///
/// The batch is materialized into a unique temporary directory and then loaded
/// through the canonical DICOM series loader.
pub fn load_dicom_series_from_named_bytes(
    files: &[(String, &[u8])],
) -> Result<LoadedVolume> {
    if files.is_empty() {
        anyhow::bail!("empty DICOM byte batch")
    }

    let temp_root = create_unique_temp_subdir("ritk_snap_dropped_dicom")?;
    for (idx, (name, bytes)) in files.iter().enumerate() {
        if !is_likely_dicom_bytes(name, bytes) {
            continue;
        }
        let file_name = sanitize_temp_filename(name, idx);
        let file_path = temp_root.join(file_name);
        std::fs::write(&file_path, bytes)
            .with_context(|| format!("failed writing dropped DICOM temp file '{}'", file_path.display()))?;
    }

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        load_dicom_volume(&temp_root)
            .with_context(|| {
                format!(
                    "failed to load dropped DICOM byte batch from '{}'",
                    temp_root.display()
                )
            })
    }))
    .unwrap_or_else(|_| {
        anyhow::bail!(
            "dropped DICOM byte batch does not form a loadable series (insufficient slice geometry or invalid frame set)"
        )
    });

    let _ = std::fs::remove_dir_all(&temp_root);
    result
}

fn create_unique_temp_subdir(prefix: &str) -> Result<PathBuf> {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("clock must be after UNIX_EPOCH")
        .as_nanos();
    let pid = std::process::id();
    let path = std::env::temp_dir().join(format!("{prefix}_{pid}_{now}"));
    std::fs::create_dir_all(&path)
        .with_context(|| format!("failed to create temp directory '{}'", path.display()))?;
    Ok(path)
}

fn sanitize_temp_filename(name: &str, index: usize) -> String {
    let trimmed = name.trim();
    if trimmed.is_empty() {
        return format!("slice_{index:04}.dcm");
    }

    let mut cleaned: String = trimmed
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '.' || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect();

    if cleaned.len() > 120 {
        cleaned.truncate(120);
    }
    if cleaned.is_empty() {
        cleaned = format!("slice_{index:04}.dcm");
    }
    if !cleaned.contains('.') {
        cleaned.push_str(".dcm");
    }
    cleaned
}

fn is_likely_dicom_bytes(name_hint: &str, bytes: &[u8]) -> bool {
    let n = name_hint.to_ascii_lowercase();
    if n.ends_with(".dcm") || n.ends_with(".dicom") || n.ends_with(".ima") || n == "dicomdir" {
        return true;
    }
    bytes.len() >= 132 && &bytes[128..132] == b"DICM"
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
        .into_vec::<f32>()
        .map_err(|e| anyhow::anyhow!("failed to extract f32 pixel data from image tensor: {e:?}"))?;

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
    let requested = folder.as_ref();
    let folder = classify_dicom_input_path(requested)
        .dicom_root()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| requested.to_path_buf());
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
            Ok(series_list) => {
                entries.extend(
                    series_list
                        .into_iter()
                        .map(SeriesEntry::from_dicom_series_info),
                );
            }
            Err(e) => {
                warn!(path = %dir.display(), error = %e, "skipping directory (not a DICOM series)");
            }
        }
    };

    // Scan the root folder itself.
    try_add(&folder, &mut entries, &mut seen_folders);

    // Walk subdirectories up to depth 5.
    for entry in WalkDir::new(&folder)
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
    use burn::tensor::{Shape, Tensor, TensorData};
    use ritk_core::image::Image;
    use ritk_core::spatial::{Direction, Point, Spacing};
    use tempfile::tempdir;

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

    #[test]
    fn test_load_volume_from_bytes_nifti_roundtrip_shape() {
        let dir = tempdir().expect("create temp dir");
        let path = dir.path().join("drop_test.nii");
        let device = <B as burn::tensor::backend::Backend>::Device::default();

        let shape = Shape::new([3, 2, 4]);
        let data = TensorData::new((0..24).map(|v| v as f32).collect::<Vec<_>>(), shape);
        let tensor = Tensor::<B, 3>::from_data(data, &device);
        let image = Image::new(
            tensor,
            Point::new([1.0, 2.0, 3.0]),
            Spacing::new([0.8, 0.9, 1.7]),
            Direction(nalgebra::SMatrix::identity()),
        );
        ritk_io::write_nifti(&path, &image).expect("write synthetic nifti");

        let bytes = std::fs::read(&path).expect("read written nifti bytes");
        let vol = load_volume_from_bytes("dropped.nii", &bytes)
            .expect("load_volume_from_bytes should load valid nifti bytes");

        assert_eq!(vol.shape, [3, 2, 4]);
        assert_eq!(vol.data.len(), 24);
        assert!(vol.spacing[0] > 0.0 && vol.spacing[1] > 0.0 && vol.spacing[2] > 0.0);
    }

    #[test]
    fn test_load_dicom_series_from_named_bytes_batch() {
        let dir = std::path::Path::new(r"D:\ritk\test_data\2_head_mri_t2\DICOM");
        if !dir.exists() {
            eprintln!(
                "SKIP test_load_dicom_series_from_named_bytes_batch: fixture absent at {:?}",
                dir
            );
            return;
        }

        let mut owned_files: Vec<(String, Vec<u8>)> = std::fs::read_dir(dir)
            .expect("read DICOM fixture directory")
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.path())
            .filter(|path| path.is_file())
            .map(|path| {
                let name = path
                    .file_name()
                    .and_then(|s| s.to_str())
                    .unwrap_or("slice.dcm")
                    .to_owned();
                let bytes = std::fs::read(&path).expect("read DICOM fixture file bytes");
                (name, bytes)
            })
            .collect();

        owned_files.sort_by(|a, b| a.0.cmp(&b.0));
        let borrowed: Vec<(String, &[u8])> = owned_files
            .iter()
            .map(|(name, bytes)| (name.clone(), bytes.as_slice()))
            .collect();
        let vol = load_dicom_series_from_named_bytes(&borrowed)
            .expect("load_dicom_series_from_named_bytes should load valid DICOM batch");

        let [d, r, c] = vol.shape;
        assert!(d > 0 && r > 0 && c > 0, "loaded DICOM shape must be non-zero");
        assert_eq!(vol.data.len(), d * r * c);
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
