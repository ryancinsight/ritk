//! DICOM series scanning, loading, and the `DicomReader` facade.

use anyhow::{bail, Context, Result};
use arrayvec::ArrayString;
use burn::tensor::backend::Backend;
use burn::tensor::{Shape, Tensor, TensorData};
use dicom::dictionary_std::tags;
use dicom::object::{FileDicomObject, InMemDicomObject};
use moirai::ParallelSlice;
use nalgebra::{Matrix3, Point3 as NaPoint3, Vector3 as NaVector3};
use ritk_core::image::Image;
use ritk_core::spatial::{Direction, Point, Spacing};
use ritk_dicom::{
    decode_frame_with, parse_file_with, DecodeFrameRequest, DicomRsBackend, PixelLayout,
};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use super::transfer_syntax::TransferSyntaxKind;
use crate::format::dicom::reader::types::{literal_arraystring, truncate_arraystring};

/// Metadata for a discovered DICOM series
#[derive(Debug, Clone)]
pub struct DicomSeriesInfo {
    pub series_instance_uid: ArrayString<64>,
    pub series_description: String,
    pub modality: ArrayString<16>,
    pub patient_id: String,
    pub file_paths: Vec<PathBuf>,
}

pub(crate) fn sort_discovered_series(series_list: &mut [DicomSeriesInfo]) {
    series_list.sort_by(|a, b| {
        a.patient_id
            .cmp(&b.patient_id)
            .then_with(|| a.modality.cmp(&b.modality))
            .then_with(|| a.series_description.cmp(&b.series_description))
            .then_with(|| a.series_instance_uid.cmp(&b.series_instance_uid))
            .then_with(|| {
                let a_first = a
                    .file_paths
                    .first()
                    .map(|p| p.to_string_lossy().to_string())
                    .unwrap_or_default();
                let b_first = b
                    .file_paths
                    .first()
                    .map(|p| p.to_string_lossy().to_string())
                    .unwrap_or_default();
                a_first.cmp(&b_first)
            })
    });
}

/// Scan a directory for DICOM series, grouping them by SeriesInstanceUID.
///
/// This function scans the directory in parallel to parse DICOM headers.
pub fn scan_dicom_directory<P: AsRef<Path>>(path: P) -> Result<Vec<DicomSeriesInfo>> {
    let path = path.as_ref();

    // Collect all file paths first
    let entries: Vec<PathBuf> = fs::read_dir(path)
        .context("Failed to read directory")?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_file())
        .collect();

    if entries.is_empty() {
        return Ok(Vec::new());
    }

    // Parallel processing to read headers
    let series_map = Arc::new(Mutex::new(
        HashMap::<ArrayString<64>, DicomSeriesInfo>::new(),
    ));

    entries.par().for_each(|file_path| {
        // Try to open as DICOM
        if let Ok(obj) = parse_file_with::<DicomRsBackend, _>(file_path) {
            let uid = match get_string(&obj, tags::SERIES_INSTANCE_UID) {
                Some(u) => match ArrayString::<64>::from(u.trim()) {
                    Ok(v) => v,
                    Err(_) => {
                        tracing::warn!(
                            "SeriesInstanceUID exceeds 64 chars, truncating: {}",
                            &u.trim()[..64]
                        );
                        truncate_arraystring::<64>(u.trim())
                    }
                },
                None => return, // Skip files without SeriesUID
            };

            let description = get_string(&obj, tags::SERIES_DESCRIPTION).unwrap_or_default();
            let modality = get_string(&obj, tags::MODALITY)
                .map(|s| {
                    let trimmed = s.trim();
                    match ArrayString::<16>::from(trimmed) {
                        Ok(v) => v,
                        Err(_) => {
                            tracing::warn!(
                                "Modality exceeds 16 chars, truncating: {}",
                                &trimmed[..16]
                            );
                            truncate_arraystring::<16>(trimmed)
                        }
                    }
                })
                .unwrap_or_else(|| literal_arraystring("OT"));
            let patient_id = get_string(&obj, tags::PATIENT_ID).unwrap_or_default();

            let mut map = series_map.lock().expect(
                "series map mutex poisoned — another thread panicked while holding the lock",
            );
            let entry = map.entry(uid).or_insert_with(|| DicomSeriesInfo {
                series_instance_uid: uid,
                series_description: description,
                modality,
                patient_id,
                file_paths: Vec::new(),
            });
            entry.file_paths.push(file_path.clone());
        }
    });

    let map = Arc::try_unwrap(series_map)
        .expect("series map Arc still has multiple owners — parallel scan must be complete")
        .into_inner()
        .expect("series map mutex must be unlocked after parallel scan");
    let mut series_list: Vec<DicomSeriesInfo> = map.into_values().collect();

    // Sort file paths within each series for determinism (though load_series will re-sort spatially)
    for series in &mut series_list {
        series.file_paths.sort();
    }
    sort_discovered_series(&mut series_list);

    Ok(series_list)
}

/// Load a specific DICOM series into a 3D Image.
///
/// Performs rigorous checks for spatial consistency (uniform spacing, orientation).
pub fn load_dicom_series<B: Backend>(
    series: &DicomSeriesInfo,
    device: &B::Device,
) -> Result<Image<B, 3>> {
    if series.file_paths.is_empty() {
        bail!("Series {} has no files", series.series_instance_uid);
    }

    // 1. Read all headers to sort spatially
    // We read sequentially here or parallel? Parallel is better.
    let mut slices: Vec<(PathBuf, FileDicomObject<InMemDicomObject>)> = series
        .file_paths
        .par()
        .map_collect(|p| {
            let obj =
                parse_file_with::<DicomRsBackend, _>(p).context("Failed to open DICOM file")?;
            Ok((p.clone(), obj))
        })
        .into_iter()
        .collect::<Result<Vec<_>>>()?;

    // 2. Determine Orientation from the first slice
    let first_obj = &slices[0].1;
    let orientation = get_f64_vec(first_obj, tags::IMAGE_ORIENTATION_PATIENT)
        .context("Missing ImageOrientationPatient in first slice")?;

    if orientation.len() != 6 {
        bail!(
            "Invalid ImageOrientationPatient length: {}",
            orientation.len()
        );
    }

    let dir_x = NaVector3::new(orientation[0], orientation[1], orientation[2]);
    let dir_y = NaVector3::new(orientation[3], orientation[4], orientation[5]);

    // Normalize to ensure valid direction cosines
    let dir_x = dir_x.normalize();
    let dir_y = dir_y.normalize();

    let dir_z = dir_x.cross(&dir_y).normalize(); // Assuming orthogonal Z for sorting logic

    // 3. Sort slices by projection onto normal vector
    slices.sort_by(|a, b| {
        let pos_a = get_position(&a.1).unwrap_or(NaPoint3::origin());
        let pos_b = get_position(&b.1).unwrap_or(NaPoint3::origin());

        let dist_a = pos_a.coords.dot(&dir_z);
        let dist_b = pos_b.coords.dot(&dir_z);

        dist_a
            .partial_cmp(&dist_b)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // 4. Validate Spatial Consistency & Calculate Spacing
    // We need to ensure all slices have the same orientation and consistent spacing.
    let first_obj = &slices[0].1;
    let rows = get_u32(first_obj, tags::ROWS).context("Missing Rows")?;
    let cols = get_u32(first_obj, tags::COLUMNS).context("Missing Columns")?;
    let pixel_spacing =
        get_f64_vec(first_obj, tags::PIXEL_SPACING).context("Missing PixelSpacing")?;
    let dy = pixel_spacing[0]; // Row spacing (between rows) -> Y spacing
    let dx = pixel_spacing[1]; // Col spacing (between cols) -> X spacing

    let origin_pos = get_position(first_obj).context("Missing ImagePositionPatient")?;

    let dz = if slices.len() > 1 {
        // Calculate average spacing and check variance
        let mut sum_spacing = 0.0;
        let mut min_spacing = f64::MAX;
        let mut max_spacing = f64::MIN;

        for i in 0..slices.len() - 1 {
            let p1 = get_position(&slices[i].1)
                .expect("slice ImagePositionPatient must be present after spatial sort validation");
            let p2 = get_position(&slices[i + 1].1)
                .expect("slice ImagePositionPatient must be present after spatial sort validation");
            let diff = p2 - p1;
            let spacing = diff.dot(&dir_z).abs(); // Projected distance

            sum_spacing += spacing;
            if spacing < min_spacing {
                min_spacing = spacing;
            }
            if spacing > max_spacing {
                max_spacing = spacing;
            }

            // Validate orientation consistency
            let current_orient =
                get_f64_vec(&slices[i + 1].1, tags::IMAGE_ORIENTATION_PATIENT).unwrap_or_default();
            if current_orient.len() == 6 {
                let cx = NaVector3::new(current_orient[0], current_orient[1], current_orient[2]);
                let cy = NaVector3::new(current_orient[3], current_orient[4], current_orient[5]);
                if (cx - dir_x).norm() > 1e-3 || (cy - dir_y).norm() > 1e-3 {
                    bail!("Inconsistent ImageOrientationPatient in series");
                }
            }
        }

        let avg_spacing = sum_spacing / (slices.len() - 1) as f64;

        // Strict tolerance check (e.g., 1%)
        if (max_spacing - min_spacing) > 0.01 * avg_spacing {
            bail!(
                "Non-uniform slice spacing detected: min={}, max={}, avg={}",
                min_spacing,
                max_spacing,
                avg_spacing
            );
        }

        avg_spacing
    } else {
        get_f64(first_obj, tags::SLICE_THICKNESS).unwrap_or(1.0)
    };

    // 5. Build Spatial Metadata
    let spacing = Spacing::new([dx, dy, dz]);
    let origin = Point::new([origin_pos.x, origin_pos.y, origin_pos.z]);
    let direction_mat = Matrix3::from_columns(&[dir_x, dir_y, dir_z]);
    let direction = Direction(direction_mat);

    // 6. Load Pixel Data in Parallel
    // Chunk buffer for parallel writing
    // We can't easily mutate a Vec in parallel without unsafe or splitting.
    // Using par_iter to decode and collect is better.
    let slice_pixels: Vec<Vec<f32>> = slices
        .par()
        .map_collect(|(_p, obj)| {
            let slope = get_f64(obj, tags::RESCALE_SLOPE).unwrap_or(1.0);
            let intercept = get_f64(obj, tags::RESCALE_INTERCEPT).unwrap_or(0.0);
            let samples_per_pixel = get_u32(obj, tags::SAMPLES_PER_PIXEL).unwrap_or(1) as usize;
            let bits_allocated = get_u32(obj, tags::BITS_ALLOCATED).unwrap_or(16) as u16;
            let pixel_representation = get_u32(obj, tags::PIXEL_REPRESENTATION).unwrap_or(0) as u16;
            let transfer_syntax = TransferSyntaxKind::from_uid(obj.meta().transfer_syntax());

            let rescaled = decode_frame_with::<DicomRsBackend>(
                obj,
                DecodeFrameRequest {
                    frame_index: 0,
                    transfer_syntax,
                    layout: PixelLayout {
                        rows: rows as usize,
                        cols: cols as usize,
                        samples_per_pixel,
                        bits_allocated,
                        pixel_representation,
                        rescale_slope: slope as f32,
                        rescale_intercept: intercept as f32,
                    },
                },
            )
            .context("Failed to decode pixel data")?
            .pixels;

            if rescaled.len() != (rows * cols) as usize {
                return Err(anyhow::anyhow!(
                    "Slice data size mismatch: expected {}, got {}",
                    rows * cols,
                    rescaled.len()
                ));
            }

            Ok(rescaled)
        })
        .into_iter()
        .collect::<Result<Vec<_>>>()?;

    // Flatten
    let depth = slices.len();
    let volume_size = depth * rows as usize * cols as usize;
    let mut flattened = Vec::with_capacity(volume_size);
    for slice in slice_pixels {
        flattened.extend(slice);
    }

    // 7. Create Tensor
    let shape = Shape::new([depth, rows as usize, cols as usize]);
    let data = TensorData::new(flattened, shape);
    let tensor = Tensor::<B, 3>::from_data(data, device);

    Ok(Image::new(tensor, origin, spacing, direction))
}

/// Convenience function to read a single series from a directory.
/// If multiple series exist, it errors out to avoid ambiguity.
pub fn read_dicom_series<B: Backend, P: AsRef<Path>>(
    path: P,
    device: &B::Device,
) -> Result<Image<B, 3>> {
    // Convert path to owned string for error messages before move
    let path_ref = path.as_ref().to_path_buf();

    let series_list = scan_dicom_directory(&path_ref)?;
    if series_list.is_empty() {
        bail!("No DICOM series found in {:?}", path_ref);
    }
    if series_list.len() > 1 {
        bail!(
            "Multiple DICOM series found in {:?}. Use scan_dicom_directory to select one.",
            path_ref
        );
    }

    load_dicom_series(&series_list[0], device)
}

// --- Helpers ---

fn get_string(obj: &FileDicomObject<InMemDicomObject>, tag: dicom::core::Tag) -> Option<String> {
    obj.element(tag).ok()?.to_str().ok().map(|s| s.to_string())
}

fn get_u32(obj: &FileDicomObject<InMemDicomObject>, tag: dicom::core::Tag) -> Option<u32> {
    obj.element(tag).ok()?.to_int::<u32>().ok()
}

fn get_f64(obj: &FileDicomObject<InMemDicomObject>, tag: dicom::core::Tag) -> Option<f64> {
    obj.element(tag).ok()?.to_float64().ok()
}

fn get_f64_vec(obj: &FileDicomObject<InMemDicomObject>, tag: dicom::core::Tag) -> Option<Vec<f64>> {
    obj.element(tag).ok()?.to_multi_float64().ok()
}

fn get_position(obj: &FileDicomObject<InMemDicomObject>) -> Option<NaPoint3<f64>> {
    let v = get_f64_vec(obj, tags::IMAGE_POSITION_PATIENT)?;
    if v.len() == 3 {
        Some(NaPoint3::new(v[0], v[1], v[2]))
    } else {
        None
    }
}

use crate::domain::ImageReader;

/// DIP boundary executing strict `ImageReader` invariants over standard DICOM datasets.
pub struct DicomReader<B: Backend> {
    device: B::Device,
}

impl<B: Backend> DicomReader<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}

impl<B: Backend> ImageReader<B, 3> for DicomReader<B> {
    fn read<P: AsRef<Path>>(&self, path: P) -> std::io::Result<Image<B, 3>> {
        read_dicom_series(path, &self.device).map_err(|e| std::io::Error::other(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_scan_empty_dir() {
        let temp = tempfile::tempdir().unwrap();
        let series = scan_dicom_directory(temp.path()).unwrap();
        assert!(series.is_empty());
    }

    #[test]
    fn discovered_series_sort_is_deterministic() {
        let mut v = vec![
            DicomSeriesInfo {
                series_instance_uid: ArrayString::from("2").unwrap(),
                series_description: "B".to_owned(),
                modality: ArrayString::from("MR").unwrap(),
                patient_id: "P2".to_owned(),
                file_paths: vec![PathBuf::from("z/2.dcm")],
            },
            DicomSeriesInfo {
                series_instance_uid: ArrayString::from("1").unwrap(),
                series_description: "A".to_owned(),
                modality: ArrayString::from("CT").unwrap(),
                patient_id: "P1".to_owned(),
                file_paths: vec![PathBuf::from("a/1.dcm")],
            },
            DicomSeriesInfo {
                series_instance_uid: ArrayString::from("3").unwrap(),
                series_description: "A".to_owned(),
                modality: ArrayString::from("CT").unwrap(),
                patient_id: "P1".to_owned(),
                file_paths: vec![PathBuf::from("b/1.dcm")],
            },
        ];

        sort_discovered_series(&mut v);

        let uids: Vec<&str> = v.iter().map(|s| s.series_instance_uid.as_str()).collect();
        assert_eq!(uids, vec!["1", "3", "2"]);
    }
}
