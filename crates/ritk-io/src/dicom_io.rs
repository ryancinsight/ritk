use anyhow::{Context, Result, anyhow, bail};
use burn::tensor::{Tensor, TensorData, Shape};
use burn::tensor::backend::Backend;
use dicom::object::{FileDicomObject, InMemDicomObject, open_file};
use dicom::pixeldata::PixelDecoder;
use dicom::dictionary_std::tags;
use ritk_core::image::Image;
use ritk_core::spatial::{Point, Spacing, Direction};
use nalgebra::{Matrix3, Vector3 as NaVector3, Point3 as NaPoint3};
use std::path::{Path, PathBuf};
use std::fs;
use std::collections::HashMap;
use rayon::prelude::*;
use std::sync::{Mutex, Arc};

/// Metadata for a discovered DICOM series
#[derive(Debug, Clone)]
pub struct DicomSeriesInfo {
    pub series_instance_uid: String,
    pub series_description: String,
    pub modality: String,
    pub patient_id: String,
    pub file_paths: Vec<PathBuf>,
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
    let series_map = Arc::new(Mutex::new(HashMap::<String, DicomSeriesInfo>::new()));

    entries.par_iter().for_each(|file_path| {
        // Try to open as DICOM
        if let Ok(obj) = open_file(file_path) {
            let uid = match get_string(&obj, tags::SERIES_INSTANCE_UID) {
                Some(u) => u,
                None => return, // Skip files without SeriesUID
            };

            let description = get_string(&obj, tags::SERIES_DESCRIPTION).unwrap_or_default();
            let modality = get_string(&obj, tags::MODALITY).unwrap_or_default();
            let patient_id = get_string(&obj, tags::PATIENT_ID).unwrap_or_default();

            let mut map = series_map.lock().unwrap();
            let entry = map.entry(uid.clone()).or_insert_with(|| DicomSeriesInfo {
                series_instance_uid: uid,
                series_description: description,
                modality,
                patient_id,
                file_paths: Vec::new(),
            });
            entry.file_paths.push(file_path.clone());
        }
    });

    let map = Arc::try_unwrap(series_map).unwrap().into_inner().unwrap();
    let mut series_list: Vec<DicomSeriesInfo> = map.into_values().collect();
    
    // Sort file paths within each series for determinism (though load_series will re-sort spatially)
    for series in &mut series_list {
        series.file_paths.sort();
    }

    Ok(series_list)
}

/// Load a specific DICOM series into a 3D Image.
///
/// Performs rigorous checks for spatial consistency (uniform spacing, orientation).
pub fn load_dicom_series<B: Backend>(series: &DicomSeriesInfo, device: &B::Device) -> Result<Image<B, 3>> {
    if series.file_paths.is_empty() {
        bail!("Series {} has no files", series.series_instance_uid);
    }

    // 1. Read all headers to sort spatially
    // We read sequentially here or parallel? Parallel is better.
    let mut slices: Vec<(PathBuf, FileDicomObject<InMemDicomObject>)> = series.file_paths.par_iter()
        .map(|p| {
            let obj = open_file(p).context("Failed to open DICOM file")?;
            Ok((p.clone(), obj))
        })
        .collect::<Result<Vec<_>>>()?;

    // 2. Determine Orientation from the first slice
    let first_obj = &slices[0].1;
    let orientation = get_f64_vec(first_obj, tags::IMAGE_ORIENTATION_PATIENT)
        .context("Missing ImageOrientationPatient in first slice")?;
    
    if orientation.len() != 6 {
        bail!("Invalid ImageOrientationPatient length: {}", orientation.len());
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
        
        dist_a.partial_cmp(&dist_b).unwrap_or(std::cmp::Ordering::Equal)
    });

    // 4. Validate Spatial Consistency & Calculate Spacing
    // We need to ensure all slices have the same orientation and consistent spacing.
    let first_obj = &slices[0].1;
    let rows = get_u32(first_obj, tags::ROWS).context("Missing Rows")?;
    let cols = get_u32(first_obj, tags::COLUMNS).context("Missing Columns")?;
    let pixel_spacing = get_f64_vec(first_obj, tags::PIXEL_SPACING).context("Missing PixelSpacing")?;
    let dy = pixel_spacing[0]; // Row spacing (between rows) -> Y spacing
    let dx = pixel_spacing[1]; // Col spacing (between cols) -> X spacing

    let origin_pos = get_position(first_obj).context("Missing ImagePositionPatient")?;
    
    let dz = if slices.len() > 1 {
        // Calculate average spacing and check variance
        let mut sum_spacing = 0.0;
        let mut min_spacing = f64::MAX;
        let mut max_spacing = f64::MIN;

        for i in 0..slices.len() - 1 {
            let p1 = get_position(&slices[i].1).unwrap();
            let p2 = get_position(&slices[i+1].1).unwrap();
            let diff = p2 - p1;
            let spacing = diff.dot(&dir_z).abs(); // Projected distance
            
            sum_spacing += spacing;
            if spacing < min_spacing { min_spacing = spacing; }
            if spacing > max_spacing { max_spacing = spacing; }
            
            // Validate orientation consistency
            let current_orient = get_f64_vec(&slices[i+1].1, tags::IMAGE_ORIENTATION_PATIENT).unwrap_or_default();
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
            bail!("Non-uniform slice spacing detected: min={}, max={}, avg={}", min_spacing, max_spacing, avg_spacing);
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
    let slice_pixels: Vec<Vec<f32>> = slices.par_iter()
        .map(|(_p, obj)| {
             let pixel_data = obj.decode_pixel_data().context("Failed to decode pixel data")?;
             let slope = get_f64(obj, tags::RESCALE_SLOPE).unwrap_or(1.0);
             let intercept = get_f64(obj, tags::RESCALE_INTERCEPT).unwrap_or(0.0);
             
             // Convert to f32
             let data = pixel_data.to_vec::<f32>().map_err(|e| anyhow!("Pixel data conversion error: {}", e))?;
             
             // Apply rescaling
             let rescaled: Vec<f32> = data.into_iter().map(|v| v * slope as f32 + intercept as f32).collect();
             
             if rescaled.len() != rows as usize * cols as usize {
                 return Err(anyhow!("Slice data size mismatch: expected {}, got {}", rows as usize * cols as usize, rescaled.len()));
             }
             
             Ok(rescaled)
        })
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
pub fn read_dicom_series<B: Backend, P: AsRef<Path>>(path: P, device: &B::Device) -> Result<Image<B, 3>> {
    // Convert path to owned string for error messages before move
    let path_ref = path.as_ref().to_path_buf();
    
    let series_list = scan_dicom_directory(&path_ref)?;
    if series_list.is_empty() {
        bail!("No DICOM series found in {:?}", path_ref);
    }
    if series_list.len() > 1 {
        bail!("Multiple DICOM series found in {:?}. Use scan_dicom_directory to select one.", path_ref);
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

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;
    // Basic compilation test
    type TestBackend = NdArray<f32>;
    
    #[test]
    fn test_scan_empty_dir() {
        let temp = tempfile::tempdir().unwrap();
        let series = scan_dicom_directory(temp.path()).unwrap();
        assert!(series.is_empty());
    }
}
