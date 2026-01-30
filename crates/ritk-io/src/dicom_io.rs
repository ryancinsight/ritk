use anyhow::{Context, Result, anyhow};
use burn::tensor::{Tensor, TensorData, Shape};
use burn::tensor::backend::Backend;
use dicom::object::{FileDicomObject, InMemDicomObject, open_file};
use dicom::pixeldata::PixelDecoder;
use dicom::dictionary_std::tags;
use ritk_core::image::Image;
use ritk_core::spatial::{Point, Spacing, Direction};
use nalgebra::{Matrix3, Vector3 as NaVector3, Point3 as NaPoint3};
use std::path::Path;
use std::fs;

/// Read a DICOM series from a directory.
/// Assumes all files in the directory belong to the same series (or filters valid ones).
/// Sorts by Instance Number or Image Position.
pub fn read_dicom_series<B: Backend, P: AsRef<Path>>(path: P, device: &B::Device) -> Result<Image<B, 3>> {
    let path = path.as_ref();
    let mut files = Vec::new();

    // 1. Scan directory for DICOM files
    for entry in fs::read_dir(path).context("Failed to read directory")? {
        let entry = entry?;
        let path = entry.path();
        if path.is_file() {
            // Try opening as DICOM
            if let Ok(obj) = open_file(&path) {
                files.push((path, obj));
            }
        }
    }

    if files.is_empty() {
        return Err(anyhow!("No DICOM files found in {:?}", path));
    }

    // 2. Determine Slice Normal from the first file
    // We assume all files in the series have the same orientation.
    let first_obj_ref = &files[0].1;
    let orientation = get_f64_vec(first_obj_ref, tags::IMAGE_ORIENTATION_PATIENT).context("Missing Orientation in first file")?;
    let dir_x = NaVector3::new(orientation[0], orientation[1], orientation[2]);
    let dir_y = NaVector3::new(orientation[3], orientation[4], orientation[5]);
    let dir_z = dir_x.cross(&dir_y).normalize();

    // 3. Sort files by projection onto normal vector (robust spatial sorting)
    files.sort_by(|a, b| {
        let pos_a_vec = get_f64_vec(&a.1, tags::IMAGE_POSITION_PATIENT).unwrap_or(vec![0.0, 0.0, 0.0]);
        let pos_b_vec = get_f64_vec(&b.1, tags::IMAGE_POSITION_PATIENT).unwrap_or(vec![0.0, 0.0, 0.0]);
        
        let pos_a = NaVector3::new(pos_a_vec[0], pos_a_vec[1], pos_a_vec[2]);
        let pos_b = NaVector3::new(pos_b_vec[0], pos_b_vec[1], pos_b_vec[2]);
        
        let dist_a = pos_a.dot(&dir_z);
        let dist_b = pos_b.dot(&dir_z);
        
        // Float comparison
        dist_a.partial_cmp(&dist_b).unwrap_or(std::cmp::Ordering::Equal)
    });

    // 4. Extract Metadata from the first slice (now sorted)
    let first_obj = &files[0].1;
    
    // Dimensions
    let rows = get_u32(first_obj, tags::ROWS).context("Missing Rows")?;
    let cols = get_u32(first_obj, tags::COLUMNS).context("Missing Columns")?;
    let depth = files.len() as u32;

    // Pixel Spacing (Row, Col) -> (y, x)? No, usually (row spacing, col spacing) which corresponds to Y and X size?
    // DICOM Pixel Spacing (0028,0030) is "Row Spacing\Col Spacing" (mm between centers of adjacent rows/cols).
    // Wait, standard is "Row Spacing \ Column Spacing".
    // Row Spacing = distance between rows (delta Y).
    // Col Spacing = distance between cols (delta X).
    let spacing_vec = get_f64_vec(first_obj, tags::PIXEL_SPACING).context("Missing Pixel Spacing")?;
    let dy = spacing_vec[0];
    let dx = spacing_vec[1];

    // Image Orientation Patient (0020,0037) -> 6 values (Xx, Xy, Xz, Yx, Yy, Yz)
    // Actually they are Row Cosines (direction of rows) and Column Cosines (direction of columns).
    // Row vector corresponds to increasing Column index (X).
    // Column vector corresponds to increasing Row index (Y).
    // So first 3 are X-axis direction, next 3 are Y-axis direction.
    let orientation = get_f64_vec(first_obj, tags::IMAGE_ORIENTATION_PATIENT).context("Missing Orientation")?;
    let dir_x = NaVector3::new(orientation[0], orientation[1], orientation[2]);
    let dir_y = NaVector3::new(orientation[3], orientation[4], orientation[5]);
    
    // Compute Z direction (Slice direction)
    // Z = X cross Y
    let dir_z = dir_x.cross(&dir_y).normalize();

    // Image Position Patient (0020,0032) -> Origin of first slice (center of top-left voxel)
    let pos_vec = get_f64_vec(first_obj, tags::IMAGE_POSITION_PATIENT).context("Missing Position")?;
    let na_origin = NaPoint3::new(pos_vec[0], pos_vec[1], pos_vec[2]);

    // Calculate Slice Spacing (dz)
    // If >1 slice, project (Pos_last - Pos_first) onto Z direction / (N-1)
    let dz = if files.len() > 1 {
        let last_obj = &files.last().unwrap().1;
        let last_pos_vec = get_f64_vec(last_obj, tags::IMAGE_POSITION_PATIENT).context("Missing last slice position")?;
        let last_pos = NaPoint3::new(last_pos_vec[0], last_pos_vec[1], last_pos_vec[2]);
        let dist = (last_pos - na_origin).dot(&dir_z);
        dist.abs() / ((files.len() - 1) as f64)
    } else {
        // Fallback to Slice Thickness or 1.0
        get_f64(first_obj, tags::SLICE_THICKNESS).unwrap_or(1.0)
    };

    // Construct Direction Matrix
    // Columns are directions of X, Y, Z axes of the image array.
    let direction_mat = Matrix3::from_columns(&[dir_x, dir_y, dir_z]);
    let direction = Direction(direction_mat);

    let spacing = Spacing::new([dx, dy, dz]);
    let origin = Point::new([na_origin.x, na_origin.y, na_origin.z]);

    // 4. Load Pixel Data
    // We assume 16-bit signed or unsigned usually for medical images.
    // We'll convert to f32.
    // Flattened buffer: Z * Y * X
    let mut buffer: Vec<f32> = Vec::with_capacity((depth * rows * cols) as usize);

    for (_p, obj) in &files {
        // Decode pixel data
        let pixel_data = obj.decode_pixel_data().context("Failed to decode pixel data")?;
        
        // Rescale Slope/Intercept
        let slope = get_f64(obj, tags::RESCALE_SLOPE).unwrap_or(1.0);
        let intercept = get_f64(obj, tags::RESCALE_INTERCEPT).unwrap_or(0.0);

        // Convert to f32 and apply rescale
        // pixel_data can be u8, u16, i16, etc.
        // We use to_vec_f32() or similar if available, otherwise manual.
        // dicom-pixeldata's DecodedPixelData usually holds raw bytes or typed vec.
        // Let's use dynamic conversion.
        
        let slice_data: Vec<f32> = match pixel_data.to_vec::<f32>() {
            Ok(v) => v,
            Err(_) => {
                // Fallback: manual conversion if needed, but to_vec should work for numeric types
                // If it fails, maybe it's multi-frame? We assume single-frame per file here.
                return Err(anyhow!("Unsupported pixel data format"));
            }
        };

        // Apply slope/intercept
        for val in slice_data {
            buffer.push(val * slope as f32 + intercept as f32);
        }
    }

    // 5. Create Tensor
    // Shape: [Batch, Channel, D, H, W] -> [1, 1, Z, Y, X]
    // Or just [Z, Y, X] for Image<B, 3> which wraps Tensor<B, D> (Z, Y, X).
    // We explicitly verify that the layout is [Z, Y, X] (Depth, Height, Width).
    
    let shape = [depth as usize, rows as usize, cols as usize];
    let data = TensorData::new(buffer, Shape::new(shape));
    let tensor = Tensor::<B, 3>::from_data(data, device);

    Ok(Image::new(tensor, origin, spacing, direction))
}

// Helpers

fn get_u32(obj: &FileDicomObject<InMemDicomObject>, tag: dicom::core::Tag) -> Option<u32> {
    obj.element(tag).ok()?.to_int::<u32>().ok()
}

fn get_f64(obj: &FileDicomObject<InMemDicomObject>, tag: dicom::core::Tag) -> Option<f64> {
    obj.element(tag).ok()?.to_float64().ok()
}

fn get_f64_vec(obj: &FileDicomObject<InMemDicomObject>, tag: dicom::core::Tag) -> Option<Vec<f64>> {
    obj.element(tag).ok()?.to_multi_float64().ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;
    use dicom::object::{FileDicomObject, StandardDataDictionary, FileMetaTableBuilder};
    use dicom::core::{DataElement, VR, PrimitiveValue};
    use tempfile::tempdir;

    type B = NdArray<f32>;

    fn create_dummy_dicom(
        path: &Path,
        instance_number: i32,
        position: [f64; 3],
        pixel_data: Vec<u16>,
        rows: u16,
        cols: u16
    ) -> Result<()> {
        let meta = FileMetaTableBuilder::new()
            .media_storage_sop_class_uid("1.2.840.10008.5.1.4.1.1.4")
            .media_storage_sop_instance_uid(format!("1.2.3.{}", instance_number))
            .transfer_syntax("1.2.840.10008.1.2.1")
            .build()?;
            
        let mut obj = FileDicomObject::new_empty_with_dict_and_meta(StandardDataDictionary, meta);

        // SOP Class UID (MR Image Storage)
        obj.put(DataElement::new(
            tags::SOP_CLASS_UID,
            VR::UI,
            PrimitiveValue::from("1.2.840.10008.5.1.4.1.1.4"),
        ));

        // Instance Number
        obj.put(DataElement::new(
            tags::INSTANCE_NUMBER,
            VR::IS,
            PrimitiveValue::from(instance_number.to_string()),
        ));

        // Image Position Patient
        obj.put(DataElement::new(
            tags::IMAGE_POSITION_PATIENT,
            VR::DS,
            PrimitiveValue::from(format!("{}\\{}\\{}", position[0], position[1], position[2])),
        ));

        // Image Orientation Patient (Identity)
        obj.put(DataElement::new(
            tags::IMAGE_ORIENTATION_PATIENT,
            VR::DS,
            PrimitiveValue::from("1\\0\\0\\0\\1\\0"),
        ));

        // Pixel Spacing (1.0 \ 1.0)
        obj.put(DataElement::new(
            tags::PIXEL_SPACING,
            VR::DS,
            PrimitiveValue::from("1.0\\1.0"),
        ));

        // Rows
        obj.put(DataElement::new(
            tags::ROWS,
            VR::US,
            PrimitiveValue::from(rows),
        ));

        // Columns
        obj.put(DataElement::new(
            tags::COLUMNS,
            VR::US,
            PrimitiveValue::from(cols),
        ));

        // Bits Allocated/Stored
        obj.put(DataElement::new(tags::BITS_ALLOCATED, VR::US, PrimitiveValue::from(16u16)));
        obj.put(DataElement::new(tags::BITS_STORED, VR::US, PrimitiveValue::from(16u16)));
        obj.put(DataElement::new(tags::HIGH_BIT, VR::US, PrimitiveValue::from(15u16)));
        obj.put(DataElement::new(tags::PIXEL_REPRESENTATION, VR::US, PrimitiveValue::from(0u16))); // Unsigned
        obj.put(DataElement::new(tags::SAMPLES_PER_PIXEL, VR::US, PrimitiveValue::from(1u16)));

        // Photometric Interpretation
        obj.put(DataElement::new(
            tags::PHOTOMETRIC_INTERPRETATION,
            VR::CS,
            PrimitiveValue::from("MONOCHROME2"),
        ));

        // Pixel Data
        // dicom-rs expects bytes.
        let mut bytes = Vec::new();
        for val in pixel_data {
            bytes.extend_from_slice(&val.to_le_bytes());
        }
        obj.put(DataElement::new(
            tags::PIXEL_DATA,
            VR::OW,
            PrimitiveValue::from(bytes),
        ));

        obj.write_to_file(path)?;
        Ok(())
    }

    #[test]
    fn test_read_dicom_series_basic() -> Result<()> {
        let dir = tempdir()?;
        let rows = 4;
        let cols = 4;
        
        // Slice 1: z=0
        let data1: Vec<u16> = vec![0; 16];
        create_dummy_dicom(&dir.path().join("1.dcm"), 1, [0.0, 0.0, 0.0], data1, rows, cols)?;

        // Slice 2: z=1
        let data2: Vec<u16> = vec![100; 16];
        create_dummy_dicom(&dir.path().join("2.dcm"), 2, [0.0, 0.0, 1.0], data2, rows, cols)?;

        let device = Default::default();
        let image = read_dicom_series::<B, _>(dir.path(), &device)?;

        // Check Shape: [Z, Y, X] = [2, 4, 4]
        assert_eq!(image.shape(), [2, 4, 4]);

        // Check Spacing: [1.0, 1.0, 1.0] (z spacing derived from position diff)
        let spacing = image.spacing();
        assert!((spacing[2] - 1.0).abs() < 1e-5);
        
        // Check Origin
        let origin = image.origin();
        assert_eq!(origin.to_vec(), vec![0.0, 0.0, 0.0]);

        Ok(())
    }
}
