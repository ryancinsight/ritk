//! DICOM series writer using dicom-rs v0.8.2.
//! Transfer syntax: Explicit VR LE. Each .dcm has 128-byte preamble + DICM magic.

use anyhow::{bail, Context, Result};
use burn::tensor::backend::Backend;
use dicom::core::smallvec::SmallVec;
use dicom::core::{DataElement, PrimitiveValue, Tag, VR};
use dicom::object::meta::FileMetaTableBuilder;
use dicom::object::InMemDicomObject;
use ritk_core::image::Image;
use std::marker::PhantomData;
use std::path::{Path, PathBuf};

fn generate_series_uid() -> String {
    let t = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!("2.25.{}", t)
}

fn generate_instance_uid(series_uid: &str, instance: usize) -> String {
    format!("{}.{}", series_uid, instance + 1)
}

pub fn write_dicom_series<B: Backend, P: AsRef<Path>>(path: P, image: &Image<B, 3>) -> Result<()> {
    let path = path.as_ref();
    let [depth, rows, cols] = image.shape();
    if depth == 0 || rows == 0 || cols == 0 {
        bail!("DICOM: depth={depth} rows={rows} cols={cols} must be >0");
    }
    let series_dir = ensure_series_directory(path)?;
    let series_uid = generate_series_uid();
    let study_uid = series_uid.clone();
    let series_instance_uid = format!("{}.1", series_uid);
    let td = image.data().clone().into_data();
    let all_data: &[f32] = td
        .as_slice::<f32>()
        .map_err(|e| anyhow::anyhow!("image tensor must contain f32 data: {:?}", e))?;
    let slice_len = rows * cols;
    for z in 0..depth {
        let slice_offset = z * slice_len;
        let slice_f32 = &all_data[slice_offset..slice_offset + slice_len];
        let (min_val, max_val) = slice_f32.iter().copied()
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(mn, mx), v| (mn.min(v), mx.max(v)));
        let (rescale_slope, rescale_intercept) = if (max_val - min_val).abs() <= f32::EPSILON {
            (1.0_f32, min_val)
        } else {
            ((max_val - min_val) / 65535.0_f32, min_val)
        };
        let pixel_u16: Vec<u16> = slice_f32.iter()
            .map(|&v| ((v - rescale_intercept) / rescale_slope).round() as u16)
            .collect();
        let sop_instance_uid = generate_instance_uid(&series_uid, z);
        let mut obj = InMemDicomObject::new_empty();
        obj.put(DataElement::new(Tag(0x0008,0x0016),VR::UI,PrimitiveValue::from("1.2.840.10008.5.1.4.1.1.7")));
        obj.put(DataElement::new(Tag(0x0008,0x0018),VR::UI,PrimitiveValue::from(sop_instance_uid.as_str())));
        obj.put(DataElement::new(Tag(0x0008,0x0060),VR::CS,PrimitiveValue::from("OT")));
        obj.put(DataElement::new(Tag(0x0020,0x000D),VR::UI,PrimitiveValue::from(study_uid.as_str())));
        obj.put(DataElement::new(Tag(0x0020,0x000E),VR::UI,PrimitiveValue::from(series_instance_uid.as_str())));
        obj.put(DataElement::new(Tag(0x0020,0x0013),VR::IS,PrimitiveValue::from(format!("{}", z+1))));
        obj.put(DataElement::new(Tag(0x0028,0x0010),VR::US,PrimitiveValue::from(rows as u16)));
        obj.put(DataElement::new(Tag(0x0028,0x0011),VR::US,PrimitiveValue::from(cols as u16)));
        obj.put(DataElement::new(Tag(0x0028,0x0100),VR::US,PrimitiveValue::from(16_u16)));
        obj.put(DataElement::new(Tag(0x0028,0x0101),VR::US,PrimitiveValue::from(16_u16)));
        obj.put(DataElement::new(Tag(0x0028,0x0102),VR::US,PrimitiveValue::from(15_u16)));
        obj.put(DataElement::new(Tag(0x0028,0x0103),VR::US,PrimitiveValue::from(0_u16)));
        obj.put(DataElement::new(Tag(0x0028,0x1053),VR::DS,PrimitiveValue::from(format!("{:.6}",rescale_slope))));
        obj.put(DataElement::new(Tag(0x0028,0x1052),VR::DS,PrimitiveValue::from(format!("{:.6}",rescale_intercept))));
        obj.put(DataElement::new(Tag(0x0028,0x0004),VR::CS,PrimitiveValue::from("MONOCHROME2")));
        obj.put(DataElement::new(Tag(0x7FE0,0x0010),VR::OW,PrimitiveValue::U16(SmallVec::from_vec(pixel_u16))));
        let file_obj = obj.with_meta(
            FileMetaTableBuilder::new()
                .media_storage_sop_class_uid("1.2.840.10008.5.1.4.1.1.7")
                .media_storage_sop_instance_uid(sop_instance_uid.as_str())
                .transfer_syntax("1.2.840.10008.1.2.1"),)
            .map_err(|e| anyhow::anyhow!("DICOM meta failed slice {z}: {e}"))?;
        let slice_path = series_dir.join(format!("slice_{z:04}.dcm"));
        file_obj.write_to_file(&slice_path)
            .map_err(|e| anyhow::anyhow!("write slice {z} failed: {e}"))?;
    }
    Ok(())
}

pub struct DicomWriter<B> { _phantom: PhantomData<B> }
impl<B> DicomWriter<B> {
    pub fn new() -> Self { Self { _phantom: PhantomData } }
    pub fn series_path<P: AsRef<Path>>(path: P) -> PathBuf { path.as_ref().to_path_buf() }
}
impl<B> Default for DicomWriter<B> { fn default() -> Self { Self::new() } }

fn ensure_series_directory(path: &Path) -> Result<PathBuf> {
    if path.exists() {
        if !path.is_dir() { bail!("DICOM output path is not a directory"); }
        return Ok(path.to_path_buf());
    }
    std::fs::create_dir_all(path)
        .with_context(|| "failed to create DICOM series output directory")?;
    Ok(path.to_path_buf())
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Shape, Tensor, TensorData};
    use ritk_core::image::Image;
    use ritk_core::spatial::{Direction, Point, Spacing};
    type Backend = burn_ndarray::NdArray<f32>;

    fn make_image(depth: usize, rows: usize, cols: usize, fill: f32) -> Image<Backend, 3> {
        let device = Default::default();
        let data = vec![fill; depth * rows * cols];
        let tensor = Tensor::<Backend, 3>::from_data(
            TensorData::new(data, Shape::new([depth, rows, cols])), &device,
        );
        Image::new(tensor, Point::new([0.0,0.0,0.0]),
            Spacing::new([1.0,1.0,1.0]), Direction::identity())
    }

    #[test]
    fn test_writer_rejects_zero_dimension() {
        let image = make_image(0, 4, 4, 0.5);
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("series");
        let result = write_dicom_series(&path, &image);
        assert!(result.is_err(), "zero depth must be rejected");
    }

    #[test]
    fn test_writer_creates_correct_number_of_slice_files() {
        let image = make_image(3, 4, 5, 0.5);
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("series");
        write_dicom_series(&path, &image).expect("write must succeed");
        assert!(path.is_dir());
        let count = std::fs::read_dir(&path).unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().and_then(|s| s.to_str()) == Some("dcm"))
            .count();
        assert_eq!(count, 3, "must produce exactly 3 .dcm files");
    }

    #[test]
    fn test_writer_slice_files_are_nonempty() {
        let image = make_image(2, 8, 8, 100.0);
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("series");
        write_dicom_series(&path, &image).expect("write must succeed");
        for entry in std::fs::read_dir(&path).unwrap().filter_map(|e| e.ok()) {
            if entry.path().extension().and_then(|s| s.to_str()) == Some("dcm") {
                let size = std::fs::metadata(entry.path()).unwrap().len();
                assert!(size > 200, "DICOM slice must be >200 bytes, got {}", size);
            }
        }
    }

    #[test]
    fn test_writer_dcm_starts_with_dicom_magic() {
        let image = make_image(1, 4, 4, 0.0);
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("series");
        write_dicom_series(&path, &image).expect("write must succeed");
        let dcm_path = path.join("slice_0000.dcm");
        let bytes = std::fs::read(&dcm_path).expect("slice file must exist");
        assert!(bytes.len() >= 132, "DICOM file must be >=132 bytes");
        assert_eq!(&bytes[128..132], b"DICM",
            "DICOM magic bytes must be present at offset 128");
    }
}
