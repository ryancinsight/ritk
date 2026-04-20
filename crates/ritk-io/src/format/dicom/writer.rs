//! DICOM series writer using dicom-rs v0.8.2.
//! Transfer syntax: Explicit VR LE. Each .dcm has 128-byte preamble + DICM magic.

use super::reader::DicomReadMetadata;
use anyhow::{bail, Context, Result};
use burn::tensor::backend::Backend;
use dicom::core::smallvec::SmallVec;
use dicom::core::{DataElement, PrimitiveValue, Tag, VR};
use dicom::object::meta::FileMetaTableBuilder;
use dicom::object::InMemDicomObject;
use ritk_core::image::Image;
use std::marker::PhantomData;
use std::path::{Path, PathBuf};

const DICOM_SOP_CLASS_SECONDARY_CAPTURE: &str = "1.2.840.10008.5.1.4.1.1.7";

fn format_triplet(value: [f64; 3]) -> String {
    format!("{:.6}\\{:.6}\\{:.6}", value[0], value[1], value[2])
}

fn format_pair(value: [f64; 2]) -> String {
    format!("{:.6}\\{:.6}", value[0], value[1])
}

fn format_six(value: [f64; 6]) -> String {
    format!(
        "{:.6}\\{:.6}\\{:.6}\\{:.6}\\{:.6}\\{:.6}",
        value[0], value[1], value[2], value[3], value[4], value[5]
    )
}

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
        let (min_val, max_val) = slice_f32
            .iter()
            .copied()
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(mn, mx), v| {
                (mn.min(v), mx.max(v))
            });
        let (rescale_slope, rescale_intercept) = if (max_val - min_val).abs() <= f32::EPSILON {
            (1.0_f32, min_val)
        } else {
            ((max_val - min_val) / 65535.0_f32, min_val)
        };
        let pixel_u16: Vec<u16> = slice_f32
            .iter()
            .map(|&v| ((v - rescale_intercept) / rescale_slope).round() as u16)
            .collect();
        let sop_instance_uid = generate_instance_uid(&series_uid, z);
        let mut obj = InMemDicomObject::new_empty();
        obj.put(DataElement::new(
            Tag(0x0008, 0x0016),
            VR::UI,
            PrimitiveValue::from("1.2.840.10008.5.1.4.1.1.7"),
        ));
        obj.put(DataElement::new(
            Tag(0x0008, 0x0018),
            VR::UI,
            PrimitiveValue::from(sop_instance_uid.as_str()),
        ));
        obj.put(DataElement::new(
            Tag(0x0008, 0x0060),
            VR::CS,
            PrimitiveValue::from("OT"),
        ));
        obj.put(DataElement::new(
            Tag(0x0020, 0x000D),
            VR::UI,
            PrimitiveValue::from(study_uid.as_str()),
        ));
        obj.put(DataElement::new(
            Tag(0x0020, 0x000E),
            VR::UI,
            PrimitiveValue::from(series_instance_uid.as_str()),
        ));
        obj.put(DataElement::new(
            Tag(0x0020, 0x0013),
            VR::IS,
            PrimitiveValue::from(format!("{}", z + 1)),
        ));
        obj.put(DataElement::new(
            Tag(0x0028, 0x0010),
            VR::US,
            PrimitiveValue::from(rows as u16),
        ));
        obj.put(DataElement::new(
            Tag(0x0028, 0x0011),
            VR::US,
            PrimitiveValue::from(cols as u16),
        ));
        obj.put(DataElement::new(
            Tag(0x0028, 0x0100),
            VR::US,
            PrimitiveValue::from(16_u16),
        ));
        obj.put(DataElement::new(
            Tag(0x0028, 0x0101),
            VR::US,
            PrimitiveValue::from(16_u16),
        ));
        obj.put(DataElement::new(
            Tag(0x0028, 0x0102),
            VR::US,
            PrimitiveValue::from(15_u16),
        ));
        obj.put(DataElement::new(
            Tag(0x0028, 0x0103),
            VR::US,
            PrimitiveValue::from(0_u16),
        ));
        obj.put(DataElement::new(
            Tag(0x0028, 0x1053),
            VR::DS,
            PrimitiveValue::from(format!("{:.6}", rescale_slope)),
        ));
        obj.put(DataElement::new(
            Tag(0x0028, 0x1052),
            VR::DS,
            PrimitiveValue::from(format!("{:.6}", rescale_intercept)),
        ));
        obj.put(DataElement::new(
            Tag(0x0028, 0x0004),
            VR::CS,
            PrimitiveValue::from("MONOCHROME2"),
        ));
        obj.put(DataElement::new(
            Tag(0x7FE0, 0x0010),
            VR::OW,
            PrimitiveValue::U16(SmallVec::from_vec(pixel_u16)),
        ));
        let file_obj = obj
            .with_meta(
                FileMetaTableBuilder::new()
                    .media_storage_sop_class_uid("1.2.840.10008.5.1.4.1.1.7")
                    .media_storage_sop_instance_uid(sop_instance_uid.as_str())
                    .transfer_syntax("1.2.840.10008.1.2.1"),
            )
            .map_err(|e| anyhow::anyhow!("DICOM meta failed slice {z}: {e}"))?;
        let slice_path = series_dir.join(format!("slice_{z:04}.dcm"));
        file_obj
            .write_to_file(&slice_path)
            .map_err(|e| anyhow::anyhow!("write slice {z} failed: {e}"))?;
    }
    Ok(())
}

/// Write a DICOM series with optional metadata propagation.
///
/// When `metadata` is `Some`, spatial reference tags (ImagePositionPatient,
/// ImageOrientationPatient, PixelSpacing, SliceThickness) and series-level
/// identifiers are written into each slice. When `None`, the writer falls
/// back to generated UIDs and default tag values (identical to
/// `write_dicom_series`).
pub fn write_dicom_series_with_metadata<B: Backend, P: AsRef<Path>>(
    path: P,
    image: &Image<B, 3>,
    metadata: Option<&DicomReadMetadata>,
) -> Result<()> {
    let path = path.as_ref();
    let [depth, rows, cols] = image.shape();
    if depth == 0 || rows == 0 || cols == 0 {
        bail!("DICOM: depth={depth} rows={rows} cols={cols} must be >0");
    }
    let series_dir = ensure_series_directory(path)?;

    // Resolve UIDs: prefer metadata, fall back to generated.
    let generated_uid = generate_series_uid();
    let series_uid = metadata
        .and_then(|m| m.series_instance_uid.as_deref())
        .unwrap_or(&generated_uid);
    let study_uid = metadata
        .and_then(|m| m.study_instance_uid.as_deref())
        .unwrap_or(&generated_uid);

    let modality = metadata.and_then(|m| m.modality.as_deref()).unwrap_or("OT");
    let photometric = metadata
        .and_then(|m| m.photometric_interpretation.as_deref())
        .unwrap_or("MONOCHROME2");

    let sop_class = DICOM_SOP_CLASS_SECONDARY_CAPTURE;

    // Spatial parameters from metadata or defaults.
    let spacing = metadata.map(|m| m.spacing).unwrap_or([1.0, 1.0, 1.0]);
    let origin = metadata.map(|m| m.origin).unwrap_or([0.0, 0.0, 0.0]);
    let direction = metadata
        .map(|m| m.direction)
        .unwrap_or([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
    // Slice normal is direction[6..9].
    let normal = [direction[6], direction[7], direction[8]];

    let td = image.data().clone().into_data();
    let all_data: &[f32] = td
        .as_slice::<f32>()
        .map_err(|e| anyhow::anyhow!("image tensor must contain f32 data: {:?}", e))?;
    let slice_len = rows * cols;

    for z in 0..depth {
        let slice_offset = z * slice_len;
        let slice_f32 = &all_data[slice_offset..slice_offset + slice_len];
        let (min_val, max_val) = slice_f32
            .iter()
            .copied()
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(mn, mx), v| {
                (mn.min(v), mx.max(v))
            });
        let (rescale_slope, rescale_intercept) = if (max_val - min_val).abs() <= f32::EPSILON {
            (1.0_f32, min_val)
        } else {
            ((max_val - min_val) / 65535.0_f32, min_val)
        };
        let pixel_u16: Vec<u16> = slice_f32
            .iter()
            .map(|&v| ((v - rescale_intercept) / rescale_slope).round() as u16)
            .collect();

        let sop_instance_uid = generate_instance_uid(series_uid, z);
        let mut obj = InMemDicomObject::new_empty();

        // --- Mandatory DICOM tags ---
        obj.put(DataElement::new(
            Tag(0x0008, 0x0016),
            VR::UI,
            PrimitiveValue::from(sop_class),
        ));
        obj.put(DataElement::new(
            Tag(0x0008, 0x0018),
            VR::UI,
            PrimitiveValue::from(sop_instance_uid.as_str()),
        ));
        obj.put(DataElement::new(
            Tag(0x0008, 0x0060),
            VR::CS,
            PrimitiveValue::from(modality),
        ));
        obj.put(DataElement::new(
            Tag(0x0020, 0x000D),
            VR::UI,
            PrimitiveValue::from(study_uid),
        ));
        obj.put(DataElement::new(
            Tag(0x0020, 0x000E),
            VR::UI,
            PrimitiveValue::from(series_uid),
        ));
        obj.put(DataElement::new(
            Tag(0x0020, 0x0013),
            VR::IS,
            PrimitiveValue::from(format!("{}", z + 1)),
        ));

        // --- Image pixel module ---
        obj.put(DataElement::new(
            Tag(0x0028, 0x0010),
            VR::US,
            PrimitiveValue::from(rows as u16),
        ));
        obj.put(DataElement::new(
            Tag(0x0028, 0x0011),
            VR::US,
            PrimitiveValue::from(cols as u16),
        ));
        obj.put(DataElement::new(
            Tag(0x0028, 0x0100),
            VR::US,
            PrimitiveValue::from(16_u16),
        ));
        obj.put(DataElement::new(
            Tag(0x0028, 0x0101),
            VR::US,
            PrimitiveValue::from(16_u16),
        ));
        obj.put(DataElement::new(
            Tag(0x0028, 0x0102),
            VR::US,
            PrimitiveValue::from(15_u16),
        ));
        obj.put(DataElement::new(
            Tag(0x0028, 0x0103),
            VR::US,
            PrimitiveValue::from(0_u16),
        ));
        obj.put(DataElement::new(
            Tag(0x0028, 0x1053),
            VR::DS,
            PrimitiveValue::from(format!("{:.6}", rescale_slope)),
        ));
        obj.put(DataElement::new(
            Tag(0x0028, 0x1052),
            VR::DS,
            PrimitiveValue::from(format!("{:.6}", rescale_intercept)),
        ));
        obj.put(DataElement::new(
            Tag(0x0028, 0x0004),
            VR::CS,
            PrimitiveValue::from(photometric),
        ));

        // --- Spatial reference tags ---
        if metadata.is_some() {
            // ImagePositionPatient: origin + z * spacing_z * normal
            let ipp_x = origin[0] + (z as f64) * spacing[2] * normal[0];
            let ipp_y = origin[1] + (z as f64) * spacing[2] * normal[1];
            let ipp_z = origin[2] + (z as f64) * spacing[2] * normal[2];
            obj.put(DataElement::new(
                Tag(0x0020, 0x0032),
                VR::DS,
                PrimitiveValue::from(format_triplet([ipp_x, ipp_y, ipp_z])),
            ));
            // ImageOrientationPatient: row and column direction cosines
            obj.put(DataElement::new(
                Tag(0x0020, 0x0037),
                VR::DS,
                PrimitiveValue::from(format_six([
                    direction[0],
                    direction[1],
                    direction[2],
                    direction[3],
                    direction[4],
                    direction[5],
                ])),
            ));
            // PixelSpacing: row spacing \ column spacing
            obj.put(DataElement::new(
                Tag(0x0028, 0x0030),
                VR::DS,
                PrimitiveValue::from(format_pair([spacing[0], spacing[1]])),
            ));
            // SliceThickness
            obj.put(DataElement::new(
                Tag(0x0018, 0x0050),
                VR::DS,
                PrimitiveValue::from(format!("{:.6}", spacing[2])),
            ));
        }

        // --- Optional series-level tags from metadata ---
        if let Some(m) = metadata {
            if let Some(ref uid) = m.frame_of_reference_uid {
                obj.put(DataElement::new(
                    Tag(0x0020, 0x0052),
                    VR::UI,
                    PrimitiveValue::from(uid.as_str()),
                ));
            }
            if let Some(ref pid) = m.patient_id {
                obj.put(DataElement::new(
                    Tag(0x0010, 0x0020),
                    VR::LO,
                    PrimitiveValue::from(pid.as_str()),
                ));
            }
            if let Some(ref pn) = m.patient_name {
                obj.put(DataElement::new(
                    Tag(0x0010, 0x0010),
                    VR::PN,
                    PrimitiveValue::from(pn.as_str()),
                ));
            }
            if let Some(ref sd) = m.study_date {
                obj.put(DataElement::new(
                    Tag(0x0008, 0x0020),
                    VR::DA,
                    PrimitiveValue::from(sd.as_str()),
                ));
            }
            if let Some(ref desc) = m.series_description {
                obj.put(DataElement::new(
                    Tag(0x0008, 0x103E),
                    VR::LO,
                    PrimitiveValue::from(desc.as_str()),
                ));
            }
            if let Some(ref sd) = m.series_date {
                obj.put(DataElement::new(
                    Tag(0x0008, 0x0021),
                    VR::DA,
                    PrimitiveValue::from(sd.as_str()),
                ));
            }
            if let Some(ref st) = m.series_time {
                obj.put(DataElement::new(
                    Tag(0x0008, 0x0031),
                    VR::TM,
                    PrimitiveValue::from(st.as_str()),
                ));
            }
            if let Some(bits) = m.bits_allocated {
                obj.put(DataElement::new(
                    Tag(0x0028, 0x0100),
                    VR::US,
                    PrimitiveValue::from(bits),
                ));
            }
            if let Some(bits) = m.bits_stored {
                obj.put(DataElement::new(
                    Tag(0x0028, 0x0101),
                    VR::US,
                    PrimitiveValue::from(bits),
                ));
            }
            if let Some(bits) = m.high_bit {
                obj.put(DataElement::new(
                    Tag(0x0028, 0x0102),
                    VR::US,
                    PrimitiveValue::from(bits),
                ));
            }
            if let Some(ref uid) = m.private_tags.get("0019,10AA") {
                obj.put(DataElement::new(
                    Tag(0x0019, 0x10AA),
                    VR::LO,
                    PrimitiveValue::from(uid.as_str()),
                ));
            }
        }

        // --- Pixel data ---
        obj.put(DataElement::new(
            Tag(0x7FE0, 0x0010),
            VR::OW,
            PrimitiveValue::U16(SmallVec::from_vec(pixel_u16)),
        ));

        let file_obj = obj
            .with_meta(
                FileMetaTableBuilder::new()
                    .media_storage_sop_class_uid(sop_class)
                    .media_storage_sop_instance_uid(sop_instance_uid.as_str())
                    .transfer_syntax("1.2.840.10008.1.2.1"),
            )
            .map_err(|e| anyhow::anyhow!("DICOM meta failed slice {z}: {e}"))?;
        let slice_path = series_dir.join(format!("slice_{z:04}.dcm"));
        file_obj
            .write_to_file(&slice_path)
            .map_err(|e| anyhow::anyhow!("write slice {z} failed: {e}"))?;
    }
    Ok(())
}

pub struct DicomWriter<B> {
    _phantom: PhantomData<B>,
}
impl<B> DicomWriter<B> {
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
    pub fn series_path<P: AsRef<Path>>(path: P) -> PathBuf {
        path.as_ref().to_path_buf()
    }
}
impl<B> Default for DicomWriter<B> {
    fn default() -> Self {
        Self::new()
    }
}

fn ensure_series_directory(path: &Path) -> Result<PathBuf> {
    if path.exists() {
        if !path.is_dir() {
            bail!("DICOM output path is not a directory");
        }
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
    use dicom::core::Tag;
    use dicom::object::open_file;
    use std::collections::HashMap;

    fn make_image(depth: usize, rows: usize, cols: usize, fill: f32) -> Image<Backend, 3> {
        let device = Default::default();
        let data = vec![fill; depth * rows * cols];
        let tensor = Tensor::<Backend, 3>::from_data(
            TensorData::new(data, Shape::new([depth, rows, cols])),
            &device,
        );
        Image::new(
            tensor,
            Point::new([0.0, 0.0, 0.0]),
            Spacing::new([1.0, 1.0, 1.0]),
            Direction::identity(),
        )
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
        let count = std::fs::read_dir(&path)
            .unwrap()
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
        assert_eq!(
            &bytes[128..132],
            b"DICM",
            "DICOM magic bytes must be present at offset 128"
        );
    }

    fn make_image_with_spatial(
        depth: usize,
        rows: usize,
        cols: usize,
        fill: f32,
        origin: [f64; 3],
        spacing: [f64; 3],
    ) -> Image<Backend, 3> {
        let device = Default::default();
        let data = vec![fill; depth * rows * cols];
        let tensor = Tensor::<Backend, 3>::from_data(
            TensorData::new(data, Shape::new([depth, rows, cols])),
            &device,
        );
        Image::new(
            tensor,
            Point::new(origin),
            Spacing::new(spacing),
            Direction::identity(),
        )
    }

    fn make_test_metadata() -> DicomReadMetadata {
        let mut private_tags = HashMap::new();
        private_tags.insert("0019,10AA".to_string(), "PRIVATE_SERIES_VALUE".to_string());

        DicomReadMetadata {
            series_instance_uid: Some("1.2.3.4.5.6.789".to_string()),
            study_instance_uid: Some("1.2.3.4.5.6.100".to_string()),
            frame_of_reference_uid: Some("1.2.3.4.5.6.200".to_string()),
            series_description: Some("Test Series".to_string()),
            modality: Some("CT".to_string()),
            patient_id: Some("PAT001".to_string()),
            patient_name: Some("Test^Patient".to_string()),
            study_date: Some("20240101".to_string()),
            series_date: Some("20240102".to_string()),
            series_time: Some("123456".to_string()),
            dimensions: [4, 4, 3],
            spacing: [0.5, 0.5, 2.5],
            origin: [10.0, 20.0, 30.0],
            direction: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            bits_allocated: Some(16),
            bits_stored: Some(16),
            high_bit: Some(15),
            photometric_interpretation: Some("MONOCHROME2".to_string()),
            slices: Vec::new(),
            private_tags,
        }
    }

    #[test]
    fn test_metadata_writer_spatial_tags_first_slice() {
        let meta = make_test_metadata();
        let image = make_image_with_spatial(3, 4, 4, 50.0, meta.origin, meta.spacing);
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("meta_series");
        write_dicom_series_with_metadata(&path, &image, Some(&meta))
            .expect("metadata write must succeed");

        let dcm_path = path.join("slice_0000.dcm");
        let obj = open_file(&dcm_path).expect("must open written DICOM");

        // ImagePositionPatient -- first slice: origin itself
        let ipp = obj
            .element(Tag(0x0020, 0x0032))
            .expect("IPP tag must exist");
        let ipp_str = ipp.to_str().unwrap();
        let ipp_vals: Vec<f64> = ipp_str
            .split('\\')
            .map(|s| s.trim().parse().unwrap())
            .collect();
        assert_eq!(ipp_vals.len(), 3);
        assert!((ipp_vals[0] - 10.0).abs() < 1e-3, "IPP x={}", ipp_vals[0]);
        assert!((ipp_vals[1] - 20.0).abs() < 1e-3, "IPP y={}", ipp_vals[1]);
        assert!((ipp_vals[2] - 30.0).abs() < 1e-3, "IPP z={}", ipp_vals[2]);

        // ImageOrientationPatient
        let iop = obj
            .element(Tag(0x0020, 0x0037))
            .expect("IOP tag must exist");
        let iop_str = iop.to_str().unwrap();
        let iop_vals: Vec<f64> = iop_str
            .split('\\')
            .map(|s| s.trim().parse().unwrap())
            .collect();
        assert_eq!(iop_vals.len(), 6);
        assert!((iop_vals[0] - 1.0).abs() < 1e-6, "IOP[0]");
        assert!((iop_vals[4] - 1.0).abs() < 1e-6, "IOP[4]");

        // PixelSpacing
        let ps = obj
            .element(Tag(0x0028, 0x0030))
            .expect("PixelSpacing must exist");
        let ps_str = ps.to_str().unwrap();
        let ps_vals: Vec<f64> = ps_str
            .split('\\')
            .map(|s| s.trim().parse().unwrap())
            .collect();
        assert_eq!(ps_vals.len(), 2);
        assert!((ps_vals[0] - 0.5).abs() < 1e-6, "PS row={}", ps_vals[0]);
        assert!((ps_vals[1] - 0.5).abs() < 1e-6, "PS col={}", ps_vals[1]);

        // SliceThickness
        let st = obj
            .element(Tag(0x0018, 0x0050))
            .expect("SliceThickness must exist");
        let st_val: f64 = st.to_str().unwrap().trim().parse().unwrap();
        assert!((st_val - 2.5).abs() < 1e-6, "SliceThickness={}", st_val);

        // Modality
        let mod_elem = obj
            .element(Tag(0x0008, 0x0060))
            .expect("Modality must exist");
        assert_eq!(mod_elem.to_str().unwrap().trim(), "CT");

        // PatientID
        let pid = obj
            .element(Tag(0x0010, 0x0020))
            .expect("PatientID must exist");
        assert_eq!(pid.to_str().unwrap().trim(), "PAT001");

        // SeriesDate / SeriesTime
        let sd = obj
            .element(Tag(0x0008, 0x0021))
            .expect("SeriesDate must exist");
        assert_eq!(sd.to_str().unwrap().trim(), "20240102");
        let st = obj
            .element(Tag(0x0008, 0x0031))
            .expect("SeriesTime must exist");
        assert_eq!(st.to_str().unwrap().trim(), "123456");

        // FrameOfReferenceUID
        let for_uid = obj
            .element(Tag(0x0020, 0x0052))
            .expect("FrameOfReferenceUID must exist");
        assert_eq!(for_uid.to_str().unwrap().trim(), "1.2.3.4.5.6.200");

        // Private tag preservation
        let private = obj
            .element(Tag(0x0019, 0x10AA))
            .expect("private tag must exist");
        assert_eq!(private.to_str().unwrap().trim(), "PRIVATE_SERIES_VALUE");
    }

    #[test]
    fn test_metadata_writer_multislice_ipp_increment() {
        let meta = make_test_metadata();
        let image = make_image_with_spatial(3, 4, 4, 75.0, meta.origin, meta.spacing);
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("multi_series");
        write_dicom_series_with_metadata(&path, &image, Some(&meta))
            .expect("metadata write must succeed");

        // Verify z-positions: origin_z + z_index * spacing_z * normal_z
        // With identity direction, normal = [0,0,1], spacing_z = 2.5, origin_z = 30.0
        let expected_z = [30.0, 32.5, 35.0];
        for (z_idx, &ez) in expected_z.iter().enumerate() {
            let dcm_path = path.join(format!("slice_{z_idx:04}.dcm"));
            let obj = open_file(&dcm_path).unwrap_or_else(|_| panic!("must open slice {z_idx}"));
            let ipp = obj.element(Tag(0x0020, 0x0032)).expect("IPP must exist");
            let ipp_str = ipp.to_str().unwrap();
            let ipp_vals: Vec<f64> = ipp_str
                .split('\\')
                .map(|s| s.trim().parse().unwrap())
                .collect();
            assert!(
                (ipp_vals[2] - ez).abs() < 1e-3,
                "slice {z_idx}: expected z={ez}, got z={}",
                ipp_vals[2]
            );
        }
    }

    #[test]
    fn test_metadata_writer_none_metadata_fallback() {
        let image = make_image(2, 4, 4, 25.0);
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("no_meta_series");
        write_dicom_series_with_metadata(&path, &image, None)
            .expect("write with None metadata must succeed");

        let dcm_path = path.join("slice_0000.dcm");
        let obj = open_file(&dcm_path).expect("must open written DICOM");

        // Without metadata, spatial tags should NOT be present.
        assert!(
            obj.element(Tag(0x0020, 0x0032)).is_err(),
            "IPP should not exist when metadata is None"
        );
        assert!(
            obj.element(Tag(0x0020, 0x0037)).is_err(),
            "IOP should not exist when metadata is None"
        );

        // Modality defaults to OT
        let mod_elem = obj
            .element(Tag(0x0008, 0x0060))
            .expect("Modality must exist");
        assert_eq!(mod_elem.to_str().unwrap().trim(), "OT");

        // Private tags are not emitted when metadata is absent.
        assert!(
            obj.element(Tag(0x0019, 0x10AA)).is_err(),
            "private tag should not exist when metadata is None"
        );
    }

    #[test]
    fn test_metadata_writer_rejects_zero_dimension() {
        let meta = make_test_metadata();
        let image = make_image(0, 4, 4, 0.0);
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("zero_series");
        let result = write_dicom_series_with_metadata(&path, &image, Some(&meta));
        assert!(result.is_err(), "zero depth must be rejected");
    }
}
