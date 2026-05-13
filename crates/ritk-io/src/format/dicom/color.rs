//! DICOM color volume loading.
//!
//! This module is the color-volume counterpart to the scalar DICOM loaders.
//! It preserves RGB samples in a typed `ColorVolume<B, 3>` instead of forcing
//! multi-sample frames through scalar `Image<B, 3>`.

use std::path::Path;

use anyhow::{bail, Context, Result};
use burn::tensor::backend::Backend;
use burn::tensor::{Shape, Tensor, TensorData};
use dicom::core::Tag;
use nalgebra::SMatrix;
use ritk_core::image::RgbVolume;
use ritk_core::spatial::{Direction, Point, Spacing};
use ritk_dicom::{
    decode_frame_with, parse_file_with, DecodeFrameRequest, DicomRsBackend, PixelLayout,
    TransferSyntaxKind,
};

use super::reader::{self, DicomReadMetadata, DicomSliceMetadata};

const RGB_CHANNELS: usize = 3;

/// Read a DICOM RGB series into a rank-4 color volume.
///
/// The returned tensor shape is `[depth, rows, cols, 3]` with interleaved RGB
/// samples in the channel axis. Only byte-addressable unsigned RGB data is
/// accepted; scalar, palette, YBR, CMYK, and signed color data are rejected.
pub fn read_dicom_color_series<B: Backend, P: AsRef<Path>>(
    path: P,
    device: &B::Device,
) -> Result<(RgbVolume<B>, DicomReadMetadata)> {
    let series = reader::scan_dicom_directory(path)?;
    load_color_from_series(series.metadata, device)
}

/// Alias matching the scalar loader naming convention.
pub fn load_dicom_color_series<B: Backend, P: AsRef<Path>>(
    path: P,
    device: &B::Device,
) -> Result<(RgbVolume<B>, DicomReadMetadata)> {
    read_dicom_color_series(path, device)
}

fn load_color_from_series<B: Backend>(
    mut metadata: DicomReadMetadata,
    device: &B::Device,
) -> Result<(RgbVolume<B>, DicomReadMetadata)> {
    let slices = metadata.slices.clone();
    if slices.is_empty() {
        bail!("DICOM color series is empty");
    }

    let rows = metadata.dimensions[0];
    let cols = metadata.dimensions[1];
    let depth = metadata.dimensions[2];
    if rows == 0 || cols == 0 || depth == 0 {
        bail!("DICOM color series has invalid zero dimensions");
    }
    if slices.len() != depth {
        bail!(
            "DICOM color series slice count {} does not match metadata depth {}",
            slices.len(),
            depth
        );
    }

    let frame_samples = rows
        .checked_mul(cols)
        .and_then(|n| n.checked_mul(RGB_CHANNELS))
        .context("DICOM color frame sample count overflow")?;
    let total_samples = frame_samples
        .checked_mul(depth)
        .context("DICOM color volume sample count overflow")?;
    let mut volume = vec![0.0_f32; total_samples];

    for (z, slice) in slices.iter().enumerate() {
        let frame = read_rgb_slice_samples(slice, rows, cols)
            .with_context(|| format!("failed to decode DICOM RGB slice {:?}", slice.path))?;
        if frame.len() != frame_samples {
            bail!(
                "DICOM RGB slice {:?} returned {} samples; expected {}",
                slice.path,
                frame.len(),
                frame_samples
            );
        }
        let start = z * frame_samples;
        volume[start..start + frame_samples].copy_from_slice(&frame);
    }

    metadata.dimensions = [rows, cols, depth];
    let tensor = Tensor::<B, 4>::from_data(
        TensorData::new(volume, Shape::new([depth, rows, cols, RGB_CHANNELS])),
        device,
    );
    let image = RgbVolume::try_new(
        tensor,
        Point::new(metadata.origin),
        Spacing::new(metadata.spacing),
        Direction(SMatrix::<f64, 3, 3>::from_column_slice(&metadata.direction)),
    )?;
    Ok((image, metadata))
}

fn read_rgb_slice_samples(
    slice: &DicomSliceMetadata,
    expected_rows: usize,
    expected_cols: usize,
) -> Result<Vec<f32>> {
    let obj = parse_file_with::<DicomRsBackend, _>(&slice.path)
        .with_context(|| format!("failed to open DICOM slice {:?}", slice.path))?;

    let transfer_syntax = obj.meta().transfer_syntax();
    let ts = TransferSyntaxKind::from_uid(transfer_syntax);
    if ts.is_compressed() && !ts.is_codec_supported() {
        bail!(
            "DICOM RGB series: compressed transfer syntax '{}' in slice {:?} is not supported",
            transfer_syntax,
            slice.path
        );
    }
    if ts.is_big_endian() {
        bail!(
            "DICOM RGB series: big-endian transfer syntax '{}' in slice {:?} is not supported",
            transfer_syntax,
            slice.path
        );
    }

    let rows = required_usize(&obj, Tag(0x0028, 0x0010), "Rows")?;
    let cols = required_usize(&obj, Tag(0x0028, 0x0011), "Columns")?;
    if rows != expected_rows || cols != expected_cols {
        bail!(
            "DICOM RGB slice {:?} dimensions {}x{} do not match series {}x{}",
            slice.path,
            rows,
            cols,
            expected_rows,
            expected_cols
        );
    }

    let samples_per_pixel = optional_usize(&obj, Tag(0x0028, 0x0002)).unwrap_or(1);
    if samples_per_pixel != RGB_CHANNELS {
        bail!(
            "DICOM color volume loader supports only RGB SamplesPerPixel=3; {:?} declares SamplesPerPixel={}",
            slice.path,
            samples_per_pixel
        );
    }

    let photometric = required_string(&obj, Tag(0x0028, 0x0004), "PhotometricInterpretation")?;
    if !photometric.trim().eq_ignore_ascii_case("RGB") {
        bail!(
            "DICOM color volume loader supports only PhotometricInterpretation=RGB; {:?} declares {}",
            slice.path,
            photometric.trim()
        );
    }
    let planar_configuration = optional_u16(&obj, Tag(0x0028, 0x0006)).unwrap_or(0);
    if planar_configuration != 0 {
        bail!(
            "DICOM RGB color volume loader supports only interleaved PlanarConfiguration=0; {:?} declares {}",
            slice.path,
            planar_configuration
        );
    }

    let bits_allocated = optional_u16(&obj, Tag(0x0028, 0x0100)).unwrap_or(slice.bits_allocated);
    if bits_allocated != 8 {
        bail!(
            "DICOM RGB color volume loader supports only BitsAllocated=8; {:?} declares {}",
            slice.path,
            bits_allocated
        );
    }
    let pixel_representation =
        optional_u16(&obj, Tag(0x0028, 0x0103)).unwrap_or(slice.pixel_representation);
    if pixel_representation != 0 {
        bail!(
            "DICOM RGB color volume loader supports only unsigned samples; {:?} declares PixelRepresentation={}",
            slice.path,
            pixel_representation
        );
    }

    let decoded = decode_frame_with::<DicomRsBackend>(
        &obj,
        DecodeFrameRequest {
            frame_index: 0,
            transfer_syntax: ts,
            layout: PixelLayout {
                rows,
                cols,
                samples_per_pixel,
                bits_allocated,
                pixel_representation,
                rescale_slope: 1.0,
                rescale_intercept: 0.0,
            },
        },
    )
    .with_context(|| format!("DICOM backend decode failed for RGB slice {:?}", slice.path))?;

    Ok(decoded.pixels)
}

fn required_usize(obj: &dicom::object::DefaultDicomObject, tag: Tag, name: &str) -> Result<usize> {
    obj.element(tag)
        .with_context(|| format!("{name} absent"))?
        .to_str()
        .with_context(|| format!("{name} unreadable"))?
        .trim()
        .parse::<usize>()
        .with_context(|| format!("{name} invalid"))
}

fn optional_usize(obj: &dicom::object::DefaultDicomObject, tag: Tag) -> Option<usize> {
    obj.element(tag)
        .ok()
        .and_then(|e| e.to_str().ok())
        .and_then(|s| s.trim().parse().ok())
}

fn optional_u16(obj: &dicom::object::DefaultDicomObject, tag: Tag) -> Option<u16> {
    obj.element(tag)
        .ok()
        .and_then(|e| e.to_str().ok())
        .and_then(|s| s.trim().parse().ok())
}

fn required_string(
    obj: &dicom::object::DefaultDicomObject,
    tag: Tag,
    name: &str,
) -> Result<String> {
    Ok(obj
        .element(tag)
        .with_context(|| format!("{name} absent"))?
        .to_str()
        .with_context(|| format!("{name} unreadable"))?
        .to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::backend::Backend;
    use burn_ndarray::NdArray;
    use dicom::core::smallvec::SmallVec;
    use dicom::core::{DataElement, PrimitiveValue, VR};
    use dicom::object::meta::FileMetaTableBuilder;
    use dicom::object::InMemDicomObject;

    type B = NdArray<f32>;

    fn write_rgb_slice(
        path: &Path,
        sop_instance_uid: &str,
        instance_number: u16,
        z_mm: f64,
        samples: &[u8],
        planar_configuration: Option<u16>,
    ) {
        assert_eq!(samples.len(), 2 * RGB_CHANNELS);
        let mut obj = InMemDicomObject::new_empty();
        obj.put(DataElement::new(
            Tag(0x0008, 0x0016),
            VR::UI,
            PrimitiveValue::from("1.2.840.10008.5.1.4.1.1.7"),
        ));
        obj.put(DataElement::new(
            Tag(0x0008, 0x0018),
            VR::UI,
            PrimitiveValue::from(sop_instance_uid),
        ));
        obj.put(DataElement::new(
            Tag(0x0008, 0x0060),
            VR::CS,
            PrimitiveValue::from("OT"),
        ));
        obj.put(DataElement::new(
            Tag(0x0020, 0x000D),
            VR::UI,
            PrimitiveValue::from("2.25.3000"),
        ));
        obj.put(DataElement::new(
            Tag(0x0020, 0x000E),
            VR::UI,
            PrimitiveValue::from("2.25.3001"),
        ));
        obj.put(DataElement::new(
            Tag(0x0020, 0x0013),
            VR::IS,
            PrimitiveValue::from(instance_number.to_string()),
        ));
        obj.put(DataElement::new(
            Tag(0x0020, 0x0032),
            VR::DS,
            PrimitiveValue::from(format!("0\\0\\{z_mm}")),
        ));
        obj.put(DataElement::new(
            Tag(0x0020, 0x0037),
            VR::DS,
            PrimitiveValue::from("1\\0\\0\\0\\1\\0"),
        ));
        obj.put(DataElement::new(
            Tag(0x0028, 0x0002),
            VR::US,
            PrimitiveValue::from(3_u16),
        ));
        obj.put(DataElement::new(
            Tag(0x0028, 0x0004),
            VR::CS,
            PrimitiveValue::from("RGB"),
        ));
        if let Some(planar_configuration) = planar_configuration {
            obj.put(DataElement::new(
                Tag(0x0028, 0x0006),
                VR::US,
                PrimitiveValue::from(planar_configuration),
            ));
        }
        obj.put(DataElement::new(
            Tag(0x0028, 0x0010),
            VR::US,
            PrimitiveValue::from(1_u16),
        ));
        obj.put(DataElement::new(
            Tag(0x0028, 0x0011),
            VR::US,
            PrimitiveValue::from(2_u16),
        ));
        obj.put(DataElement::new(
            Tag(0x0028, 0x0030),
            VR::DS,
            PrimitiveValue::from("0.5\\0.25"),
        ));
        obj.put(DataElement::new(
            Tag(0x0028, 0x0100),
            VR::US,
            PrimitiveValue::from(8_u16),
        ));
        obj.put(DataElement::new(
            Tag(0x0028, 0x0103),
            VR::US,
            PrimitiveValue::from(0_u16),
        ));
        obj.put(DataElement::new(
            Tag(0x7FE0, 0x0010),
            VR::OB,
            PrimitiveValue::U8(SmallVec::from_vec(samples.to_vec())),
        ));

        obj.with_meta(
            FileMetaTableBuilder::new()
                .media_storage_sop_class_uid("1.2.840.10008.5.1.4.1.1.7")
                .media_storage_sop_instance_uid(sop_instance_uid)
                .transfer_syntax("1.2.840.10008.1.2.1"),
        )
        .expect("file meta must be valid")
        .write_to_file(path)
        .expect("DICOM RGB slice must be written");
    }

    #[test]
    fn read_dicom_color_series_preserves_interleaved_rgb_samples() {
        let dir = tempfile::tempdir().expect("tempdir");
        write_rgb_slice(
            &dir.path().join("slice1.dcm"),
            "2.25.3101",
            1,
            0.0,
            &[255, 0, 0, 0, 255, 0],
            Some(0),
        );
        write_rgb_slice(
            &dir.path().join("slice2.dcm"),
            "2.25.3102",
            2,
            2.0,
            &[0, 0, 255, 255, 255, 255],
            Some(0),
        );
        let device = <B as Backend>::Device::default();

        let (volume, metadata) =
            read_dicom_color_series::<B, _>(dir.path(), &device).expect("RGB load must succeed");

        assert_eq!(volume.shape(), [2, 1, 2, 3]);
        assert_eq!(volume.spatial_shape(), [2, 1, 2]);
        assert_eq!(metadata.dimensions, [1, 2, 2]);
        assert_eq!(metadata.photometric_interpretation.as_deref(), Some("RGB"));
        let data = volume.data().clone().to_data();
        let samples = data.as_slice::<f32>().expect("f32 tensor data");
        assert_eq!(
            samples,
            &[255.0, 0.0, 0.0, 0.0, 255.0, 0.0, 0.0, 0.0, 255.0, 255.0, 255.0, 255.0]
        );
    }

    #[test]
    fn read_dicom_color_series_rejects_scalar_samples() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("mono.dcm");
        let mut obj = InMemDicomObject::new_empty();
        obj.put(DataElement::new(
            Tag(0x0008, 0x0016),
            VR::UI,
            PrimitiveValue::from("1.2.840.10008.5.1.4.1.1.7"),
        ));
        obj.put(DataElement::new(
            Tag(0x0008, 0x0018),
            VR::UI,
            PrimitiveValue::from("2.25.3201"),
        ));
        obj.put(DataElement::new(
            Tag(0x0020, 0x000E),
            VR::UI,
            PrimitiveValue::from("2.25.3202"),
        ));
        obj.put(DataElement::new(
            Tag(0x0028, 0x0002),
            VR::US,
            PrimitiveValue::from(1_u16),
        ));
        obj.put(DataElement::new(
            Tag(0x0028, 0x0004),
            VR::CS,
            PrimitiveValue::from("MONOCHROME2"),
        ));
        obj.put(DataElement::new(
            Tag(0x0028, 0x0010),
            VR::US,
            PrimitiveValue::from(1_u16),
        ));
        obj.put(DataElement::new(
            Tag(0x0028, 0x0011),
            VR::US,
            PrimitiveValue::from(2_u16),
        ));
        obj.put(DataElement::new(
            Tag(0x0028, 0x0100),
            VR::US,
            PrimitiveValue::from(8_u16),
        ));
        obj.put(DataElement::new(
            Tag(0x0028, 0x0103),
            VR::US,
            PrimitiveValue::from(0_u16),
        ));
        obj.put(DataElement::new(
            Tag(0x7FE0, 0x0010),
            VR::OB,
            PrimitiveValue::U8(SmallVec::from_vec(vec![1, 2])),
        ));
        obj.with_meta(
            FileMetaTableBuilder::new()
                .media_storage_sop_class_uid("1.2.840.10008.5.1.4.1.1.7")
                .media_storage_sop_instance_uid("2.25.3201")
                .transfer_syntax("1.2.840.10008.1.2.1"),
        )
        .expect("file meta must be valid")
        .write_to_file(&path)
        .expect("scalar DICOM must be written");
        let device = <B as Backend>::Device::default();

        let err = read_dicom_color_series::<B, _>(dir.path(), &device).unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.contains("SamplesPerPixel=1"),
            "expected RGB sample-count rejection, got {msg}"
        );
    }

    #[test]
    fn read_dicom_color_series_rejects_planar_rgb_samples() {
        let dir = tempfile::tempdir().expect("tempdir");
        write_rgb_slice(
            &dir.path().join("planar.dcm"),
            "2.25.3151",
            1,
            0.0,
            &[255, 0, 0, 0, 255, 0],
            Some(1),
        );
        let device = <B as Backend>::Device::default();

        let err = read_dicom_color_series::<B, _>(dir.path(), &device).unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.contains("PlanarConfiguration=0") && msg.contains("declares 1"),
            "expected planar RGB rejection, got {msg}"
        );
    }
}
