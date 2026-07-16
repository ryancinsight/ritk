//! Series loading — `load_dicom_series`, `read_dicom_series`, and the `DicomReader` facade.

use anyhow::{bail, Context, Result};
use coeus_core::ComputeBackend;
use dicom::dictionary_std::tags;
use dicom::object::{FileDicomObject, InMemDicomObject};
use ritk_core::image::Image;
use ritk_dicom::{
    decode_frame_with, parse_file_with, DecodeFrameRequest, DicomRsBackend, PixelLayout,
    PixelSignedness,
};
use ritk_image::native::Image as NativeImage;
use ritk_image::tensor::backend::Backend;

use ritk_image::tensor::{Shape, TensorData, Tensor};
use ritk_spatial::{Direction, Point, Spacing, Vector};
use std::path::{Path, PathBuf};

use crate::format::dicom::transfer_syntax::TransferSyntaxKind;

use super::scan::scan_dicom_directory;
use super::types::DicomSeriesInfo;

/// Load a specific DICOM series into a 3D Image.
///
/// Performs rigorous checks for spatial consistency (uniform spacing, orientation).
pub fn load_dicom_series<B: Backend>(
    series: &DicomSeriesInfo,
    device: &B::Device,
) -> Result<Image<B, 3>> {
    let decoded = decode_series(series)?;
    let tensor = Tensor::<B, 3>::from_data(
        TensorData::new(decoded.voxels, Shape::new(decoded.shape)),
        device,
    );

    Ok(Image::new(
        tensor,
        decoded.origin,
        decoded.spacing,
        decoded.direction,
    ))
}

/// Load a specific DICOM series into a native Coeus-backed 3D Image.
///
/// Performs the same spatial consistency checks and pixel decode as
/// [`load_dicom_series`], but constructs the image on the Atlas native tensor
/// substrate instead of a Burn tensor.
pub fn load_native_dicom_series<B: ComputeBackend>(
    series: &DicomSeriesInfo,
    backend: &B,
) -> Result<NativeImage<f32, B, 3>> {
    let decoded = decode_series(series)?;
    NativeImage::from_flat_on(
        decoded.voxels,
        decoded.shape,
        decoded.origin,
        decoded.spacing,
        decoded.direction,
        backend,
    )
}

struct DecodedDicomSeries {
    voxels: Vec<f32>,
    shape: [usize; 3],
    origin: Point<3>,
    spacing: Spacing<3>,
    direction: Direction<3>,
}

fn decode_series(series: &DicomSeriesInfo) -> Result<DecodedDicomSeries> {
    if series.file_paths.is_empty() {
        bail!("Series {} has no files", series.series_instance_uid);
    }

    // 1. Read all headers to sort spatially
    let mut slices: Vec<(PathBuf, FileDicomObject<InMemDicomObject>)> =
        moirai::map_collect_index_with::<moirai::Adaptive, _, _>(series.file_paths.len(), |i| {
            let p = &series.file_paths[i];
            let obj =
                parse_file_with::<DicomRsBackend, _>(p).context("Failed to open DICOM file")?;
            Ok((p.clone(), obj))
        })
        .into_iter()
        .collect::<Result<Vec<_>>>()?;

    // 2. Determine Orientation from the first slice
    let first_obj = &slices[0].1;
    let orientation = get_scalar_vec(first_obj, tags::IMAGE_ORIENTATION_PATIENT)
        .context("Missing ImageOrientationPatient in first slice")?;

    if orientation.len() != 6 {
        bail!(
            "Invalid ImageOrientationPatient length: {}",
            orientation.len()
        );
    }

    let dir_x = Vector::new([orientation[0], orientation[1], orientation[2]]);
    let dir_y = Vector::new([orientation[3], orientation[4], orientation[5]]);

    // Normalize to ensure valid direction cosines
    let dir_x = dir_x
        .normalized()
        .context("Invalid zero-length ImageOrientationPatient row direction")?;
    let dir_y = dir_y
        .normalized()
        .context("Invalid zero-length ImageOrientationPatient column direction")?;

    let dir_z = dir_x
        .cross(&dir_y)
        .normalized()
        .context("Invalid parallel ImageOrientationPatient directions")?;

    // 3. Sort slices by projection onto normal vector
    slices.sort_by(|a, b| {
        let pos_a = get_position(&a.1).unwrap_or(Point::origin());
        let pos_b = get_position(&b.1).unwrap_or(Point::origin());

        let dist_a = Vector::new(pos_a.to_array()).dot(&dir_z);
        let dist_b = Vector::new(pos_b.to_array()).dot(&dir_z);

        dist_a
            .partial_cmp(&dist_b)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // 4. Validate Spatial Consistency & Calculate Spacing
    let first_obj = &slices[0].1;
    let rows = element_as_u32(first_obj, tags::ROWS).context("Missing Rows")?;
    let cols = element_as_u32(first_obj, tags::COLUMNS).context("Missing Columns")?;
    let pixel_spacing =
        get_scalar_vec(first_obj, tags::PIXEL_SPACING).context("Missing PixelSpacing")?;
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
            let current_orient = get_scalar_vec(&slices[i + 1].1, tags::IMAGE_ORIENTATION_PATIENT)
                .unwrap_or_default();
            if current_orient.len() == 6 {
                let cx = Vector::new([current_orient[0], current_orient[1], current_orient[2]])
                    .normalized()
                    .context("Invalid zero-length ImageOrientationPatient row direction")?;
                let cy = Vector::new([current_orient[3], current_orient[4], current_orient[5]])
                    .normalized()
                    .context("Invalid zero-length ImageOrientationPatient column direction")?;
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
        get_scalar(first_obj, tags::SLICE_THICKNESS).unwrap_or(1.0)
    };

    // 5. Build Spatial Metadata
    let spacing = Spacing::new([dx, dy, dz]);
    let origin = Point::new(origin_pos.to_array());
    let direction = Direction::from_columns([dir_x, dir_y, dir_z]);

    // 6. Load Pixel Data in Parallel
    let slice_pixels: Vec<Vec<f32>> =
        moirai::map_collect_index_with::<moirai::Adaptive, _, _>(slices.len(), |i| {
            let obj = &slices[i].1;
            let slope = get_scalar(obj, tags::RESCALE_SLOPE).unwrap_or(1.0);
            let intercept = get_scalar(obj, tags::RESCALE_INTERCEPT).unwrap_or(0.0);
            let samples_per_pixel =
                element_as_u32(obj, tags::SAMPLES_PER_PIXEL).unwrap_or(1) as usize;
            let bits_allocated = element_as_u32(obj, tags::BITS_ALLOCATED).unwrap_or(16) as u16;
            let pixel_representation: PixelSignedness = PixelSignedness::try_from(
                element_as_u32(obj, tags::PIXEL_REPRESENTATION).unwrap_or(0) as u16,
            )
            .unwrap_or(PixelSignedness::Unsigned);
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

    Ok(DecodedDicomSeries {
        voxels: flattened,
        shape: [depth, rows as usize, cols as usize],
        origin,
        spacing,
        direction,
    })
}

/// Convenience function to read a single series from a directory.
/// If multiple series exist, it errors out to avoid ambiguity.
pub fn read_dicom_series<B: Backend, P: AsRef<Path>>(
    path: P,
    device: &B::Device,
) -> Result<Image<B, 3>> {
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

/// Convenience function to read a single series into a native Coeus-backed image.
///
/// If multiple series exist, it errors out to avoid ambiguity.
pub fn read_native_dicom_series<B: ComputeBackend, P: AsRef<Path>>(
    path: P,
    backend: &B,
) -> Result<NativeImage<f32, B, 3>> {
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

    load_native_dicom_series(&series_list[0], backend)
}

// --- Helpers ---

fn element_as_u32(obj: &FileDicomObject<InMemDicomObject>, tag: dicom::core::Tag) -> Option<u32> {
    obj.element(tag).ok()?.to_int::<u32>().ok()
}

fn get_scalar(obj: &FileDicomObject<InMemDicomObject>, tag: dicom::core::Tag) -> Option<f64> {
    obj.element(tag).ok()?.to_float64().ok()
}

fn get_scalar_vec(
    obj: &FileDicomObject<InMemDicomObject>,
    tag: dicom::core::Tag,
) -> Option<Vec<f64>> {
    obj.element(tag).ok()?.to_multi_float64().ok()
}

fn get_position(obj: &FileDicomObject<InMemDicomObject>) -> Option<Point<3>> {
    let v = get_scalar_vec(obj, tags::IMAGE_POSITION_PATIENT)?;
    if v.len() == 3 {
        Some(Point::new([v[0], v[1], v[2]]))
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

impl<B: Backend> ImageReader<Image<B, 3>> for DicomReader<B> {
    fn read<P: AsRef<Path>>(&self, path: P) -> std::io::Result<Image<B, 3>> {
        read_dicom_series(path, &self.device).map_err(|e| std::io::Error::other(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::{load_dicom_series, load_native_dicom_series};
    use coeus_core::SequentialBackend;
    use ritk_core::image::Image;
    
use ritk_image::tensor::{Shape, TensorData, Tensor};
    use ritk_spatial::{Direction, Point, Spacing};
    use std::collections::HashMap;

    #[test]
    fn native_series_loader_matches_legacy_loader() {
        type B = burn_ndarray::SequentialBackend;

        let dir = tempfile::tempdir().expect("tempdir");
        let series_path = dir.path().join("series_native_parity");

        let (depth, rows, cols) = (3usize, 3usize, 4usize);
        let values: Vec<f32> = (0..(depth * rows * cols))
            .map(|i| i as f32 * 0.25 + 2.0)
            .collect();
        let device = <B as ritk_image::tensor::backend::Backend>::Device::default();
        let tensor = Tensor::<B, 3>::from_data(
            (values, ([depth, rows, cols])),
            &backend,
        );
        let image = Image::<B, 3>::new(
            tensor,
            Point::new([1.0, 2.0, 3.0]),
            Spacing::new([1.5, 0.75, 0.5]),
            Direction::identity(),
        );

        let meta = crate::format::dicom::DicomReadMetadata {
            series_instance_uid: Some("2.25.71001".try_into().unwrap()),
            study_instance_uid: Some("2.25.71002".try_into().unwrap()),
            frame_of_reference_uid: None,
            series_description: None,
            modality: Some("CT".try_into().unwrap()),
            patient_id: None,
            patient_name: None,
            study_date: None,
            series_date: None,
            series_time: None,
            dimensions: [rows, cols, depth],
            spacing: [1.5, 0.75, 0.5],
            origin: [1.0, 2.0, 3.0],
            direction: [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
            bits_allocated: Some(16),
            bits_stored: Some(16),
            high_bit: Some(15),
            photometric_interpretation: Some("MONOCHROME2".try_into().unwrap()),
            slices: Vec::new(),
            private_tags: HashMap::new(),
            preservation: crate::format::dicom::DicomPreservationSet::new(),
            patient_weight_kg: None,
            decay_correction: None,
            radionuclide_total_dose_bq: None,
            radiopharmaceutical_start_time: None,
            radionuclide_half_life_s: None,
        };
        crate::format::dicom::writer::write_dicom_series_with_metadata(
            &series_path,
            &image,
            Some(&meta),
        )
        .expect("write_dicom_series_with_metadata");
        let series = crate::format::dicom::scan_dicom_directory(&series_path)
            .expect("scan series")
            .pop()
            .expect("one series");

        let legacy = load_dicom_series::<B>(&series, &backend).expect("legacy load");
        let native =
            load_native_dicom_series(&series, &SequentialBackend).expect("native series load");

        assert_eq!(native.shape(), legacy.shape());
        legacy.with_data_slice(|legacy_values: &[f32]| {
            assert_eq!(
                native.data_slice().expect("native contiguous data"),
                legacy_values,
                "native series facade must use the same decoded voxels"
            );
        });
        assert_eq!(native.origin().to_array(), legacy.origin().to_array());
        assert_eq!(native.spacing().to_array(), legacy.spacing().to_array());
        for row in 0..3 {
            for col in 0..3 {
                assert_eq!(
                    native.direction()[(row, col)],
                    legacy.direction()[(row, col)],
                    "direction[{row},{col}]"
                );
            }
        }
    }
}
