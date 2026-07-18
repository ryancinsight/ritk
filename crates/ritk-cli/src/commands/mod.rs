//! Shared command infrastructure for the RITK CLI.
//!
//! Declares the five subcommand modules and provides the shared IO helpers
//! (`infer_format`, `read_image`, `write_image`, `write_image_inferred`) and
//! the concrete `Backend` type alias used throughout all command handlers.

pub mod convert;
pub mod filter;
pub mod normalize;
pub mod register;
pub mod resample;
pub mod segment;
pub mod stats;
pub mod viewer;

use anyhow::{anyhow, bail, Context, Result};
use burn_ndarray::NdArray;
use coeus_core::SequentialBackend;
use ritk_image::native::Image as NativeImage;
use ritk_image::tensor::Backend as BurnBackend;
use ritk_image::Image;
use ritk_io::{is_rgb_dicom_series, ImageFormat};
use ritk_io::{ImageReader, ImageWriter};
use std::path::Path;

// â”€â”€ Shared backend â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// CPU backend used by every CLI command.
///
/// `NdArray<f32>` requires no GPU runtime and produces deterministic results,
/// which is appropriate for a CLI tool that must run on any host.
pub(crate) type Backend = NdArray<f32>;

/// Atlas-native CPU backend used for the native I/O path (ADR 0003).
///
/// `SequentialBackend` mirrors [`Backend`]'s rationale â€” no GPU runtime,
/// deterministic â€” and keeps the native path's dependency surface minimal
/// while Burn is still present. Zero-sized; constructed fresh per call.
pub(crate) type NativeBackend = SequentialBackend;

// â”€â”€ Format inference â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Infer the image format from a file-system path.
///
/// Delegates to [`ritk_io::ImageFormat::from_path`] as the SSOT for extensionâ†’format mapping.
pub(crate) fn infer_format(path: &Path) -> Option<ImageFormat> {
    ImageFormat::from_path(path)
}

// â”€â”€ Read helper â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Read a 3-D medical image from `path`, inferring the format from the
/// file extension.
///
/// # Errors
/// Returns an error when the extension is unrecognised or the underlying
/// reader fails.
pub(crate) fn read_image(path: &Path) -> Result<Image<Backend, 3>> {
    let fmt = infer_format(path)
        .ok_or_else(|| anyhow!("Cannot infer input format from path: {}", path.display()))?;

    match fmt {
        ImageFormat::Dicom => {
            if is_rgb_dicom_series(path).unwrap_or(false) {
                bail!(
                    "RGB DICOM colour series are not supported by the CLI. \
                     Use `ritk-snap` (the graphical viewer) to load and inspect RGB DICOM volumes."
                );
            }
            read_image_native(path).map(native_image_to_burn)
        }
        ImageFormat::Vtk => read_image_native(path).map(native_image_to_burn),
        ImageFormat::NIfTI
        | ImageFormat::MetaImage
        | ImageFormat::Nrrd
        | ImageFormat::Png
        | ImageFormat::Mgh
        | ImageFormat::Tiff
        | ImageFormat::Jpeg
        | ImageFormat::Analyze => read_image_native(path).map(native_image_to_burn) }
}

// â”€â”€ Write helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Write `image` to `path` using the explicitly supplied `format`.
///
/// Accepted formats: `NIfTI`, `MetaImage`, `Nrrd`, `Mgh`, `Tiff`, `Vtk`,
/// `Jpeg`, `Analyze`.
/// `Png` is recognised but unsupported (returns a descriptive `Err`).
/// `Dicom` write is supported via [`ritk_io::write_dicom_series`].
///
/// # Errors
/// Returns an error when the format is unsupported or the writer fails.
pub(crate) fn write_image(
    path: &Path,
    image: &Image<Backend, 3>,
    format: ImageFormat,
) -> Result<()> {
    match format {
        ImageFormat::NIfTI => ritk_io::write_nifti::<Backend, _>(path, image)
            .with_context(|| format!("Failed to write NIfTI file: {}", path.display())),
        ImageFormat::MetaImage => ritk_io::write_metaimage::<Backend, _>(path, image)
            .with_context(|| format!("Failed to write MetaImage file: {}", path.display())),
        ImageFormat::Nrrd => ritk_io::write_nrrd::<Backend, _>(path, image)
            .with_context(|| format!("Failed to write NRRD file: {}", path.display())),
        ImageFormat::Mgh => ritk_io::write_mgh::<Backend, _>(image, path)
            .with_context(|| format!("Failed to write MGH file: {}", path.display())),
        ImageFormat::Tiff | ImageFormat::Jpeg | ImageFormat::Vtk => {
            let native = burn_image_to_native(image)?;
            write_image_native(path, &native, format)
        }
        ImageFormat::Png => Err(anyhow!(
            "PNG output is not supported: ritk-io has no write_png implementation. \
             Convert to NIfTI, MetaImage, or NRRD instead."
        )),
        ImageFormat::Dicom => ritk_io::write_dicom_series::<Backend, _>(path, image)
            .with_context(|| format!("Failed to write DICOM series to: {}", path.display())),
        ImageFormat::Analyze => ritk_io::write_analyze::<Backend, _>(path, image)
            .with_context(|| format!("Failed to write Analyze file: {}", path.display())) }
}

/// Write `image` to `path`, inferring the output format from the path extension.
///
/// Delegates to [`write_image`] after resolving the format.
///
/// # Errors
/// Returns an error when the extension is unrecognised or the writer fails.
pub(crate) fn write_image_inferred(path: &Path, image: &Image<Backend, 3>) -> Result<()> {
    let fmt = infer_format(path)
        .ok_or_else(|| anyhow!("Cannot infer output format from path: {}", path.display()))?;
    write_image(path, image, fmt)
}

// â”€â”€ Atlas-native I/O (ADR 0003 Phase A) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
//
// Parallel native helpers coexisting with the legacy helpers above during the
// cutover (ADR 0003). VTK now uses native reader/writer contracts. DICOM has a
// native reader but still lacks a native writer.

/// True when `fmt` has an Atlas-native reader (ADR 0003 Phase A coverage).
pub(crate) fn is_native_read_capable(fmt: ImageFormat) -> bool {
    matches!(
        fmt,
        ImageFormat::NIfTI
            | ImageFormat::Nrrd
            | ImageFormat::Analyze
            | ImageFormat::Mgh
            | ImageFormat::MetaImage
            | ImageFormat::Tiff
            | ImageFormat::Jpeg
            | ImageFormat::Vtk
            | ImageFormat::Png
            | ImageFormat::Dicom
    )
}

/// True when `fmt` has an Atlas-native writer (ADR 0003 Phase A coverage).
///
/// Narrower than [`is_native_read_capable`]: PNG has no native writer, matching
/// the Burn path (which also cannot write PNG â€” see [`write_image`]).
pub(crate) fn is_native_write_capable(fmt: ImageFormat) -> bool {
    matches!(
        fmt,
        ImageFormat::NIfTI
            | ImageFormat::Nrrd
            | ImageFormat::Analyze
            | ImageFormat::Mgh
            | ImageFormat::MetaImage
            | ImageFormat::Tiff
            | ImageFormat::Jpeg
            | ImageFormat::Vtk
    )
}

/// Read a 3-D medical image from `path` via the Atlas-native path.
///
/// Callers must first check [`is_native_read_capable`] on the inferred
/// format; formats without a native reader return a descriptive `Err` rather
/// than panicking, since the capability check is a contract, not a proof.
///
/// # Errors
/// Returns an error when the extension is unrecognised, the format has no
/// native reader, or the underlying reader fails.
pub(crate) fn read_image_native(path: &Path) -> Result<NativeImage<f32, NativeBackend, 3>> {
    let backend = NativeBackend::default();
    let fmt = infer_format(path)
        .ok_or_else(|| anyhow!("Cannot infer input format from path: {}", path.display()))?;

    match fmt {
        ImageFormat::NIfTI => ImageReader::read(
            &ritk_io::format::nifti::native::NiftiReader::new(backend),
            path,
        )
        .with_context(|| format!("Failed to read NIfTI file (native): {}", path.display())),
        ImageFormat::MetaImage => ImageReader::read(
            &ritk_io::format::metaimage::native::MetaImageReader::new(backend),
            path,
        )
        .with_context(|| format!("Failed to read MetaImage file (native): {}", path.display())),
        ImageFormat::Nrrd => ImageReader::read(
            &ritk_io::format::nrrd::native::NrrdReader::new(backend),
            path,
        )
        .with_context(|| format!("Failed to read NRRD file (native): {}", path.display())),
        ImageFormat::Png => {
            ImageReader::read(&ritk_io::format::png::native::PngReader::new(backend), path)
                .with_context(|| format!("Failed to read PNG file (native): {}", path.display()))
        }
        ImageFormat::Mgh => {
            ImageReader::read(&ritk_io::format::mgh::native::MghReader::new(backend), path)
                .with_context(|| format!("Failed to read MGH file (native): {}", path.display()))
        }
        ImageFormat::Tiff => ImageReader::read(
            &ritk_io::format::tiff::native::TiffReader::new(backend),
            path,
        )
        .with_context(|| format!("Failed to read TIFF file (native): {}", path.display())),
        ImageFormat::Jpeg => ImageReader::read(
            &ritk_io::format::jpeg::native::JpegReader::new(backend),
            path,
        )
        .with_context(|| format!("Failed to read JPEG file (native): {}", path.display())),
        ImageFormat::Vtk => {
            ImageReader::read(&ritk_io::format::vtk::native::VtkReader::new(backend), path)
                .with_context(|| format!("Failed to read VTK file (native): {}", path.display()))
        }
        ImageFormat::Analyze => ImageReader::read(
            &ritk_io::format::analyze::native::AnalyzeReader::new(backend),
            path,
        )
        .with_context(|| format!("Failed to read Analyze file (native): {}", path.display())),
        ImageFormat::Dicom => ImageReader::read(
            &ritk_io::format::dicom::native::DicomReader::new(backend),
            path,
        )
        .with_context(|| format!("Failed to read DICOM series (native): {}", path.display())) }
}

/// Bridge a native image into the Burn image type required by command
/// pipelines that have not migrated past ADR 0003 Phase A.
pub(crate) fn native_image_to_burn(image: NativeImage<f32, NativeBackend, 3>) -> Image<Backend, 3> {
    let backend = NativeBackend::default();
    let shape = image.shape();
    let origin = *image.origin();
    let spacing = *image.spacing();
    let direction = *image.direction();
    let values = image.data_cow_on(&backend).into_owned();
    let device: <Backend as BurnBackend>::Device = Default::default();

    Image::from_flat_on(values, shape, origin, spacing, direction, &device)
}

/// Bridge a legacy image into the native I/O contract at the CLI boundary.
pub(crate) fn burn_image_to_native(
    image: &Image<Backend, 3>,
) -> Result<NativeImage<f32, NativeBackend, 3>> {
    let backend = NativeBackend::default();
    NativeImage::from_flat_on(
        image.try_data_vec()?,
        image.shape(),
        *image.origin(),
        *image.spacing(),
        *image.direction(),
        &backend,
    )
    .context("cannot convert legacy image to native I/O image")
}

/// Write `image` to `path` via the Atlas-native path, using the explicitly
/// supplied `format`.
///
/// Callers must first check [`is_native_write_capable`] on the target format.
///
/// # Errors
/// Returns an error when the format has no native writer or the writer fails.
pub(crate) fn write_image_native(
    path: &Path,
    image: &NativeImage<f32, NativeBackend, 3>,
    format: ImageFormat,
) -> Result<()> {
    let backend = NativeBackend::default();
    match format {
        ImageFormat::NIfTI => ImageWriter::write(
            &ritk_io::format::nifti::native::NiftiWriter::new(backend),
            path,
            image,
        )
        .with_context(|| format!("Failed to write NIfTI file (native): {}", path.display())),
        ImageFormat::MetaImage => ImageWriter::write(
            &ritk_io::format::metaimage::native::MetaImageWriter::new(backend),
            path,
            image,
        )
        .with_context(|| {
            format!(
                "Failed to write MetaImage file (native): {}",
                path.display()
            )
        }),
        ImageFormat::Nrrd => ImageWriter::write(
            &ritk_io::format::nrrd::native::NrrdWriter::new(backend),
            path,
            image,
        )
        .with_context(|| format!("Failed to write NRRD file (native): {}", path.display())),
        ImageFormat::Mgh => ImageWriter::write(
            &ritk_io::format::mgh::native::MghWriter::new(backend),
            path,
            image,
        )
        .with_context(|| format!("Failed to write MGH file (native): {}", path.display())),
        ImageFormat::Tiff => ImageWriter::write(
            &ritk_io::format::tiff::native::TiffWriter::new(backend),
            path,
            image,
        )
        .with_context(|| format!("Failed to write TIFF file (native): {}", path.display())),
        ImageFormat::Jpeg => ImageWriter::write(
            &ritk_io::format::jpeg::native::JpegWriter::new(backend),
            path,
            image,
        )
        .with_context(|| format!("Failed to write JPEG file (native): {}", path.display())),
        ImageFormat::Vtk => ImageWriter::write(
            &ritk_io::format::vtk::native::VtkWriter::new(backend),
            path,
            image,
        )
        .with_context(|| format!("Failed to write VTK file (native): {}", path.display())),
        ImageFormat::Analyze => ImageWriter::write(
            &ritk_io::format::analyze::native::AnalyzeWriter::new(backend),
            path,
            image,
        )
        .with_context(|| format!("Failed to write Analyze file (native): {}", path.display())),
        ImageFormat::Png | ImageFormat::Dicom => Err(anyhow!(
            "{:?} has no Atlas-native writer; check is_native_write_capable first",
            format
        )) }
}

// â”€â”€ Tests â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

#[cfg(test)]
mod tests {
    use super::*;
    use ritk_io::ImageFormat;

    #[test]
    fn test_infer_format_nifti_single_ext() {
        assert_eq!(
            infer_format(Path::new("brain.nii")),
            Some(ImageFormat::NIfTI)
        );
    }

    #[test]
    fn test_infer_format_nifti_compound_ext() {
        assert_eq!(
            infer_format(Path::new("brain.nii.gz")),
            Some(ImageFormat::NIfTI)
        );
    }

    #[test]
    fn test_infer_format_metaimage_mha() {
        assert_eq!(
            infer_format(Path::new("scan.mha")),
            Some(ImageFormat::MetaImage)
        );
    }

    #[test]
    fn test_infer_format_metaimage_mhd() {
        assert_eq!(
            infer_format(Path::new("scan.mhd")),
            Some(ImageFormat::MetaImage)
        );
    }

    #[test]
    fn test_infer_format_nrrd() {
        assert_eq!(
            infer_format(Path::new("volume.nrrd")),
            Some(ImageFormat::Nrrd)
        );
    }

    #[test]
    fn test_infer_format_nhdr() {
        assert_eq!(
            infer_format(Path::new("volume.nhdr")),
            Some(ImageFormat::Nrrd)
        );
    }

    #[test]
    fn test_infer_format_png() {
        assert_eq!(infer_format(Path::new("slice.png")), Some(ImageFormat::Png));
    }

    #[test]
    fn test_infer_format_dicom_dcm() {
        assert_eq!(infer_format(Path::new("001.dcm")), Some(ImageFormat::Dicom));
    }

    #[test]
    fn test_infer_format_mgh() {
        assert_eq!(infer_format(Path::new("brain.mgh")), Some(ImageFormat::Mgh));
    }

    #[test]
    fn test_infer_format_mgz() {
        assert_eq!(infer_format(Path::new("brain.mgz")), Some(ImageFormat::Mgh));
    }

    #[test]
    fn test_infer_format_tiff() {
        assert_eq!(
            infer_format(Path::new("scan.tiff")),
            Some(ImageFormat::Tiff)
        );
    }

    #[test]
    fn test_infer_format_tif() {
        assert_eq!(infer_format(Path::new("scan.tif")), Some(ImageFormat::Tiff));
    }

    #[test]
    fn test_infer_format_vtk() {
        assert_eq!(infer_format(Path::new("model.vtk")), Some(ImageFormat::Vtk));
    }

    #[test]
    fn test_infer_format_jpeg() {
        assert_eq!(
            infer_format(Path::new("photo.jpeg")),
            Some(ImageFormat::Jpeg)
        );
    }

    #[test]
    fn test_infer_format_jpg() {
        assert_eq!(
            infer_format(Path::new("photo.jpg")),
            Some(ImageFormat::Jpeg)
        );
    }

    #[test]
    fn test_infer_format_analyze_hdr() {
        assert_eq!(
            infer_format(Path::new("brain.hdr")),
            Some(ImageFormat::Analyze)
        );
    }

    #[test]
    fn test_infer_format_analyze_img() {
        assert_eq!(
            infer_format(Path::new("brain.img")),
            Some(ImageFormat::Analyze)
        );
    }

    #[test]
    fn test_infer_format_unknown_returns_none() {
        assert_eq!(infer_format(Path::new("data.xyz")), None);
    }

    #[test]
    fn test_infer_format_no_extension_returns_none() {
        assert_eq!(infer_format(Path::new("scandata")), None);
    }

    #[test]
    fn test_native_read_capability_tracks_dicom_cutover() {
        assert!(
            is_native_read_capable(ImageFormat::Dicom),
            "DICOM reads must route through the native reader"
        );
        assert!(
            !is_native_write_capable(ImageFormat::Dicom),
            "DICOM writes remain on the legacy writer until a native writer exists"
        );
        assert!(is_native_read_capable(ImageFormat::Vtk));
    }

    #[test]
    fn test_native_image_to_burn_preserves_values_and_metadata() {
        use ritk_spatial::{Direction, Point, Spacing};

        let backend = NativeBackend::default();
        let shape = [2, 2, 3];
        let values = (0..12).map(|v| v as f32 + 0.25).collect::<Vec<_>>();
        let origin = Point::new([1.0, 2.0, 3.0]);
        let spacing = Spacing::new([0.5, 0.75, 1.25]);
        let direction = Direction::identity();
        let native = NativeImage::<f32, NativeBackend, 3>::from_flat_on(
            values.clone(),
            shape,
            origin,
            spacing,
            direction,
            &backend,
        )
        .expect("native image construction must preserve valid shape");

        let burn = native_image_to_burn(native);

        assert_eq!(burn.shape(), shape);
        assert_eq!(*burn.origin(), origin);
        assert_eq!(*burn.spacing(), spacing);
        assert_eq!(*burn.direction(), direction);
        assert_eq!(burn.try_data_vec().unwrap(), values);
    }

    #[test]
    fn test_dicom_native_read_matches_legacy_reader_and_shared_helper() {
        use ritk_image::tensor::{Shape, Tensor };
        use ritk_image::Image;
        use ritk_spatial::{Direction, Point, Spacing};

        let dir = tempfile::tempdir().unwrap();
        let series_path = dir.path().join("series.dicom");
        let values = (0..8).map(|v| v as f32).collect::<Vec<_>>();
        let shape = [2, 2, 2];
        let origin = Point::new([0.0; 3]);
        let spacing = Spacing::new([1.0; 3]);
        let direction = Direction::identity();
        let device: <Backend as BurnBackend>::Device = Default::default();
        let tensor = Tensor::<f32, Backend>::from_data(
            ::new(values.clone(), Shape::new(shape)),
            &device,
        );
        let image = Image::new(tensor, origin, spacing, direction);

        ritk_io::write_dicom_series::<Backend, _>(&series_path, &image)
            .expect("DICOM fixture write must succeed");

        let (legacy, _) =
            ritk_io::load_dicom_series_with_metadata::<Backend, _>(&series_path, &device)
                .expect("legacy metadata-rich read");
        let native = read_image_native(&series_path).expect("native read");
        let shared = read_image(&series_path).expect("shared helper read");

        assert_eq!(native.shape(), legacy.shape());
        assert_eq!(*native.origin(), *legacy.origin());
        assert_eq!(*native.spacing(), *legacy.spacing());
        assert_eq!(*native.direction(), *legacy.direction());
        assert_eq!(
            native.data_vec_on(&NativeBackend::default()),
            legacy.try_data_vec().unwrap()
        );
        assert_eq!(shared.shape(), legacy.shape());
        assert_eq!(*shared.origin(), *legacy.origin());
        assert_eq!(*shared.spacing(), *legacy.spacing());
        assert_eq!(*shared.direction(), *legacy.direction());
        assert_eq!(
            shared.try_data_vec().unwrap(),
            legacy.try_data_vec().unwrap()
        );
    }

    #[test]
    fn test_write_image_png_returns_err() {
        use ritk_image::tensor::{Shape, Tensor };
        use ritk_image::Image;
        use ritk_spatial::{Direction, Point, Spacing};

        let device: <Backend as BurnBackend>::Device = Default::default();
        let td = ::new(vec![0.0f32; 8], Shape::new([2, 2, 2]));
        let tensor = Tensor::<f32, Backend>::from_data(td, &device);
        let image = Image::new(
            tensor,
            Point::new([0.0; 3]),
            Spacing::new([1.0; 3]),
            Direction::identity(),
        );
        let result = write_image(Path::new("out.png"), &image, ImageFormat::Png);
        assert!(result.is_err(), "PNG write must return an error");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("PNG output is not supported"),
            "error message must explain PNG limitation, got: {msg}"
        );
    }

    #[test]
    fn test_write_image_dicom_creates_directory() {
        use ritk_image::tensor::{Shape, Tensor };
        use ritk_image::Image;
        use ritk_spatial::{Direction, Point, Spacing};
        let dir = tempfile::tempdir().unwrap();
        let out_path = dir.path().join("dicom_series");
        let device: <Backend as BurnBackend>::Device = Default::default();
        let td = ::new(vec![0.0f32; 8], Shape::new([2, 2, 2]));
        let tensor = Tensor::<f32, Backend>::from_data(td, &device);
        let image = Image::new(
            tensor,
            Point::new([0.0; 3]),
            Spacing::new([1.0; 3]),
            Direction::identity(),
        );
        let result = write_image(&out_path, &image, ImageFormat::Dicom);
        assert!(
            result.is_ok(),
            "DICOM write must succeed: {:?}",
            result.err()
        );
        assert!(out_path.is_dir(), "DICOM output directory must exist");
    }

    #[test]
    fn test_write_image_vtk_succeeds() {
        use ritk_image::tensor::{Shape, Tensor };
        use ritk_image::Image;
        use ritk_spatial::{Direction, Point, Spacing};

        let dir = tempfile::tempdir().unwrap();
        let out_path = dir.path().join("out.vtk");

        let device: <Backend as BurnBackend>::Device = Default::default();
        let td = ::new(vec![1.0f32; 8], Shape::new([2, 2, 2]));
        let tensor = Tensor::<f32, Backend>::from_data(td, &device);
        let image = Image::new(
            tensor,
            Point::new([0.0; 3]),
            Spacing::new([1.0; 3]),
            Direction::identity(),
        );
        let result = write_image(&out_path, &image, ImageFormat::Vtk);
        assert!(result.is_ok(), "VTK write must succeed: {:?}", result.err());
        assert!(out_path.exists(), "VTK output file must exist");
        assert!(
            out_path.metadata().unwrap().len() > 0,
            "VTK file must be non-empty"
        );
    }

    #[test]
    fn test_write_image_jpeg_depth_gt_one_returns_error() {
        use ritk_image::tensor::{Shape, Tensor };
        use ritk_image::Image;
        use ritk_spatial::{Direction, Point, Spacing};

        let device: <Backend as BurnBackend>::Device = Default::default();
        // A depth of two violates JPEG's two-dimensional image contract.
        let td = ::new(vec![128.0f32; 8], Shape::new([2, 2, 2]));
        let tensor = Tensor::<f32, Backend>::from_data(td, &device);
        let image = Image::new(
            tensor,
            Point::new([0.0; 3]),
            Spacing::new([1.0; 3]),
            Direction::identity(),
        );
        let dir = tempfile::tempdir().unwrap();
        let out_path = dir.path().join("out.jpg");
        let result = write_image(&out_path, &image, ImageFormat::Jpeg);
        assert!(result.is_err(), "JPEG write with depth=2 must fail");
        let err = result.unwrap_err();
        let msg = format!("{:#}", err);
        assert!(
            msg.contains("depth=2"),
            "error message must name the rejected depth, got: {msg}"
        );
    }

    #[test]
    fn test_write_image_jpeg_2d_succeeds() {
        use ritk_image::tensor::{Shape, Tensor };
        use ritk_image::Image;
        use ritk_spatial::{Direction, Point, Spacing};

        let dir = tempfile::tempdir().unwrap();
        let out_path = dir.path().join("out.jpg");

        let device: <Backend as BurnBackend>::Device = Default::default();
        // nz=1 is valid for JPEG
        let td = ::new(vec![128.0f32; 4], Shape::new([1, 2, 2]));
        let tensor = Tensor::<f32, Backend>::from_data(td, &device);
        let image = Image::new(
            tensor,
            Point::new([0.0; 3]),
            Spacing::new([1.0; 3]),
            Direction::identity(),
        );
        let result = write_image(&out_path, &image, ImageFormat::Jpeg);
        assert!(
            result.is_ok(),
            "JPEG write with nz=1 must succeed: {:?}",
            result.err()
        );
        assert!(out_path.exists(), "JPEG output file must exist");
        assert!(
            out_path.metadata().unwrap().len() > 0,
            "JPEG file must be non-empty"
        );
    }
}
