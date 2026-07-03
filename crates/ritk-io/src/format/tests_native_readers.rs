//! Differential coverage for the Atlas-native reader implementors of the
//! unified [`crate::domain::ImageReader`] contract.
//!
//! One harness, all formats: read the *same file* through the native trait
//! reader and the Burn free-function reader and assert exact voxel and shape
//! equality. Comparing two readers of one file (rather than reader output vs.
//! original values) makes the oracle independent of any write-side lossiness
//! (e.g. JPEG quantization) — the native adapter must decode identically to
//! the verified Burn path, byte-for-byte of the decoded stream.

use crate::domain::ImageReader;
use burn::tensor::backend::Backend;
use burn_ndarray::NdArray;
use coeus_core::SequentialBackend;
use ritk_core::image::Image as BurnImage;
use ritk_image::native::Image as NativeImage;
use ritk_spatial::{Direction, Point, Spacing};
use std::path::Path;

type BurnBackend = NdArray<f32>;

/// Read `path` through the native trait `reader` and the Burn `read_burn`
/// free function; assert identical shape and exact voxel equality.
fn assert_native_reader_matches_burn<R>(
    path: &Path,
    reader: &R,
    read_burn: impl Fn(&Path) -> anyhow::Result<BurnImage<BurnBackend, 3>>,
) where
    R: ImageReader<NativeImage<f32, SequentialBackend, 3>>,
{
    let native: NativeImage<f32, SequentialBackend, 3> =
        reader.read(path).expect("native trait read");
    let burn = read_burn(path).expect("burn read");

    assert_eq!(native.shape(), burn.shape(), "shape parity");
    let native_vals = native.data_slice().expect("contiguous native data");
    let burn_vals = burn.try_data_vec().expect("burn host data");
    assert_eq!(
        native_vals,
        burn_vals.as_slice(),
        "native trait reader must decode identically to the burn reader"
    );
}

/// A small anisotropic Burn test volume for the formats with Burn writers.
fn burn_volume(dims: [usize; 3]) -> BurnImage<BurnBackend, 3> {
    use burn::tensor::{Shape, Tensor, TensorData};
    let n = dims[0] * dims[1] * dims[2];
    let voxels: Vec<f32> = (0..n).map(|i| i as f32 * 0.5 - 4.0).collect();
    let device = <BurnBackend as Backend>::Device::default();
    let tensor = Tensor::<BurnBackend, 3>::from_data(
        TensorData::new(voxels, Shape::new(dims)),
        &device,
    );
    BurnImage::new(
        tensor,
        Point::new([1.0, -2.0, 3.0]),
        Spacing::new([2.0, 1.5, 0.75]),
        Direction::identity(),
    )
}

fn burn_device() -> <BurnBackend as Backend>::Device {
    <BurnBackend as Backend>::Device::default()
}

/// Round-trip a native volume through the unified [`crate::domain::ImageWriter`]
/// then [`crate::domain::ImageReader`] adapters; assert exact voxel + shape
/// parity. Exercises the native writer adapter end-to-end (the reader adapters
/// are covered by the differential tests above).
fn assert_native_writer_reader_round_trips<W, R>(path: &std::path::Path, writer: &W, reader: &R)
where
    W: crate::domain::ImageWriter<NativeImage<f32, SequentialBackend, 3>>,
    R: ImageReader<NativeImage<f32, SequentialBackend, 3>>,
{
    let dims = [2usize, 3, 4];
    let n = dims[0] * dims[1] * dims[2];
    let voxels: Vec<f32> = (0..n).map(|i| i as f32 * 0.5 - 4.0).collect();
    let image = NativeImage::from_flat_on(
        voxels.clone(),
        dims,
        Point::new([5.0, -10.0, 15.0]),
        Spacing::new([1.5, 0.75, 0.9]),
        Direction::identity(),
        &SequentialBackend,
    )
    .expect("native image");

    writer.write(path, &image).expect("contract write");
    let loaded: NativeImage<f32, SequentialBackend, 3> = reader.read(path).expect("contract read");

    assert_eq!(loaded.shape(), dims, "shape parity");
    assert_eq!(
        loaded.data_slice().expect("contiguous"),
        voxels.as_slice(),
        "native writer→reader contract must preserve voxels exactly"
    );
}

#[test]
fn native_nrrd_writer_reader_contract_round_trips() {
    let dir = tempfile::tempdir().expect("tempdir");
    assert_native_writer_reader_round_trips(
        &dir.path().join("contract.nrrd"),
        &super::nrrd::native::NrrdWriter::new(SequentialBackend),
        &super::nrrd::native::NrrdReader::new(SequentialBackend),
    );
}

#[test]
fn native_analyze_writer_reader_contract_round_trips() {
    let dir = tempfile::tempdir().expect("tempdir");
    assert_native_writer_reader_round_trips(
        &dir.path().join("contract.hdr"),
        &super::analyze::native::AnalyzeWriter::new(SequentialBackend),
        &super::analyze::native::AnalyzeReader::new(SequentialBackend),
    );
}

#[test]
fn native_mgh_writer_reader_contract_round_trips() {
    let dir = tempfile::tempdir().expect("tempdir");
    assert_native_writer_reader_round_trips(
        &dir.path().join("contract.mgh"),
        &super::mgh::native::MghWriter::new(SequentialBackend),
        &super::mgh::native::MghReader::new(SequentialBackend),
    );
}

#[test]
fn native_metaimage_writer_reader_contract_round_trips() {
    let dir = tempfile::tempdir().expect("tempdir");
    assert_native_writer_reader_round_trips(
        &dir.path().join("contract.mha"),
        &super::metaimage::native::MetaImageWriter::new(SequentialBackend),
        &super::metaimage::native::MetaImageReader::new(SequentialBackend),
    );
}

#[test]
fn native_minc_writer_reader_contract_round_trips() {
    let dir = tempfile::tempdir().expect("tempdir");
    assert_native_writer_reader_round_trips(
        &dir.path().join("contract.mnc"),
        &super::minc::native::MincWriter::new(SequentialBackend),
        &super::minc::native::MincReader::new(SequentialBackend),
    );
}

#[test]
fn native_tiff_writer_reader_contract_round_trips() {
    let dir = tempfile::tempdir().expect("tempdir");
    assert_native_writer_reader_round_trips(
        &dir.path().join("contract.tiff"),
        &super::tiff::native::TiffWriter::new(SequentialBackend),
        &super::tiff::native::TiffReader::new(SequentialBackend),
    );
}

#[test]
fn native_mgh_reader_matches_burn() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("vol.mgh");
    ritk_mgh::write_mgh(&burn_volume([2, 3, 4]), &path).expect("mgh write");
    assert_native_reader_matches_burn(
        &path,
        &super::mgh::native::MghReader::new(SequentialBackend),
        |p| ritk_mgh::read_mgh::<BurnBackend, _>(p, &burn_device()),
    );
}

#[test]
fn native_metaimage_reader_matches_burn() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("vol.mha");
    ritk_metaimage::write_metaimage(&path, &burn_volume([2, 3, 4])).expect("mha write");
    assert_native_reader_matches_burn(
        &path,
        &super::metaimage::native::MetaImageReader::new(SequentialBackend),
        |p| ritk_metaimage::read_metaimage::<BurnBackend, _>(p, &burn_device()),
    );
}

#[test]
fn native_minc_reader_matches_burn() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("vol.mnc");
    ritk_minc::write_minc(&burn_volume([2, 3, 4]), &path).expect("minc write");
    assert_native_reader_matches_burn(
        &path,
        &super::minc::native::MincReader::new(SequentialBackend),
        |p| ritk_minc::read_minc::<BurnBackend, _>(p, &burn_device()),
    );
}

#[test]
fn native_nrrd_reader_matches_burn() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("vol.nrrd");
    ritk_nrrd::write_nrrd(&path, &burn_volume([2, 3, 4])).expect("nrrd write");
    assert_native_reader_matches_burn(
        &path,
        &super::nrrd::native::NrrdReader::new(SequentialBackend),
        |p| ritk_nrrd::read_nrrd::<BurnBackend, _>(p, &burn_device()),
    );
}

#[test]
fn native_analyze_reader_matches_burn() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("vol.hdr");
    ritk_analyze::write_analyze(&path, &burn_volume([2, 3, 4])).expect("analyze write");
    assert_native_reader_matches_burn(
        &path,
        &super::analyze::native::AnalyzeReader::new(SequentialBackend),
        |p| ritk_analyze::read_analyze::<BurnBackend, _>(p, &burn_device()),
    );
}

#[test]
fn native_tiff_reader_matches_burn() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("vol.tiff");
    ritk_tiff::write_tiff(&burn_volume([2, 3, 4]), &path).expect("tiff write");
    assert_native_reader_matches_burn(
        &path,
        &super::tiff::native::TiffReader::new(SequentialBackend),
        |p| ritk_tiff::read_tiff::<BurnBackend, _>(p, &burn_device()),
    );
}

#[test]
fn native_jpeg_reader_matches_burn() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("slice.jpg");
    ritk_jpeg::write_jpeg(&path, &burn_volume([1, 8, 12])).expect("jpeg write");
    assert_native_reader_matches_burn(
        &path,
        &super::jpeg::native::JpegReader::new(SequentialBackend),
        |p| ritk_jpeg::read_jpeg::<BurnBackend, _>(p, &burn_device()),
    );
}

/// Write a synthetic 8-bit grayscale PNG (no Burn PNG writer exists).
fn write_gray_png(path: &Path, width: u32, height: u32, seed: u8) {
    let img = image::GrayImage::from_fn(width, height, |x, y| {
        image::Luma([((x * 7 + y * 13) as u8).wrapping_add(seed)])
    });
    img.save(path).expect("png save");
}

#[test]
fn native_png_reader_matches_burn() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("slice.png");
    write_gray_png(&path, 12, 8, 3);
    assert_native_reader_matches_burn(
        &path,
        &super::png::native::PngReader::new(SequentialBackend),
        |p| ritk_png::read_png_to_image::<BurnBackend, _>(p, &burn_device()),
    );
}

#[test]
fn native_png_series_reader_matches_burn() {
    let dir = tempfile::tempdir().expect("tempdir");
    write_gray_png(&dir.path().join("s000.png"), 6, 4, 11);
    write_gray_png(&dir.path().join("s001.png"), 6, 4, 71);
    assert_native_reader_matches_burn(
        dir.path(),
        &super::png::native::PngSeriesReader::new(SequentialBackend),
        |p| ritk_png::read_png_series::<BurnBackend, _>(p, &burn_device()),
    );
}
