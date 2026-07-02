//! Differential coverage for the Coeus reader implementors of the
//! [`crate::domain::coeus`] contract.
//!
//! One harness, all formats: read the *same file* through the Coeus trait
//! reader and the Burn free-function reader and assert exact voxel and shape
//! equality. Comparing two readers of one file (rather than reader output vs.
//! original values) makes the oracle independent of any write-side lossiness
//! (e.g. JPEG quantization) — the Coeus adapter must decode identically to
//! the verified Burn path, byte-for-byte of the decoded stream.

use crate::domain::coeus::CoeusImageReader;
use burn::tensor::backend::Backend;
use burn_ndarray::NdArray;
use coeus_core::SequentialBackend;
use ritk_core::image::Image as BurnImage;
use ritk_image::coeus::Image as CoeusImage;
use ritk_spatial::{Direction, Point, Spacing};
use std::path::Path;

type BurnBackend = NdArray<f32>;

/// Read `path` through the Coeus trait `reader` and the Burn `read_burn`
/// free function; assert identical shape and exact voxel equality.
fn assert_coeus_reader_matches_burn<R>(
    path: &Path,
    reader: &R,
    read_burn: impl Fn(&Path) -> anyhow::Result<BurnImage<BurnBackend, 3>>,
) where
    R: CoeusImageReader<f32, SequentialBackend, 3>,
{
    let coeus: CoeusImage<f32, SequentialBackend, 3> =
        reader.read(path).expect("coeus trait read");
    let burn = read_burn(path).expect("burn read");

    assert_eq!(coeus.shape(), burn.shape(), "shape parity");
    let coeus_vals = coeus.data_slice().expect("contiguous coeus data");
    let burn_vals = burn.try_data_vec().expect("burn host data");
    assert_eq!(
        coeus_vals,
        burn_vals.as_slice(),
        "coeus trait reader must decode identically to the burn reader"
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

#[test]
fn coeus_mgh_reader_matches_burn() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("vol.mgh");
    ritk_mgh::write_mgh(&burn_volume([2, 3, 4]), &path).expect("mgh write");
    assert_coeus_reader_matches_burn(
        &path,
        &super::mgh::CoeusMghReader::new(SequentialBackend),
        |p| ritk_mgh::read_mgh::<BurnBackend, _>(p, &burn_device()),
    );
}

#[test]
fn coeus_metaimage_reader_matches_burn() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("vol.mha");
    ritk_metaimage::write_metaimage(&path, &burn_volume([2, 3, 4])).expect("mha write");
    assert_coeus_reader_matches_burn(
        &path,
        &super::metaimage::CoeusMetaImageReader::new(SequentialBackend),
        |p| ritk_metaimage::read_metaimage::<BurnBackend, _>(p, &burn_device()),
    );
}

#[test]
fn coeus_minc_reader_matches_burn() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("vol.mnc");
    ritk_minc::write_minc(&burn_volume([2, 3, 4]), &path).expect("minc write");
    assert_coeus_reader_matches_burn(
        &path,
        &super::minc::CoeusMincReader::new(SequentialBackend),
        |p| ritk_minc::read_minc::<BurnBackend, _>(p, &burn_device()),
    );
}

#[test]
fn coeus_tiff_reader_matches_burn() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("vol.tiff");
    ritk_tiff::write_tiff(&burn_volume([2, 3, 4]), &path).expect("tiff write");
    assert_coeus_reader_matches_burn(
        &path,
        &super::tiff::CoeusTiffReader::new(SequentialBackend),
        |p| ritk_tiff::read_tiff::<BurnBackend, _>(p, &burn_device()),
    );
}

#[test]
fn coeus_jpeg_reader_matches_burn() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("slice.jpg");
    ritk_jpeg::write_jpeg(&path, &burn_volume([1, 8, 12])).expect("jpeg write");
    assert_coeus_reader_matches_burn(
        &path,
        &super::jpeg::CoeusJpegReader::new(SequentialBackend),
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
fn coeus_png_reader_matches_burn() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("slice.png");
    write_gray_png(&path, 12, 8, 3);
    assert_coeus_reader_matches_burn(
        &path,
        &super::png::CoeusPngReader::new(SequentialBackend),
        |p| ritk_png::read_png_to_image::<BurnBackend, _>(p, &burn_device()),
    );
}

#[test]
fn coeus_png_series_reader_matches_burn() {
    let dir = tempfile::tempdir().expect("tempdir");
    write_gray_png(&dir.path().join("s000.png"), 6, 4, 11);
    write_gray_png(&dir.path().join("s001.png"), 6, 4, 71);
    assert_coeus_reader_matches_burn(
        dir.path(),
        &super::png::CoeusPngSeriesReader::new(SequentialBackend),
        |p| ritk_png::read_png_series::<BurnBackend, _>(p, &burn_device()),
    );
}
