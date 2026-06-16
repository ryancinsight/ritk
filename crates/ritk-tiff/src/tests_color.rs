use super::*;
use burn::tensor::backend::Backend;
use burn_ndarray::NdArray;
use std::fs::File;
use std::io::BufWriter;
use tempfile::tempdir;
use tiff::encoder::{colortype, TiffEncoder};

type TestBackend = NdArray<f32>;

fn write_rgb8_pages(path: &Path, width: u32, height: u32, pages: &[Vec<u8>]) -> Result<()> {
    let file = File::create(path)?;
    let writer = BufWriter::new(file);
    let mut encoder = TiffEncoder::new(writer)?;
    for page in pages {
        encoder.write_image::<colortype::RGB8>(width, height, page)?;
    }
    Ok(())
}

fn write_gray8_page(path: &Path, width: u32, height: u32, pixels: &[u8]) -> Result<()> {
    let file = File::create(path)?;
    let writer = BufWriter::new(file);
    let mut encoder = TiffEncoder::new(writer)?;
    encoder.write_image::<colortype::Gray8>(width, height, pixels)?;
    Ok(())
}

fn volume_values(volume: &RgbVolume<TestBackend>) -> Vec<f32> {
    volume.with_data_slice(|s| s.to_vec())
}

#[test]
fn read_tiff_color_to_volume_preserves_rgb_page_stack() -> Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("rgb_stack.tiff");
    let page0 = vec![1, 2, 3, 4, 5, 6];
    let page1 = vec![10, 20, 30, 40, 50, 60];
    write_rgb8_pages(&path, 2, 1, &[page0.clone(), page1.clone()])?;
    let device = <TestBackend as Backend>::Device::default();

    let volume = read_tiff_color_to_volume::<TestBackend, _>(&path, &device)?;

    assert_eq!(volume.shape(), [2, 1, 2, 3]);
    assert_eq!(volume.spatial_shape(), [2, 1, 2]);
    assert_eq!(
        volume_values(&volume),
        page0
            .iter()
            .chain(page1.iter())
            .map(|&v| v as f32)
            .collect::<Vec<_>>()
    );
    assert_eq!(
        [volume.origin()[0], volume.origin()[1], volume.origin()[2]],
        [0.0, 0.0, 0.0]
    );
    assert_eq!(
        [
            volume.spacing()[0],
            volume.spacing()[1],
            volume.spacing()[2]
        ],
        [1.0, 1.0, 1.0]
    );
    Ok(())
}

#[test]
fn read_tiff_color_to_volume_rejects_grayscale_tiff() -> Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("gray.tiff");
    write_gray8_page(&path, 2, 1, &[7, 9])?;
    let device = <TestBackend as Backend>::Device::default();

    let err = read_tiff_color_to_volume::<TestBackend, _>(&path, &device).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("supports only RGB pages"),
        "expected RGB page rejection, got {msg}"
    );
    Ok(())
}

#[test]
fn tiff_color_reader_delegates_to_rgb_loader() -> Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("reader.tiff");
    write_rgb8_pages(&path, 1, 1, &[vec![32, 128, 224]])?;
    let reader = TiffColorReader::<TestBackend>::new(Default::default());

    let volume = reader.read_volume(&path)?;

    assert_eq!(volume.shape(), [1, 1, 1, 3]);
    assert_eq!(volume_values(&volume), vec![32.0, 128.0, 224.0]);
    Ok(())
}
