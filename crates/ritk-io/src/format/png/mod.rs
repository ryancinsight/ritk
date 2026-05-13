pub use ritk_png::{
    read_png_color_series, read_png_color_to_volume, read_png_series, read_png_to_image,
    PngColorReader, PngColorSeriesReader, PngReader, PngSeriesReader,
};

use crate::domain::ImageReader;
use burn::tensor::backend::Backend;
use ritk_core::image::Image;
use std::path::Path;

impl<B: Backend> ImageReader<B, 3> for PngReader<B> {
    fn read<P: AsRef<Path>>(&self, path: P) -> std::io::Result<Image<B, 3>> {
        self.read_image(path)
            .map_err(|e| std::io::Error::other(e.to_string()))
    }
}

impl<B: Backend> ImageReader<B, 3> for PngSeriesReader<B> {
    fn read<P: AsRef<Path>>(&self, path: P) -> std::io::Result<Image<B, 3>> {
        self.read_image(path)
            .map_err(|e| std::io::Error::other(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::{PngReader, PngSeriesReader};
    use crate::domain::ImageReader;
    use burn::tensor::backend::Backend;
    use burn_ndarray::NdArray;
    use std::path::Path;
    use tempfile::tempdir;

    type TestBackend = NdArray<f32>;

    fn write_gray_png(path: &Path, width: u32, height: u32, pixels: &[u8]) {
        let image = image::GrayImage::from_raw(width, height, pixels.to_vec())
            .expect("test image dimensions must match pixel count");
        image.save(path).expect("test PNG write must succeed");
    }

    fn tensor_values(image: &ritk_core::image::Image<TestBackend, 3>) -> Vec<f32> {
        let data = image.data().clone().to_data();
        data.as_slice::<f32>().unwrap().to_vec()
    }

    #[test]
    fn png_reader_adapter_delegates_to_authoritative_crate() -> anyhow::Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("slice.png");
        write_gray_png(&path, 2, 1, &[9, 10]);

        let device: <TestBackend as Backend>::Device = Default::default();
        let reader = PngReader::<TestBackend>::new(device);
        let image = ImageReader::<TestBackend, 3>::read(&reader, &path)?;

        assert_eq!(image.shape(), [1, 1, 2]);
        assert_eq!(tensor_values(&image), vec![9.0, 10.0]);
        Ok(())
    }

    #[test]
    fn png_series_reader_adapter_delegates_to_authoritative_crate() -> anyhow::Result<()> {
        let dir = tempdir()?;
        write_gray_png(&dir.path().join("slice2.png"), 1, 1, &[2]);
        write_gray_png(&dir.path().join("slice1.png"), 1, 1, &[1]);

        let device: <TestBackend as Backend>::Device = Default::default();
        let reader = PngSeriesReader::<TestBackend>::new(device);
        let image = ImageReader::<TestBackend, 3>::read(&reader, dir.path())?;

        assert_eq!(image.shape(), [2, 1, 1]);
        assert_eq!(tensor_values(&image), vec![1.0, 2.0]);
        Ok(())
    }
}
