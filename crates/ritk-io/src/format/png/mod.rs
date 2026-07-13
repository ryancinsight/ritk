pub use ritk_png::{
    read_png_color_series, read_png_color_to_volume, PngColorReader, PngColorSeriesReader,
};

use crate::domain::ImageReader;
use anyhow::Result;
use coeus_core::SequentialBackend;
use ritk_core::image::Image;
use ritk_image::tensor::backend::Backend;
use ritk_image::tensor::{Shape, Tensor, TensorData};
use std::path::Path;

fn native_to_legacy<B: Backend>(
    native: ritk_image::native::Image<f32, SequentialBackend, 3>,
    device: &B::Device,
) -> Image<B, 3> {
    let tensor = Tensor::<B, 3>::from_data(
        TensorData::new(
            native.data_cow_on(&SequentialBackend).into_owned(),
            Shape::new(native.shape()),
        ),
        device,
    );
    Image::new(
        tensor,
        *native.origin(),
        *native.spacing(),
        *native.direction(),
    )
}

/// Reads one PNG through the native provider and converts at this legacy boundary.
pub fn read_png_to_image<B: Backend, P: AsRef<Path>>(
    path: P,
    device: &B::Device,
) -> Result<Image<B, 3>> {
    ritk_png::read_png_to_image(path, &SequentialBackend)
        .map(|native| native_to_legacy(native, device))
}

/// Reads a PNG series through the native provider and converts at this legacy boundary.
pub fn read_png_series<B: Backend, P: AsRef<Path>>(
    path: P,
    device: &B::Device,
) -> Result<Image<B, 3>> {
    ritk_png::read_png_series(path, &SequentialBackend)
        .map(|native| native_to_legacy(native, device))
}

/// Device-bound legacy PNG reader.
pub struct PngReader<B: Backend> {
    device: B::Device,
}

impl<B: Backend> PngReader<B> {
    /// Creates a reader for `device`.
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }

    /// Reads one PNG into the legacy image substrate.
    pub fn read_image<P: AsRef<Path>>(&self, path: P) -> Result<Image<B, 3>> {
        read_png_to_image(path, &self.device)
    }
}

/// Device-bound legacy PNG series reader.
pub struct PngSeriesReader<B: Backend> {
    device: B::Device,
}

impl<B: Backend> PngSeriesReader<B> {
    /// Creates a series reader for `device`.
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }

    /// Reads a PNG series into the legacy image substrate.
    pub fn read_image<P: AsRef<Path>>(&self, path: P) -> Result<Image<B, 3>> {
        read_png_series(path, &self.device)
    }
}

impl<B: Backend> ImageReader<Image<B, 3>> for PngReader<B> {
    fn read<P: AsRef<Path>>(&self, path: P) -> std::io::Result<Image<B, 3>> {
        self.read_image(path)
            .map_err(|e| std::io::Error::other(e.to_string()))
    }
}

impl<B: Backend> ImageReader<Image<B, 3>> for PngSeriesReader<B> {
    fn read<P: AsRef<Path>>(&self, path: P) -> std::io::Result<Image<B, 3>> {
        self.read_image(path)
            .map_err(|e| std::io::Error::other(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::{PngReader, PngSeriesReader};
    use crate::domain::ImageReader;
    use burn_ndarray::NdArray;
    use ritk_core::image::Image;
    use ritk_image::tensor::backend::Backend;
    use std::path::Path;
    use tempfile::tempdir;

    type TestBackend = NdArray<f32>;

    fn write_gray_png(path: &Path, width: u32, height: u32, pixels: &[u8]) {
        let image = image::GrayImage::from_raw(width, height, pixels.to_vec())
            .expect("test image dimensions must match pixel count");
        image.save(path).expect("test PNG write must succeed");
    }

    fn tensor_values(image: &ritk_core::image::Image<TestBackend, 3>) -> Vec<f32> {
        image.data_slice().into_owned()
    }

    #[test]
    fn png_reader_adapter_delegates_to_authoritative_crate() -> anyhow::Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("slice.png");
        write_gray_png(&path, 2, 1, &[9, 10]);

        let device: <TestBackend as Backend>::Device = Default::default();
        let reader = PngReader::<TestBackend>::new(device);
        let image = ImageReader::<Image<TestBackend, 3>>::read(&reader, &path)?;

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
        let image = ImageReader::<Image<TestBackend, 3>>::read(&reader, dir.path())?;

        assert_eq!(image.shape(), [2, 1, 1]);
        assert_eq!(tensor_values(&image), vec![1.0, 2.0]);
        Ok(())
    }
}

/// Atlas-native-substrate implementors of [`crate::domain::ImageReader`].
///
/// Transitional module: names inside are the plain end-state names; the
/// module itself disambiguates from the Burn types during coexistence and
/// folds away when the Burn path is deleted (ADR 0002).
pub mod native {
    use crate::domain::{to_io_err, ImageReader};
    use coeus_core::ComputeBackend;
    use ritk_image::native::Image;
    use std::path::Path;

    /// Backend-bound Atlas-native reader (counterpart of the Burn [`super::PngReader`]).
    pub struct PngReader<B: ComputeBackend> {
        backend: B,
    }

    impl<B: ComputeBackend> PngReader<B> {
        /// Create a reader that constructs images on `backend`.
        pub fn new(backend: B) -> Self {
            Self { backend }
        }
    }

    impl<B: ComputeBackend> ImageReader<Image<f32, B, 3>> for PngReader<B> {
        fn read<P: AsRef<Path>>(&self, path: P) -> std::io::Result<Image<f32, B, 3>> {
            ritk_png::read_png_to_image(path, &self.backend).map_err(to_io_err)
        }
    }

    /// Backend-bound Atlas-native reader (counterpart of the Burn [`super::PngSeriesReader`]).
    pub struct PngSeriesReader<B: ComputeBackend> {
        backend: B,
    }

    impl<B: ComputeBackend> PngSeriesReader<B> {
        /// Create a reader that constructs images on `backend`.
        pub fn new(backend: B) -> Self {
            Self { backend }
        }
    }

    impl<B: ComputeBackend> ImageReader<Image<f32, B, 3>> for PngSeriesReader<B> {
        fn read<P: AsRef<Path>>(&self, path: P) -> std::io::Result<Image<f32, B, 3>> {
            ritk_png::read_png_series(path, &self.backend).map_err(to_io_err)
        }
    }
}
