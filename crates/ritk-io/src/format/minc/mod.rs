pub use ritk_minc::{read_minc, write_minc, MincReader, MincWriter};

use crate::domain::{ImageReader, ImageWriter};
use ritk_image::tensor::backend::Backend;
use ritk_core::image::Image;
use std::path::Path;

impl<B: Backend> ImageReader<Image<B, 3>> for MincReader<B> {
    fn read<P: AsRef<Path>>(&self, path: P) -> std::io::Result<Image<B, 3>> {
        self.read_image(path)
            .map_err(|e| std::io::Error::other(e.to_string()))
    }
}

impl<B: Backend> ImageWriter<Image<B, 3>> for MincWriter {
    fn write<P: AsRef<Path>>(&self, path: P, image: &Image<B, 3>) -> std::io::Result<()> {
        write_minc(image, path).map_err(|e| std::io::Error::other(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::{MincReader, MincWriter};
    use crate::domain::{ImageReader, ImageWriter};
    use ritk_image::tensor::backend::Backend;
    use ritk_image::tensor::{Shape, Tensor, TensorData};
    use burn_ndarray::NdArray;
    use ritk_core::image::Image;
    use ritk_spatial::{Direction, Point, Spacing};
    use tempfile::tempdir;

    type TestBackend = NdArray<f32>;

    fn make_image(
        device: &<TestBackend as Backend>::Device,
        shape: [usize; 3],
        values: Vec<f32>,
    ) -> Image<TestBackend, 3> {
        let td = TensorData::new(values, Shape::new(shape));
        let tensor = Tensor::<TestBackend, 3>::from_data(td, device);
        Image::new(
            tensor,
            Point::new([0.0, 0.0, 0.0]),
            Spacing::new([1.0, 1.0, 1.0]),
            Direction::identity(),
        )
    }

    #[test]
    fn minc_writer_adapter_produces_hdf5_signature() -> anyhow::Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("adapter.mnc");
        let device: <TestBackend as Backend>::Device = Default::default();

        let image = make_image(&device, [2, 2, 2], vec![1.0f32; 8]);
        let writer = MincWriter;
        ImageWriter::<Image<TestBackend, 3>>::write(&writer, &path, &image)?;

        assert!(path.exists(), "adapter must create file");
        let bytes = std::fs::read(&path)?;
        assert_eq!(
            &bytes[0..8],
            b"\x89HDF\r\n\x1a\n",
            "output must start with HDF5 signature"
        );

        Ok(())
    }

    #[test]
    fn minc_reader_adapter_requires_valid_hdf5() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("bad.mnc");
        std::fs::write(&path, b"not an hdf5 file").unwrap();
        let device: <TestBackend as Backend>::Device = Default::default();
        let reader = MincReader::<TestBackend>::new(device);
        let result = ImageReader::<Image<TestBackend, 3>>::read(&reader, &path);
        assert!(result.is_err(), "reading invalid HDF5 must fail");
    }
}

/// Atlas-native-substrate implementors of [`crate::domain::ImageReader`].
///
/// Transitional module: names inside are the plain end-state names; the
/// module itself disambiguates from the Burn types during coexistence and
/// folds away when the Burn path is deleted (ADR 0002).
pub mod native {
    use crate::domain::{to_io_err, ImageReader, ImageWriter};
    use coeus_core::{ComputeBackend, CpuAddressableStorage};
    use ritk_image::native::Image;
    use std::path::Path;

    /// Backend-bound Atlas-native reader (counterpart of the Burn [`super::MincReader`]).
    pub struct MincReader<B: ComputeBackend> {
        backend: B,
    }

    impl<B: ComputeBackend> MincReader<B> {
        /// Create a reader that constructs images on `backend`.
        pub fn new(backend: B) -> Self {
            Self { backend }
        }
    }

    impl<B: ComputeBackend> ImageReader<Image<f32, B, 3>> for MincReader<B> {
        fn read<P: AsRef<Path>>(&self, path: P) -> std::io::Result<Image<f32, B, 3>> {
            ritk_minc::native::read_minc(path, &self.backend).map_err(to_io_err)
        }
    }

    /// Backend-bound Atlas-native writer (counterpart of the Burn writer).
    pub struct MincWriter<B: ComputeBackend> {
        backend: B,
    }

    impl<B: ComputeBackend> MincWriter<B> {
        /// Create a writer that extracts host data via `backend`.
        pub fn new(backend: B) -> Self {
            Self { backend }
        }
    }

    impl<B> ImageWriter<Image<f32, B, 3>> for MincWriter<B>
    where
        B: ComputeBackend + Default,
        B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
    {
        fn write<P: AsRef<Path>>(&self, path: P, image: &Image<f32, B, 3>) -> std::io::Result<()> {
            ritk_minc::native::write_minc(image, path, &self.backend).map_err(to_io_err)
        }
    }
}
