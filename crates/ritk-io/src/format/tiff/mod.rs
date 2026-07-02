pub use ritk_tiff::{
    read_tiff, read_tiff_color_to_volume, write_tiff, TiffColorReader, TiffReader, TiffWriter,
};

use crate::domain::{ImageReader, ImageWriter};
use burn::tensor::backend::Backend;
use ritk_core::image::Image;
use std::path::Path;

impl<B: Backend> ImageReader<Image<B, 3>> for TiffReader<B> {
    fn read<P: AsRef<Path>>(&self, path: P) -> std::io::Result<Image<B, 3>> {
        self.read_image(path)
            .map_err(|e| std::io::Error::other(e.to_string()))
    }
}

impl<B: Backend> ImageWriter<Image<B, 3>> for TiffWriter {
    fn write<P: AsRef<Path>>(&self, path: P, image: &Image<B, 3>) -> std::io::Result<()> {
        write_tiff(image, path).map_err(|e| std::io::Error::other(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::{TiffReader, TiffWriter};
    use crate::domain::{ImageReader, ImageWriter};
    use burn::tensor::backend::Backend;
    use burn::tensor::{Shape, Tensor, TensorData};
    use burn_ndarray::NdArray;
    use ritk_core::image::Image;
    use ritk_spatial::{Direction, Point, Spacing};
    use tempfile::tempdir;

    type TestBackend = NdArray<f32>;

    fn image_from_values(
        device: &<TestBackend as Backend>::Device,
        shape: [usize; 3],
        values: Vec<f32>,
    ) -> Image<TestBackend, 3> {
        let tensor_data = TensorData::new(values, Shape::new(shape));
        let tensor = Tensor::<TestBackend, 3>::from_data(tensor_data, device);
        Image::new(
            tensor,
            Point::new([0.0, 0.0, 0.0]),
            Spacing::new([1.0, 1.0, 1.0]),
            Direction::identity(),
        )
    }

    #[test]
    fn tiff_reader_writer_adapters_delegate_to_authoritative_crate() -> anyhow::Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("adapter.tiff");
        let device: <TestBackend as Backend>::Device = Default::default();

        // Write via ImageWriter adapter.
        let image = image_from_values(&device, [2, 3, 4], vec![1.5f32; 2 * 3 * 4]);
        let writer = TiffWriter;
        ImageWriter::<Image<TestBackend, 3>>::write(&writer, &path, &image)?;

        // Read via ImageReader adapter.
        let reader = TiffReader::<TestBackend>::new(device);
        let loaded = ImageReader::<Image<TestBackend, 3>>::read(&reader, &path)?;

        assert_eq!(
            loaded.shape(),
            [2, 3, 4],
            "shape must be preserved through adapter round-trip"
        );

        loaded.with_data_slice(|loaded_vals| {
            for (i, &v) in loaded_vals.iter().enumerate() {
                assert!(
                    (v - 1.5).abs() < 1e-6,
                    "voxel[{}]: expected 1.5, got {}",
                    i,
                    v,
                );
            }
        });
        Ok(())
    }
}

/// Atlas-native-substrate implementors of [`crate::domain::ImageReader`].
///
/// Transitional module: names inside are the plain end-state names; the
/// module itself disambiguates from the Burn types during coexistence and
/// folds away when the Burn path is deleted (ADR 0002).
#[cfg(feature = "coeus")]
pub mod native {
    use crate::domain::{to_io_err, ImageReader};
    use coeus_core::ComputeBackend;
    use ritk_image::coeus::Image;
    use std::path::Path;

    /// Backend-bound Atlas-native reader (counterpart of the Burn [`super::TiffReader`]).
    pub struct TiffReader<B: ComputeBackend> {
        backend: B,
    }

    impl<B: ComputeBackend> TiffReader<B> {
        /// Create a reader that constructs images on `backend`.
        pub fn new(backend: B) -> Self {
            Self { backend }
        }
    }

    impl<B: ComputeBackend> ImageReader<Image<f32, B, 3>> for TiffReader<B> {
        fn read<P: AsRef<Path>>(&self, path: P) -> std::io::Result<Image<f32, B, 3>> {
            ritk_tiff::read_tiff_coeus(path, &self.backend).map_err(to_io_err)
        }
    }
}
