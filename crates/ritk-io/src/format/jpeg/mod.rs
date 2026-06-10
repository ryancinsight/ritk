pub use ritk_jpeg::{
    read_jpeg, read_jpeg_color_to_volume, write_jpeg, JpegColorReader, JpegReader, JpegWriter,
};

use crate::domain::{ImageReader, ImageWriter};
use burn::tensor::backend::Backend;
use ritk_core::image::Image;
use std::path::Path;

impl<B: Backend> ImageReader<B, 3> for JpegReader<B> {
    fn read<P: AsRef<Path>>(&self, path: P) -> std::io::Result<Image<B, 3>> {
        self.read_image(path)
            .map_err(|e| std::io::Error::other(e.to_string()))
    }
}

impl<B: Backend> ImageWriter<B, 3> for JpegWriter<B> {
    fn write<P: AsRef<Path>>(&self, path: P, image: &Image<B, 3>) -> std::io::Result<()> {
        self.write_image(path, image)
            .map_err(|e| std::io::Error::other(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::{JpegReader, JpegWriter};
    use crate::domain::{ImageReader, ImageWriter};
    use burn::tensor::backend::Backend;
    use burn::tensor::{Shape, Tensor, TensorData};
    use burn_ndarray::NdArray;
    use ritk_core::image::Image;
    use ritk_core::spatial::{Direction, Point, Spacing};
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

    fn tensor_values(image: &Image<TestBackend, 3>) -> Vec<f32> {
        image.data_slice().into_owned()
    }

    #[test]
    fn jpeg_reader_writer_adapters_delegate_to_authoritative_crate() -> anyhow::Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("adapter.jpg");
        let device: <TestBackend as Backend>::Device = Default::default();
        let image = image_from_values(&device, [1, 1, 3], vec![16.0, 128.0, 240.0]);

        let writer = JpegWriter::<TestBackend>::default();
        ImageWriter::<TestBackend, 3>::write(&writer, &path, &image)?;

        let reader = JpegReader::<TestBackend>::new(device);
        let loaded = ImageReader::<TestBackend, 3>::read(&reader, &path)?;

        assert_eq!(loaded.shape(), [1, 1, 3]);
        let values = tensor_values(&loaded);
        assert_eq!(values.len(), 3);
        assert!(values[0] <= 24.0, "expected dark pixel, got {}", values[0]);
        assert!(
            (values[1] - 128.0).abs() <= 12.0,
            "expected mid pixel near 128, got {}",
            values[1]
        );
        assert!(
            values[2] >= 228.0,
            "expected bright pixel, got {}",
            values[2]
        );
        Ok(())
    }
}
