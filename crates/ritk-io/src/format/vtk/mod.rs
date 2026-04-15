//! VTK legacy structured points format support.
//!
//! Provides `read_vtk` / `write_vtk` free functions and `VtkReader` /
//! `VtkWriter` types that implement the `ImageReader` / `ImageWriter`
//! DIP abstractions.
//!
//! ## Format Scope
//!
//! Only `DATASET STRUCTURED_POINTS` with scalar point data is supported.
//! The reader handles both `ASCII` and `BINARY` encodings. The writer
//! always emits `BINARY` with big-endian `float` scalars.

pub mod reader;
pub mod writer;

pub use reader::read_vtk;
pub use writer::write_vtk;

use crate::domain::{ImageReader, ImageWriter};
use burn::tensor::backend::Backend;
use ritk_core::image::Image;
use std::path::Path;

/// DIP boundary implementing `ImageReader` for VTK legacy structured points.
pub struct VtkReader<B: Backend> {
    device: B::Device,
}

impl<B: Backend> VtkReader<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}

impl<B: Backend> ImageReader<B, 3> for VtkReader<B> {
    fn read<P: AsRef<Path>>(&self, path: P) -> std::io::Result<Image<B, 3>> {
        read_vtk(path, &self.device)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))
    }
}

/// DIP boundary implementing `ImageWriter` for VTK legacy structured points.
pub struct VtkWriter<B: Backend> {
    _marker: std::marker::PhantomData<B>,
}

impl<B: Backend> Default for VtkWriter<B> {
    fn default() -> Self {
        Self {
            _marker: std::marker::PhantomData,
        }
    }
}

impl<B: Backend> ImageWriter<B, 3> for VtkWriter<B> {
    fn write<P: AsRef<Path>>(&self, path: P, image: &Image<B, 3>) -> std::io::Result<()> {
        write_vtk(path, image)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Shape, Tensor, TensorData};
    use burn_ndarray::NdArray;
    use ritk_core::spatial::{Direction, Point, Spacing};
    use std::io::Write;
    use tempfile::tempdir;

    type TestBackend = NdArray<f32>;

    // -----------------------------------------------------------------------
    // Helper: build an Image<TestBackend, 3> with analytically derived voxel
    // values. Each voxel at index (z, y, x) gets value z*100 + y*10 + x,
    // providing unique, deterministic, verifiable data across all positions.
    // -----------------------------------------------------------------------
    fn make_test_image(
        nx: usize,
        ny: usize,
        nz: usize,
        origin: [f64; 3],
        spacing: [f64; 3],
    ) -> Image<TestBackend, 3> {
        let total = nx * ny * nz;
        let mut data = Vec::with_capacity(total);
        // Tensor shape [nz, ny, nx], linear index = z*ny*nx + y*nx + x
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    data.push((z * 100 + y * 10 + x) as f32);
                }
            }
        }
        let shape = Shape::new([nz, ny, nx]);
        let tensor_data = TensorData::new(data, shape);
        let device = Default::default();
        let tensor = Tensor::<TestBackend, 3>::from_data(tensor_data, &device);

        Image::new(
            tensor,
            Point::new(origin),
            Spacing::new(spacing),
            Direction::identity(),
        )
    }

    /// Extract the flat f32 vector from an image tensor for value-semantic
    /// comparison.
    fn image_to_vec(image: &Image<TestBackend, 3>) -> Vec<f32> {
        let td = image.data().clone().to_data();
        td.as_slice::<f32>().unwrap().to_vec()
    }

    // =======================================================================
    // 1. Round-trip: write then read, verify exact voxel equality
    // =======================================================================
    #[test]
    fn test_roundtrip_voxel_values() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("roundtrip.vtk");
        let device: <TestBackend as Backend>::Device = Default::default();

        // 4×5×6 image (nx=4, ny=5, nz=6)
        let image = make_test_image(4, 5, 6, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let original_data = image_to_vec(&image);
        assert_eq!(original_data.len(), 4 * 5 * 6);

        write_vtk(&path, &image).unwrap();
        let loaded = read_vtk::<TestBackend, _>(&path, &device).unwrap();
        let loaded_data = image_to_vec(&loaded);

        assert_eq!(loaded.shape(), [6, 5, 4]); // [nz, ny, nx]
        assert_eq!(original_data.len(), loaded_data.len());

        // Exact f32 equality: write emits big-endian f32, read decodes
        // big-endian f32. No precision loss occurs.
        for (i, (&orig, &read)) in original_data.iter().zip(loaded_data.iter()).enumerate() {
            assert_eq!(
                orig, read,
                "voxel mismatch at linear index {}: wrote {}, read {}",
                i, orig, read
            );
        }
    }

    // =======================================================================
    // 2. Spatial metadata preservation across round-trip
    // =======================================================================
    #[test]
    fn test_roundtrip_spatial_metadata() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("spatial.vtk");
        let device: <TestBackend as Backend>::Device = Default::default();

        let origin_vals = [10.5, -20.25, 30.125];
        let spacing_vals = [0.5, 0.75, 1.25];
        let image = make_test_image(3, 4, 5, origin_vals, spacing_vals);

        write_vtk(&path, &image).unwrap();
        let loaded = read_vtk::<TestBackend, _>(&path, &device).unwrap();

        let lo = loaded.origin();
        let ls = loaded.spacing();
        let ld = loaded.direction();

        let eps = 1e-9;

        // Origin [X, Y, Z]
        assert!(
            (lo[0] - origin_vals[0]).abs() < eps,
            "origin X: expected {}, got {}",
            origin_vals[0],
            lo[0]
        );
        assert!(
            (lo[1] - origin_vals[1]).abs() < eps,
            "origin Y: expected {}, got {}",
            origin_vals[1],
            lo[1]
        );
        assert!(
            (lo[2] - origin_vals[2]).abs() < eps,
            "origin Z: expected {}, got {}",
            origin_vals[2],
            lo[2]
        );

        // Spacing [X, Y, Z]
        assert!(
            (ls[0] - spacing_vals[0]).abs() < eps,
            "spacing X: expected {}, got {}",
            spacing_vals[0],
            ls[0]
        );
        assert!(
            (ls[1] - spacing_vals[1]).abs() < eps,
            "spacing Y: expected {}, got {}",
            spacing_vals[1],
            ls[1]
        );
        assert!(
            (ls[2] - spacing_vals[2]).abs() < eps,
            "spacing Z: expected {}, got {}",
            spacing_vals[2],
            ls[2]
        );

        // Direction must be identity (VTK structured points has no direction)
        assert_eq!(*ld, Direction::identity());
    }

    // =======================================================================
    // 3. ASCII read test
    // =======================================================================
    #[test]
    fn test_ascii_read() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("ascii.vtk");
        let device: <TestBackend as Backend>::Device = Default::default();

        // Construct a valid ASCII VTK file for a 2×3×2 image (nx=2, ny=3, nz=2)
        let nx = 2usize;
        let ny = 3usize;
        let nz = 2usize;
        let total = nx * ny * nz; // 12

        let mut content = String::new();
        content.push_str("# vtk DataFile Version 2.0\n");
        content.push_str("Test ASCII VTK\n");
        content.push_str("ASCII\n");
        content.push_str("DATASET STRUCTURED_POINTS\n");
        content.push_str(&format!("DIMENSIONS {} {} {}\n", nx, ny, nz));
        content.push_str("ORIGIN 1.0 2.0 3.0\n");
        content.push_str("SPACING 0.5 0.5 0.5\n");
        content.push_str(&format!("POINT_DATA {}\n", total));
        content.push_str("SCALARS intensity float 1\n");
        content.push_str("LOOKUP_TABLE default\n");

        // Write values: z*100 + y*10 + x per the same scheme as make_test_image
        let mut values = Vec::with_capacity(total);
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    values.push(format!("{}", (z * 100 + y * 10 + x) as f32));
                }
            }
        }
        // Write 4 values per line to also test multi-token line parsing
        for chunk in values.chunks(4) {
            content.push_str(&chunk.join(" "));
            content.push('\n');
        }

        std::fs::write(&path, &content).unwrap();

        let loaded = read_vtk::<TestBackend, _>(&path, &device).unwrap();

        assert_eq!(loaded.shape(), [nz, ny, nx]);

        let lo = loaded.origin();
        let ls = loaded.spacing();
        let eps = 1e-9;
        assert!((lo[0] - 1.0).abs() < eps);
        assert!((lo[1] - 2.0).abs() < eps);
        assert!((lo[2] - 3.0).abs() < eps);
        assert!((ls[0] - 0.5).abs() < eps);
        assert!((ls[1] - 0.5).abs() < eps);
        assert!((ls[2] - 0.5).abs() < eps);

        // Verify each voxel value
        let data = image_to_vec(&loaded);
        assert_eq!(data.len(), total);
        let mut idx = 0;
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let expected = (z * 100 + y * 10 + x) as f32;
                    assert_eq!(
                        data[idx], expected,
                        "ASCII voxel mismatch at ({},{},{}): got {}, expected {}",
                        x, y, z, data[idx], expected
                    );
                    idx += 1;
                }
            }
        }
    }

    // =======================================================================
    // 4a. Error: empty file
    // =======================================================================
    #[test]
    fn test_error_empty_file() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("empty.vtk");
        let device: <TestBackend as Backend>::Device = Default::default();

        std::fs::File::create(&path).unwrap();

        let result = read_vtk::<TestBackend, _>(&path, &device);
        assert!(result.is_err());
        let err_msg = format!("{:#}", result.unwrap_err());
        assert!(
            err_msg.contains("EOF") || err_msg.contains("header") || err_msg.contains("version"),
            "error message should mention header/version/EOF, got: {}",
            err_msg
        );
    }

    // =======================================================================
    // 4b. Error: wrong dataset type
    // =======================================================================
    #[test]
    fn test_error_wrong_dataset_type() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("polydata.vtk");
        let device: <TestBackend as Backend>::Device = Default::default();

        let content = "\
# vtk DataFile Version 3.0
wrong dataset test
ASCII
DATASET POLYDATA
";
        std::fs::write(&path, content).unwrap();

        let result = read_vtk::<TestBackend, _>(&path, &device);
        assert!(result.is_err());
        let err_msg = format!("{:#}", result.unwrap_err());
        assert!(
            err_msg.contains("STRUCTURED_POINTS") || err_msg.contains("unsupported"),
            "error should mention STRUCTURED_POINTS, got: {}",
            err_msg
        );
    }

    // =======================================================================
    // 4c. Error: truncated binary data
    // =======================================================================
    #[test]
    fn test_error_truncated_binary() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("truncated.vtk");
        let device: <TestBackend as Backend>::Device = Default::default();

        let mut file = std::fs::File::create(&path).unwrap();
        // Write a valid header for a 2×2×2 = 8 voxels image but supply
        // only 2 float values (8 bytes instead of 32).
        write!(file, "# vtk DataFile Version 3.0\n").unwrap();
        write!(file, "truncated test\n").unwrap();
        write!(file, "BINARY\n").unwrap();
        write!(file, "DATASET STRUCTURED_POINTS\n").unwrap();
        write!(file, "DIMENSIONS 2 2 2\n").unwrap();
        write!(file, "ORIGIN 0 0 0\n").unwrap();
        write!(file, "SPACING 1 1 1\n").unwrap();
        write!(file, "POINT_DATA 8\n").unwrap();
        write!(file, "SCALARS scalars float 1\n").unwrap();
        write!(file, "LOOKUP_TABLE default\n").unwrap();
        // Only 8 bytes = 2 floats instead of 32 bytes = 8 floats
        let v1: f32 = 1.0;
        let v2: f32 = 2.0;
        file.write_all(&v1.to_be_bytes()).unwrap();
        file.write_all(&v2.to_be_bytes()).unwrap();
        file.flush().unwrap();
        drop(file);

        let result = read_vtk::<TestBackend, _>(&path, &device);
        assert!(result.is_err());
        let err_msg = format!("{:#}", result.unwrap_err());
        assert!(
            err_msg.contains("binary") || err_msg.contains("read") || err_msg.contains("bytes"),
            "error should mention binary data reading failure, got: {}",
            err_msg
        );
    }

    // =======================================================================
    // 4d. Error: non-existent file
    // =======================================================================
    #[test]
    fn test_error_nonexistent_file() {
        let device: <TestBackend as Backend>::Device = Default::default();
        let result = read_vtk::<TestBackend, _>("/tmp/nonexistent_vtk_file_ritk_test.vtk", &device);
        assert!(result.is_err());
    }

    // =======================================================================
    // 5. Boundary: single-voxel image (1×1×1)
    // =======================================================================
    #[test]
    fn test_single_voxel_roundtrip() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("single.vtk");
        let device: <TestBackend as Backend>::Device = Default::default();

        let origin_vals = [-5.0, 10.0, 15.5];
        let spacing_vals = [2.0, 3.0, 4.0];
        let voxel_value: f32 = 42.5;

        let shape = Shape::new([1, 1, 1]); // [nz, ny, nx]
        let tensor_data = TensorData::new(vec![voxel_value], shape);
        let tensor = Tensor::<TestBackend, 3>::from_data(tensor_data, &device);
        let image = Image::new(
            tensor,
            Point::new(origin_vals),
            Spacing::new(spacing_vals),
            Direction::identity(),
        );

        write_vtk(&path, &image).unwrap();
        let loaded = read_vtk::<TestBackend, _>(&path, &device).unwrap();

        assert_eq!(loaded.shape(), [1, 1, 1]);

        let data = image_to_vec(&loaded);
        assert_eq!(data.len(), 1);
        assert_eq!(data[0], voxel_value, "single voxel value mismatch");

        let lo = loaded.origin();
        let ls = loaded.spacing();
        let eps = 1e-9;
        assert!((lo[0] - origin_vals[0]).abs() < eps);
        assert!((lo[1] - origin_vals[1]).abs() < eps);
        assert!((lo[2] - origin_vals[2]).abs() < eps);
        assert!((ls[0] - spacing_vals[0]).abs() < eps);
        assert!((ls[1] - spacing_vals[1]).abs() < eps);
        assert!((ls[2] - spacing_vals[2]).abs() < eps);
    }

    // =======================================================================
    // 6. DIP trait wiring: VtkReader / VtkWriter compile and execute
    // =======================================================================
    #[test]
    fn test_dip_trait_roundtrip() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("dip.vtk");
        let device: <TestBackend as Backend>::Device = Default::default();

        let image = make_test_image(3, 3, 3, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let original_data = image_to_vec(&image);

        let writer = VtkWriter::<TestBackend>::default();
        writer.write(&path, &image).unwrap();

        let reader = VtkReader::<TestBackend>::new(device);
        let loaded = reader.read(&path).unwrap();
        let loaded_data = image_to_vec(&loaded);

        assert_eq!(original_data, loaded_data);
    }

    // =======================================================================
    // 7. ASCII file with blank lines and different version number
    // =======================================================================
    #[test]
    fn test_ascii_with_blank_lines_and_version_5() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("blanks.vtk");
        let device: <TestBackend as Backend>::Device = Default::default();

        // Version 5.1, blank lines interspersed
        let content = "\
# vtk DataFile Version 5.1
blank line stress test

ASCII

DATASET STRUCTURED_POINTS

DIMENSIONS 2 2 1
ORIGIN 0 0 0

SPACING 1 1 1

POINT_DATA 4
SCALARS data float 1

LOOKUP_TABLE default
10.0 20.0
30.0 40.0
";
        std::fs::write(&path, content).unwrap();

        let loaded = read_vtk::<TestBackend, _>(&path, &device).unwrap();
        assert_eq!(loaded.shape(), [1, 2, 2]);

        let data = image_to_vec(&loaded);
        assert_eq!(data.len(), 4);
        assert_eq!(data[0], 10.0);
        assert_eq!(data[1], 20.0);
        assert_eq!(data[2], 30.0);
        assert_eq!(data[3], 40.0);
    }

    // =======================================================================
    // 8. ASPECT_RATIO synonym for SPACING (legacy VTK 1.x)
    // =======================================================================
    #[test]
    fn test_aspect_ratio_synonym() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("aspect.vtk");
        let device: <TestBackend as Backend>::Device = Default::default();

        let content = "\
# vtk DataFile Version 1.0
aspect ratio test
ASCII
DATASET STRUCTURED_POINTS
DIMENSIONS 1 1 1
ORIGIN 0 0 0
ASPECT_RATIO 2.0 3.0 4.0
POINT_DATA 1
SCALARS s float 1
LOOKUP_TABLE default
7.0
";
        std::fs::write(&path, content).unwrap();

        let loaded = read_vtk::<TestBackend, _>(&path, &device).unwrap();
        let ls = loaded.spacing();
        let eps = 1e-9;
        assert!((ls[0] - 2.0).abs() < eps);
        assert!((ls[1] - 3.0).abs() < eps);
        assert!((ls[2] - 4.0).abs() < eps);

        let data = image_to_vec(&loaded);
        assert_eq!(data[0], 7.0);
    }

    // =======================================================================
    // 9. Negative and fractional voxel values preserved exactly
    // =======================================================================
    #[test]
    fn test_negative_and_fractional_values() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("negfrac.vtk");
        let device: <TestBackend as Backend>::Device = Default::default();

        let values: Vec<f32> = vec![
            -1.0,
            0.0,
            1.0,
            -0.5,
            std::f32::consts::PI,
            std::f32::consts::E,
            -1000.125,
            999.875,
        ];
        let shape = Shape::new([2, 2, 2]); // nz=2, ny=2, nx=2
        let tensor_data = TensorData::new(values.clone(), shape);
        let tensor = Tensor::<TestBackend, 3>::from_data(tensor_data, &device);
        let image = Image::new(
            tensor,
            Point::new([0.0, 0.0, 0.0]),
            Spacing::new([1.0, 1.0, 1.0]),
            Direction::identity(),
        );

        write_vtk(&path, &image).unwrap();
        let loaded = read_vtk::<TestBackend, _>(&path, &device).unwrap();
        let loaded_data = image_to_vec(&loaded);

        assert_eq!(values.len(), loaded_data.len());
        for (i, (&orig, &read)) in values.iter().zip(loaded_data.iter()).enumerate() {
            assert_eq!(
                orig.to_bits(),
                read.to_bits(),
                "bit-exact mismatch at index {}: {} vs {}",
                i,
                orig,
                read
            );
        }
    }

    // =======================================================================
    // 10. POINT_DATA / DIMENSIONS mismatch is rejected
    // =======================================================================
    #[test]
    fn test_error_point_data_dimension_mismatch() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("mismatch.vtk");
        let device: <TestBackend as Backend>::Device = Default::default();

        let content = "\
# vtk DataFile Version 3.0
mismatch test
ASCII
DATASET STRUCTURED_POINTS
DIMENSIONS 3 3 3
ORIGIN 0 0 0
SPACING 1 1 1
POINT_DATA 10
SCALARS s float 1
LOOKUP_TABLE default
1 2 3 4 5 6 7 8 9 10
";
        std::fs::write(&path, content).unwrap();

        let result = read_vtk::<TestBackend, _>(&path, &device);
        assert!(result.is_err());
        let err_msg = format!("{:#}", result.unwrap_err());
        assert!(
            err_msg.contains("POINT_DATA") || err_msg.contains("DIMENSIONS"),
            "error should mention count mismatch, got: {}",
            err_msg
        );
    }
}
