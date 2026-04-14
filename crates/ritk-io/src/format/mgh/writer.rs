//! MGH / MGZ writer for 3-D volumetric images.
//!
//! Writes an `Image<B, 3>` as an MGH or MGZ file.  The output data type is
//! always `MRI_FLOAT` (type code 3, IEEE 754 big-endian f32).  Gzip
//! compression is applied when the output path has extension `.mgz` or
//! `.mgh.gz`.
//!
//! # Header
//!
//! The 284-byte big-endian header is populated as follows:
//!
//! | Offset | Type   | Value                                         |
//! |--------|--------|-----------------------------------------------|
//! | 0      | i32    | version = 1                                   |
//! | 4      | i32    | width = nx  (tensor axis 2)                   |
//! | 8      | i32    | height = ny (tensor axis 1)                   |
//! | 12     | i32    | depth = nz  (tensor axis 0)                   |
//! | 16     | i32    | nframes = 1                                   |
//! | 20     | i32    | type = 3 (MRI\_FLOAT)                         |
//! | 24     | i32    | dof = 0                                       |
//! | 28     | i16    | goodRASFlag = 1                               |
//! | 30     | f32×3  | spacing\[0\], spacing\[1\], spacing\[2\]     |
//! | 42     | f32×9  | direction cosines (column-major)              |
//! | 78     | f32×3  | c\_r, c\_a, c\_s (RAS center)                 |
//! | 90     | —      | zero padding to byte 284                      |
//!
//! # RAS Center Derivation
//!
//! ```text
//! Mdc   = Direction matrix (3×3, columns are axis direction cosines)
//! D     = diag(spacing[0], spacing[1], spacing[2])
//! h     = [(nx−1)/2, (ny−1)/2, (nz−1)/2]^T
//! c_ras = origin + Mdc · D · h
//! ```
//!
//! This is the inverse of the origin derivation used by the reader:
//! `origin = c_ras − Mdc · D · h`.
//!
//! # Data Section
//!
//! Voxel data is written as big-endian f32 values starting at byte 284.
//! The RITK tensor `[nz, ny, nx]` row-major layout maps directly to the
//! MGH Fortran (column-major) layout without permutation because
//! `z·ny·nx + y·nx + x ≡ x + y·nx + z·nx·ny`.

use super::{is_gzip_path, MRI_FLOAT, PADDING_LEN, VERSION};
use anyhow::{anyhow, Context, Result};
use burn::tensor::backend::Backend;
use flate2::write::GzEncoder;
use flate2::Compression;
use ritk_core::image::Image;
use std::io::{BufWriter, Write};
use std::path::Path;

// ── Big-endian primitive write helpers ────────────────────────────────────────

/// Write a big-endian `i32` to `w`.
fn write_i32_be<W: Write>(w: &mut W, val: i32) -> Result<()> {
    w.write_all(&val.to_be_bytes())
        .context("Failed to write i32 BE")
}

/// Write a big-endian `i16` to `w`.
fn write_i16_be<W: Write>(w: &mut W, val: i16) -> Result<()> {
    w.write_all(&val.to_be_bytes())
        .context("Failed to write i16 BE")
}

/// Write a big-endian `f32` to `w`.
fn write_f32_be<W: Write>(w: &mut W, val: f32) -> Result<()> {
    w.write_all(&val.to_be_bytes())
        .context("Failed to write f32 BE")
}

// ── Public API ───────────────────────────────────────────────────────────────

/// Write a 3-D `Image` as an MGH or MGZ file.
///
/// # Gzip compression
///
/// If `path` ends in `.mgz` or `.mgh.gz`, the entire output stream is
/// gzip-compressed with default compression level.  Otherwise
/// uncompressed MGH is written.
///
/// # Errors
///
/// Returns `Err` when:
/// - The file cannot be created.
/// - Tensor data cannot be extracted as `f32`.
/// - An I/O error occurs during writing.
pub fn write_mgh<B: Backend, P: AsRef<Path>>(image: &Image<B, 3>, path: P) -> Result<()> {
    let path = path.as_ref();
    let file = std::fs::File::create(path)
        .with_context(|| format!("Cannot create MGH/MGZ file {:?}", path))?;

    if is_gzip_path(path) {
        let mut encoder = GzEncoder::new(BufWriter::new(file), Compression::default());
        write_mgh_to_writer(image, &mut encoder)?;
        encoder.finish().context("Failed to finalize gzip stream")?;
    } else {
        let mut writer = BufWriter::new(file);
        write_mgh_to_writer(image, &mut writer)?;
        writer.flush().context("Failed to flush MGH output")?;
    }
    Ok(())
}

// ── Core writer ──────────────────────────────────────────────────────────────

/// Write the MGH binary stream (header + data) to any `Write` sink.
///
/// # Algorithm
///
/// 1. Extract `[nz, ny, nx]` from the tensor shape and map to
///    `width = nx`, `height = ny`, `depth = nz`.
/// 2. Compute the RAS center from origin, direction, and spacing.
/// 3. Emit the 284-byte big-endian header.
/// 4. Emit voxel data as big-endian f32.
fn write_mgh_to_writer<B: Backend, W: Write>(image: &Image<B, 3>, writer: &mut W) -> Result<()> {
    let shape = image.shape(); // [nz, ny, nx]
    let nz = shape[0];
    let ny = shape[1];
    let nx = shape[2];

    // ── Header (bytes 0–283) ──────────────────────────────────────────────
    write_i32_be(writer, VERSION)?; // version
    write_i32_be(writer, nx as i32)?; // width
    write_i32_be(writer, ny as i32)?; // height
    write_i32_be(writer, nz as i32)?; // depth
    write_i32_be(writer, 1)?; // nframes
    write_i32_be(writer, MRI_FLOAT)?; // type
    write_i32_be(writer, 0)?; // dof
    write_i16_be(writer, 1)?; // goodRASFlag

    // Spacing
    let spacing = image.spacing();
    write_f32_be(writer, spacing[0] as f32)?; // spacingX
    write_f32_be(writer, spacing[1] as f32)?; // spacingY
    write_f32_be(writer, spacing[2] as f32)?; // spacingZ

    // Direction cosines — column-major: x column, then y column, then z column.
    // Direction<3>.0 is an SMatrix<f64, 3, 3> where column i is the direction
    // of the i-th image axis in RAS physical space.
    let dir = image.direction().0;
    for col in 0..3 {
        for row in 0..3 {
            write_f32_be(writer, dir[(row, col)] as f32)?;
        }
    }

    // RAS center: c_ras = origin + Mdc · D · h
    //
    // Where:
    //   Mdc = direction matrix (3×3)
    //   D   = diag(spacing[0], spacing[1], spacing[2])
    //   h   = [(nx−1)/2, (ny−1)/2, (nz−1)/2]^T
    let origin = image.origin();
    let half_dim = nalgebra::Vector3::new(
        (nx as f64 - 1.0) / 2.0,
        (ny as f64 - 1.0) / 2.0,
        (nz as f64 - 1.0) / 2.0,
    );
    let scaled_half = nalgebra::Vector3::new(
        spacing[0] * half_dim[0],
        spacing[1] * half_dim[1],
        spacing[2] * half_dim[2],
    );
    let offset = dir * scaled_half;
    let c_ras = nalgebra::Vector3::new(origin[0], origin[1], origin[2]) + offset;

    write_f32_be(writer, c_ras[0] as f32)?;
    write_f32_be(writer, c_ras[1] as f32)?;
    write_f32_be(writer, c_ras[2] as f32)?;

    // Padding to 284 bytes (90 bytes written so far).
    writer
        .write_all(&[0u8; PADDING_LEN])
        .context("Failed to write MGH header padding")?;

    // ── Voxel data (big-endian f32) ───────────────────────────────────────
    let tensor_data = image.data().clone().to_data();
    let f32_slice = tensor_data
        .as_slice::<f32>()
        .map_err(|e| anyhow!("Failed to extract f32 slice from tensor: {:?}", e))?;

    let n_voxels = nx * ny * nz;
    if f32_slice.len() != n_voxels {
        return Err(anyhow!(
            "Tensor data length {} does not match shape [{}, {}, {}] = {} voxels",
            f32_slice.len(),
            nz,
            ny,
            nx,
            n_voxels
        ));
    }

    // Bulk-convert to big-endian byte buffer for a single write syscall.
    let mut data_buf = vec![0u8; n_voxels * 4];
    for (i, &val) in f32_slice.iter().enumerate() {
        let off = i * 4;
        data_buf[off..off + 4].copy_from_slice(&val.to_be_bytes());
    }
    writer
        .write_all(&data_buf)
        .context("Failed to write MGH voxel data")?;

    Ok(())
}

// ── Public writer struct ─────────────────────────────────────────────────────

/// Stateless writer for MGH / MGZ files.
///
/// Provides a struct-based API that delegates to [`write_mgh`].
pub struct MghWriter;

impl MghWriter {
    /// Write `image` to the MGH or MGZ file at `path`.
    ///
    /// See [`write_mgh`] for full documentation.
    pub fn write<B: Backend, P: AsRef<Path>>(image: &Image<B, 3>, path: P) -> Result<()> {
        write_mgh(image, path)
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::super::HEADER_SIZE;
    use super::*;
    use crate::format::mgh::read_mgh;
    use burn::tensor::backend::Backend;
    use burn::tensor::{Shape, Tensor, TensorData};
    use burn_ndarray::NdArray;
    use ritk_core::image::Image;
    use ritk_core::spatial::{Direction, Point, Spacing};
    use tempfile::tempdir;

    type TestBackend = NdArray<f32>;

    /// Build an `Image<TestBackend, 3>` with default spatial metadata.
    fn make_image(data: Vec<f32>, nz: usize, ny: usize, nx: usize) -> Image<TestBackend, 3> {
        let device: <TestBackend as Backend>::Device = Default::default();
        let td = TensorData::new(data, Shape::new([nz, ny, nx]));
        let tensor = Tensor::<TestBackend, 3>::from_data(td, &device);
        Image::new(
            tensor,
            Point::new([0.0, 0.0, 0.0]),
            Spacing::new([1.0, 1.0, 1.0]),
            Direction::identity(),
        )
    }

    /// Build an `Image<TestBackend, 3>` with explicit spatial metadata.
    fn make_image_with_spatial(
        data: Vec<f32>,
        nz: usize,
        ny: usize,
        nx: usize,
        origin: Point<3>,
        spacing: Spacing<3>,
        direction: Direction<3>,
    ) -> Image<TestBackend, 3> {
        let device: <TestBackend as Backend>::Device = Default::default();
        let td = TensorData::new(data, Shape::new([nz, ny, nx]));
        let tensor = Tensor::<TestBackend, 3>::from_data(td, &device);
        Image::new(tensor, origin, spacing, direction)
    }

    /// Build a complete MGH file as a byte vector for data-type reader tests.
    ///
    /// Identical to the helper in `reader::tests` — duplicated here so that
    /// `writer::tests` is self-contained.
    fn build_mgh_bytes(
        version: i32,
        dims: [i32; 3],
        mri_type: i32,
        spacing: [f32; 3],
        dir_cols: [[f32; 3]; 3],
        c_ras: [f32; 3],
        data: &[u8],
    ) -> Vec<u8> {
        let mut buf = Vec::with_capacity(HEADER_SIZE + data.len());
        buf.extend_from_slice(&version.to_be_bytes());
        buf.extend_from_slice(&dims[0].to_be_bytes());
        buf.extend_from_slice(&dims[1].to_be_bytes());
        buf.extend_from_slice(&dims[2].to_be_bytes());
        buf.extend_from_slice(&1_i32.to_be_bytes()); // nframes
        buf.extend_from_slice(&mri_type.to_be_bytes());
        buf.extend_from_slice(&0_i32.to_be_bytes()); // dof
        buf.extend_from_slice(&1_i16.to_be_bytes()); // goodRASFlag
        for &s in &spacing {
            buf.extend_from_slice(&s.to_be_bytes());
        }
        for col in &dir_cols {
            for &v in col {
                buf.extend_from_slice(&v.to_be_bytes());
            }
        }
        for &c in &c_ras {
            buf.extend_from_slice(&c.to_be_bytes());
        }
        debug_assert_eq!(buf.len(), 90);
        buf.resize(HEADER_SIZE, 0u8);
        buf.extend_from_slice(data);
        buf
    }

    const IDENTITY_DIR: [[f32; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

    // ── Round-trip: f32 values ────────────────────────────────────────────

    /// Write known f32 values, read back, and verify every voxel is
    /// bit-identical.
    #[test]
    fn test_round_trip_f32_values() -> Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("rt_f32.mgh");
        let device: <TestBackend as Backend>::Device = Default::default();

        let nz = 2usize;
        let ny = 3usize;
        let nx = 5usize;
        // Voxel i = i · e / 13.  Transcendental scaling prevents accidental
        // alignment to simple floating-point representations.
        let data_vec: Vec<f32> = (0..(nz * ny * nx) as u32)
            .map(|i| (i as f32) * std::f32::consts::E / 13.0)
            .collect();

        let image = make_image(data_vec.clone(), nz, ny, nx);
        write_mgh(&image, &path)?;

        let loaded = read_mgh::<TestBackend, _>(&path, &device)?;
        assert_eq!(loaded.shape(), [nz, ny, nx]);

        let td = loaded.data().clone().to_data();
        let loaded_vals = td.as_slice::<f32>().unwrap();
        assert_eq!(loaded_vals.len(), data_vec.len());
        for (i, (&got, &exp)) in loaded_vals.iter().zip(data_vec.iter()).enumerate() {
            assert_eq!(
                got.to_bits(),
                exp.to_bits(),
                "voxel[{}]: expected {} ({:#010x}), got {} ({:#010x})",
                i,
                exp,
                exp.to_bits(),
                got,
                got.to_bits()
            );
        }
        Ok(())
    }

    // ── MGZ compression round-trip ───────────────────────────────────────

    /// Write as `.mgz`, verify gzip magic bytes, read back, and compare.
    #[test]
    fn test_round_trip_mgz() -> Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("rt.mgz");
        let device: <TestBackend as Backend>::Device = Default::default();

        let nz = 3usize;
        let ny = 4usize;
        let nx = 2usize;
        let data_vec: Vec<f32> = (0..(nz * ny * nx) as u32)
            .map(|i| (i as f32).sqrt() + 0.5)
            .collect();

        let image = make_image(data_vec.clone(), nz, ny, nx);
        write_mgh(&image, &path)?;

        // Verify file begins with gzip magic 0x1f 0x8b.
        let file_bytes = std::fs::read(&path)?;
        assert_eq!(file_bytes[0], 0x1f, "First byte must be gzip magic 0x1f");
        assert_eq!(file_bytes[1], 0x8b, "Second byte must be gzip magic 0x8b");

        let loaded = read_mgh::<TestBackend, _>(&path, &device)?;
        assert_eq!(loaded.shape(), [nz, ny, nx]);

        let td = loaded.data().clone().to_data();
        let loaded_vals = td.as_slice::<f32>().unwrap();
        assert_eq!(loaded_vals.len(), data_vec.len());
        for (i, (&got, &exp)) in loaded_vals.iter().zip(data_vec.iter()).enumerate() {
            assert_eq!(
                got.to_bits(),
                exp.to_bits(),
                "voxel[{}]: expected {}, got {}",
                i,
                exp,
                got
            );
        }
        Ok(())
    }

    // ── .mgh.gz extension ────────────────────────────────────────────────

    /// The `.mgh.gz` extension must also trigger gzip compression.
    #[test]
    fn test_round_trip_mgh_gz_extension() -> Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("test.mgh.gz");
        let device: <TestBackend as Backend>::Device = Default::default();

        let data_vec: Vec<f32> = (0..8u32).map(|i| i as f32 * 2.5).collect();
        let image = make_image(data_vec.clone(), 2, 2, 2);
        write_mgh(&image, &path)?;

        // Confirm gzip magic.
        let bytes = std::fs::read(&path)?;
        assert_eq!(bytes[0], 0x1f);
        assert_eq!(bytes[1], 0x8b);

        let loaded = read_mgh::<TestBackend, _>(&path, &device)?;
        let td = loaded.data().clone().to_data();
        let loaded_vals = td.as_slice::<f32>().unwrap();
        for (i, (&got, &exp)) in loaded_vals.iter().zip(data_vec.iter()).enumerate() {
            assert_eq!(got.to_bits(), exp.to_bits(), "voxel[{}]", i);
        }
        Ok(())
    }

    // ── Non-default spatial metadata round-trip ──────────────────────────

    /// Write with a 90° rotation around Z and anisotropic spacing, read
    /// back, and verify spatial metadata against analytically derived
    /// reference values.
    ///
    /// # Analytical derivation (all values exact in f32)
    ///
    /// ```text
    /// Mdc = | 0  -1  0 |    spacing = [0.5, 0.75, 1.25]
    ///       | 1   0  0 |    dims    = [nx=4, ny=3, nz=2]
    ///       | 0   0  1 |    origin  = [10.75, 19.25, 29.375]
    ///
    /// h      = [1.5, 1.0, 0.5]
    /// D·h    = [0.75, 0.75, 0.625]
    /// Mdc·D·h = [−0.75, 0.75, 0.625]
    ///
    /// c_ras = origin + Mdc·D·h = [10.0, 20.0, 30.0]
    ///
    /// Re-derived origin = c_ras − Mdc·D·h = [10.75, 19.25, 29.375]  ✓
    /// ```
    #[test]
    fn test_round_trip_nondefault_spatial() -> Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("spatial_rt.mgh");
        let device: <TestBackend as Backend>::Device = Default::default();

        let nz = 2usize;
        let ny = 3usize;
        let nx = 4usize;

        // 90° rotation around Z — every matrix entry exact in f32/f64.
        let mut dir_mat = nalgebra::SMatrix::<f64, 3, 3>::zeros();
        dir_mat[(0, 0)] = 0.0;
        dir_mat[(0, 1)] = -1.0;
        dir_mat[(0, 2)] = 0.0;
        dir_mat[(1, 0)] = 1.0;
        dir_mat[(1, 1)] = 0.0;
        dir_mat[(1, 2)] = 0.0;
        dir_mat[(2, 0)] = 0.0;
        dir_mat[(2, 1)] = 0.0;
        dir_mat[(2, 2)] = 1.0;
        let direction = Direction(dir_mat);
        let spacing = Spacing::new([0.5, 0.75, 1.25]);
        let origin = Point::new([10.75, 19.25, 29.375]);

        let n = nz * ny * nx;
        let data_vec: Vec<f32> = (0..n as u32).map(|i| i as f32 * 0.1 + 1.0).collect();

        let image =
            make_image_with_spatial(data_vec.clone(), nz, ny, nx, origin, spacing, direction);

        write_mgh(&image, &path)?;
        let loaded = read_mgh::<TestBackend, _>(&path, &device)?;
        assert_eq!(loaded.shape(), [nz, ny, nx]);

        // ── Spacing ───────────────────────────────────────────────────────
        let sp = loaded.spacing();
        assert!(
            (sp[0] - 0.5).abs() < 1e-6,
            "spacing[0]: expected 0.5, got {}",
            sp[0]
        );
        assert!(
            (sp[1] - 0.75).abs() < 1e-6,
            "spacing[1]: expected 0.75, got {}",
            sp[1]
        );
        assert!(
            (sp[2] - 1.25).abs() < 1e-6,
            "spacing[2]: expected 1.25, got {}",
            sp[2]
        );

        // ── Direction ─────────────────────────────────────────────────────
        let d = loaded.direction();
        assert!((d[(0, 0)] - 0.0).abs() < 1e-6, "d(0,0)={}", d[(0, 0)]);
        assert!((d[(1, 0)] - 1.0).abs() < 1e-6, "d(1,0)={}", d[(1, 0)]);
        assert!((d[(2, 0)] - 0.0).abs() < 1e-6, "d(2,0)={}", d[(2, 0)]);
        assert!((d[(0, 1)] - (-1.0)).abs() < 1e-6, "d(0,1)={}", d[(0, 1)]);
        assert!((d[(1, 1)] - 0.0).abs() < 1e-6, "d(1,1)={}", d[(1, 1)]);
        assert!((d[(2, 1)] - 0.0).abs() < 1e-6, "d(2,1)={}", d[(2, 1)]);
        assert!((d[(0, 2)] - 0.0).abs() < 1e-6, "d(0,2)={}", d[(0, 2)]);
        assert!((d[(1, 2)] - 0.0).abs() < 1e-6, "d(1,2)={}", d[(1, 2)]);
        assert!((d[(2, 2)] - 1.0).abs() < 1e-6, "d(2,2)={}", d[(2, 2)]);

        // ── Origin ────────────────────────────────────────────────────────
        let o = loaded.origin();
        assert!(
            (o[0] - 10.75).abs() < 1e-5,
            "origin[0]: expected 10.75, got {}",
            o[0]
        );
        assert!(
            (o[1] - 19.25).abs() < 1e-5,
            "origin[1]: expected 19.25, got {}",
            o[1]
        );
        assert!(
            (o[2] - 29.375).abs() < 1e-5,
            "origin[2]: expected 29.375, got {}",
            o[2]
        );

        // ── Voxel data ───────────────────────────────────────────────────
        let td = loaded.data().clone().to_data();
        let loaded_vals = td.as_slice::<f32>().unwrap();
        assert_eq!(loaded_vals.len(), data_vec.len());
        for (i, (&got, &exp)) in loaded_vals.iter().zip(data_vec.iter()).enumerate() {
            assert_eq!(
                got.to_bits(),
                exp.to_bits(),
                "voxel[{}]: expected {}, got {}",
                i,
                exp,
                got
            );
        }
        Ok(())
    }

    // ── MghWriter struct delegates ───────────────────────────────────────

    #[test]
    fn test_writer_struct_delegates() -> Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("struct.mgh");
        let device: <TestBackend as Backend>::Device = Default::default();

        let image = make_image(vec![1.0f32; 2 * 2 * 2], 2, 2, 2);
        MghWriter::write(&image, &path)?;

        assert!(path.exists(), "Output file must exist");
        let meta = std::fs::metadata(&path)?;
        assert!(meta.len() > 0, "Output file must be non-empty");

        // Verify data survives round-trip through the struct API.
        let loaded = read_mgh::<TestBackend, _>(&path, &device)?;
        assert_eq!(loaded.shape(), [2, 2, 2]);
        let td = loaded.data().clone().to_data();
        let loaded_vals = td.as_slice::<f32>().unwrap();
        for (i, &got) in loaded_vals.iter().enumerate() {
            assert_eq!(got, 1.0f32, "voxel[{}]: expected 1.0, got {}", i, got);
        }
        Ok(())
    }

    // ── Header byte-level validation ─────────────────────────────────────

    /// Write an uncompressed .mgh file and inspect the raw header bytes to
    /// verify the binary layout matches the MGH specification.
    #[test]
    fn test_header_binary_layout() -> Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("header_check.mgh");

        let nz = 2usize;
        let ny = 3usize;
        let nx = 5usize;
        let n = nz * ny * nx;
        let data_vec: Vec<f32> = (0..n as u32).map(|i| i as f32).collect();

        let spacing = Spacing::new([0.5, 1.0, 2.0]);
        let image = make_image_with_spatial(
            data_vec,
            nz,
            ny,
            nx,
            Point::new([0.0, 0.0, 0.0]),
            spacing,
            Direction::identity(),
        );
        write_mgh(&image, &path)?;

        let raw = std::fs::read(&path)?;

        // File must be at least header + data.
        assert_eq!(raw.len(), HEADER_SIZE + n * 4);

        // Version = 1
        assert_eq!(i32::from_be_bytes(raw[0..4].try_into().unwrap()), 1);
        // Width = nx = 5
        assert_eq!(i32::from_be_bytes(raw[4..8].try_into().unwrap()), 5);
        // Height = ny = 3
        assert_eq!(i32::from_be_bytes(raw[8..12].try_into().unwrap()), 3);
        // Depth = nz = 2
        assert_eq!(i32::from_be_bytes(raw[12..16].try_into().unwrap()), 2);
        // nframes = 1
        assert_eq!(i32::from_be_bytes(raw[16..20].try_into().unwrap()), 1);
        // type = MRI_FLOAT = 3
        assert_eq!(i32::from_be_bytes(raw[20..24].try_into().unwrap()), 3);
        // dof = 0
        assert_eq!(i32::from_be_bytes(raw[24..28].try_into().unwrap()), 0);
        // goodRASFlag = 1
        assert_eq!(i16::from_be_bytes(raw[28..30].try_into().unwrap()), 1);

        // Spacing values
        let sp_x = f32::from_be_bytes(raw[30..34].try_into().unwrap());
        let sp_y = f32::from_be_bytes(raw[34..38].try_into().unwrap());
        let sp_z = f32::from_be_bytes(raw[38..42].try_into().unwrap());
        assert_eq!(sp_x, 0.5f32);
        assert_eq!(sp_y, 1.0f32);
        assert_eq!(sp_z, 2.0f32);

        // Direction cosines for identity matrix, column-major.
        // Column 0: [1, 0, 0] at bytes 42..54
        assert_eq!(f32::from_be_bytes(raw[42..46].try_into().unwrap()), 1.0);
        assert_eq!(f32::from_be_bytes(raw[46..50].try_into().unwrap()), 0.0);
        assert_eq!(f32::from_be_bytes(raw[50..54].try_into().unwrap()), 0.0);
        // Column 1: [0, 1, 0] at bytes 54..66
        assert_eq!(f32::from_be_bytes(raw[54..58].try_into().unwrap()), 0.0);
        assert_eq!(f32::from_be_bytes(raw[58..62].try_into().unwrap()), 1.0);
        assert_eq!(f32::from_be_bytes(raw[62..66].try_into().unwrap()), 0.0);
        // Column 2: [0, 0, 1] at bytes 66..78
        assert_eq!(f32::from_be_bytes(raw[66..70].try_into().unwrap()), 0.0);
        assert_eq!(f32::from_be_bytes(raw[70..74].try_into().unwrap()), 0.0);
        assert_eq!(f32::from_be_bytes(raw[74..78].try_into().unwrap()), 1.0);

        // RAS center: c_ras = origin + Mdc · D · h
        //   origin = [0, 0, 0], Mdc = I, D = diag(0.5, 1.0, 2.0)
        //   h = [(5−1)/2, (3−1)/2, (2−1)/2] = [2.0, 1.0, 0.5]
        //   D·h = [1.0, 1.0, 1.0]
        //   c_ras = [0+1.0, 0+1.0, 0+1.0] = [1.0, 1.0, 1.0]
        let c_r = f32::from_be_bytes(raw[78..82].try_into().unwrap());
        let c_a = f32::from_be_bytes(raw[82..86].try_into().unwrap());
        let c_s = f32::from_be_bytes(raw[86..90].try_into().unwrap());
        assert!((c_r - 1.0).abs() < 1e-6, "c_r: expected 1.0, got {}", c_r);
        assert!((c_a - 1.0).abs() < 1e-6, "c_a: expected 1.0, got {}", c_a);
        assert!((c_s - 1.0).abs() < 1e-6, "c_s: expected 1.0, got {}", c_s);

        // Padding bytes 90..284 must all be zero.
        for (i, &b) in raw[90..HEADER_SIZE].iter().enumerate() {
            assert_eq!(b, 0, "Padding byte at offset {} is non-zero: {}", 90 + i, b);
        }

        // First data voxel (index 0) at byte 284.
        let first_voxel = f32::from_be_bytes(raw[284..288].try_into().unwrap());
        assert_eq!(first_voxel, 0.0f32);

        // Last data voxel (index 29) at byte 284 + 29*4 = 400.
        let last_voxel = f32::from_be_bytes(raw[400..404].try_into().unwrap());
        assert_eq!(last_voxel, 29.0f32);

        Ok(())
    }

    // ── All four data types via crafted binary + reader ──────────────────

    /// Craft MGH files for every supported data type (u8, i16, i32, f32),
    /// read each with the reader, and verify the f32 conversion against
    /// analytically derived expected values.
    #[test]
    fn test_all_four_data_types_readable() -> Result<()> {
        use super::super::{MRI_INT, MRI_SHORT, MRI_UCHAR};

        let dir = tempdir()?;
        let device: <TestBackend as Backend>::Device = Default::default();

        // ── MRI_UCHAR (u8) ────────────────────────────────────────────────
        {
            let path = dir.path().join("types_u8.mgh");
            let vals: Vec<u8> = vec![0, 50, 100, 150, 200, 250, 128, 64];
            let expected: Vec<f32> = vals.iter().map(|&v| v as f32).collect();
            let mgh = build_mgh_bytes(
                1,
                [2, 2, 2],
                MRI_UCHAR,
                [1.0, 1.0, 1.0],
                IDENTITY_DIR,
                [0.0, 0.0, 0.0],
                &vals,
            );
            std::fs::write(&path, &mgh)?;

            let image = read_mgh::<TestBackend, _>(&path, &device)?;
            let td = image.data().clone().to_data();
            let loaded = td.as_slice::<f32>().unwrap();
            assert_eq!(loaded.len(), expected.len());
            for (i, (&got, &exp)) in loaded.iter().zip(expected.iter()).enumerate() {
                assert_eq!(got, exp, "u8 voxel[{}]: expected {}, got {}", i, exp, got);
            }
        }

        // ── MRI_SHORT (i16) ───────────────────────────────────────────────
        {
            let path = dir.path().join("types_i16.mgh");
            let vals: Vec<i16> = vec![-32000, -1000, 0, 1000, 5000, 10000, -5000, 32000];
            let expected: Vec<f32> = vals.iter().map(|&v| v as f32).collect();
            let data_bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_be_bytes()).collect();
            let mgh = build_mgh_bytes(
                1,
                [2, 2, 2],
                MRI_SHORT,
                [1.0, 1.0, 1.0],
                IDENTITY_DIR,
                [0.0, 0.0, 0.0],
                &data_bytes,
            );
            std::fs::write(&path, &mgh)?;

            let image = read_mgh::<TestBackend, _>(&path, &device)?;
            let td = image.data().clone().to_data();
            let loaded = td.as_slice::<f32>().unwrap();
            assert_eq!(loaded.len(), expected.len());
            for (i, (&got, &exp)) in loaded.iter().zip(expected.iter()).enumerate() {
                assert_eq!(got, exp, "i16 voxel[{}]: expected {}, got {}", i, exp, got);
            }
        }

        // ── MRI_INT (i32) ─────────────────────────────────────────────────
        {
            let path = dir.path().join("types_i32.mgh");
            // All values within ±2^24 so i32→f32 cast is exact.
            let vals: Vec<i32> = vec![-100_000, -1, 0, 1, 50_000, 100_000, -50_000, 12345];
            let expected: Vec<f32> = vals.iter().map(|&v| v as f32).collect();
            let data_bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_be_bytes()).collect();
            let mgh = build_mgh_bytes(
                1,
                [2, 2, 2],
                MRI_INT,
                [1.0, 1.0, 1.0],
                IDENTITY_DIR,
                [0.0, 0.0, 0.0],
                &data_bytes,
            );
            std::fs::write(&path, &mgh)?;

            let image = read_mgh::<TestBackend, _>(&path, &device)?;
            let td = image.data().clone().to_data();
            let loaded = td.as_slice::<f32>().unwrap();
            assert_eq!(loaded.len(), expected.len());
            for (i, (&got, &exp)) in loaded.iter().zip(expected.iter()).enumerate() {
                assert_eq!(got, exp, "i32 voxel[{}]: expected {}, got {}", i, exp, got);
            }
        }

        // ── MRI_FLOAT (f32) ───────────────────────────────────────────────
        {
            let path = dir.path().join("types_f32.mgh");
            let vals: Vec<f32> = vec![
                std::f32::consts::PI,
                -std::f32::consts::E,
                0.0,
                f32::MIN_POSITIVE,
                1.0 / 7.0,
                std::f32::consts::SQRT_2,
                -123456.789,
                std::f32::consts::LN_2,
            ];
            let data_bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_be_bytes()).collect();
            let mgh = build_mgh_bytes(
                1,
                [2, 2, 2],
                super::super::MRI_FLOAT,
                [1.0, 1.0, 1.0],
                IDENTITY_DIR,
                [0.0, 0.0, 0.0],
                &data_bytes,
            );
            std::fs::write(&path, &mgh)?;

            let image = read_mgh::<TestBackend, _>(&path, &device)?;
            let td = image.data().clone().to_data();
            let loaded = td.as_slice::<f32>().unwrap();
            assert_eq!(loaded.len(), vals.len());
            for (i, (&got, &exp)) in loaded.iter().zip(vals.iter()).enumerate() {
                assert_eq!(
                    got.to_bits(),
                    exp.to_bits(),
                    "f32 voxel[{}]: expected {} ({:#010x}), got {} ({:#010x})",
                    i,
                    exp,
                    exp.to_bits(),
                    got,
                    got.to_bits()
                );
            }
        }

        Ok(())
    }

    // ── Negative test: invalid version ───────────────────────────────────

    /// A file with `version != 1` must be rejected with an error that
    /// mentions "version".
    #[test]
    fn test_invalid_version_rejected() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("bad_ver.mgh");
        let device: <TestBackend as Backend>::Device = Default::default();

        let mgh = build_mgh_bytes(
            99, // invalid version
            [2, 2, 2],
            super::super::MRI_FLOAT,
            [1.0, 1.0, 1.0],
            IDENTITY_DIR,
            [0.0, 0.0, 0.0],
            &vec![0u8; 2 * 2 * 2 * 4],
        );
        std::fs::write(&path, &mgh).unwrap();

        let result = read_mgh::<TestBackend, _>(&path, &device);
        assert!(result.is_err(), "Reading invalid version must fail");
        let msg = format!("{:#}", result.unwrap_err());
        assert!(
            msg.contains("version"),
            "Error must mention 'version', got: {}",
            msg
        );
    }

    // ── File size validation ─────────────────────────────────────────────

    /// The output file must be at least header + data bytes.
    #[test]
    fn test_file_contains_full_payload() -> Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("payload.mgh");

        let nz = 2usize;
        let ny = 3usize;
        let nx = 4usize;
        let n_voxels = nz * ny * nx;

        let image = make_image(vec![1.0f32; n_voxels], nz, ny, nx);
        write_mgh(&image, &path)?;

        let file_size = std::fs::metadata(&path)?.len();
        let expected = (HEADER_SIZE + n_voxels * 4) as u64;
        assert_eq!(
            file_size,
            expected,
            "File size {} must equal header ({}) + data ({}) = {}",
            file_size,
            HEADER_SIZE,
            n_voxels * 4,
            expected
        );
        Ok(())
    }

    // ── Edge-case voxel values ───────────────────────────────────────────

    /// Verify that subnormal, zero, negative-zero, and large-magnitude f32
    /// values survive the write-read round-trip bit-identically.
    #[test]
    fn test_edge_case_values_round_trip() -> Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("edges.mgh");
        let device: <TestBackend as Backend>::Device = Default::default();

        let values: Vec<f32> = vec![
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::MAX,
            f32::MIN,
            std::f32::consts::PI,
        ];
        assert_eq!(values.len(), 8);

        let image = make_image(values.clone(), 2, 2, 2);
        write_mgh(&image, &path)?;

        let loaded = read_mgh::<TestBackend, _>(&path, &device)?;
        let td = loaded.data().clone().to_data();
        let loaded_vals = td.as_slice::<f32>().unwrap();
        assert_eq!(loaded_vals.len(), values.len());
        for (i, (&got, &expected)) in loaded_vals.iter().zip(values.iter()).enumerate() {
            assert_eq!(
                got.to_bits(),
                expected.to_bits(),
                "voxel[{}]: expected {} (bits {:#010x}), got {} (bits {:#010x})",
                i,
                expected,
                expected.to_bits(),
                got,
                got.to_bits()
            );
        }
        Ok(())
    }

    // ── MGZ round-trip with non-default spatial ──────────────────────────

    /// Combined test: gzip compression AND non-default spatial metadata.
    #[test]
    fn test_round_trip_mgz_with_spatial() -> Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("spatial.mgz");
        let device: <TestBackend as Backend>::Device = Default::default();

        let nz = 2usize;
        let ny = 2usize;
        let nx = 2usize;

        let spacing = Spacing::new([2.0, 3.0, 4.0]);
        let origin = Point::new([100.0, 200.0, 300.0]);
        let direction = Direction::identity();

        let data_vec: Vec<f32> = (0..8u32).map(|i| i as f32 * 10.0).collect();
        let image =
            make_image_with_spatial(data_vec.clone(), nz, ny, nx, origin, spacing, direction);

        write_mgh(&image, &path)?;

        // Confirm gzip.
        let bytes = std::fs::read(&path)?;
        assert_eq!(bytes[0], 0x1f);
        assert_eq!(bytes[1], 0x8b);

        let loaded = read_mgh::<TestBackend, _>(&path, &device)?;
        assert_eq!(loaded.shape(), [nz, ny, nx]);

        // Spacing.
        let sp = loaded.spacing();
        assert!((sp[0] - 2.0).abs() < 1e-6);
        assert!((sp[1] - 3.0).abs() < 1e-6);
        assert!((sp[2] - 4.0).abs() < 1e-6);

        // Origin.
        // c_ras = [100, 200, 300] + I * diag(2,3,4) * [0.5, 0.5, 0.5]
        //       = [100+1, 200+1.5, 300+2] = [101, 201.5, 302]
        // origin_rt = [101, 201.5, 302] - [1, 1.5, 2] = [100, 200, 300]
        let o = loaded.origin();
        assert!(
            (o[0] - 100.0).abs() < 1e-4,
            "origin[0]: expected 100, got {}",
            o[0]
        );
        assert!(
            (o[1] - 200.0).abs() < 1e-4,
            "origin[1]: expected 200, got {}",
            o[1]
        );
        assert!(
            (o[2] - 300.0).abs() < 1e-4,
            "origin[2]: expected 300, got {}",
            o[2]
        );

        // Voxel data.
        let td = loaded.data().clone().to_data();
        let loaded_vals = td.as_slice::<f32>().unwrap();
        for (i, (&got, &exp)) in loaded_vals.iter().zip(data_vec.iter()).enumerate() {
            assert_eq!(got.to_bits(), exp.to_bits(), "voxel[{}]", i);
        }

        Ok(())
    }
}
