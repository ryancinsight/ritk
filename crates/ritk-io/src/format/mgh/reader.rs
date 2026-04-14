//! MGH / MGZ reader for 3-D volumetric images.
//!
//! # MGH Binary Layout (284-byte header, big-endian)
//!
//! | Offset | Type   | Field                                          |
//! |--------|--------|------------------------------------------------|
//! | 0      | i32    | version (must be 1)                            |
//! | 4      | i32    | width (X dimension)                            |
//! | 8      | i32    | height (Y dimension)                           |
//! | 12     | i32    | depth (Z dimension)                            |
//! | 16     | i32    | nframes (typically 1)                          |
//! | 20     | i32    | data type (0=u8, 1=i32, 3=f32, 4=i16)         |
//! | 24     | i32    | dof (degrees of freedom)                       |
//! | 28     | i16    | goodRASFlag (1 = valid RAS metadata)           |
//! | 30     | f32×3  | voxel spacing (X, Y, Z) in mm                  |
//! | 42     | f32×9  | direction cosines (3 columns, column-major)    |
//! | 78     | f32×3  | RAS center of volume (c\_r, c\_a, c\_s)        |
//! | 90     | —      | zero padding to byte 284                       |
//!
//! # Data Section
//!
//! Starts at byte 284.  Voxels are stored in Fortran (column-major) order:
//! X varies fastest, then Y, then Z, then frame.  This maps directly to
//! RITK's row-major `[nz, ny, nx]` tensor layout without permutation
//! because `x + y·nx + z·nx·ny ≡ z·ny·nx + y·nx + x`.
//!
//! # Origin Derivation
//!
//! ```text
//! Mdc    = [x_ras, y_ras, z_ras]          (3×3 direction cosine matrix)
//! D      = diag(spacing_x, spacing_y, spacing_z)
//! h      = [(width−1)/2, (height−1)/2, (depth−1)/2]^T
//! origin = c_ras − Mdc · D · h
//! ```
//!
//! # Proof of index equivalence (Fortran ↔ RITK row-major)
//!
//! MGH linear index for voxel `(x, y, z)`:
//!   `i_mgh = x + y·nx + z·nx·ny`
//!
//! RITK tensor `[nz, ny, nx]` row-major linear index for element `[z, y, x]`:
//!   `i_ritk = z·(ny·nx) + y·nx + x`
//!
//! Since `ny·nx = nx·ny`, these are identical: `i_mgh ≡ i_ritk`.
//! Therefore the raw byte stream from the MGH data section can be loaded
//! directly into the RITK tensor without any axis permutation.

use super::{is_gzip_path, MRI_FLOAT, MRI_INT, MRI_SHORT, MRI_UCHAR, PADDING_LEN, VERSION};
use anyhow::{bail, Context, Result};
use burn::tensor::backend::Backend;
use burn::tensor::{Shape, Tensor, TensorData};
use flate2::read::GzDecoder;
use nalgebra::SMatrix;
use ritk_core::image::Image;
use ritk_core::spatial::{Direction, Point, Spacing};
use std::io::{BufReader, Read};
use std::path::Path;

// ── Big-endian primitive read helpers ─────────────────────────────────────────

/// Read a big-endian `i32` from `r`.
fn read_i32_be<R: Read>(r: &mut R) -> Result<i32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf).context("Failed to read i32 BE")?;
    Ok(i32::from_be_bytes(buf))
}

/// Read a big-endian `i16` from `r`.
fn read_i16_be<R: Read>(r: &mut R) -> Result<i16> {
    let mut buf = [0u8; 2];
    r.read_exact(&mut buf).context("Failed to read i16 BE")?;
    Ok(i16::from_be_bytes(buf))
}

/// Read a big-endian `f32` from `r`.
fn read_f32_be<R: Read>(r: &mut R) -> Result<f32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf).context("Failed to read f32 BE")?;
    Ok(f32::from_be_bytes(buf))
}

/// Number of bytes occupied by a single voxel for the given MGH type code.
///
/// # Errors
///
/// Returns `Err` for unrecognised type codes.
fn bytes_per_voxel(mri_type: i32) -> Result<usize> {
    match mri_type {
        MRI_UCHAR => Ok(1),
        MRI_SHORT => Ok(2),
        MRI_INT | MRI_FLOAT => Ok(4),
        other => bail!("Unsupported MGH data type code: {}", other),
    }
}

// ── Public API ───────────────────────────────────────────────────────────────

/// Read an MGH or MGZ file into a 3-D `Image`.
///
/// # Gzip detection
///
/// Files with extension `.mgz` or `.mgh.gz` are decompressed with gzip
/// before parsing.  All other extensions are treated as uncompressed MGH.
///
/// # Errors
///
/// Returns `Err` when:
/// - The file cannot be opened.
/// - The version field is not 1.
/// - The data type code is unrecognised.
/// - The file is truncated (fewer bytes than header + data).
pub fn read_mgh<B: Backend, P: AsRef<Path>>(path: P, device: &B::Device) -> Result<Image<B, 3>> {
    let path = path.as_ref();
    let file = std::fs::File::open(path)
        .with_context(|| format!("Cannot open MGH/MGZ file {:?}", path))?;

    if is_gzip_path(path) {
        let gz = GzDecoder::new(BufReader::new(file));
        let mut reader = BufReader::new(gz);
        read_mgh_from_reader(&mut reader, device)
            .with_context(|| format!("Failed to parse MGZ file {:?}", path))
    } else {
        let mut reader = BufReader::new(file);
        read_mgh_from_reader(&mut reader, device)
            .with_context(|| format!("Failed to parse MGH file {:?}", path))
    }
}

// ── Core reader ──────────────────────────────────────────────────────────────

/// Parse an MGH byte stream (header + data) into an `Image<B, 3>`.
///
/// # Algorithm
///
/// 1. Read the 284-byte header and validate the version field.
/// 2. Extract spatial metadata (spacing, direction, RAS center) if
///    `goodRASFlag == 1`; otherwise use identity defaults.
/// 3. Compute origin from the RAS center via the formula documented in the
///    module-level doc.
/// 4. Read the voxel data for frame 0 and convert to `f32`.
/// 5. Construct `Image<B, 3>` with shape `[nz, ny, nx]`.
fn read_mgh_from_reader<B: Backend, R: Read>(
    reader: &mut R,
    device: &B::Device,
) -> Result<Image<B, 3>> {
    // ── Fixed-position header fields (bytes 0–27) ─────────────────────────
    let version = read_i32_be(reader)?;
    if version != VERSION {
        bail!(
            "Invalid MGH version: expected {}, found {}",
            VERSION,
            version
        );
    }

    let width = read_i32_be(reader)?;
    let height = read_i32_be(reader)?;
    let depth = read_i32_be(reader)?;
    let nframes = read_i32_be(reader)?;
    let mri_type = read_i32_be(reader)?;
    let _dof = read_i32_be(reader)?;

    if width <= 0 || height <= 0 || depth <= 0 {
        bail!(
            "Invalid MGH dimensions: width={}, height={}, depth={}",
            width,
            height,
            depth
        );
    }
    if nframes <= 0 {
        bail!("Invalid MGH nframes: {}", nframes);
    }
    if nframes > 1 {
        tracing::warn!(
            nframes,
            "MGH file contains multiple frames; only frame 0 will be loaded"
        );
    }

    // ── Spatial metadata block (bytes 28–89) ──────────────────────────────
    let good_ras_flag = read_i16_be(reader)?;

    // The spatial fields are always present in the 284-byte header regardless
    // of the flag value.  Read them unconditionally so the stream position
    // advances to the padding region.
    let spacing_x = read_f32_be(reader)?;
    let spacing_y = read_f32_be(reader)?;
    let spacing_z = read_f32_be(reader)?;

    let x_r = read_f32_be(reader)?;
    let x_a = read_f32_be(reader)?;
    let x_s = read_f32_be(reader)?;
    let y_r = read_f32_be(reader)?;
    let y_a = read_f32_be(reader)?;
    let y_s = read_f32_be(reader)?;
    let z_r = read_f32_be(reader)?;
    let z_a = read_f32_be(reader)?;
    let z_s = read_f32_be(reader)?;

    let c_r = read_f32_be(reader)?;
    let c_a = read_f32_be(reader)?;
    let c_s = read_f32_be(reader)?;

    // ── Padding (bytes 90–283) ────────────────────────────────────────────
    let mut _padding = [0u8; PADDING_LEN];
    reader
        .read_exact(&mut _padding)
        .context("Failed to read MGH header padding")?;

    // ── Derive spatial metadata ───────────────────────────────────────────
    let (spacing, direction, origin) = if good_ras_flag == 1 {
        let sp = Spacing::new([spacing_x as f64, spacing_y as f64, spacing_z as f64]);

        let dir_matrix = SMatrix::<f64, 3, 3>::from_columns(&[
            nalgebra::Vector3::new(x_r as f64, x_a as f64, x_s as f64),
            nalgebra::Vector3::new(y_r as f64, y_a as f64, y_s as f64),
            nalgebra::Vector3::new(z_r as f64, z_a as f64, z_s as f64),
        ]);
        let dir = Direction(dir_matrix);

        // origin = c_ras − Mdc · D · h
        let half_dim = nalgebra::Vector3::new(
            (width as f64 - 1.0) / 2.0,
            (height as f64 - 1.0) / 2.0,
            (depth as f64 - 1.0) / 2.0,
        );
        let scaled_half = nalgebra::Vector3::new(
            sp[0] * half_dim[0],
            sp[1] * half_dim[1],
            sp[2] * half_dim[2],
        );
        let offset = dir_matrix * scaled_half;
        let c_ras = nalgebra::Vector3::new(c_r as f64, c_a as f64, c_s as f64);
        let origin_vec = c_ras - offset;

        (
            sp,
            dir,
            Point::new([origin_vec[0], origin_vec[1], origin_vec[2]]),
        )
    } else {
        // No valid RAS info — use RITK default spatial metadata.
        (
            Spacing::new([1.0, 1.0, 1.0]),
            Direction::identity(),
            Point::new([0.0, 0.0, 0.0]),
        )
    };

    // ── Voxel data (frame 0 only) ─────────────────────────────────────────
    let nx = width as usize;
    let ny = height as usize;
    let nz = depth as usize;
    let n_voxels = nx
        .checked_mul(ny)
        .and_then(|v| v.checked_mul(nz))
        .ok_or_else(|| anyhow::anyhow!("Volume dimensions overflow: {}×{}×{}", nx, ny, nz))?;

    let bpv = bytes_per_voxel(mri_type)?;
    let data_size = n_voxels.checked_mul(bpv).ok_or_else(|| {
        anyhow::anyhow!("Data size overflow: {} voxels × {} bytes", n_voxels, bpv)
    })?;

    let mut raw = vec![0u8; data_size];
    reader
        .read_exact(&mut raw)
        .context("Failed to read MGH voxel data")?;

    let f32_data: Vec<f32> = match mri_type {
        MRI_UCHAR => raw.iter().map(|&b| b as f32).collect(),
        MRI_SHORT => raw
            .chunks_exact(2)
            .map(|c| i16::from_be_bytes([c[0], c[1]]) as f32)
            .collect(),
        MRI_INT => raw
            .chunks_exact(4)
            .map(|c| i32::from_be_bytes([c[0], c[1], c[2], c[3]]) as f32)
            .collect(),
        MRI_FLOAT => raw
            .chunks_exact(4)
            .map(|c| f32::from_be_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        _ => unreachable!("bytes_per_voxel already validated the type code"),
    };

    // ── Construct Image ───────────────────────────────────────────────────
    let tensor_data = TensorData::new(f32_data, Shape::new([nz, ny, nx]));
    let tensor = Tensor::<B, 3>::from_data(tensor_data, device);

    Ok(Image::new(tensor, origin, spacing, direction))
}

// ── Public reader struct ─────────────────────────────────────────────────────

/// Stateless reader for MGH / MGZ files.
///
/// Provides a struct-based API that delegates to [`read_mgh`].
pub struct MghReader;

impl MghReader {
    /// Read an MGH or MGZ file into a 3-D `Image`.
    ///
    /// See [`read_mgh`] for full documentation.
    pub fn read<B: Backend, P: AsRef<Path>>(path: P, device: &B::Device) -> Result<Image<B, 3>> {
        read_mgh(path, device)
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::super::HEADER_SIZE;
    use super::*;
    use burn::tensor::backend::Backend;
    use burn_ndarray::NdArray;
    use flate2::write::GzEncoder;
    use flate2::Compression;
    use std::io::Write;
    use tempfile::tempdir;

    type TestBackend = NdArray<f32>;

    /// Build a complete MGH file as a byte vector.
    ///
    /// # Parameters
    ///
    /// - `version`: header version field (1 = valid).
    /// - `dims`: `[width, height, depth]`.
    /// - `mri_type`: data type code (0, 1, 3, or 4).
    /// - `spacing`: `[sx, sy, sz]` in mm.
    /// - `dir_cols`: direction cosines as three column vectors
    ///   `[[x_r,x_a,x_s], [y_r,y_a,y_s], [z_r,z_a,z_s]]`.
    /// - `c_ras`: RAS center `[c_r, c_a, c_s]`.
    /// - `data`: raw voxel bytes (big-endian, exact byte count).
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

        // Fixed fields (bytes 0–27): 7 × i32
        buf.extend_from_slice(&version.to_be_bytes());
        buf.extend_from_slice(&dims[0].to_be_bytes()); // width
        buf.extend_from_slice(&dims[1].to_be_bytes()); // height
        buf.extend_from_slice(&dims[2].to_be_bytes()); // depth
        buf.extend_from_slice(&1_i32.to_be_bytes()); // nframes
        buf.extend_from_slice(&mri_type.to_be_bytes());
        buf.extend_from_slice(&0_i32.to_be_bytes()); // dof

        // goodRASFlag (bytes 28–29)
        buf.extend_from_slice(&1_i16.to_be_bytes());

        // Spacing (bytes 30–41)
        for &s in &spacing {
            buf.extend_from_slice(&s.to_be_bytes());
        }

        // Direction cosines column-major (bytes 42–77)
        for col in &dir_cols {
            for &v in col {
                buf.extend_from_slice(&v.to_be_bytes());
            }
        }

        // RAS center (bytes 78–89)
        for &c in &c_ras {
            buf.extend_from_slice(&c.to_be_bytes());
        }

        // Assert we are at byte 90 before padding
        debug_assert_eq!(buf.len(), 90, "Header fields must occupy exactly 90 bytes");

        // Padding to 284 bytes
        buf.resize(HEADER_SIZE, 0u8);

        // Voxel data
        buf.extend_from_slice(data);
        buf
    }

    /// Identity direction columns for `build_mgh_bytes`.
    const IDENTITY_DIR: [[f32; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

    // ── Data type: MRI_FLOAT (f32) ───────────────────────────────────────

    /// Read a crafted MRI_FLOAT file and verify bit-exact voxel values.
    #[test]
    fn test_read_f32_data() -> Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("f32.mgh");
        let device: <TestBackend as Backend>::Device = Default::default();

        // Analytically chosen constants (transcendental and rational).
        let values: Vec<f32> = vec![
            std::f32::consts::PI,
            std::f32::consts::E,
            std::f32::consts::SQRT_2,
            std::f32::consts::LN_2,
            1.0 / 7.0,
            -std::f32::consts::FRAC_PI_2,
            2.0 * std::f32::consts::E,
            1.0 / 3.0,
        ];
        let data_bytes: Vec<u8> = values.iter().flat_map(|v| v.to_be_bytes()).collect();

        let mgh = build_mgh_bytes(
            1,
            [2, 2, 2],
            MRI_FLOAT,
            [1.0, 1.0, 1.0],
            IDENTITY_DIR,
            [0.0, 0.0, 0.0],
            &data_bytes,
        );
        std::fs::write(&path, &mgh)?;

        let image = read_mgh::<TestBackend, _>(&path, &device)?;
        assert_eq!(image.shape(), [2, 2, 2]);

        let td = image.data().clone().to_data();
        let loaded = td.as_slice::<f32>().unwrap();
        assert_eq!(loaded.len(), values.len());
        for (i, (&got, &expected)) in loaded.iter().zip(values.iter()).enumerate() {
            assert_eq!(
                got.to_bits(),
                expected.to_bits(),
                "f32 voxel[{}]: expected {} ({:#010x}), got {} ({:#010x})",
                i,
                expected,
                expected.to_bits(),
                got,
                got.to_bits()
            );
        }
        Ok(())
    }

    // ── Data type: MRI_UCHAR (u8) ───────────────────────────────────────

    /// Read a crafted MRI_UCHAR file and verify u8→f32 conversion.
    #[test]
    fn test_read_u8_data() -> Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("u8.mgh");
        let device: <TestBackend as Backend>::Device = Default::default();

        // 2×3×2 = 12 voxels.  Values: 10·i for i in 0..12, all ≤ 110 < 256.
        let u8_vals: Vec<u8> = (0u8..12).map(|i| i * 10).collect();
        let expected: Vec<f32> = u8_vals.iter().map(|&v| v as f32).collect();

        let mgh = build_mgh_bytes(
            1,
            [2, 3, 2],
            MRI_UCHAR,
            [1.0, 1.0, 1.0],
            IDENTITY_DIR,
            [0.0, 0.0, 0.0],
            &u8_vals,
        );
        std::fs::write(&path, &mgh)?;

        let image = read_mgh::<TestBackend, _>(&path, &device)?;
        assert_eq!(image.shape(), [2, 3, 2]);

        let td = image.data().clone().to_data();
        let loaded = td.as_slice::<f32>().unwrap();
        assert_eq!(loaded.len(), 12);
        for (i, (&got, &exp)) in loaded.iter().zip(expected.iter()).enumerate() {
            assert_eq!(got, exp, "u8 voxel[{}]: expected {}, got {}", i, exp, got);
        }
        Ok(())
    }

    // ── Data type: MRI_SHORT (i16) ──────────────────────────────────────

    /// Read a crafted MRI_SHORT file and verify i16→f32 conversion.
    ///
    /// Values span the negative and positive range of i16 to exercise
    /// sign-extension and big-endian byte ordering.
    #[test]
    fn test_read_i16_data() -> Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("i16.mgh");
        let device: <TestBackend as Backend>::Device = Default::default();

        let i16_vals: Vec<i16> = vec![
            -1000, -100, 0, 100, 200, 300, 400, 500, -500, -200, 150, 750,
        ];
        let expected: Vec<f32> = i16_vals.iter().map(|&v| v as f32).collect();
        let data_bytes: Vec<u8> = i16_vals.iter().flat_map(|v| v.to_be_bytes()).collect();

        let mgh = build_mgh_bytes(
            1,
            [2, 3, 2],
            MRI_SHORT,
            [1.0, 1.0, 1.0],
            IDENTITY_DIR,
            [0.0, 0.0, 0.0],
            &data_bytes,
        );
        std::fs::write(&path, &mgh)?;

        let image = read_mgh::<TestBackend, _>(&path, &device)?;
        assert_eq!(image.shape(), [2, 3, 2]);

        let td = image.data().clone().to_data();
        let loaded = td.as_slice::<f32>().unwrap();
        assert_eq!(loaded.len(), 12);
        for (i, (&got, &exp)) in loaded.iter().zip(expected.iter()).enumerate() {
            assert_eq!(got, exp, "i16 voxel[{}]: expected {}, got {}", i, exp, got);
        }
        Ok(())
    }

    // ── Data type: MRI_INT (i32) ────────────────────────────────────────

    /// Read a crafted MRI_INT file and verify i32→f32 conversion.
    ///
    /// All test values are within ±2^24 so the i32→f32 cast is exact.
    #[test]
    fn test_read_i32_data() -> Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("i32.mgh");
        let device: <TestBackend as Backend>::Device = Default::default();

        let i32_vals: Vec<i32> = vec![
            -100_000, -10_000, 0, 10_000, 20_000, 30_000, 40_000, 50_000, -50_000, -20_000, 15_000,
            75_000,
        ];
        let expected: Vec<f32> = i32_vals.iter().map(|&v| v as f32).collect();
        let data_bytes: Vec<u8> = i32_vals.iter().flat_map(|v| v.to_be_bytes()).collect();

        let mgh = build_mgh_bytes(
            1,
            [2, 3, 2],
            MRI_INT,
            [1.0, 1.0, 1.0],
            IDENTITY_DIR,
            [0.0, 0.0, 0.0],
            &data_bytes,
        );
        std::fs::write(&path, &mgh)?;

        let image = read_mgh::<TestBackend, _>(&path, &device)?;
        assert_eq!(image.shape(), [2, 3, 2]);

        let td = image.data().clone().to_data();
        let loaded = td.as_slice::<f32>().unwrap();
        assert_eq!(loaded.len(), 12);
        for (i, (&got, &exp)) in loaded.iter().zip(expected.iter()).enumerate() {
            assert_eq!(got, exp, "i32 voxel[{}]: expected {}, got {}", i, exp, got);
        }
        Ok(())
    }

    // ── Negative test: invalid version ──────────────────────────────────

    /// A file with `version != 1` must be rejected with an error that
    /// mentions "version".
    #[test]
    fn test_read_invalid_version() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("bad_version.mgh");
        let device: <TestBackend as Backend>::Device = Default::default();

        let mgh = build_mgh_bytes(
            2, // invalid
            [2, 2, 2],
            MRI_FLOAT,
            [1.0, 1.0, 1.0],
            IDENTITY_DIR,
            [0.0, 0.0, 0.0],
            &vec![0u8; 2 * 2 * 2 * 4],
        );
        std::fs::write(&path, &mgh).unwrap();

        let result = read_mgh::<TestBackend, _>(&path, &device);
        assert!(result.is_err(), "Reading invalid version must fail");
        let err_msg = format!("{:#}", result.unwrap_err());
        assert!(
            err_msg.contains("version"),
            "Error must mention 'version', got: {}",
            err_msg
        );
    }

    // ── MGZ (gzip) reader test ──────────────────────────────────────────

    /// Gzip-compress a crafted MGH file, write as `.mgz`, read it back,
    /// and verify bit-exact voxel values.
    #[test]
    fn test_read_mgz() -> Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("test.mgz");
        let device: <TestBackend as Backend>::Device = Default::default();

        // Values: i·1.5 + 0.25 for i in 0..8.
        let values: Vec<f32> = (0..8).map(|i| (i as f32) * 1.5 + 0.25).collect();
        let data_bytes: Vec<u8> = values.iter().flat_map(|v| v.to_be_bytes()).collect();

        let mgh = build_mgh_bytes(
            1,
            [2, 2, 2],
            MRI_FLOAT,
            [1.0, 1.0, 1.0],
            IDENTITY_DIR,
            [0.0, 0.0, 0.0],
            &data_bytes,
        );

        // Gzip-compress the MGH stream.
        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(&mgh)?;
        let mgz_bytes = encoder.finish()?;
        std::fs::write(&path, &mgz_bytes)?;

        let image = read_mgh::<TestBackend, _>(&path, &device)?;
        assert_eq!(image.shape(), [2, 2, 2]);

        let td = image.data().clone().to_data();
        let loaded = td.as_slice::<f32>().unwrap();
        assert_eq!(loaded.len(), 8);
        for (i, (&got, &exp)) in loaded.iter().zip(values.iter()).enumerate() {
            assert_eq!(
                got.to_bits(),
                exp.to_bits(),
                "mgz voxel[{}]: expected {}, got {}",
                i,
                exp,
                got
            );
        }
        Ok(())
    }

    // ── .mgh.gz extension ───────────────────────────────────────────────

    /// Verify that the `.mgh.gz` extension is also recognised as gzip.
    #[test]
    fn test_read_mgh_gz_extension() -> Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("test.mgh.gz");
        let device: <TestBackend as Backend>::Device = Default::default();

        let values: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let data_bytes: Vec<u8> = values.iter().flat_map(|v| v.to_be_bytes()).collect();

        let mgh = build_mgh_bytes(
            1,
            [2, 2, 2],
            MRI_FLOAT,
            [1.0, 1.0, 1.0],
            IDENTITY_DIR,
            [0.0, 0.0, 0.0],
            &data_bytes,
        );

        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(&mgh)?;
        std::fs::write(&path, encoder.finish()?)?;

        let image = read_mgh::<TestBackend, _>(&path, &device)?;
        let td = image.data().clone().to_data();
        let loaded = td.as_slice::<f32>().unwrap();
        for (i, (&got, &exp)) in loaded.iter().zip(values.iter()).enumerate() {
            assert_eq!(got, exp, "voxel[{}]", i);
        }
        Ok(())
    }

    // ── Non-default spatial metadata ────────────────────────────────────

    /// Craft a file with a 90° rotation around Z and anisotropic spacing,
    /// then verify origin, spacing, and direction against analytically
    /// derived reference values.
    ///
    /// # Analytical derivation
    ///
    /// ```text
    /// Mdc = | 0  -1  0 |    spacing = [0.5, 0.75, 1.25]
    ///       | 1   0  0 |    dims    = [4, 3, 2] (width, height, depth)
    ///       | 0   0  1 |    c_ras   = [10, 20, 30]
    ///
    /// half = [(4−1)/2, (3−1)/2, (2−1)/2] = [1.5, 1.0, 0.5]
    /// D·half = [0.5·1.5, 0.75·1.0, 1.25·0.5] = [0.75, 0.75, 0.625]
    ///
    /// Mdc · D·half:
    ///   row 0:  0·0.75 + (−1)·0.75 + 0·0.625 = −0.75
    ///   row 1:  1·0.75 +   0·0.75  + 0·0.625 =  0.75
    ///   row 2:  0·0.75 +   0·0.75  + 1·0.625 =  0.625
    ///
    /// origin = [10−(−0.75), 20−0.75, 30−0.625] = [10.75, 19.25, 29.375]
    /// ```
    #[test]
    fn test_read_nondefault_spatial() -> Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("spatial.mgh");
        let device: <TestBackend as Backend>::Device = Default::default();

        let spacing_f32 = [0.5f32, 0.75, 1.25];
        // 90-degree rotation around Z: x→+y, y→−x, z→z.
        let dir_cols: [[f32; 3]; 3] = [
            [0.0, 1.0, 0.0],  // x column
            [-1.0, 0.0, 0.0], // y column
            [0.0, 0.0, 1.0],  // z column
        ];
        let c_ras = [10.0f32, 20.0, 30.0];
        let dims = [4i32, 3, 2]; // width=4, height=3, depth=2
        let n_voxels = 4 * 3 * 2;
        let data_bytes: Vec<u8> = (0..n_voxels)
            .map(|i| (i as f32) * 0.1)
            .flat_map(|v| v.to_be_bytes())
            .collect();

        let mgh = build_mgh_bytes(
            1,
            dims,
            MRI_FLOAT,
            spacing_f32,
            dir_cols,
            c_ras,
            &data_bytes,
        );
        std::fs::write(&path, &mgh)?;

        let image = read_mgh::<TestBackend, _>(&path, &device)?;
        // Tensor shape is [nz=depth, ny=height, nx=width].
        assert_eq!(image.shape(), [2, 3, 4]);

        // ── Spacing ───────────────────────────────────────────────────────
        let sp = image.spacing();
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
        let d = image.direction();
        // Column 0: [0, 1, 0]
        assert!((d[(0, 0)] - 0.0).abs() < 1e-6, "d(0,0)={}", d[(0, 0)]);
        assert!((d[(1, 0)] - 1.0).abs() < 1e-6, "d(1,0)={}", d[(1, 0)]);
        assert!((d[(2, 0)] - 0.0).abs() < 1e-6, "d(2,0)={}", d[(2, 0)]);
        // Column 1: [-1, 0, 0]
        assert!((d[(0, 1)] - (-1.0)).abs() < 1e-6, "d(0,1)={}", d[(0, 1)]);
        assert!((d[(1, 1)] - 0.0).abs() < 1e-6, "d(1,1)={}", d[(1, 1)]);
        assert!((d[(2, 1)] - 0.0).abs() < 1e-6, "d(2,1)={}", d[(2, 1)]);
        // Column 2: [0, 0, 1]
        assert!((d[(0, 2)] - 0.0).abs() < 1e-6, "d(0,2)={}", d[(0, 2)]);
        assert!((d[(1, 2)] - 0.0).abs() < 1e-6, "d(1,2)={}", d[(1, 2)]);
        assert!((d[(2, 2)] - 1.0).abs() < 1e-6, "d(2,2)={}", d[(2, 2)]);

        // ── Origin (analytical reference above) ───────────────────────────
        let o = image.origin();
        assert!(
            (o[0] - 10.75).abs() < 1e-6,
            "origin[0]: expected 10.75, got {}",
            o[0]
        );
        assert!(
            (o[1] - 19.25).abs() < 1e-6,
            "origin[1]: expected 19.25, got {}",
            o[1]
        );
        assert!(
            (o[2] - 29.375).abs() < 1e-6,
            "origin[2]: expected 29.375, got {}",
            o[2]
        );

        Ok(())
    }

    // ── goodRASFlag == 0: default spatial metadata ──────────────────────

    /// When `goodRASFlag` is 0, the reader must use identity direction,
    /// unit spacing, and zero origin.
    #[test]
    fn test_read_good_ras_flag_zero() -> Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("no_ras.mgh");
        let device: <TestBackend as Backend>::Device = Default::default();

        // Build header manually with goodRASFlag = 0.
        let n_voxels = 2 * 2 * 2;
        let values: Vec<f32> = (0..n_voxels).map(|i| i as f32).collect();
        let data_bytes: Vec<u8> = values.iter().flat_map(|v| v.to_be_bytes()).collect();

        let mut buf = Vec::with_capacity(HEADER_SIZE + data_bytes.len());
        buf.extend_from_slice(&1_i32.to_be_bytes()); // version
        buf.extend_from_slice(&2_i32.to_be_bytes()); // width
        buf.extend_from_slice(&2_i32.to_be_bytes()); // height
        buf.extend_from_slice(&2_i32.to_be_bytes()); // depth
        buf.extend_from_slice(&1_i32.to_be_bytes()); // nframes
        buf.extend_from_slice(&MRI_FLOAT.to_be_bytes());
        buf.extend_from_slice(&0_i32.to_be_bytes()); // dof
        buf.extend_from_slice(&0_i16.to_be_bytes()); // goodRASFlag = 0
                                                     // Fill remaining spatial fields with garbage (must be ignored).
        for _ in 0..15 {
            buf.extend_from_slice(&99.9f32.to_be_bytes());
        }
        buf.resize(HEADER_SIZE, 0u8);
        buf.extend_from_slice(&data_bytes);

        std::fs::write(&path, &buf)?;

        let image = read_mgh::<TestBackend, _>(&path, &device)?;
        assert_eq!(image.shape(), [2, 2, 2]);

        // Default spatial metadata.
        let sp = image.spacing();
        assert_eq!(sp[0], 1.0);
        assert_eq!(sp[1], 1.0);
        assert_eq!(sp[2], 1.0);

        let d = image.direction();
        assert_eq!(d[(0, 0)], 1.0);
        assert_eq!(d[(1, 1)], 1.0);
        assert_eq!(d[(2, 2)], 1.0);
        assert_eq!(d[(0, 1)], 0.0);

        let o = image.origin();
        assert_eq!(o[0], 0.0);
        assert_eq!(o[1], 0.0);
        assert_eq!(o[2], 0.0);

        // Voxel data must still be correct.
        let td = image.data().clone().to_data();
        let loaded = td.as_slice::<f32>().unwrap();
        for (i, (&got, &exp)) in loaded.iter().zip(values.iter()).enumerate() {
            assert_eq!(got, exp, "voxel[{}]", i);
        }
        Ok(())
    }

    // ── Round-trip: write then read ─────────────────────────────────────

    /// Write an image with [`write_mgh`], read it back, and verify
    /// bit-exact voxel data and approximate spatial metadata.
    #[test]
    fn test_round_trip_basic() -> Result<()> {
        use crate::format::mgh::write_mgh;

        let dir = tempdir()?;
        let path = dir.path().join("roundtrip.mgh");
        let device: <TestBackend as Backend>::Device = Default::default();

        let nz = 3usize;
        let ny = 4usize;
        let nx = 5usize;
        let data_vec: Vec<f32> = (0..(nz * ny * nx) as u32)
            .map(|i| i as f32 * std::f32::consts::PI / 11.0)
            .collect();

        let tensor_data = TensorData::new(data_vec.clone(), Shape::new([nz, ny, nx]));
        let tensor = Tensor::<TestBackend, 3>::from_data(tensor_data, &device);
        let image = Image::new(
            tensor,
            Point::new([0.0, 0.0, 0.0]),
            Spacing::new([1.0, 1.0, 1.0]),
            Direction::identity(),
        );

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
                "voxel[{}]: expected {}, got {}",
                i,
                exp,
                got
            );
        }
        Ok(())
    }

    // ── Round-trip with MGZ compression ─────────────────────────────────

    #[test]
    fn test_round_trip_mgz() -> Result<()> {
        use crate::format::mgh::write_mgh;

        let dir = tempdir()?;
        let path = dir.path().join("roundtrip.mgz");
        let device: <TestBackend as Backend>::Device = Default::default();

        let nz = 2usize;
        let ny = 3usize;
        let nx = 4usize;
        let data_vec: Vec<f32> = (0..(nz * ny * nx) as u32)
            .map(|i| (i as f32).sqrt() + 0.5)
            .collect();

        let tensor_data = TensorData::new(data_vec.clone(), Shape::new([nz, ny, nx]));
        let tensor = Tensor::<TestBackend, 3>::from_data(tensor_data, &device);
        let image = Image::new(
            tensor,
            Point::new([1.0, 2.0, 3.0]),
            Spacing::new([0.5, 0.5, 0.5]),
            Direction::identity(),
        );

        write_mgh(&image, &path)?;

        // Confirm the file is actually gzip-compressed.
        let file_bytes = std::fs::read(&path)?;
        assert_eq!(file_bytes[0], 0x1f, "First byte must be gzip magic 0x1f");
        assert_eq!(file_bytes[1], 0x8b, "Second byte must be gzip magic 0x8b");

        let loaded = read_mgh::<TestBackend, _>(&path, &device)?;
        assert_eq!(loaded.shape(), [nz, ny, nx]);

        let td = loaded.data().clone().to_data();
        let loaded_vals = td.as_slice::<f32>().unwrap();
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

    // ── Round-trip with non-default spatial metadata ─────────────────────

    /// Write with 90° Z-rotation and anisotropic spacing, read back, and
    /// compare spatial metadata against analytically derived values.
    #[test]
    fn test_round_trip_nondefault_spatial() -> Result<()> {
        use crate::format::mgh::write_mgh;

        let dir = tempdir()?;
        let path = dir.path().join("spatial_rt.mgh");
        let device: <TestBackend as Backend>::Device = Default::default();

        let nz = 2usize;
        let ny = 3usize;
        let nx = 4usize;

        // 90° rotation around Z — every matrix entry exact in f32.
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

        let tensor_data = TensorData::new(data_vec.clone(), Shape::new([nz, ny, nx]));
        let tensor = Tensor::<TestBackend, 3>::from_data(tensor_data, &device);
        let image = Image::new(tensor, origin, spacing, direction);

        write_mgh(&image, &path)?;
        let loaded = read_mgh::<TestBackend, _>(&path, &device)?;
        assert_eq!(loaded.shape(), [nz, ny, nx]);

        // Spacing (f32-exact values round-trip without loss).
        let sp = loaded.spacing();
        assert!((sp[0] - 0.5).abs() < 1e-6, "spacing[0]={}", sp[0]);
        assert!((sp[1] - 0.75).abs() < 1e-6, "spacing[1]={}", sp[1]);
        assert!((sp[2] - 1.25).abs() < 1e-6, "spacing[2]={}", sp[2]);

        // Direction.
        let d = loaded.direction();
        assert!((d[(0, 0)] - 0.0).abs() < 1e-6);
        assert!((d[(1, 0)] - 1.0).abs() < 1e-6);
        assert!((d[(0, 1)] - (-1.0)).abs() < 1e-6);
        assert!((d[(1, 1)] - 0.0).abs() < 1e-6);
        assert!((d[(2, 2)] - 1.0).abs() < 1e-6);

        // Origin.
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

        // Voxel data.
        let td = loaded.data().clone().to_data();
        let loaded_vals = td.as_slice::<f32>().unwrap();
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

    // ── MghReader struct ────────────────────────────────────────────────

    #[test]
    fn test_mgh_reader_struct_delegates() -> Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("struct.mgh");
        let device: <TestBackend as Backend>::Device = Default::default();

        let values: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let data_bytes: Vec<u8> = values.iter().flat_map(|v| v.to_be_bytes()).collect();
        let mgh = build_mgh_bytes(
            1,
            [2, 2, 2],
            MRI_FLOAT,
            [1.0, 1.0, 1.0],
            IDENTITY_DIR,
            [0.0, 0.0, 0.0],
            &data_bytes,
        );
        std::fs::write(&path, &mgh)?;

        let image = MghReader::read::<TestBackend, _>(&path, &device)?;
        assert_eq!(image.shape(), [2, 2, 2]);

        let td = image.data().clone().to_data();
        let loaded = td.as_slice::<f32>().unwrap();
        for (i, (&got, &exp)) in loaded.iter().zip(values.iter()).enumerate() {
            assert_eq!(got, exp, "voxel[{}]", i);
        }
        Ok(())
    }

    // ── Negative test: unsupported data type ────────────────────────────

    #[test]
    fn test_read_unsupported_type_code() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("bad_type.mgh");
        let device: <TestBackend as Backend>::Device = Default::default();

        let mgh = build_mgh_bytes(
            1,
            [2, 2, 2],
            99, // invalid type code
            [1.0, 1.0, 1.0],
            IDENTITY_DIR,
            [0.0, 0.0, 0.0],
            &vec![0u8; 2 * 2 * 2], // arbitrary
        );
        std::fs::write(&path, &mgh).unwrap();

        let result = read_mgh::<TestBackend, _>(&path, &device);
        assert!(result.is_err(), "Unsupported type code must fail");
        let err_msg = format!("{:#}", result.unwrap_err());
        assert!(
            err_msg.contains("data type"),
            "Error must mention 'data type', got: {}",
            err_msg
        );
    }

    // ── Negative test: truncated file ───────────────────────────────────

    #[test]
    fn test_read_truncated_file() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("truncated.mgh");
        let device: <TestBackend as Backend>::Device = Default::default();

        // Write only a partial header (100 bytes out of 284).
        let partial = vec![0u8; 100];
        // Patch version=1 at offset 0 so the version check passes before
        // the truncation is detected.
        let mut buf = partial;
        buf[0..4].copy_from_slice(&1_i32.to_be_bytes());
        buf[4..8].copy_from_slice(&2_i32.to_be_bytes()); // width
        buf[8..12].copy_from_slice(&2_i32.to_be_bytes()); // height
        buf[12..16].copy_from_slice(&2_i32.to_be_bytes()); // depth
        buf[16..20].copy_from_slice(&1_i32.to_be_bytes()); // nframes
        buf[20..24].copy_from_slice(&MRI_FLOAT.to_be_bytes());
        std::fs::write(&path, &buf).unwrap();

        let result = read_mgh::<TestBackend, _>(&path, &device);
        assert!(result.is_err(), "Truncated file must fail");
    }
}
