use super::convert::checked_f64_to_f32;
use super::convert::{f64_affine_to_f32, f64_to_f32, f64x4_to_f32x4};
use super::raw::{
    read_array, read_f32x4_as_f64, read_f64x4, read_field, write_f32x4, write_f64x4, write_field,
};
use super::validate::{
    checked_lane, checked_spatial_pixdim, dims_for_version, qfac_from_pixdim,
    qform_quaternion_scalar, validate_bitpix, validate_dims, validate_i64_vox_offset,
    validate_vox_offset, volume_count,
};
use super::{NiftiDatatype, NiftiLane};
use anyhow::{anyhow, bail, Context, Result};
use consus_core::ByteOrder;

const NIFTI1_HEADER_LEN: usize = 348;
const NIFTI1_SINGLE_FILE_VOX_OFFSET: usize = 352;
const NIFTI2_HEADER_LEN: usize = 540;
const NIFTI2_SINGLE_FILE_VOX_OFFSET: usize = 544;
const NIFTI1_MAGIC_SINGLE_FILE: [u8; 4] = *b"n+1\0";
const NIFTI2_MAGIC_SINGLE_FILE: [u8; 8] = *b"n+2\0\r\n\x1a\n";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum HeaderVersion {
    One,
    Two,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum HeaderAxis {
    Volume,
    Acquisition,
}

impl HeaderVersion {
    pub(super) const fn single_file_vox_offset(self) -> usize {
        match self {
            Self::One => NIFTI1_SINGLE_FILE_VOX_OFFSET,
            Self::Two => NIFTI2_SINGLE_FILE_VOX_OFFSET,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct NiftiHeader {
    pub(crate) version: HeaderVersion,
    pub(crate) dim: [usize; 8],
    pub(crate) datatype: NiftiDatatype,
    pub(crate) pixdim: [f64; 8],
    pub(crate) vox_offset: usize,
    pub(crate) qform_code: i32,
    pub(crate) sform_code: i32,
    pub(crate) quatern_b: f64,
    pub(crate) quatern_c: f64,
    pub(crate) quatern_d: f64,
    pub(crate) quatern_x: f64,
    pub(crate) quatern_y: f64,
    pub(crate) quatern_z: f64,
    pub(crate) srow_x: [f64; 4],
    pub(crate) srow_y: [f64; 4],
    pub(crate) srow_z: [f64; 4],
    pub(crate) scl_slope: f64,
    pub(crate) scl_inter: f64,
    pub(crate) xyzt_units: i32,
    endian: ByteOrder,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct HeaderDims {
    pub(crate) nx: usize,
    pub(crate) ny: usize,
    pub(crate) nz: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct HeaderSpatial {
    pub(crate) pixdim: [f64; 8],
    pub(crate) srow_x: [f64; 4],
    pub(crate) srow_y: [f64; 4],
    pub(crate) srow_z: [f64; 4],
}

impl NiftiHeader {
    /// Build a NIfTI-1 header for a single volume.
    #[cfg(test)]
    pub(crate) fn new_volume(
        dims: HeaderDims,
        datatype: NiftiDatatype,
        spatial: HeaderSpatial,
    ) -> Result<Self> {
        Self::new_with_version(
            HeaderVersion::One,
            dims,
            1,
            HeaderAxis::Volume,
            datatype,
            spatial,
        )
    }

    /// Build a header for `volumes` values sharing the `dims` spatial grid.
    ///
    /// `axis` selects rank 3 or rank 4 independently of volume count, allowing
    /// a one-entry acquisition axis to remain rank 4.
    pub(crate) fn new_with_version(
        version: HeaderVersion,
        dims: HeaderDims,
        volumes: usize,
        axis: HeaderAxis,
        datatype: NiftiDatatype,
        spatial: HeaderSpatial,
    ) -> Result<Self> {
        let dim = dims_for_version(version, dims, volumes, axis)?;

        Ok(Self {
            version,
            dim,
            datatype,
            pixdim: spatial.pixdim,
            vox_offset: version.single_file_vox_offset(),
            qform_code: 0,
            sform_code: 1,
            quatern_b: 0.0,
            quatern_c: 0.0,
            quatern_d: 0.0,
            quatern_x: 0.0,
            quatern_y: 0.0,
            quatern_z: 0.0,
            srow_x: spatial.srow_x,
            srow_y: spatial.srow_y,
            srow_z: spatial.srow_z,
            scl_slope: 1.0,
            scl_inter: 0.0,
            xyzt_units: 2,
            endian: ByteOrder::LittleEndian,
        })
    }

    pub(crate) fn parse(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < 4 {
            bail!(
                "NIfTI header requires at least 4 bytes, got {}",
                bytes.len()
            );
        }

        let little = read_field::<i32>(bytes, 0, ByteOrder::LittleEndian)?;
        let big = read_field::<i32>(bytes, 0, ByteOrder::BigEndian)?;
        match (little, big) {
            (348, _) => Self::parse_nifti1(bytes, ByteOrder::LittleEndian),
            (_, 348) => Self::parse_nifti1(bytes, ByteOrder::BigEndian),
            (540, _) => Self::parse_nifti2(bytes, ByteOrder::LittleEndian),
            (_, 540) => Self::parse_nifti2(bytes, ByteOrder::BigEndian),
            _ => bail!("Invalid NIfTI sizeof_hdr; expected 348 or 540"),
        }
    }

    fn parse_nifti1(bytes: &[u8], endian: ByteOrder) -> Result<Self> {
        if bytes.len() < NIFTI1_HEADER_LEN {
            bail!(
                "NIfTI-1 header requires {NIFTI1_HEADER_LEN} bytes, got {}",
                bytes.len()
            );
        }

        let magic = read_array::<4>(bytes, 344)?;
        if magic != NIFTI1_MAGIC_SINGLE_FILE {
            bail!("Unsupported NIfTI-1 magic; expected single-file n+1");
        }

        let mut dim = [0_usize; 8];
        for (index, slot) in dim.iter_mut().enumerate() {
            let raw = read_field::<i16>(bytes, 40 + index * 2, endian)?;
            *slot = usize::try_from(raw)
                .with_context(|| format!("NIfTI-1 dim[{index}] must be non-negative, got {raw}"))?;
        }
        validate_dims(dim)?;

        let datatype = NiftiDatatype::from_code(read_field::<i16>(bytes, 70, endian)?)?;
        validate_bitpix(datatype, read_field::<i16>(bytes, 72, endian)?)?;

        let mut pixdim = [0.0_f64; 8];
        for (index, slot) in pixdim.iter_mut().enumerate() {
            *slot = f64::from(read_field::<f32>(bytes, 76 + index * 4, endian)?);
        }

        let vox_offset = f64::from(read_field::<f32>(bytes, 108, endian)?);
        let vox_offset = validate_vox_offset(HeaderVersion::One, vox_offset)?;

        Ok(Self {
            version: HeaderVersion::One,
            dim,
            datatype,
            pixdim,
            vox_offset,
            qform_code: i32::from(read_field::<i16>(bytes, 252, endian)?),
            sform_code: i32::from(read_field::<i16>(bytes, 254, endian)?),
            quatern_b: f64::from(read_field::<f32>(bytes, 256, endian)?),
            quatern_c: f64::from(read_field::<f32>(bytes, 260, endian)?),
            quatern_d: f64::from(read_field::<f32>(bytes, 264, endian)?),
            quatern_x: f64::from(read_field::<f32>(bytes, 268, endian)?),
            quatern_y: f64::from(read_field::<f32>(bytes, 272, endian)?),
            quatern_z: f64::from(read_field::<f32>(bytes, 276, endian)?),
            srow_x: read_f32x4_as_f64(bytes, 280, endian)?,
            srow_y: read_f32x4_as_f64(bytes, 296, endian)?,
            srow_z: read_f32x4_as_f64(bytes, 312, endian)?,
            scl_slope: f64::from(read_field::<f32>(bytes, 112, endian)?),
            scl_inter: f64::from(read_field::<f32>(bytes, 116, endian)?),
            xyzt_units: i32::from(bytes[123]),
            endian,
        })
    }

    fn parse_nifti2(bytes: &[u8], endian: ByteOrder) -> Result<Self> {
        if bytes.len() < NIFTI2_HEADER_LEN {
            bail!(
                "NIfTI-2 header requires {NIFTI2_HEADER_LEN} bytes, got {}",
                bytes.len()
            );
        }

        let magic = read_array::<8>(bytes, 4)?;
        if magic != NIFTI2_MAGIC_SINGLE_FILE {
            bail!("Unsupported NIfTI-2 magic; expected single-file n+2");
        }

        let mut dim = [0_usize; 8];
        for (index, slot) in dim.iter_mut().enumerate() {
            let raw = read_field::<i64>(bytes, 16 + index * 8, endian)?;
            *slot = usize::try_from(raw).with_context(|| {
                format!("NIfTI-2 dim[{index}] must be non-negative and fit usize, got {raw}")
            })?;
        }
        validate_dims(dim)?;

        let datatype = NiftiDatatype::from_code(read_field::<i16>(bytes, 12, endian)?)?;
        validate_bitpix(datatype, read_field::<i16>(bytes, 14, endian)?)?;

        let mut pixdim = [0.0_f64; 8];
        for (index, slot) in pixdim.iter_mut().enumerate() {
            *slot = read_field::<f64>(bytes, 104 + index * 8, endian)?;
        }

        let vox_offset =
            validate_i64_vox_offset(HeaderVersion::Two, read_field::<i64>(bytes, 168, endian)?)?;

        Ok(Self {
            version: HeaderVersion::Two,
            dim,
            datatype,
            pixdim,
            vox_offset,
            qform_code: read_field::<i32>(bytes, 344, endian)?,
            sform_code: read_field::<i32>(bytes, 348, endian)?,
            quatern_b: read_field::<f64>(bytes, 352, endian)?,
            quatern_c: read_field::<f64>(bytes, 360, endian)?,
            quatern_d: read_field::<f64>(bytes, 368, endian)?,
            quatern_x: read_field::<f64>(bytes, 376, endian)?,
            quatern_y: read_field::<f64>(bytes, 384, endian)?,
            quatern_z: read_field::<f64>(bytes, 392, endian)?,
            srow_x: read_f64x4(bytes, 400, endian)?,
            srow_y: read_f64x4(bytes, 432, endian)?,
            srow_z: read_f64x4(bytes, 464, endian)?,
            scl_slope: read_field::<f64>(bytes, 176, endian)?,
            scl_inter: read_field::<f64>(bytes, 184, endian)?,
            xyzt_units: read_field::<i32>(bytes, 500, endian)?,
            endian,
        })
    }

    pub(crate) fn validate_for_encoding(&self) -> Result<()> {
        match self.version {
            HeaderVersion::One => {
                for (index, value) in self.dim.iter().copied().enumerate() {
                    i16::try_from(value)
                        .with_context(|| format!("NIfTI-1 dim[{index}] exceeds i16 capacity"))?;
                }
                i16::try_from(self.qform_code).context("NIfTI-1 qform_code exceeds i16")?;
                i16::try_from(self.sform_code).context("NIfTI-1 sform_code exceeds i16")?;
                u8::try_from(self.xyzt_units).context("NIfTI-1 xyzt_units exceeds u8")?;

                let offset =
                    u32::try_from(self.vox_offset).context("NIfTI-1 vox_offset exceeds u32")?;
                let encoded_offset = checked_f64_to_f32(f64::from(offset), "vox_offset")?;
                if f64::from(encoded_offset) != f64::from(offset) {
                    bail!("NIfTI-1 vox_offset cannot be represented as an exact f32 integer");
                }

                for (index, value) in self.pixdim.iter().copied().enumerate() {
                    let narrowed = checked_f64_to_f32(value, "pixdim")?;
                    if (1..=3).contains(&index) && narrowed <= 0.0 {
                        bail!("NIfTI-1 pixdim[{index}] is not positive after f32 encoding");
                    }
                }
                for value in [
                    self.quatern_b,
                    self.quatern_c,
                    self.quatern_d,
                    self.quatern_x,
                    self.quatern_y,
                    self.quatern_z,
                    self.scl_slope,
                    self.scl_inter,
                ] {
                    checked_f64_to_f32(value, "header field")?;
                }
                for row in [self.srow_x, self.srow_y, self.srow_z] {
                    for value in row {
                        checked_f64_to_f32(value, "srow")?;
                    }
                }
                if self.scl_slope != 0.0 && checked_f64_to_f32(self.scl_slope, "scl_slope")? == 0.0
                {
                    bail!("NIfTI-1 scl_slope underflows to zero and disables scaling");
                }
            }
            HeaderVersion::Two => {
                for (index, value) in self.dim.iter().copied().enumerate() {
                    i64::try_from(value)
                        .with_context(|| format!("NIfTI-2 dim[{index}] exceeds i64 capacity"))?;
                }
                i64::try_from(self.vox_offset).context("NIfTI-2 vox_offset exceeds i64")?;
                for (field, value) in self.float_fields() {
                    if !value.is_finite() {
                        bail!("NIfTI-2 {field} must be finite, got {value}");
                    }
                }
            }
        }
        Ok(())
    }

    fn float_fields(&self) -> impl Iterator<Item = (&'static str, f64)> + '_ {
        self.pixdim
            .iter()
            .copied()
            .map(|value| ("pixdim", value))
            .chain(
                [
                    self.quatern_b,
                    self.quatern_c,
                    self.quatern_d,
                    self.quatern_x,
                    self.quatern_y,
                    self.quatern_z,
                    self.scl_slope,
                    self.scl_inter,
                ]
                .into_iter()
                .map(|value| ("header field", value)),
            )
            .chain(
                [self.srow_x, self.srow_y, self.srow_z]
                    .into_iter()
                    .flatten()
                    .map(|value| ("srow", value)),
            )
    }

    pub(crate) fn encode(&self) -> Vec<u8> {
        match self.version {
            HeaderVersion::One => self.encode_nifti1().to_vec(),
            HeaderVersion::Two => self.encode_nifti2().to_vec(),
        }
    }

    fn encode_nifti1(&self) -> [u8; NIFTI1_HEADER_LEN] {
        let mut out = [0_u8; NIFTI1_HEADER_LEN];
        write_field::<i32>(&mut out, 0, 348);
        for (index, value) in self.dim.iter().copied().enumerate() {
            write_field::<i16>(
                &mut out,
                40 + index * 2,
                i16::try_from(value)
                    .expect("invariant: NIfTI-1 header dims are validated at construction"),
            );
        }
        write_field::<i16>(&mut out, 70, self.datatype.code());
        write_field::<i16>(&mut out, 72, self.datatype.bitpix());
        for (index, value) in self.pixdim.iter().copied().enumerate() {
            write_field::<f32>(&mut out, 76 + index * 4, f64_to_f32(value, "pixdim"));
        }
        let vox_offset = u32::try_from(self.vox_offset)
            .expect("invariant: NIfTI-1 vox_offset fits u32 after validation");
        write_field::<f32>(
            &mut out,
            108,
            f64_to_f32(f64::from(vox_offset), "vox_offset"),
        );
        write_field::<f32>(&mut out, 112, f64_to_f32(self.scl_slope, "scl_slope"));
        write_field::<f32>(&mut out, 116, f64_to_f32(self.scl_inter, "scl_inter"));
        out[123] = u8::try_from(self.xyzt_units)
            .expect("invariant: NIfTI-1 xyzt_units is set to a u8-compatible value");
        write_field::<i16>(
            &mut out,
            252,
            i16::try_from(self.qform_code).expect("invariant: NIfTI-1 qform_code fits i16"),
        );
        write_field::<i16>(
            &mut out,
            254,
            i16::try_from(self.sform_code).expect("invariant: NIfTI-1 sform_code fits i16"),
        );
        write_field::<f32>(&mut out, 256, f64_to_f32(self.quatern_b, "quatern_b"));
        write_field::<f32>(&mut out, 260, f64_to_f32(self.quatern_c, "quatern_c"));
        write_field::<f32>(&mut out, 264, f64_to_f32(self.quatern_d, "quatern_d"));
        write_field::<f32>(&mut out, 268, f64_to_f32(self.quatern_x, "quatern_x"));
        write_field::<f32>(&mut out, 272, f64_to_f32(self.quatern_y, "quatern_y"));
        write_field::<f32>(&mut out, 276, f64_to_f32(self.quatern_z, "quatern_z"));
        write_f32x4(&mut out, 280, self.srow_x);
        write_f32x4(&mut out, 296, self.srow_y);
        write_f32x4(&mut out, 312, self.srow_z);
        out[344..348].copy_from_slice(&NIFTI1_MAGIC_SINGLE_FILE);
        out
    }

    fn encode_nifti2(&self) -> [u8; NIFTI2_HEADER_LEN] {
        let mut out = [0_u8; NIFTI2_HEADER_LEN];
        write_field::<i32>(&mut out, 0, 540);
        out[4..12].copy_from_slice(&NIFTI2_MAGIC_SINGLE_FILE);
        write_field::<i16>(&mut out, 12, self.datatype.code());
        write_field::<i16>(&mut out, 14, self.datatype.bitpix());
        for (index, value) in self.dim.iter().copied().enumerate() {
            write_field::<i64>(
                &mut out,
                16 + index * 8,
                i64::try_from(value)
                    .expect("invariant: NIfTI-2 header dims are validated at construction"),
            );
        }
        for (index, value) in self.pixdim.iter().copied().enumerate() {
            write_field::<f64>(&mut out, 104 + index * 8, value);
        }
        write_field::<i64>(
            &mut out,
            168,
            i64::try_from(self.vox_offset).expect("invariant: vox_offset fits i64"),
        );
        write_field::<f64>(&mut out, 176, self.scl_slope);
        write_field::<f64>(&mut out, 184, self.scl_inter);
        write_field::<i32>(&mut out, 344, self.qform_code);
        write_field::<i32>(&mut out, 348, self.sform_code);
        write_field::<f64>(&mut out, 352, self.quatern_b);
        write_field::<f64>(&mut out, 360, self.quatern_c);
        write_field::<f64>(&mut out, 368, self.quatern_d);
        write_field::<f64>(&mut out, 376, self.quatern_x);
        write_field::<f64>(&mut out, 384, self.quatern_y);
        write_field::<f64>(&mut out, 392, self.quatern_z);
        write_f64x4(&mut out, 400, self.srow_x);
        write_f64x4(&mut out, 432, self.srow_y);
        write_f64x4(&mut out, 464, self.srow_z);
        write_field::<i32>(&mut out, 500, self.xyzt_units);
        out
    }

    /// Decodes one lane of `N` raw bytes in the header's byte order.
    ///
    /// The byte order is a runtime property of the file and the datatype is a
    /// runtime enum, so the scalar enters as a type parameter rather than as
    /// four functions that differ only in which one they name.
    pub(crate) fn read_lane<T, const N: usize>(&self, raw: [u8; N]) -> T
    where
        T: NiftiLane<N>,
    {
        match self.endian {
            ByteOrder::LittleEndian => T::from_little_endian(raw),
            ByteOrder::BigEndian => T::from_big_endian(raw),
        }
    }

    pub(crate) fn read_f32_voxel(&self, raw: &[u8]) -> Result<f32> {
        Ok(match self.datatype {
            NiftiDatatype::Uint8 => f32::from(raw[0]),
            NiftiDatatype::Int16 => f32::from(self.read_lane::<i16, 2>(checked_lane::<2>(raw)?)),
            NiftiDatatype::Int32 => self.read_lane::<i32, 4>(checked_lane::<4>(raw)?) as f32,
            NiftiDatatype::Float32 => self.read_lane::<f32, 4>(checked_lane::<4>(raw)?),
            NiftiDatatype::Uint32 => self.read_lane::<u32, 4>(checked_lane::<4>(raw)?) as f32,
            NiftiDatatype::Int8
            | NiftiDatatype::Uint16
            | NiftiDatatype::Uint64
            | NiftiDatatype::Int64
            | NiftiDatatype::Float64 => {
                bail!(
                    "NIfTI image convenience reader does not support stored datatype {:?}",
                    self.datatype
                )
            }
        })
    }

    pub(crate) fn read_label_voxel(&self, raw: &[u8]) -> Result<u32> {
        Ok(match self.datatype {
            NiftiDatatype::Uint8 => u32::from(raw[0]),
            NiftiDatatype::Int16 => {
                let value = self.read_lane::<i16, 2>(checked_lane::<2>(raw)?);
                u32::try_from(value).with_context(|| {
                    format!("NIfTI label voxel must be non-negative, got {value}")
                })?
            }
            NiftiDatatype::Int32 => {
                let value = self.read_lane::<i32, 4>(checked_lane::<4>(raw)?);
                u32::try_from(value).with_context(|| {
                    format!("NIfTI label voxel must be non-negative, got {value}")
                })?
            }
            NiftiDatatype::Float32 => self
                .read_lane::<f32, 4>(checked_lane::<4>(raw)?)
                .max(0.0)
                .round() as u32,
            NiftiDatatype::Uint32 => self.read_lane::<u32, 4>(checked_lane::<4>(raw)?),
            NiftiDatatype::Int8
            | NiftiDatatype::Uint16
            | NiftiDatatype::Uint64
            | NiftiDatatype::Int64
            | NiftiDatatype::Float64 => {
                bail!(
                    "NIfTI label convenience reader does not support stored datatype {:?}",
                    self.datatype
                )
            }
        })
    }

    pub(crate) fn affine(&self) -> Result<[[f32; 4]; 4]> {
        if self.sform_code > 0 {
            Ok([
                f64x4_to_f32x4(self.srow_x, "srow_x")?,
                f64x4_to_f32x4(self.srow_y, "srow_y")?,
                f64x4_to_f32x4(self.srow_z, "srow_z")?,
                [0.0, 0.0, 0.0, 1.0],
            ])
        } else if self.qform_code > 0 {
            let b = self.quatern_b;
            let c = self.quatern_c;
            let d = self.quatern_d;
            let a = qform_quaternion_scalar(b, c, d)?;
            let qfac = qfac_from_pixdim(self.pixdim[0])?;
            let [dx, dy, dz_abs] = checked_spatial_pixdim(self.pixdim)?;
            let dz = dz_abs * qfac;

            let r11 = a * a + b * b - c * c - d * d;
            let r12 = 2.0 * b * c - 2.0 * a * d;
            let r13 = 2.0 * b * d + 2.0 * a * c;
            let r21 = 2.0 * b * c + 2.0 * a * d;
            let r22 = a * a + c * c - b * b - d * d;
            let r23 = 2.0 * c * d - 2.0 * a * b;
            let r31 = 2.0 * b * d - 2.0 * a * c;
            let r32 = 2.0 * c * d + 2.0 * a * b;
            let r33 = a * a + d * d - c * c - b * b;

            let affine = [
                [r11 * dx, r12 * dy, r13 * dz, self.quatern_x],
                [r21 * dx, r22 * dy, r23 * dz, self.quatern_y],
                [r31 * dx, r32 * dy, r33 * dz, self.quatern_z],
                [0.0, 0.0, 0.0, 1.0],
            ];
            f64_affine_to_f32(affine)
        } else {
            let [dx, dy, dz] = checked_spatial_pixdim(self.pixdim)?;
            Ok([
                [f64_to_f32(dx, "pixdim[1]"), 0.0, 0.0, 0.0],
                [0.0, f64_to_f32(dy, "pixdim[2]"), 0.0, 0.0],
                [0.0, 0.0, f64_to_f32(dz, "pixdim[3]"), 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ])
        }
    }

    /// Volumes this header declares on its spatial grid.
    ///
    /// One for a rank-3 volume header; `dim[4]` for a rank-4 acquisition series.
    pub(crate) fn volume_count(&self) -> usize {
        volume_count(self.dim)
    }

    /// Voxels in one volume of this header's spatial grid.
    pub(crate) fn voxels_per_volume(&self) -> Result<usize> {
        crate::shape::checked_voxel_count(self.dim[1], self.dim[2], self.dim[3])
    }

    /// Byte range of the whole payload, spanning every declared volume.
    pub(crate) fn volume_byte_range(&self, byte_len: usize) -> Result<std::ops::Range<usize>> {
        let voxel_count = self
            .voxels_per_volume()?
            .checked_mul(self.volume_count())
            .ok_or_else(|| anyhow!("NIfTI series voxel count overflows usize"))?;
        let data_len = voxel_count
            .checked_mul(self.datatype.byte_width())
            .ok_or_else(|| anyhow!("NIfTI data byte count overflows usize"))?;
        let end = self
            .vox_offset
            .checked_add(data_len)
            .ok_or_else(|| anyhow!("NIfTI data byte range overflows usize"))?;
        if byte_len < end {
            bail!("NIfTI payload truncated: need {end} bytes, got {byte_len}");
        }
        Ok(self.vox_offset..end)
    }
}

#[cfg(test)]
pub(crate) fn write_single_file_bytes(header: &NiftiHeader, data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(header.vox_offset + data.len());
    out.extend_from_slice(&header.encode());
    out.extend_from_slice(&[0, 0, 0, 0]);
    out.extend_from_slice(data);
    out
}
