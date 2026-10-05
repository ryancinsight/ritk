use super::convert::{
    encode_header_scalar, intensity_from_signed_voxel, intensity_from_unsigned_voxel,
    label_value_from_intensity, validate_nifti1_scalar,
};
use super::error::NiftiHeaderError;
use super::raw::{
    read_array, read_f32x4_as_f64, read_f64x4, read_field, write_f32x4, write_f64x4, write_field,
};
use super::validate::{
    checked_lane, checked_spatial_pixdim, dims_for_version, qfac_from_pixdim,
    qform_quaternion_scalar, validate_bitpix, validate_dims, validate_i64_vox_offset,
    validate_vox_offset, volume_count,
};
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

impl HeaderVersion {
    pub(super) const fn single_file_vox_offset(self) -> usize {
        match self {
            Self::One => NIFTI1_SINGLE_FILE_VOX_OFFSET,
            Self::Two => NIFTI2_SINGLE_FILE_VOX_OFFSET,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum NiftiDatatype {
    Uint8,
    Int8,
    Uint16,
    Int16,
    Uint32,
    Int32,
    Float32,
    Float64,
    Int64,
    Uint64,
}

impl NiftiDatatype {
    pub(crate) const fn code(self) -> i16 {
        match self {
            Self::Uint8 => 2,
            Self::Int8 => 256,
            Self::Uint16 => 512,
            Self::Int16 => 4,
            Self::Uint32 => 768,
            Self::Int32 => 8,
            Self::Float32 => 16,
            Self::Float64 => 64,
            Self::Int64 => 1024,
            Self::Uint64 => 1280,
        }
    }

    pub(super) const fn bitpix(self) -> i16 {
        match self {
            Self::Uint8 | Self::Int8 => 8,
            Self::Uint16 | Self::Int16 => 16,
            Self::Uint32 | Self::Int32 | Self::Float32 => 32,
            Self::Float64 | Self::Int64 | Self::Uint64 => 64,
        }
    }

    pub(crate) const fn byte_width(self) -> usize {
        match self {
            Self::Uint8 | Self::Int8 => 1,
            Self::Uint16 | Self::Int16 => 2,
            Self::Uint32 | Self::Int32 | Self::Float32 => 4,
            Self::Float64 | Self::Int64 | Self::Uint64 => 8,
        }
    }

    pub(crate) fn from_code(code: i16) -> Result<Self> {
        match code {
            2 => Ok(Self::Uint8),
            256 => Ok(Self::Int8),
            512 => Ok(Self::Uint16),
            4 => Ok(Self::Int16),
            768 => Ok(Self::Uint32),
            8 => Ok(Self::Int32),
            16 => Ok(Self::Float32),
            64 => Ok(Self::Float64),
            1024 => Ok(Self::Int64),
            1280 => Ok(Self::Uint64),
            _ => bail!("Unsupported NIfTI datatype code {code}"),
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
        Self::new_with_version(HeaderVersion::One, dims, 1, datatype, spatial)
    }

    /// Build a header for `volumes` volumes sharing the `dims` spatial grid.
    ///
    /// One volume produces a rank-3 header; more produce a rank-4 header with
    /// the count in `dim[4]`.
    pub(crate) fn new_with_version(
        version: HeaderVersion,
        dims: HeaderDims,
        volumes: usize,
        datatype: NiftiDatatype,
        spatial: HeaderSpatial,
    ) -> Result<Self> {
        let dim = dims_for_version(version, dims, volumes)?;

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
            *slot = usize::from(read_field::<u16>(bytes, 40 + index * 2, endian)?);
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

    pub(crate) fn encode(&self) -> std::result::Result<Vec<u8>, NiftiHeaderError> {
        match self.version {
            HeaderVersion::One => {
                self.validate_nifti1_fields()?;
                Ok(self.encode_nifti1().to_vec())
            }
            HeaderVersion::Two => Ok(self.encode_nifti2().to_vec()),
        }
    }

    fn validate_nifti1_fields(&self) -> std::result::Result<(), NiftiHeaderError> {
        for value in self.pixdim {
            validate_nifti1_scalar(value, "pixdim").map_err(NiftiHeaderError::from)?;
        }
        for row in [&self.srow_x, &self.srow_y, &self.srow_z] {
            for value in row {
                validate_nifti1_scalar(*value, "sform").map_err(NiftiHeaderError::from)?;
            }
        }
        for (value, field) in [
            (self.scl_slope, "scl_slope"),
            (self.scl_inter, "scl_inter"),
            (self.quatern_b, "quatern_b"),
            (self.quatern_c, "quatern_c"),
            (self.quatern_d, "quatern_d"),
            (self.quatern_x, "quatern_x"),
            (self.quatern_y, "quatern_y"),
            (self.quatern_z, "quatern_z"),
        ] {
            validate_nifti1_scalar(value, field).map_err(NiftiHeaderError::from)?;
        }
        Ok(())
    }

    fn encode_nifti1(&self) -> [u8; NIFTI1_HEADER_LEN] {
        let mut out = [0_u8; NIFTI1_HEADER_LEN];
        write_field::<i32>(&mut out, 0, 348);
        for (index, value) in self.dim.iter().copied().enumerate() {
            write_field::<u16>(
                &mut out,
                40 + index * 2,
                u16::try_from(value)
                    .expect("invariant: NIfTI-1 header dims are validated at construction"),
            );
        }
        write_field::<i16>(&mut out, 70, self.datatype.code());
        write_field::<i16>(&mut out, 72, self.datatype.bitpix());
        for (index, value) in self.pixdim.iter().copied().enumerate() {
            write_field::<f32>(&mut out, 76 + index * 4, encode_header_scalar(value));
        }
        let voxel_offset =
            u32::try_from(self.vox_offset).expect("invariant: NIfTI-1 voxel offset fits u32");
        write_field::<f32>(&mut out, 108, encode_header_scalar(f64::from(voxel_offset)));
        write_field::<f32>(&mut out, 112, encode_header_scalar(self.scl_slope));
        write_field::<f32>(&mut out, 116, encode_header_scalar(self.scl_inter));
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
        write_field::<f32>(&mut out, 256, encode_header_scalar(self.quatern_b));
        write_field::<f32>(&mut out, 260, encode_header_scalar(self.quatern_c));
        write_field::<f32>(&mut out, 264, encode_header_scalar(self.quatern_d));
        write_field::<f32>(&mut out, 268, encode_header_scalar(self.quatern_x));
        write_field::<f32>(&mut out, 272, encode_header_scalar(self.quatern_y));
        write_field::<f32>(&mut out, 276, encode_header_scalar(self.quatern_z));
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
            NiftiDatatype::Int32 => {
                intensity_from_signed_voxel(self.read_lane::<i32, 4>(checked_lane::<4>(raw)?))
            }
            NiftiDatatype::Float32 => self.read_lane::<f32, 4>(checked_lane::<4>(raw)?),
            NiftiDatatype::Uint32 => {
                intensity_from_unsigned_voxel(self.read_lane::<u32, 4>(checked_lane::<4>(raw)?))
            }
            NiftiDatatype::Int8
            | NiftiDatatype::Uint16
            | NiftiDatatype::Float64
            | NiftiDatatype::Int64
            | NiftiDatatype::Uint64 => {
                bail!(
                    "NIfTI datatype {} requires the stored-sample reader",
                    self.datatype.code()
                )
            }
        })
    }

    pub(crate) fn read_label_voxel(&self, raw: &[u8]) -> Result<u32> {
        Ok(match self.datatype {
            NiftiDatatype::Uint8 => u32::from(raw[0]),
            NiftiDatatype::Int8 => {
                let value = self.read_lane::<i8, 1>(checked_lane::<1>(raw)?);
                u32::try_from(value).with_context(|| {
                    format!("NIfTI label voxel must be non-negative, got {value}")
                })?
            }
            NiftiDatatype::Uint16 => u32::from(self.read_lane::<u16, 2>(checked_lane::<2>(raw)?)),
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
            NiftiDatatype::Float32 => {
                label_value_from_intensity(self.read_lane::<f32, 4>(checked_lane::<4>(raw)?))
            }
            NiftiDatatype::Uint32 => self.read_lane::<u32, 4>(checked_lane::<4>(raw)?),
            NiftiDatatype::Int64 => {
                let value = self.read_lane::<i64, 8>(checked_lane::<8>(raw)?);
                u32::try_from(value)
                    .with_context(|| format!("NIfTI label voxel must fit UInt32, got {value}"))?
            }
            NiftiDatatype::Uint64 => {
                let value = self.read_lane::<u64, 8>(checked_lane::<8>(raw)?);
                u32::try_from(value)
                    .with_context(|| format!("NIfTI label voxel must fit UInt32, got {value}"))?
            }
            NiftiDatatype::Float64 => {
                bail!("NIfTI Float64 labels require an explicit label conversion")
            }
        })
    }

    pub(crate) fn affine(&self) -> Result<[[f64; 4]; 4]> {
        if self.sform_code > 0 {
            Ok([self.srow_x, self.srow_y, self.srow_z, [0.0, 0.0, 0.0, 1.0]])
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
            Ok(affine)
        } else {
            let [dx, dy, dz] = checked_spatial_pixdim(self.pixdim)?;
            Ok([
                [dx, 0.0, 0.0, 0.0],
                [0.0, dy, 0.0, 0.0],
                [0.0, 0.0, dz, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ])
        }
    }

    pub(crate) const fn byte_order(&self) -> ByteOrder {
        self.endian
    }

    pub(crate) fn spatial_unit_scale(&self) -> Result<f64> {
        match self.xyzt_units & 0x07 {
            1 => Ok(1000.0),
            2 => Ok(1.0),
            3 => Ok(0.001),
            units => bail!("NIfTI spatial units code {units} is unknown or unsupported"),
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
    out.extend_from_slice(
        &header
            .encode()
            .expect("test header fields are representable in the selected NIfTI version"),
    );
    out.extend_from_slice(&[0, 0, 0, 0]);
    out.extend_from_slice(data);
    out
}

#[cfg(test)]
mod tests {
    use super::{HeaderDims, HeaderSpatial, HeaderVersion, NiftiDatatype, NiftiHeader};

    #[test]
    fn header_round_trip_preserves_nifti1_core_fields() {
        let header = NiftiHeader::new_volume(
            HeaderDims {
                nx: 4,
                ny: 3,
                nz: 2,
            },
            NiftiDatatype::Float32,
            HeaderSpatial {
                pixdim: [1.0, 0.75, 1.5, 2.0, 1.0, 1.0, 1.0, 1.0],
                srow_x: [-0.75, 0.0, 0.0, -11.0],
                srow_y: [0.0, -1.5, 0.0, 7.5],
                srow_z: [0.0, 0.0, 2.0, 3.25],
            },
        )
        .expect("valid header");

        let parsed = NiftiHeader::parse(
            &header
                .encode()
                .expect("NIfTI-1 test header fields are representable"),
        )
        .expect("encoded header parses");
        assert_eq!(parsed.version, HeaderVersion::One);
        assert_eq!(parsed.dim, [3, 4, 3, 2, 1, 1, 1, 1]);
        assert_eq!(parsed.datatype, NiftiDatatype::Float32);
        assert_eq!(parsed.srow_x, [-0.75, 0.0, 0.0, -11.0]);
        assert_eq!(parsed.vox_offset, 352);
    }

    #[test]
    fn header_round_trip_preserves_nifti2_core_fields() {
        let header = NiftiHeader::new_with_version(
            HeaderVersion::Two,
            HeaderDims {
                nx: 70_000,
                ny: 3,
                nz: 2,
            },
            1,
            NiftiDatatype::Uint32,
            HeaderSpatial {
                pixdim: [1.0, 0.75, 1.5, 2.0, 1.0, 1.0, 1.0, 1.0],
                srow_x: [-0.75, 0.0, 0.0, -11.0],
                srow_y: [0.0, -1.5, 0.0, 7.5],
                srow_z: [0.0, 0.0, 2.0, 3.25],
            },
        )
        .expect("valid header");

        let parsed = NiftiHeader::parse(
            &header
                .encode()
                .expect("NIfTI-2 test header fields are encodable"),
        )
        .expect("encoded header parses");
        assert_eq!(parsed.version, HeaderVersion::Two);
        assert_eq!(parsed.dim, [3, 70_000, 3, 2, 1, 1, 1, 1]);
        assert_eq!(parsed.datatype, NiftiDatatype::Uint32);
        assert_eq!(parsed.srow_x, [-0.75, 0.0, 0.0, -11.0]);
        assert_eq!(parsed.vox_offset, 544);
    }

    #[test]
    fn nifti1_rejects_dimensions_above_u16() {
        let err = NiftiHeader::new_volume(
            HeaderDims {
                nx: 70_000,
                ny: 1,
                nz: 1,
            },
            NiftiDatatype::Float32,
            HeaderSpatial {
                pixdim: [1.0; 8],
                srow_x: [1.0, 0.0, 0.0, 0.0],
                srow_y: [0.0, 1.0, 0.0, 0.0],
                srow_z: [0.0, 0.0, 1.0, 0.0],
            },
        )
        .expect_err("NIfTI-1 dimensions above u16 must be rejected");

        assert!(
            err.to_string().contains("u16"),
            "error must name NIfTI-1 dimension bound: {err}"
        );
    }

    #[test]
    fn nifti1_rejects_finite_values_outside_binary32_range() {
        let mut header = NiftiHeader::new_volume(
            HeaderDims {
                nx: 1,
                ny: 1,
                nz: 1,
            },
            NiftiDatatype::Float32,
            HeaderSpatial {
                pixdim: [1.0; 8],
                srow_x: [1.0, 0.0, 0.0, 0.0],
                srow_y: [0.0, 1.0, 0.0, 0.0],
                srow_z: [0.0, 0.0, 1.0, 0.0],
            },
        )
        .expect("valid NIfTI-1 header");
        header.srow_x[3] = f64::MAX;
        let error = header
            .encode()
            .expect_err("out-of-range geometry cannot be encoded as NIfTI-1");
        assert!(error.to_string().contains("NIfTI-1 sform"));
    }
}

/// A NIfTI header scalar decodable from `N` raw bytes of either byte order.
///
/// Sealed by visibility: the set is exactly the scalars `NiftiDatatype` admits,
/// and it grows here when that enum grows.
pub(crate) trait NiftiLane<const N: usize>: Sized {
    /// Decodes the value from little-endian bytes.
    fn from_little_endian(raw: [u8; N]) -> Self;
    /// Decodes the value from big-endian bytes.
    fn from_big_endian(raw: [u8; N]) -> Self;
}

macro_rules! nifti_lane {
    ($($scalar:ty => $width:literal),+ $(,)?) => {
        $(impl NiftiLane<$width> for $scalar {
            #[inline]
            fn from_little_endian(raw: [u8; $width]) -> Self {
                Self::from_le_bytes(raw)
            }

            #[inline]
            fn from_big_endian(raw: [u8; $width]) -> Self {
                Self::from_be_bytes(raw)
            }
        })+
    };
}

// The impls are byte-for-byte identical past the scalar and its width, and the
// bodies are the std constructors themselves; a declarative macro is the one
// place this file needs to name each scalar, and it is generated once here
// rather than written out four times.
nifti_lane!(i8 => 1, u16 => 2, i16 => 2, u32 => 4, i32 => 4, u64 => 8, i64 => 8, f32 => 4, f64 => 8);
