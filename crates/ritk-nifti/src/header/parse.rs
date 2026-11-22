//! NIfTI single-file header decoding in either byte order.

use super::raw::{read_array, read_f32x4_as_f64, read_f64x4, read_field};
use super::types::{
    HeaderVersion, NiftiHeader, NIFTI1_HEADER_LEN, NIFTI1_MAGIC_SINGLE_FILE, NIFTI2_HEADER_LEN,
    NIFTI2_MAGIC_SINGLE_FILE,
};
use super::validate::{
    validate_bitpix, validate_dims, validate_i64_vox_offset, validate_vox_offset,
};
use super::NiftiDatatype;
use anyhow::{bail, Context, Result};
use consus_core::ByteOrder;

impl NiftiHeader {
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
}
