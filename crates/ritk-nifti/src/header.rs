use anyhow::{anyhow, bail, Context, Result};

const HEADER_LEN: usize = 348;
const SINGLE_FILE_VOX_OFFSET: usize = 352;
const MAGIC_NII: [u8; 4] = *b"n+1\0";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum NiftiDatatype {
    Float32,
    Uint32,
}

impl NiftiDatatype {
    pub(crate) const fn code(self) -> i16 {
        match self {
            Self::Float32 => 16,
            Self::Uint32 => 768,
        }
    }

    const fn bitpix(self) -> i16 {
        match self {
            Self::Float32 | Self::Uint32 => 32,
        }
    }

    pub(crate) fn from_code(code: i16) -> Result<Self> {
        match code {
            16 => Ok(Self::Float32),
            768 => Ok(Self::Uint32),
            _ => bail!("Unsupported NIfTI datatype code {code}"),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Endian {
    Little,
    Big,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct NiftiHeader {
    pub(crate) dim: [u16; 8],
    pub(crate) datatype: NiftiDatatype,
    pub(crate) pixdim: [f32; 8],
    pub(crate) vox_offset: usize,
    pub(crate) qform_code: i16,
    pub(crate) sform_code: i16,
    pub(crate) quatern_b: f32,
    pub(crate) quatern_c: f32,
    pub(crate) quatern_d: f32,
    pub(crate) quatern_x: f32,
    pub(crate) quatern_y: f32,
    pub(crate) quatern_z: f32,
    pub(crate) srow_x: [f32; 4],
    pub(crate) srow_y: [f32; 4],
    pub(crate) srow_z: [f32; 4],
    pub(crate) xyzt_units: u8,
    endian: Endian,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct HeaderDims {
    pub(crate) nx: usize,
    pub(crate) ny: usize,
    pub(crate) nz: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct HeaderSpatial {
    pub(crate) pixdim: [f32; 8],
    pub(crate) srow_x: [f32; 4],
    pub(crate) srow_y: [f32; 4],
    pub(crate) srow_z: [f32; 4],
}

impl NiftiHeader {
    pub(crate) fn new_3d(
        dims: HeaderDims,
        datatype: NiftiDatatype,
        spatial: HeaderSpatial,
    ) -> Result<Self> {
        let nx = u16::try_from(dims.nx).context("NIfTI nx exceeds u16 header capacity")?;
        let ny = u16::try_from(dims.ny).context("NIfTI ny exceeds u16 header capacity")?;
        let nz = u16::try_from(dims.nz).context("NIfTI nz exceeds u16 header capacity")?;

        Ok(Self {
            dim: [3, nx, ny, nz, 1, 1, 1, 1],
            datatype,
            pixdim: spatial.pixdim,
            vox_offset: SINGLE_FILE_VOX_OFFSET,
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
            xyzt_units: 2,
            endian: Endian::Little,
        })
    }

    pub(crate) fn parse(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < HEADER_LEN {
            bail!(
                "NIfTI header requires {HEADER_LEN} bytes, got {}",
                bytes.len()
            );
        }

        let little = i32::from_le_bytes(read_array::<4>(bytes, 0)?);
        let big = i32::from_be_bytes(read_array::<4>(bytes, 0)?);
        let endian = match (little, big) {
            (348, _) => Endian::Little,
            (_, 348) => Endian::Big,
            _ => bail!("Invalid NIfTI sizeof_hdr; expected 348"),
        };

        let magic = read_array::<4>(bytes, 344)?;
        if magic != MAGIC_NII {
            bail!("Unsupported NIfTI magic; expected single-file n+1");
        }

        let mut dim = [0_u16; 8];
        for (index, slot) in dim.iter_mut().enumerate() {
            *slot = read_u16(bytes, 40 + index * 2, endian)?;
        }
        if dim[0] != 3 {
            bail!("Expected 3-D NIfTI volume, found {} dimensions", dim[0]);
        }
        for (axis, value) in dim.iter().enumerate().take(4).skip(1) {
            if *value == 0 {
                bail!("NIfTI dim[{axis}] must be positive");
            }
        }

        let datatype = NiftiDatatype::from_code(read_i16(bytes, 70, endian)?)?;
        let bitpix = read_i16(bytes, 72, endian)?;
        if bitpix != datatype.bitpix() {
            bail!(
                "NIfTI bitpix {bitpix} does not match datatype {}",
                datatype.code()
            );
        }

        let mut pixdim = [0.0_f32; 8];
        for (index, slot) in pixdim.iter_mut().enumerate() {
            *slot = read_f32(bytes, 76 + index * 4, endian)?;
        }

        let vox_offset = read_f32(bytes, 108, endian)?;
        if !vox_offset.is_finite() || vox_offset < SINGLE_FILE_VOX_OFFSET as f32 {
            bail!("NIfTI vox_offset must be at least {SINGLE_FILE_VOX_OFFSET}, got {vox_offset}");
        }
        let vox_offset = vox_offset as usize;

        let qform_code = read_i16(bytes, 252, endian)?;
        let sform_code = read_i16(bytes, 254, endian)?;
        let quatern_b = read_f32(bytes, 256, endian)?;
        let quatern_c = read_f32(bytes, 260, endian)?;
        let quatern_d = read_f32(bytes, 264, endian)?;
        let quatern_x = read_f32(bytes, 268, endian)?;
        let quatern_y = read_f32(bytes, 272, endian)?;
        let quatern_z = read_f32(bytes, 276, endian)?;
        let srow_x = read_f32x4(bytes, 280, endian)?;
        let srow_y = read_f32x4(bytes, 296, endian)?;
        let srow_z = read_f32x4(bytes, 312, endian)?;
        let xyzt_units = bytes[123];

        Ok(Self {
            dim,
            datatype,
            pixdim,
            vox_offset,
            qform_code,
            sform_code,
            quatern_b,
            quatern_c,
            quatern_d,
            quatern_x,
            quatern_y,
            quatern_z,
            srow_x,
            srow_y,
            srow_z,
            xyzt_units,
            endian,
        })
    }

    pub(crate) fn encode(&self) -> [u8; HEADER_LEN] {
        let mut out = [0_u8; HEADER_LEN];
        write_i32(&mut out, 0, 348);
        for (index, value) in self.dim.iter().copied().enumerate() {
            write_u16(&mut out, 40 + index * 2, value);
        }
        write_i16(&mut out, 70, self.datatype.code());
        write_i16(&mut out, 72, self.datatype.bitpix());
        for (index, value) in self.pixdim.iter().copied().enumerate() {
            write_f32(&mut out, 76 + index * 4, value);
        }
        write_f32(&mut out, 108, self.vox_offset as f32);
        write_f32(&mut out, 112, 1.0);
        out[123] = self.xyzt_units;
        write_i16(&mut out, 252, self.qform_code);
        write_i16(&mut out, 254, self.sform_code);
        write_f32(&mut out, 256, self.quatern_b);
        write_f32(&mut out, 260, self.quatern_c);
        write_f32(&mut out, 264, self.quatern_d);
        write_f32(&mut out, 268, self.quatern_x);
        write_f32(&mut out, 272, self.quatern_y);
        write_f32(&mut out, 276, self.quatern_z);
        write_f32x4(&mut out, 280, self.srow_x);
        write_f32x4(&mut out, 296, self.srow_y);
        write_f32x4(&mut out, 312, self.srow_z);
        out[344..348].copy_from_slice(&MAGIC_NII);
        out
    }

    pub(crate) fn affine(&self) -> Result<[[f32; 4]; 4]> {
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

            Ok([
                [r11 * dx, r12 * dy, r13 * dz, self.quatern_x],
                [r21 * dx, r22 * dy, r23 * dz, self.quatern_y],
                [r31 * dx, r32 * dy, r33 * dz, self.quatern_z],
                [0.0, 0.0, 0.0, 1.0],
            ])
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

    pub(crate) fn volume_byte_range(&self, byte_len: usize) -> Result<std::ops::Range<usize>> {
        let voxel_count = crate::shape::checked_voxel_count(
            usize::from(self.dim[1]),
            usize::from(self.dim[2]),
            usize::from(self.dim[3]),
        )?;
        let data_len = voxel_count
            .checked_mul(4)
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

pub(crate) fn qfac_from_pixdim(value: f32) -> Result<f32> {
    if !value.is_finite() {
        bail!("NIfTI pixdim[0] qfac must be finite, got {value}");
    }

    if value == 0.0 || value == 1.0 {
        Ok(1.0)
    } else if value == -1.0 {
        Ok(-1.0)
    } else {
        bail!("NIfTI pixdim[0] qfac must be -1, 0, or 1, got {value}");
    }
}

pub(crate) fn checked_spatial_pixdim(pixdim: [f32; 8]) -> Result<[f32; 3]> {
    let spatial = [pixdim[1], pixdim[2], pixdim[3]];
    for (offset, value) in spatial.iter().enumerate() {
        let index = offset + 1;
        if !value.is_finite() || *value <= 0.0 {
            bail!("NIfTI pixdim[{index}] must be positive and finite, got {value}");
        }
    }

    Ok(spatial)
}

pub(crate) fn qform_quaternion_scalar(b: f32, c: f32, d: f32) -> Result<f32> {
    for (name, value) in [("b", b), ("c", c), ("d", d)] {
        if !value.is_finite() {
            bail!("NIfTI qform quaternion {name} must be finite, got {value}");
        }
    }

    let squared_vector_norm = b.mul_add(b, c.mul_add(c, d * d));
    if squared_vector_norm > 1.0 + 1.0e-5 {
        bail!("NIfTI qform quaternion vector norm squared must be <= 1, got {squared_vector_norm}");
    }

    Ok((1.0 - squared_vector_norm).max(0.0).sqrt())
}

fn read_array<const N: usize>(bytes: &[u8], offset: usize) -> Result<[u8; N]> {
    bytes
        .get(offset..offset + N)
        .ok_or_else(|| anyhow!("NIfTI header truncated at byte {offset}"))?
        .try_into()
        .map_err(|_| anyhow!("NIfTI header field width mismatch at byte {offset}"))
}

fn read_u16(bytes: &[u8], offset: usize, endian: Endian) -> Result<u16> {
    let raw = read_array::<2>(bytes, offset)?;
    Ok(match endian {
        Endian::Little => u16::from_le_bytes(raw),
        Endian::Big => u16::from_be_bytes(raw),
    })
}

fn read_i16(bytes: &[u8], offset: usize, endian: Endian) -> Result<i16> {
    let raw = read_array::<2>(bytes, offset)?;
    Ok(match endian {
        Endian::Little => i16::from_le_bytes(raw),
        Endian::Big => i16::from_be_bytes(raw),
    })
}

fn read_f32(bytes: &[u8], offset: usize, endian: Endian) -> Result<f32> {
    let raw = read_array::<4>(bytes, offset)?;
    Ok(match endian {
        Endian::Little => f32::from_le_bytes(raw),
        Endian::Big => f32::from_be_bytes(raw),
    })
}

fn read_f32x4(bytes: &[u8], offset: usize, endian: Endian) -> Result<[f32; 4]> {
    Ok([
        read_f32(bytes, offset, endian)?,
        read_f32(bytes, offset + 4, endian)?,
        read_f32(bytes, offset + 8, endian)?,
        read_f32(bytes, offset + 12, endian)?,
    ])
}

fn write_i32(out: &mut [u8], offset: usize, value: i32) {
    out[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn write_u16(out: &mut [u8], offset: usize, value: u16) {
    out[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
}

fn write_i16(out: &mut [u8], offset: usize, value: i16) {
    out[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
}

fn write_f32(out: &mut [u8], offset: usize, value: f32) {
    out[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn write_f32x4(out: &mut [u8], offset: usize, values: [f32; 4]) {
    for (index, value) in values.into_iter().enumerate() {
        write_f32(out, offset + index * 4, value);
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

#[cfg(test)]
mod tests {
    use super::{
        checked_spatial_pixdim, qfac_from_pixdim, qform_quaternion_scalar, HeaderDims,
        HeaderSpatial, NiftiDatatype, NiftiHeader,
    };

    #[test]
    fn header_round_trip_preserves_core_fields() {
        let header = NiftiHeader::new_3d(
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

        let parsed = NiftiHeader::parse(&header.encode()).expect("encoded header parses");
        assert_eq!(parsed.dim, [3, 4, 3, 2, 1, 1, 1, 1]);
        assert_eq!(parsed.datatype, NiftiDatatype::Float32);
        assert_eq!(parsed.srow_x, [-0.75, 0.0, 0.0, -11.0]);
        assert_eq!(parsed.vox_offset, 352);
    }

    #[test]
    fn checked_spatial_pixdim_rejects_zero() {
        let err = checked_spatial_pixdim([1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0])
            .expect_err("zero NIfTI spacing must be rejected");

        assert!(
            err.to_string().contains("pixdim[2]"),
            "error must name offending pixdim index: {err}"
        );
    }

    #[test]
    fn qfac_accepts_standard_values() {
        assert_eq!(qfac_from_pixdim(0.0).expect("0 qfac maps to +1"), 1.0);
        assert_eq!(qfac_from_pixdim(1.0).expect("+1 qfac is valid"), 1.0);
        assert_eq!(qfac_from_pixdim(-1.0).expect("-1 qfac is valid"), -1.0);
    }

    #[test]
    fn qfac_rejects_non_standard_value() {
        let err = qfac_from_pixdim(2.0).expect_err("non-standard qfac must be rejected");

        assert!(
            err.to_string().contains("qfac"),
            "error must name qfac invariant: {err}"
        );
    }

    #[test]
    fn qform_quaternion_rejects_impossible_norm() {
        let err = qform_quaternion_scalar(1.0, 1.0, 0.0)
            .expect_err("impossible qform quaternion must be rejected");

        assert!(
            err.to_string().contains("norm squared"),
            "error must name quaternion norm invariant: {err}"
        );
    }
}
