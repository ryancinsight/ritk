use anyhow::{anyhow, bail, Context, Result};

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
    const fn single_file_vox_offset(self) -> usize {
        match self {
            Self::One => NIFTI1_SINGLE_FILE_VOX_OFFSET,
            Self::Two => NIFTI2_SINGLE_FILE_VOX_OFFSET,
        }
    }
}

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
    pub(crate) xyzt_units: i32,
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
    pub(crate) pixdim: [f64; 8],
    pub(crate) srow_x: [f64; 4],
    pub(crate) srow_y: [f64; 4],
    pub(crate) srow_z: [f64; 4],
}

impl NiftiHeader {
    #[cfg(test)]
    pub(crate) fn new_3d(
        dims: HeaderDims,
        datatype: NiftiDatatype,
        spatial: HeaderSpatial,
    ) -> Result<Self> {
        Self::new_3d_with_version(HeaderVersion::One, dims, datatype, spatial)
    }

    pub(crate) fn new_3d_with_version(
        version: HeaderVersion,
        dims: HeaderDims,
        datatype: NiftiDatatype,
        spatial: HeaderSpatial,
    ) -> Result<Self> {
        let dim = dims_for_version(version, dims)?;

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
            xyzt_units: 2,
            endian: Endian::Little,
        })
    }

    pub(crate) fn parse(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < 4 {
            bail!(
                "NIfTI header requires at least 4 bytes, got {}",
                bytes.len()
            );
        }

        let little = i32::from_le_bytes(read_array::<4>(bytes, 0)?);
        let big = i32::from_be_bytes(read_array::<4>(bytes, 0)?);
        match (little, big) {
            (348, _) => Self::parse_nifti1(bytes, Endian::Little),
            (_, 348) => Self::parse_nifti1(bytes, Endian::Big),
            (540, _) => Self::parse_nifti2(bytes, Endian::Little),
            (_, 540) => Self::parse_nifti2(bytes, Endian::Big),
            _ => bail!("Invalid NIfTI sizeof_hdr; expected 348 or 540"),
        }
    }

    fn parse_nifti1(bytes: &[u8], endian: Endian) -> Result<Self> {
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
            *slot = usize::from(read_u16(bytes, 40 + index * 2, endian)?);
        }
        validate_3d_dims(dim)?;

        let datatype = NiftiDatatype::from_code(read_i16(bytes, 70, endian)?)?;
        validate_bitpix(datatype, read_i16(bytes, 72, endian)?)?;

        let mut pixdim = [0.0_f64; 8];
        for (index, slot) in pixdim.iter_mut().enumerate() {
            *slot = f64::from(read_f32(bytes, 76 + index * 4, endian)?);
        }

        let vox_offset = f64::from(read_f32(bytes, 108, endian)?);
        let vox_offset = validate_vox_offset(HeaderVersion::One, vox_offset)?;

        Ok(Self {
            version: HeaderVersion::One,
            dim,
            datatype,
            pixdim,
            vox_offset,
            qform_code: i32::from(read_i16(bytes, 252, endian)?),
            sform_code: i32::from(read_i16(bytes, 254, endian)?),
            quatern_b: f64::from(read_f32(bytes, 256, endian)?),
            quatern_c: f64::from(read_f32(bytes, 260, endian)?),
            quatern_d: f64::from(read_f32(bytes, 264, endian)?),
            quatern_x: f64::from(read_f32(bytes, 268, endian)?),
            quatern_y: f64::from(read_f32(bytes, 272, endian)?),
            quatern_z: f64::from(read_f32(bytes, 276, endian)?),
            srow_x: read_f32x4_as_f64(bytes, 280, endian)?,
            srow_y: read_f32x4_as_f64(bytes, 296, endian)?,
            srow_z: read_f32x4_as_f64(bytes, 312, endian)?,
            xyzt_units: i32::from(bytes[123]),
            endian,
        })
    }

    fn parse_nifti2(bytes: &[u8], endian: Endian) -> Result<Self> {
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
            let raw = read_i64(bytes, 16 + index * 8, endian)?;
            *slot = usize::try_from(raw).with_context(|| {
                format!("NIfTI-2 dim[{index}] must be non-negative and fit usize, got {raw}")
            })?;
        }
        validate_3d_dims(dim)?;

        let datatype = NiftiDatatype::from_code(read_i16(bytes, 12, endian)?)?;
        validate_bitpix(datatype, read_i16(bytes, 14, endian)?)?;

        let mut pixdim = [0.0_f64; 8];
        for (index, slot) in pixdim.iter_mut().enumerate() {
            *slot = read_f64(bytes, 104 + index * 8, endian)?;
        }

        let vox_offset =
            validate_i64_vox_offset(HeaderVersion::Two, read_i64(bytes, 168, endian)?)?;

        Ok(Self {
            version: HeaderVersion::Two,
            dim,
            datatype,
            pixdim,
            vox_offset,
            qform_code: read_i32(bytes, 344, endian)?,
            sform_code: read_i32(bytes, 348, endian)?,
            quatern_b: read_f64(bytes, 352, endian)?,
            quatern_c: read_f64(bytes, 360, endian)?,
            quatern_d: read_f64(bytes, 368, endian)?,
            quatern_x: read_f64(bytes, 376, endian)?,
            quatern_y: read_f64(bytes, 384, endian)?,
            quatern_z: read_f64(bytes, 392, endian)?,
            srow_x: read_f64x4(bytes, 400, endian)?,
            srow_y: read_f64x4(bytes, 432, endian)?,
            srow_z: read_f64x4(bytes, 464, endian)?,
            xyzt_units: read_i32(bytes, 500, endian)?,
            endian,
        })
    }

    pub(crate) fn encode(&self) -> Vec<u8> {
        match self.version {
            HeaderVersion::One => self.encode_nifti1().to_vec(),
            HeaderVersion::Two => self.encode_nifti2().to_vec(),
        }
    }

    fn encode_nifti1(&self) -> [u8; NIFTI1_HEADER_LEN] {
        let mut out = [0_u8; NIFTI1_HEADER_LEN];
        write_i32(&mut out, 0, 348);
        for (index, value) in self.dim.iter().copied().enumerate() {
            write_u16(
                &mut out,
                40 + index * 2,
                u16::try_from(value)
                    .expect("invariant: NIfTI-1 header dims are validated at construction"),
            );
        }
        write_i16(&mut out, 70, self.datatype.code());
        write_i16(&mut out, 72, self.datatype.bitpix());
        for (index, value) in self.pixdim.iter().copied().enumerate() {
            write_f32(&mut out, 76 + index * 4, f64_to_f32(value, "pixdim"));
        }
        write_f32(
            &mut out,
            108,
            f64_to_f32(self.vox_offset as f64, "vox_offset"),
        );
        write_f32(&mut out, 112, 1.0);
        out[123] = u8::try_from(self.xyzt_units)
            .expect("invariant: NIfTI-1 xyzt_units is set to a u8-compatible value");
        write_i16(
            &mut out,
            252,
            i16::try_from(self.qform_code).expect("invariant: NIfTI-1 qform_code fits i16"),
        );
        write_i16(
            &mut out,
            254,
            i16::try_from(self.sform_code).expect("invariant: NIfTI-1 sform_code fits i16"),
        );
        write_f32(&mut out, 256, f64_to_f32(self.quatern_b, "quatern_b"));
        write_f32(&mut out, 260, f64_to_f32(self.quatern_c, "quatern_c"));
        write_f32(&mut out, 264, f64_to_f32(self.quatern_d, "quatern_d"));
        write_f32(&mut out, 268, f64_to_f32(self.quatern_x, "quatern_x"));
        write_f32(&mut out, 272, f64_to_f32(self.quatern_y, "quatern_y"));
        write_f32(&mut out, 276, f64_to_f32(self.quatern_z, "quatern_z"));
        write_f32x4(&mut out, 280, self.srow_x);
        write_f32x4(&mut out, 296, self.srow_y);
        write_f32x4(&mut out, 312, self.srow_z);
        out[344..348].copy_from_slice(&NIFTI1_MAGIC_SINGLE_FILE);
        out
    }

    fn encode_nifti2(&self) -> [u8; NIFTI2_HEADER_LEN] {
        let mut out = [0_u8; NIFTI2_HEADER_LEN];
        write_i32(&mut out, 0, 540);
        out[4..12].copy_from_slice(&NIFTI2_MAGIC_SINGLE_FILE);
        write_i16(&mut out, 12, self.datatype.code());
        write_i16(&mut out, 14, self.datatype.bitpix());
        for (index, value) in self.dim.iter().copied().enumerate() {
            write_i64(
                &mut out,
                16 + index * 8,
                i64::try_from(value)
                    .expect("invariant: NIfTI-2 header dims are validated at construction"),
            );
        }
        for (index, value) in self.pixdim.iter().copied().enumerate() {
            write_f64(&mut out, 104 + index * 8, value);
        }
        write_i64(
            &mut out,
            168,
            i64::try_from(self.vox_offset).expect("invariant: vox_offset fits i64"),
        );
        write_f64(&mut out, 176, 1.0);
        write_i32(&mut out, 344, self.qform_code);
        write_i32(&mut out, 348, self.sform_code);
        write_f64(&mut out, 352, self.quatern_b);
        write_f64(&mut out, 360, self.quatern_c);
        write_f64(&mut out, 368, self.quatern_d);
        write_f64(&mut out, 376, self.quatern_x);
        write_f64(&mut out, 384, self.quatern_y);
        write_f64(&mut out, 392, self.quatern_z);
        write_f64x4(&mut out, 400, self.srow_x);
        write_f64x4(&mut out, 432, self.srow_y);
        write_f64x4(&mut out, 464, self.srow_z);
        write_i32(&mut out, 500, self.xyzt_units);
        out
    }

    pub(crate) fn read_f32_lane(&self, raw: [u8; 4]) -> f32 {
        match self.endian {
            Endian::Little => f32::from_le_bytes(raw),
            Endian::Big => f32::from_be_bytes(raw),
        }
    }

    pub(crate) fn read_u32_lane(&self, raw: [u8; 4]) -> u32 {
        match self.endian {
            Endian::Little => u32::from_le_bytes(raw),
            Endian::Big => u32::from_be_bytes(raw),
        }
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

    pub(crate) fn volume_byte_range(&self, byte_len: usize) -> Result<std::ops::Range<usize>> {
        let voxel_count = crate::shape::checked_voxel_count(self.dim[1], self.dim[2], self.dim[3])?;
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

pub(crate) fn qfac_from_pixdim(value: f64) -> Result<f64> {
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

pub(crate) fn checked_spatial_pixdim(pixdim: [f64; 8]) -> Result<[f64; 3]> {
    let spatial = [pixdim[1], pixdim[2], pixdim[3]];
    for (offset, value) in spatial.iter().enumerate() {
        let index = offset + 1;
        if !value.is_finite() || *value <= 0.0 {
            bail!("NIfTI pixdim[{index}] must be positive and finite, got {value}");
        }
    }

    Ok(spatial)
}

pub(crate) fn qform_quaternion_scalar(b: f64, c: f64, d: f64) -> Result<f64> {
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

fn dims_for_version(version: HeaderVersion, dims: HeaderDims) -> Result<[usize; 8]> {
    if matches!(version, HeaderVersion::One) {
        u16::try_from(dims.nx).context("NIfTI-1 nx exceeds u16 header capacity")?;
        u16::try_from(dims.ny).context("NIfTI-1 ny exceeds u16 header capacity")?;
        u16::try_from(dims.nz).context("NIfTI-1 nz exceeds u16 header capacity")?;
    } else {
        i64::try_from(dims.nx).context("NIfTI-2 nx exceeds i64 header capacity")?;
        i64::try_from(dims.ny).context("NIfTI-2 ny exceeds i64 header capacity")?;
        i64::try_from(dims.nz).context("NIfTI-2 nz exceeds i64 header capacity")?;
    }

    Ok([3, dims.nx, dims.ny, dims.nz, 1, 1, 1, 1])
}

fn validate_3d_dims(dim: [usize; 8]) -> Result<()> {
    if dim[0] != 3 {
        bail!("Expected 3-D NIfTI volume, found {} dimensions", dim[0]);
    }
    for (axis, value) in dim.iter().enumerate().take(4).skip(1) {
        if *value == 0 {
            bail!("NIfTI dim[{axis}] must be positive");
        }
    }
    Ok(())
}

fn validate_bitpix(datatype: NiftiDatatype, bitpix: i16) -> Result<()> {
    if bitpix != datatype.bitpix() {
        bail!(
            "NIfTI bitpix {bitpix} does not match datatype {}",
            datatype.code()
        );
    }
    Ok(())
}

fn validate_vox_offset(version: HeaderVersion, vox_offset: f64) -> Result<usize> {
    let minimum = version.single_file_vox_offset();
    if !vox_offset.is_finite() || vox_offset < minimum as f64 {
        bail!("NIfTI vox_offset must be at least {minimum}, got {vox_offset}");
    }
    if vox_offset.fract() != 0.0 {
        bail!("NIfTI vox_offset must be an integer byte offset, got {vox_offset}");
    }

    usize::try_from(vox_offset as u128)
        .map_err(|_| anyhow!("NIfTI vox_offset does not fit usize, got {vox_offset}"))
}

fn validate_i64_vox_offset(version: HeaderVersion, vox_offset: i64) -> Result<usize> {
    let minimum = version.single_file_vox_offset();
    let value = usize::try_from(vox_offset)
        .map_err(|_| anyhow!("NIfTI vox_offset must be non-negative, got {vox_offset}"))?;
    if value < minimum {
        bail!("NIfTI vox_offset must be at least {minimum}, got {vox_offset}");
    }
    Ok(value)
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

fn read_i32(bytes: &[u8], offset: usize, endian: Endian) -> Result<i32> {
    let raw = read_array::<4>(bytes, offset)?;
    Ok(match endian {
        Endian::Little => i32::from_le_bytes(raw),
        Endian::Big => i32::from_be_bytes(raw),
    })
}

fn read_i64(bytes: &[u8], offset: usize, endian: Endian) -> Result<i64> {
    let raw = read_array::<8>(bytes, offset)?;
    Ok(match endian {
        Endian::Little => i64::from_le_bytes(raw),
        Endian::Big => i64::from_be_bytes(raw),
    })
}

fn read_f32(bytes: &[u8], offset: usize, endian: Endian) -> Result<f32> {
    let raw = read_array::<4>(bytes, offset)?;
    Ok(match endian {
        Endian::Little => f32::from_le_bytes(raw),
        Endian::Big => f32::from_be_bytes(raw),
    })
}

fn read_f64(bytes: &[u8], offset: usize, endian: Endian) -> Result<f64> {
    let raw = read_array::<8>(bytes, offset)?;
    Ok(match endian {
        Endian::Little => f64::from_le_bytes(raw),
        Endian::Big => f64::from_be_bytes(raw),
    })
}

fn read_f32x4_as_f64(bytes: &[u8], offset: usize, endian: Endian) -> Result<[f64; 4]> {
    Ok([
        f64::from(read_f32(bytes, offset, endian)?),
        f64::from(read_f32(bytes, offset + 4, endian)?),
        f64::from(read_f32(bytes, offset + 8, endian)?),
        f64::from(read_f32(bytes, offset + 12, endian)?),
    ])
}

fn read_f64x4(bytes: &[u8], offset: usize, endian: Endian) -> Result<[f64; 4]> {
    Ok([
        read_f64(bytes, offset, endian)?,
        read_f64(bytes, offset + 8, endian)?,
        read_f64(bytes, offset + 16, endian)?,
        read_f64(bytes, offset + 24, endian)?,
    ])
}

fn write_i32(out: &mut [u8], offset: usize, value: i32) {
    out[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn write_i64(out: &mut [u8], offset: usize, value: i64) {
    out[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
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

fn write_f64(out: &mut [u8], offset: usize, value: f64) {
    out[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

fn write_f32x4(out: &mut [u8], offset: usize, values: [f64; 4]) {
    for (index, value) in values.into_iter().enumerate() {
        write_f32(out, offset + index * 4, f64_to_f32(value, "srow"));
    }
}

fn write_f64x4(out: &mut [u8], offset: usize, values: [f64; 4]) {
    for (index, value) in values.into_iter().enumerate() {
        write_f64(out, offset + index * 8, value);
    }
}

fn f64_to_f32(value: f64, field: &str) -> f32 {
    assert!(
        value.is_finite() && value >= f64::from(f32::MIN) && value <= f64::from(f32::MAX),
        "invariant: {field} must be finite and f32-representable for NIfTI-1 encoding"
    );
    value as f32
}

fn f64x4_to_f32x4(values: [f64; 4], field: &str) -> Result<[f32; 4]> {
    Ok([
        checked_f64_to_f32(values[0], field)?,
        checked_f64_to_f32(values[1], field)?,
        checked_f64_to_f32(values[2], field)?,
        checked_f64_to_f32(values[3], field)?,
    ])
}

fn f64_affine_to_f32(values: [[f64; 4]; 4]) -> Result<[[f32; 4]; 4]> {
    let mut out = [[0.0_f32; 4]; 4];
    for row in 0..4 {
        for col in 0..4 {
            out[row][col] = checked_f64_to_f32(values[row][col], "affine")?;
        }
    }
    Ok(out)
}

fn checked_f64_to_f32(value: f64, field: &str) -> Result<f32> {
    if !value.is_finite() || value < f64::from(f32::MIN) || value > f64::from(f32::MAX) {
        bail!("NIfTI {field} value must be finite and f32-representable, got {value}");
    }
    Ok(value as f32)
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
        HeaderSpatial, HeaderVersion, NiftiDatatype, NiftiHeader,
    };

    #[test]
    fn header_round_trip_preserves_nifti1_core_fields() {
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
        assert_eq!(parsed.version, HeaderVersion::One);
        assert_eq!(parsed.dim, [3, 4, 3, 2, 1, 1, 1, 1]);
        assert_eq!(parsed.datatype, NiftiDatatype::Float32);
        assert_eq!(parsed.srow_x, [-0.75, 0.0, 0.0, -11.0]);
        assert_eq!(parsed.vox_offset, 352);
    }

    #[test]
    fn header_round_trip_preserves_nifti2_core_fields() {
        let header = NiftiHeader::new_3d_with_version(
            HeaderVersion::Two,
            HeaderDims {
                nx: 70_000,
                ny: 3,
                nz: 2,
            },
            NiftiDatatype::Uint32,
            HeaderSpatial {
                pixdim: [1.0, 0.75, 1.5, 2.0, 1.0, 1.0, 1.0, 1.0],
                srow_x: [-0.75, 0.0, 0.0, -11.0],
                srow_y: [0.0, -1.5, 0.0, 7.5],
                srow_z: [0.0, 0.0, 2.0, 3.25],
            },
        )
        .expect("valid header");

        let parsed = NiftiHeader::parse(&header.encode()).expect("encoded header parses");
        assert_eq!(parsed.version, HeaderVersion::Two);
        assert_eq!(parsed.dim, [3, 70_000, 3, 2, 1, 1, 1, 1]);
        assert_eq!(parsed.datatype, NiftiDatatype::Uint32);
        assert_eq!(parsed.srow_x, [-0.75, 0.0, 0.0, -11.0]);
        assert_eq!(parsed.vox_offset, 544);
    }

    #[test]
    fn nifti1_rejects_dimensions_above_u16() {
        let err = NiftiHeader::new_3d(
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
