use super::convert::{f64_affine_to_f32, f64_to_f32, f64x4_to_f32x4};
use super::validate::{
    checked_spatial_pixdim, dims_for_version, qfac_from_pixdim, qform_quaternion_scalar,
    volume_count,
};
use anyhow::{anyhow, bail, Result};
use consus_core::ByteOrder;
use ritk_codecs::sample::{Rescale, SampleType};

pub(super) const NIFTI1_HEADER_LEN: usize = 348;
const NIFTI1_SINGLE_FILE_VOX_OFFSET: usize = 352;
pub(super) const NIFTI2_HEADER_LEN: usize = 540;
const NIFTI2_SINGLE_FILE_VOX_OFFSET: usize = 544;
pub(super) const NIFTI1_MAGIC_SINGLE_FILE: [u8; 4] = *b"n+1\0";
pub(super) const NIFTI2_MAGIC_SINGLE_FILE: [u8; 8] = *b"n+2\0\r\n\x1a\n";

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

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct NiftiHeader {
    pub(crate) version: HeaderVersion,
    pub(crate) dim: [usize; 8],
    pub(crate) sample_type: SampleType,
    pub(crate) rescale: Rescale,
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
    pub(super) endian: ByteOrder,
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
        sample_type: SampleType,
        spatial: HeaderSpatial,
    ) -> Result<Self> {
        Self::new_with_version(HeaderVersion::One, dims, 1, sample_type, spatial)
    }

    /// Build a header for `volumes` volumes of `sample_type` sharing the `dims`
    /// spatial grid, declaring no rescale.
    ///
    /// One volume produces a rank-3 header; more produce a rank-4 header with
    /// the count in `dim[4]`.
    pub(crate) fn new_with_version(
        version: HeaderVersion,
        dims: HeaderDims,
        volumes: usize,
        sample_type: SampleType,
        spatial: HeaderSpatial,
    ) -> Result<Self> {
        let dim = dims_for_version(version, dims, volumes)?;

        Ok(Self {
            version,
            dim,
            sample_type,
            rescale: Rescale::IDENTITY,
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
            endian: ByteOrder::LittleEndian,
        })
    }

    /// The byte order of the header and of the samples after it.
    pub(crate) fn byte_order(&self) -> ByteOrder {
        self.endian
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
            .checked_mul(self.sample_type.byte_width())
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

#[cfg(test)]
mod tests {
    use super::{HeaderDims, HeaderSpatial, HeaderVersion, NiftiHeader};
    use ritk_codecs::sample::{Rescale, SampleType};

    #[test]
    fn header_round_trip_preserves_nifti1_core_fields() {
        let header = NiftiHeader::new_volume(
            HeaderDims {
                nx: 4,
                ny: 3,
                nz: 2,
            },
            SampleType::F32,
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
        assert_eq!(parsed.sample_type, SampleType::F32);
        assert_eq!(parsed.rescale, Rescale::IDENTITY);
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
            SampleType::U32,
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
        assert_eq!(parsed.sample_type, SampleType::U32);
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
            SampleType::F32,
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
}
