//! Field-validation predicates for NIfTI header parsing and construction.
//!
//! Each predicate names the violated invariant and the offending value in its
//! error, per the project error-handling discipline.

use super::{HeaderDims, HeaderVersion, NiftiDatatype};
use anyhow::{anyhow, bail, Context, Result};

pub(super) fn checked_lane<const N: usize>(raw: &[u8]) -> Result<[u8; N]> {
    raw.try_into().map_err(|_| {
        anyhow!(
            "NIfTI voxel lane width mismatch: expected {N}, got {}",
            raw.len()
        )
    })
}

pub(super) fn qfac_from_pixdim(value: f64) -> Result<f64> {
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

pub(super) fn checked_spatial_pixdim(pixdim: [f64; 8]) -> Result<[f64; 3]> {
    let spatial = [pixdim[1], pixdim[2], pixdim[3]];
    for (offset, value) in spatial.iter().enumerate() {
        let index = offset + 1;
        if !value.is_finite() || *value <= 0.0 {
            bail!("NIfTI pixdim[{index}] must be positive and finite, got {value}");
        }
    }

    Ok(spatial)
}

pub(super) fn qform_quaternion_scalar(b: f64, c: f64, d: f64) -> Result<f64> {
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

pub(super) fn dims_for_version(version: HeaderVersion, dims: HeaderDims) -> Result<[usize; 8]> {
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

pub(super) fn validate_3d_dims(dim: [usize; 8]) -> Result<()> {
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

pub(super) fn validate_bitpix(datatype: NiftiDatatype, bitpix: i16) -> Result<()> {
    if bitpix != datatype.bitpix() {
        bail!(
            "NIfTI bitpix {bitpix} does not match datatype {}",
            datatype.code()
        );
    }
    Ok(())
}

pub(super) fn validate_vox_offset(version: HeaderVersion, vox_offset: f64) -> Result<usize> {
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

pub(super) fn validate_i64_vox_offset(version: HeaderVersion, vox_offset: i64) -> Result<usize> {
    let minimum = version.single_file_vox_offset();
    let value = usize::try_from(vox_offset)
        .map_err(|_| anyhow!("NIfTI vox_offset must be non-negative, got {vox_offset}"))?;
    if value < minimum {
        bail!("NIfTI vox_offset must be at least {minimum}, got {vox_offset}");
    }
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::{checked_spatial_pixdim, qfac_from_pixdim, qform_quaternion_scalar};

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
