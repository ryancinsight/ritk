//! NIfTI `datatype` codes for the fixed-width sample types.
//!
//! Codes are the `DT_*` constants of `nifti1.h`, shared by NIfTI-1 and
//! NIfTI-2. The complex, RGB, and 128-bit codes are outside the ten sample
//! types and are rejected.

use anyhow::{bail, Result};
use ritk_codecs::sample::SampleType;

/// The `datatype` code `nifti1.h` assigns to `sample_type`.
pub(super) const fn datatype_code(sample_type: SampleType) -> i16 {
    match sample_type {
        SampleType::U8 => 2,
        SampleType::I16 => 4,
        SampleType::I32 => 8,
        SampleType::F32 => 16,
        SampleType::F64 => 64,
        SampleType::I8 => 256,
        SampleType::U16 => 512,
        SampleType::U32 => 768,
        SampleType::I64 => 1024,
        SampleType::U64 => 1280,
    }
}

/// The sample type a `datatype` code names.
///
/// # Errors
///
/// Returns an error for a code outside the ten fixed-width sample types.
pub(super) fn sample_type_from_code(code: i16) -> Result<SampleType> {
    match SampleType::ALL
        .into_iter()
        .find(|&sample_type| datatype_code(sample_type) == code)
    {
        Some(sample_type) => Ok(sample_type),
        None => bail!(
            "Unsupported NIfTI datatype code {code}; supported codes are the \
             fixed-width integer and float types"
        ),
    }
}

/// The `bitpix` field `nifti1.h` requires beside `sample_type`.
pub(super) const fn bitpix(sample_type: SampleType) -> i16 {
    // The widest sample is 8 bytes, so the bit count is at most 64.
    (sample_type.byte_width() * 8) as i16
}

#[cfg(test)]
mod tests {
    use super::{bitpix, datatype_code, sample_type_from_code};
    use ritk_codecs::sample::SampleType;

    /// The `DT_*` values of `nifti1.h`, written out independently of the map.
    #[test]
    fn codes_match_nifti1_h() {
        assert_eq!(
            SampleType::ALL.map(datatype_code),
            [2, 256, 512, 4, 768, 8, 1280, 1024, 16, 64]
        );
        assert_eq!(
            SampleType::ALL.map(bitpix),
            [8, 8, 16, 16, 32, 32, 64, 64, 32, 64]
        );
    }

    #[test]
    fn every_code_maps_back_to_its_type() {
        for sample_type in SampleType::ALL {
            assert_eq!(
                sample_type_from_code(datatype_code(sample_type)).expect("supported code"),
                sample_type
            );
        }
    }

    /// `DT_COMPLEX64` (32), `DT_RGB24` (128), `DT_FLOAT128` (1536), and an
    /// undefined code.
    #[test]
    fn codes_outside_the_sample_types_are_rejected() {
        for code in [0, 32, 128, 1536, 2304, -2] {
            let err = sample_type_from_code(code).expect_err("unsupported code");
            assert!(err.to_string().contains(&code.to_string()), "{err}");
        }
    }
}
