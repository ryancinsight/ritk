//! Checked DICOM frame decoding and exact integer output for JPEG 2000 Part 1.
use super::{
    image::{decode_j2k_fragment, is_soc},
    organization::{self, Jpeg2000Profile},
    read_limits::check_read_budget,
    Jpeg2000DecodeError,
};
use crate::{ImageReadBudget, PixelLayout};
use anyhow::anyhow;

/// Decode a DICOM-encapsulated JPEG 2000 J2K codestream fragment.
///
/// # Arguments
/// - `fragment`: raw bytes of the encapsulated pixel data item.
/// - `layout`: pixel geometry and DICOM rescale parameters.
/// - `budget`: ceilings for the encoded fragment and the decoded `f32` output.
///
/// # Errors
/// Returns [`Jpeg2000DecodeError::Budget`] when the fragment length or the
/// output storage implied by `layout` exceeds `budget`, before the codestream
/// is parsed or any output is allocated.
///
/// Returns [`Jpeg2000DecodeError::OutputAllocation`] when the allocator cannot
/// reserve the output frame the budget admitted.
///
/// Returns [`Jpeg2000DecodeError::Codestream`] if:
/// - `fragment` does not begin with the SOC marker (0xFF4F).
/// - SIZ or SOT geometry is invalid, or identifies a tile outside the image's tile grid.
/// - the codestream declares an incompatible DICOM profile or component geometry.
/// - packet or coefficient decoding fails.
/// - a marker segment or packet is truncated, EOC or a declared tile is
///   missing, or decoded component metadata does not match `layout`.
/// - component precision exceeds the Part 1 limit of 38 bits.
pub fn decode_jpeg2000_fragment(
    fragment: &[u8],
    layout: PixelLayout,
    budget: ImageReadBudget,
) -> Result<Vec<f32>, Jpeg2000DecodeError> {
    decode_jpeg2000_with_profile(
        fragment,
        layout,
        budget,
        Jpeg2000Profile::Part1 {
            lossless_only: false,
        },
    )
}

/// Read the component precision from the JPEG 2000 main header.
///
/// # Errors
/// Returns an error on a malformed header or a non-monochrome component layout.
pub fn grayscale_component_precision(fragment: &[u8]) -> anyhow::Result<u32> {
    let siz = super::organization::size_marker(fragment)?;
    let [component] = siz.components.as_slice() else {
        anyhow::bail!("expected one grayscale JPEG 2000 component")
    };
    Ok(component.precision())
}

/// Decode stored integer samples before modality rescale, retaining all Part 1 component bits.
/// The encoded fragment and eight-byte-per-sample output are subject to the caller's budget.
///
/// # Errors
/// Returns the same structural, coding-profile and resource errors as [`decode_jpeg2000_fragment`].
pub fn decode_jpeg2000_stored_fragment(
    fragment: &[u8],
    layout: PixelLayout,
    budget: ImageReadBudget,
) -> Result<Vec<i64>, Jpeg2000DecodeError> {
    decode_jpeg2000_stored_with_profile(
        fragment,
        layout,
        budget,
        Jpeg2000Profile::Part1 {
            lossless_only: false,
        },
    )
}

/// Decode a DICOM Part 1 or HTJ2K frame with the specified profile and modality rescaling.
///
/// # Errors
/// Returns an error for an invalid codestream, incompatible DICOM profile or exceeded read budget.
pub fn decode_jpeg2000_with_profile(
    fragment: &[u8],
    layout: PixelLayout,
    budget: ImageReadBudget,
    profile: Jpeg2000Profile,
) -> Result<Vec<f32>, Jpeg2000DecodeError> {
    decode_samples(fragment, layout, budget, profile, |raw| {
        (raw as f64 * f64::from(layout.rescale_slope) + f64::from(layout.rescale_intercept)) as f32
    })
}

/// Decode Part 1 or HTJ2K into exact stored integers before modality rescale.
///
/// # Errors
/// Returns the same structural and profile errors as [`decode_jpeg2000_with_profile`].
pub fn decode_jpeg2000_stored_with_profile(
    fragment: &[u8],
    layout: PixelLayout,
    budget: ImageReadBudget,
    profile: Jpeg2000Profile,
) -> Result<Vec<i64>, Jpeg2000DecodeError> {
    decode_samples(fragment, layout, budget, profile, |raw| raw)
}

fn decode_samples<T: Copy + Default>(
    fragment: &[u8],
    layout: PixelLayout,
    budget: ImageReadBudget,
    profile: Jpeg2000Profile,
    convert: impl Fn(i64) -> T,
) -> Result<Vec<T>, Jpeg2000DecodeError> {
    if !is_jpeg2000_codestream(fragment) {
        return Err(Jpeg2000DecodeError::Codestream(anyhow!(
            "JPEG 2000 fragment does not begin with SOC marker 0xFF4F \
             (first 2 bytes: {:02X?})",
            &fragment[..fragment.len().min(2)]
        )));
    }
    check_read_budget(fragment.len(), layout, budget, size_of::<T>())?;
    let count = layout
        .samples_per_frame()
        .map_err(Jpeg2000DecodeError::Codestream)?;
    if count > isize::MAX as usize / size_of::<T>() {
        // try_reserve_exact returns CapacityOverflow without contacting the allocator.
        if let Err(source) = Vec::<T>::new().try_reserve_exact(count) {
            return Err(Jpeg2000DecodeError::OutputAllocation {
                samples: count,
                source,
            });
        }
    }
    let decoded = (|| {
        let plan = organization::inspect(fragment, layout, profile)?;
        if plan.scalar_packets {
            decode_j2k_fragment(fragment, layout, &plan, convert)
        } else {
            super::extended::decode(fragment, layout, &plan.header.siz, convert)
        }
    })();
    decoded.map_err(|error: anyhow::Error| {
        error
            .downcast()
            .unwrap_or_else(Jpeg2000DecodeError::Codestream)
    })
}

/// Returns `true` if `fragment` begins with the J2K SOC marker (`0xFF 0x4F`).
///
/// A bare DICOM JPEG 2000 codestream always starts with SOC (ISO 15444-1 §A.3).
#[inline]
pub(crate) fn is_jpeg2000_codestream(fragment: &[u8]) -> bool {
    is_soc(fragment)
}
