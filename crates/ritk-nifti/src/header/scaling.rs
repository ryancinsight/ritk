//! The `scl_slope`/`scl_inter` pair as a [`Rescale`].

use anyhow::{bail, Result};
use ritk_codecs::sample::Rescale;

/// The rescale a header's `scl_slope` and `scl_inter` fields declare.
///
/// `nifti1.h`: "If the scl_slope field is nonzero, then each voxel value in
/// the dataset should be scaled as y = scl_slope * x + scl_inter". A zero or
/// non-finite slope therefore declares no rescale, whatever the intercept;
/// writers that never set the fields leave both zero. A valid slope beside a
/// non-finite intercept has no meaning and is rejected. These are the rules
/// nibabel's `Nifti1Header.get_slope_inter` applies.
///
/// # Errors
///
/// Returns an error when the slope is valid and the intercept is not finite.
pub(super) fn rescale_from_fields(slope: f64, intercept: f64) -> Result<Rescale> {
    if slope == 0.0 || !slope.is_finite() {
        return Ok(Rescale::IDENTITY);
    }
    if !intercept.is_finite() {
        bail!("NIfTI scl_slope {slope} is valid but scl_inter {intercept} is not finite");
    }
    Ok(Rescale::new(slope, intercept)?)
}

#[cfg(test)]
mod tests {
    use super::rescale_from_fields;
    use ritk_codecs::sample::Rescale;

    #[test]
    fn zero_or_non_finite_slope_is_no_rescale() {
        for (slope, intercept) in [
            (0.0, 0.0),
            (0.0, 5.0),
            (-0.0, f64::NAN),
            (f64::NAN, 3.0),
            (f64::INFINITY, 0.0),
        ] {
            assert_eq!(
                rescale_from_fields(slope, intercept).expect("no rescale"),
                Rescale::IDENTITY,
                "slope {slope}, intercept {intercept}"
            );
        }
    }

    #[test]
    fn a_valid_slope_keeps_both_coefficients() {
        let rescale = rescale_from_fields(0.5, -1024.0).expect("valid rescale");
        assert_eq!((rescale.slope(), rescale.intercept()), (0.5, -1024.0));
    }

    #[test]
    fn a_valid_slope_with_a_non_finite_intercept_is_rejected() {
        let err = rescale_from_fields(2.0, f64::NAN).expect_err("invalid intercept");
        assert!(err.to_string().contains("scl_inter"), "{err}");
    }
}
