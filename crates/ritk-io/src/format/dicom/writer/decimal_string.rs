use super::error::DicomWriteError;
use anyhow::Result;

const MAXIMUM_DS_BYTES: usize = 16;

/// Format a finite number as a DICOM Decimal String within its wire limit.
///
/// [DICOM PS3.5, Section 6.2](https://dicom.nema.org/medical/dicom/current/output/chtml/part05/sect_6.2.html)
/// limits each DS component to 16 bytes and permits fixed-point or exponent
/// notation. The shortest round-tripping representation is retained when it
/// fits; otherwise fixed and exponent forms are compared by relative error,
/// keeping the closest representable value within that limit. Exponent
/// notation with nine significant digits fits 16 bytes for every finite f64;
/// rounding its mantissa contributes at most 5×10⁻⁹ relative error.
pub(crate) fn format_dicom_decimal(value: f64) -> Result<String> {
    if !value.is_finite() {
        return Err(DicomWriteError::NonFiniteDecimalStringValue.into());
    }
    let shortest = value.to_string();
    if shortest.len() <= MAXIMUM_DS_BYTES {
        return Ok(shortest);
    }

    let mut best = None;
    for fractional_digits in 0..=MAXIMUM_DS_BYTES {
        consider_candidate(value, format!("{value:.fractional_digits$}"), &mut best);
    }
    for fractional_digits in 0..=MAXIMUM_DS_BYTES {
        consider_candidate(value, format!("{value:.fractional_digits$E}"), &mut best);
    }
    best.map(|(formatted, _)| formatted)
        .ok_or_else(|| DicomWriteError::DecimalStringValueOutOfRange.into())
}

fn consider_candidate(value: f64, candidate: String, best: &mut Option<(String, f64)>) {
    if candidate.len() > MAXIMUM_DS_BYTES {
        return;
    }
    let Ok(parsed) = candidate.parse::<f64>() else {
        return;
    };
    if !parsed.is_finite() {
        return;
    }
    let error = if value == 0.0 {
        (parsed - value).abs()
    } else {
        (parsed - value).abs() / value.abs()
    };
    match best {
        Some((current, current_error))
            if error.total_cmp(current_error).is_gt()
                || (error.total_cmp(current_error).is_eq() && candidate.len() >= current.len()) => {
        }
        _ => *best = Some((candidate, error)),
    }
}

pub(crate) fn format_triplet(value: [f64; 3]) -> Result<String> {
    Ok(format!(
        "{}\\{}\\{}",
        format_dicom_decimal(value[0])?,
        format_dicom_decimal(value[1])?,
        format_dicom_decimal(value[2])?
    ))
}

pub(crate) fn format_pair(value: [f64; 2]) -> Result<String> {
    Ok(format!(
        "{}\\{}",
        format_dicom_decimal(value[0])?,
        format_dicom_decimal(value[1])?
    ))
}

pub(crate) fn format_six(value: [f64; 6]) -> Result<String> {
    Ok(format!(
        "{}\\{}\\{}\\{}\\{}\\{}",
        format_dicom_decimal(value[0])?,
        format_dicom_decimal(value[1])?,
        format_dicom_decimal(value[2])?,
        format_dicom_decimal(value[3])?,
        format_dicom_decimal(value[4])?,
        format_dicom_decimal(value[5])?
    ))
}

#[cfg(test)]
mod tests {
    use super::{format_dicom_decimal, MAXIMUM_DS_BYTES};
    use crate::format::dicom::writer::DicomWriteError;

    #[test]
    fn decimal_strings_round_trip_values_with_short_encodings() {
        for value in [0.0, -0.0, 0.5, 123_456.75, 1.0e-30] {
            let encoded = format_dicom_decimal(value).expect("finite DS value");
            let decoded = encoded.parse::<f64>().expect("DS parses as a number");
            assert_eq!(
                decoded.to_bits(),
                value.to_bits(),
                "encoded {value} as {encoded}"
            );
            assert!(encoded.len() <= MAXIMUM_DS_BYTES);
        }
    }

    #[test]
    fn decimal_strings_fit_the_wire_limit_at_numeric_extremes() {
        for value in [
            f64::MAX,
            -f64::MAX,
            f64::MIN_POSITIVE,
            -f64::MIN_POSITIVE,
            f64::from_bits(1),
        ] {
            let encoded = format_dicom_decimal(value).expect("finite DS value");
            let decoded = encoded.parse::<f64>().expect("DS parses as a number");
            let relative_error = (decoded - value).abs() / value.abs();
            assert!(encoded.len() <= MAXIMUM_DS_BYTES, "{encoded}");
            // A 16-byte DS carries at least nine significant digits in
            // exponent form, whose nearest-decimal error is at most 0.5e-8.
            assert!(
                relative_error <= 5.0e-9,
                "relative error {relative_error} for {value} encoded as {encoded}"
            );
        }
    }

    #[test]
    fn decimal_string_components_fit_the_wire_limit() {
        let encoded = super::format_six([
            f64::MAX,
            f64::MIN_POSITIVE,
            f64::from_bits(1),
            -f64::MAX,
            -f64::MIN_POSITIVE,
            -f64::from_bits(1),
        ])
        .expect("finite DS components");
        for component in encoded.split('\\') {
            assert!(component.len() <= MAXIMUM_DS_BYTES, "{component}");
            assert!(component.parse::<f64>().expect("DS parses").is_finite());
        }
    }

    #[test]
    fn decimal_strings_reject_non_finite_values() {
        let error = format_dicom_decimal(f64::INFINITY).expect_err("infinity is not DS");
        assert!(matches!(
            error.downcast_ref::<DicomWriteError>(),
            Some(DicomWriteError::NonFiniteDecimalStringValue)
        ));
    }
}
