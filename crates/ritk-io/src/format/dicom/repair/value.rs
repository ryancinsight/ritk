//! Rewrites of single values that older or careless writers encode outside
//! the value representation's grammar.
//!
//! Each function takes one value with its padding already removed and returns
//! the conformant spelling of the same value, or `None` when the value is
//! already conformant or no rewrite is known. A rewrite never changes what the
//! value denotes:
//!
//! - DA: PS3.5 Table 6.2-1 fixes `YYYYMMDD` and names the ACR-NEMA form
//!   `YYYY.MM.DD` as a spelling readers may meet.
//! - TM: the same table fixes `HHMMSS.FFFFFF` with trailing components
//!   optional and names the ACR-NEMA form `HH:MM:SS.frac`.
//! - UI: PS3.5 section 9.1 forbids leading zeros in a component and states
//!   that they carry no meaning, so `1.02.3` and `1.2.3` name one object.

use crate::format::dicom::identity::uid_is_valid;

fn digits(text: &str) -> bool {
    !text.is_empty() && text.bytes().all(|byte| byte.is_ascii_digit())
}

fn in_range(text: &str, low: u8, high: u8) -> bool {
    text.parse::<u8>()
        .is_ok_and(|number| (low..=high).contains(&number))
}

/// `YYYYMMDD` with a month of 01 to 12 and a day of 01 to 31.
pub(super) fn date_is_valid(value: &str) -> bool {
    value.len() == 8
        && digits(value)
        && in_range(&value[4..6], 1, 12)
        && in_range(&value[6..8], 1, 31)
}

/// `HH`, `HHMM`, `HHMMSS` or `HHMMSS.F` with one to six fraction digits;
/// seconds reach 60 for a leap second.
pub(super) fn time_is_valid(value: &str) -> bool {
    let (clock, fraction) = match value.split_once('.') {
        Some((clock, fraction)) => (clock, Some(fraction)),
        None => (value, None),
    };
    let clock_ok = digits(clock)
        && matches!(clock.len(), 2 | 4 | 6)
        && in_range(&clock[..2], 0, 23)
        && clock.get(2..4).is_none_or(|minute| in_range(minute, 0, 59))
        && clock.get(4..6).is_none_or(|second| in_range(second, 0, 60));
    clock_ok
        && fraction
            .is_none_or(|fraction| clock.len() == 6 && digits(fraction) && fraction.len() <= 6)
}

/// `YYYY.MM.DD` as `YYYYMMDD`.
pub(super) fn date(value: &str) -> Option<String> {
    if date_is_valid(value) {
        return None;
    }
    let mut parts = value.split('.');
    let (year, month, day) = (parts.next()?, parts.next()?, parts.next()?);
    let lengths_fit = year.len() == 4 && month.len() == 2 && day.len() == 2;
    let repaired = [year, month, day].concat();
    (parts.next().is_none() && lengths_fit && date_is_valid(&repaired)).then_some(repaired)
}

/// `HH:MM`, `HH:MM:SS` or `HH:MM:SS.frac` without the colons.
pub(super) fn time(value: &str) -> Option<String> {
    if time_is_valid(value) || !value.contains(':') {
        return None;
    }
    let (clock, fraction) = match value.split_once('.') {
        Some((clock, fraction)) => (clock, Some(fraction)),
        None => (value, None),
    };
    let parts: Vec<&str> = clock.split(':').collect();
    let two_digit_parts = matches!(parts.len(), 2 | 3) && parts.iter().all(|part| part.len() == 2);
    let mut repaired = parts.concat();
    if let Some(fraction) = fraction {
        repaired.push('.');
        repaired.push_str(fraction);
    }
    (two_digit_parts && time_is_valid(&repaired)).then_some(repaired)
}

/// A UID with surrounding spaces removed and leading zeros dropped from each
/// component.
pub(super) fn uid(value: &str) -> Option<String> {
    if uid_is_valid(value) {
        return None;
    }
    let repaired = value
        .trim_matches(|c: char| c == ' ' || c == '\0')
        .split('.')
        .map(|component| {
            let significant = component.trim_start_matches('0');
            if significant.is_empty() && !component.is_empty() {
                "0"
            } else {
                significant
            }
        })
        .collect::<Vec<&str>>()
        .join(".");
    uid_is_valid(&repaired).then_some(repaired)
}

#[cfg(test)]
mod tests;
