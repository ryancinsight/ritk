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

/// Days in `month` of `year` in the Gregorian calendar, the calendar PS3.5
/// Table 6.2-1 fixes for DA; zero for a month outside 1 to 12.
fn days_in_month(year: u16, month: u8) -> u8 {
    let leap = year.is_multiple_of(4) && (!year.is_multiple_of(100) || year.is_multiple_of(400));
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if leap => 29,
        2 => 28,
        _ => 0,
    }
}

/// `YYYYMMDD` naming a day of the Gregorian calendar.
pub(super) fn date_is_valid(value: &str) -> bool {
    if value.len() != 8 || !digits(value) {
        return false;
    }
    let (Ok(year), Ok(month), Ok(day)) = (
        value[..4].parse::<u16>(),
        value[4..6].parse::<u8>(),
        value[6..].parse::<u8>(),
    ) else {
        return false;
    };
    (1..=days_in_month(year, month)).contains(&day)
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
    let mut parts = value.split('.');
    let (year, month, day) = (parts.next()?, parts.next()?, parts.next()?);
    let lengths_fit = year.len() == 4 && month.len() == 2 && day.len() == 2;
    let repaired = [year, month, day].concat();
    (parts.next().is_none() && lengths_fit && date_is_valid(&repaired)).then_some(repaired)
}

/// `HH:MM`, `HH:MM:SS` or `HH:MM:SS.frac` without the colons.
pub(super) fn time(value: &str) -> Option<String> {
    if !value.contains(':') {
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
