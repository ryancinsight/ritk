//! DICOM DT admission for the elapsed time used in PET decay correction.

use std::time::Duration;

use horae_civil_time::CivilDate;

const SECONDS_PER_MINUTE: i64 = 60;
const SECONDS_PER_HOUR: i64 = 60 * SECONDS_PER_MINUTE;
const SECONDS_PER_DAY: i64 = 24 * SECONDS_PER_HOUR;
const MICROS_PER_SECOND: i64 = 1_000_000;

pub(super) struct Timestamp {
    micros: i64,
    has_offset: bool,
}

impl Timestamp {
    pub(super) fn parse(text: &str) -> Option<Self> {
        if !text.is_ascii() || text.len() < 14 {
            return None;
        }
        let (clock, suffix) = text.split_at(14);
        if !clock.bytes().all(|byte| byte.is_ascii_digit()) {
            return None;
        }
        let date = CivilDate::new(
            clock[..4].parse().ok()?,
            clock[4..6].parse().ok()?,
            clock[6..8].parse().ok()?,
        )
        .ok()?;
        let hour: i64 = clock[8..10].parse().ok()?;
        let minute: i64 = clock[10..12].parse().ok()?;
        let second: i64 = clock[12..].parse().ok()?;
        if hour >= 24 || minute >= 60 || second >= 60 {
            return None;
        }
        let offset_index = suffix.find(['+', '-']).unwrap_or(suffix.len());
        let (fraction, offset_text) = suffix.split_at(offset_index);
        let micros = fraction_micros(fraction)?;
        let offset = offset_seconds(offset_text)?;
        // CivilDate admits four-digit years. Even the full supported civil
        // span is under 3.2e17 microseconds and is representable by i64.
        let seconds = date.unix_days() * SECONDS_PER_DAY
            + hour * SECONDS_PER_HOUR
            + minute * SECONDS_PER_MINUTE
            + second
            - offset;
        Some(Self {
            micros: seconds * MICROS_PER_SECOND + micros,
            has_offset: !offset_text.is_empty(),
        })
    }

    pub(super) fn seconds_since(self, start: Self) -> Option<f64> {
        if self.has_offset != start.has_offset {
            return None;
        }
        let elapsed = u64::try_from(self.micros - start.micros).ok()?;
        Some(Duration::from_micros(elapsed).as_secs_f64())
    }
}

fn fraction_micros(text: &str) -> Option<i64> {
    if text.is_empty() {
        return Some(0);
    }
    let digits = text.strip_prefix('.')?;
    if !(1..=6).contains(&digits.len()) || !digits.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    let value: i64 = digits.parse().ok()?;
    Some(value * 10_i64.pow(u32::try_from(6 - digits.len()).ok()?))
}

fn offset_seconds(text: &str) -> Option<i64> {
    if text.is_empty() {
        return Some(0);
    }
    let (sign, digits) = text.split_at(1);
    if digits.len() != 4 || !digits.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    let hour: i64 = digits[..2].parse().ok()?;
    let minute: i64 = digits[2..].parse().ok()?;
    let max_hour = if sign == "-" { 12 } else { 14 };
    if minute >= 60 || hour > max_hour || (hour == max_hour && minute != 0) {
        return None;
    }
    let seconds = hour * SECONDS_PER_HOUR + minute * SECONDS_PER_MINUTE;
    Some(if sign == "-" { -seconds } else { seconds })
}

#[cfg(test)]
mod tests {
    use super::Timestamp;

    #[test]
    fn pet_decay_clock_retains_microseconds_and_offset() {
        let start = Timestamp::parse("20241103013000.125000-0400").unwrap();
        let end = Timestamp::parse("20241103013000.625000-0500").unwrap();
        assert_eq!(end.seconds_since(start), Some(3600.5));
        assert!(Timestamp::parse("20241103013000-1260").is_none());
        assert!(Timestamp::parse("20240230013000").is_none());
        assert!(Timestamp::parse("202411030130").is_none());
        let local = Timestamp::parse("20241103013000").unwrap();
        let zoned = Timestamp::parse("20241103013000+0000").unwrap();
        assert_eq!(local.seconds_since(zoned), None);
    }
}
