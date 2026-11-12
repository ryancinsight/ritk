use anyhow::{bail, ensure, Result};
use dicom::core::{Tag, VR};

/// User-entered replacement values. Empty optional values explicitly clear tags.
/// Text is restricted to printable ASCII to avoid changing the source charset.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct EditorFields {
    /// Required replacement Patient ID, up to 64 ASCII characters.
    pub patient_id: String,
    /// Replacement Patient Name in DICOM PN form, up to 64 characters.
    pub patient_name: String,
    /// M, F, O, or empty (unknown).
    pub sex: String,
    /// Whole years, 0 through 999; empty clears Patient Age.
    pub age_years: String,
    /// YYYYMMDD, YYYY-MM-DD or empty.
    pub birth_date: String,
    /// Positive finite centimetres, converted to DICOM metres.
    pub height_cm: String,
    /// Positive finite kilograms.
    pub weight_kg: String,
    /// YYYYMMDD, YYYY-MM-DD or empty.
    pub study_date: String,
    /// HHMM, HHMMSS, HH:MM, HH:MM:SS or empty, without fractional seconds.
    pub study_time: String,
    /// Replacement Study ID, up to 16 characters.
    pub study_id: String,
    /// Replacement accession number, up to 16 characters.
    pub accession_number: String,
}

impl EditorFields {
    /// Validate every field before creating any output.
    pub fn validate(&self) -> Result<()> {
        self.elements().map(|_| ())
    }

    pub(super) fn elements(&self) -> Result<Vec<(Tag, VR, String)>> {
        text(&self.patient_id, 64, "Patient ID")?;
        ensure!(!self.patient_id.trim().is_empty(), "Patient ID is required");
        text(&self.patient_name, 64, "Patient Name")?;
        ensure!(
            self.patient_name.split('=').count() <= 3
                && self
                    .patient_name
                    .split('=')
                    .all(|group| group.split('^').count() <= 5),
            "Patient Name has too many DICOM name components"
        );
        ensure!(
            matches!(self.sex.as_str(), "" | "M" | "F" | "O"),
            "Sex must be M, F, O or empty"
        );
        text(&self.study_id, 16, "Study ID")?;
        text(&self.accession_number, 16, "Accession Number")?;
        let age = if self.age_years.is_empty() {
            String::new()
        } else {
            ensure!(
                self.age_years.bytes().all(|b| b.is_ascii_digit()),
                "Age must contain whole years"
            );
            let n: u16 = self.age_years.parse()?;
            ensure!(n <= 999, "Age must be at most 999 years");
            format!("{n:03}Y")
        };
        Ok(vec![
            (Tag(0x0010, 0x0020), VR::LO, self.patient_id.clone()),
            (Tag(0x0010, 0x0010), VR::PN, self.patient_name.clone()),
            (Tag(0x0010, 0x0040), VR::CS, self.sex.clone()),
            (Tag(0x0010, 0x1010), VR::AS, age),
            (Tag(0x0010, 0x0030), VR::DA, date(&self.birth_date)?),
            (
                Tag(0x0010, 0x1020),
                VR::DS,
                decimal(&self.height_cm, 100.0, "Height")?,
            ),
            (
                Tag(0x0010, 0x1030),
                VR::DS,
                decimal(&self.weight_kg, 1.0, "Weight")?,
            ),
            (Tag(0x0008, 0x0020), VR::DA, date(&self.study_date)?),
            (Tag(0x0008, 0x0030), VR::TM, time(&self.study_time)?),
            (Tag(0x0020, 0x0010), VR::SH, self.study_id.clone()),
            (Tag(0x0008, 0x0050), VR::SH, self.accession_number.clone()),
        ])
    }
}

fn text(s: &str, max: usize, name: &str) -> Result<()> {
    ensure!(
        s.len() <= max && s.bytes().all(|b| (32..=126).contains(&b) && b != b'\\'),
        "{name} must be a single printable ASCII value of at most {max} characters"
    );
    Ok(())
}

fn date(s: &str) -> Result<String> {
    if s.is_empty() {
        return Ok(String::new());
    }
    let value = if s.len() == 10 && s.as_bytes()[4] == b'-' && s.as_bytes()[7] == b'-' {
        s.replace('-', "")
    } else {
        s.to_owned()
    };
    ensure!(
        value.len() == 8 && value.bytes().all(|b| b.is_ascii_digit()),
        "Dates must be YYYYMMDD or YYYY-MM-DD"
    );
    let year: u32 = value[0..4].parse()?;
    let month: u32 = value[4..6].parse()?;
    let day: u32 = value[6..8].parse()?;
    let leap = year.is_multiple_of(4) && (!year.is_multiple_of(100) || year.is_multiple_of(400));
    let days = match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 => {
            if leap {
                29
            } else {
                28
            }
        }
        _ => 0,
    };
    ensure!(
        year > 0 && day > 0 && day <= days,
        "Date is not a valid calendar date"
    );
    Ok(value)
}

fn time(s: &str) -> Result<String> {
    if s.is_empty() {
        return Ok(String::new());
    }
    let value = match s.len() {
        4 | 6 => s.to_owned(),
        5 if s.bytes().enumerate().all(|(i, b)| {
            if i == 2 {
                b == b':'
            } else {
                b.is_ascii_digit()
            }
        }) =>
        {
            s.replace(':', "")
        }
        8 if s.bytes().enumerate().all(|(i, b)| {
            if i == 2 || i == 5 {
                b == b':'
            } else {
                b.is_ascii_digit()
            }
        }) =>
        {
            s.replace(':', "")
        }
        _ => bail!("Time must be HHMM, HHMMSS, HH:MM or HH:MM:SS"),
    };
    ensure!(
        matches!(value.len(), 4 | 6) && value.bytes().all(|b| b.is_ascii_digit()),
        "Time must contain four or six digits"
    );
    let hour: u8 = value[0..2].parse()?;
    let minute: u8 = value[2..4].parse()?;
    let second: u8 = if value.len() == 6 {
        value[4..6].parse()?
    } else {
        0
    };
    ensure!(
        hour < 24 && minute < 60 && second < 60,
        "Time is outside a valid clock range"
    );
    Ok(value)
}

fn decimal(s: &str, divisor: f64, name: &str) -> Result<String> {
    if s.is_empty() {
        return Ok(String::new());
    }
    let value: f64 = s.parse()?;
    ensure!(
        value.is_finite() && value > 0.0,
        "{name} must be positive and finite"
    );
    let converted = value / divisor;
    ensure!(converted > 0.0, "{name} underflows its DICOM unit");
    let encoded = converted.to_string();
    ensure!(
        encoded.len() <= 16,
        "{name} exceeds DICOM decimal precision; use fewer digits"
    );
    Ok(encoded)
}
