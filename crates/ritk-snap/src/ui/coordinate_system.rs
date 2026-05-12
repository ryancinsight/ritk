//! Coordinate-system utilities for medical image display.
//!
//! This module provides SSOT helpers for:
//! - Anatomical frame conversion (`LPS` <-> `RAS`)
//! - DICOM patient position parsing (`(0018,5100)`)

/// Anatomical coordinate frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnatomicalFrame {
    /// Left-Posterior-Superior patient frame (DICOM/ITK convention).
    Lps,
    /// Right-Anterior-Superior patient frame (NIfTI/FSL convention).
    Ras,
}

impl AnatomicalFrame {
    /// Short display label.
    pub fn label(self) -> &'static str {
        match self {
            Self::Lps => "LPS",
            Self::Ras => "RAS",
        }
    }
}

/// Convert a point from LPS to RAS.
///
/// Relation:
/// - `R = -L`
/// - `A = -P`
/// - `S =  S`
#[inline]
pub fn lps_to_ras(lps: [f64; 3]) -> [f64; 3] {
    [-lps[0], -lps[1], lps[2]]
}

/// Convert a point from RAS to LPS.
///
/// This is the same sign flip as [`lps_to_ras`].
#[inline]
pub fn ras_to_lps(ras: [f64; 3]) -> [f64; 3] {
    [-ras[0], -ras[1], ras[2]]
}

/// Format a physical point in the requested anatomical frame.
pub fn format_point_mm(point: [f64; 3], frame: AnatomicalFrame) -> String {
    format!(
        "{} ({:.2}, {:.2}, {:.2}) mm",
        frame.label(),
        point[0],
        point[1],
        point[2]
    )
}

/// DICOM patient position code (`(0018,5100)`) classification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PatientPosition {
    HeadFirstSupine,
    HeadFirstProne,
    FeetFirstSupine,
    FeetFirstProne,
    HeadFirstDecubitusRight,
    HeadFirstDecubitusLeft,
    FeetFirstDecubitusRight,
    FeetFirstDecubitusLeft,
    Unknown(String),
}

impl PatientPosition {
    /// Parse a DICOM patient position code, e.g. `HFS`, `FFP`, `HFDL`.
    pub fn from_dicom_code(code: &str) -> Self {
        let c = code.trim().to_ascii_uppercase();
        match c.as_str() {
            "HFS" => Self::HeadFirstSupine,
            "HFP" => Self::HeadFirstProne,
            "FFS" => Self::FeetFirstSupine,
            "FFP" => Self::FeetFirstProne,
            "HFDR" => Self::HeadFirstDecubitusRight,
            "HFDL" => Self::HeadFirstDecubitusLeft,
            "FFDR" => Self::FeetFirstDecubitusRight,
            "FFDL" => Self::FeetFirstDecubitusLeft,
            _ => Self::Unknown(c),
        }
    }

    /// Canonical code string.
    pub fn code(&self) -> &str {
        match self {
            Self::HeadFirstSupine => "HFS",
            Self::HeadFirstProne => "HFP",
            Self::FeetFirstSupine => "FFS",
            Self::FeetFirstProne => "FFP",
            Self::HeadFirstDecubitusRight => "HFDR",
            Self::HeadFirstDecubitusLeft => "HFDL",
            Self::FeetFirstDecubitusRight => "FFDR",
            Self::FeetFirstDecubitusLeft => "FFDL",
            Self::Unknown(code) => code.as_str(),
        }
    }

    /// Human-readable display label.
    pub fn label(&self) -> &'static str {
        match self {
            Self::HeadFirstSupine => "Head-First Supine",
            Self::HeadFirstProne => "Head-First Prone",
            Self::FeetFirstSupine => "Feet-First Supine",
            Self::FeetFirstProne => "Feet-First Prone",
            Self::HeadFirstDecubitusRight => "Head-First Decubitus Right",
            Self::HeadFirstDecubitusLeft => "Head-First Decubitus Left",
            Self::FeetFirstDecubitusRight => "Feet-First Decubitus Right",
            Self::FeetFirstDecubitusLeft => "Feet-First Decubitus Left",
            Self::Unknown(_) => "Unknown Position",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lps_to_ras_is_sign_flip_xy_only() {
        let lps = [10.0, -20.0, 30.0];
        assert_eq!(lps_to_ras(lps), [-10.0, 20.0, 30.0]);
    }

    #[test]
    fn ras_to_lps_is_inverse_of_lps_to_ras() {
        let lps = [12.5, 3.0, -7.25];
        let ras = lps_to_ras(lps);
        assert_eq!(ras_to_lps(ras), lps);
    }

    #[test]
    fn patient_position_parser_maps_standard_codes() {
        assert_eq!(PatientPosition::from_dicom_code("hfs"), PatientPosition::HeadFirstSupine);
        assert_eq!(PatientPosition::from_dicom_code("FFP"), PatientPosition::FeetFirstProne);
        assert_eq!(
            PatientPosition::from_dicom_code("HFDL"),
            PatientPosition::HeadFirstDecubitusLeft
        );
    }

    #[test]
    fn patient_position_parser_preserves_unknown_code() {
        assert_eq!(
            PatientPosition::from_dicom_code("XYZ"),
            PatientPosition::Unknown("XYZ".to_string())
        );
    }
}
