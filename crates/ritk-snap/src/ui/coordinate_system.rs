//! Coordinate-system utilities for medical image display.
//!
//! This module provides SSOT helpers for:
//! - Anatomical frame conversion (`LPS` <-> `RAS`)
//! - DICOM patient position parsing (`(0018,5100)`)

// Re-export PatientPosition from ritk-io (SSOT — eliminates duplicate enum).
pub use ritk_io::PatientPosition;

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
/// - `S = S`
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
        use ritk_io::literal_arraystring;
        assert_eq!(
            PatientPosition::from_dicom_code("XYZ"),
            PatientPosition::Unknown(literal_arraystring::<4>("XYZ"))
        );
    }
}
