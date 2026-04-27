//! Transfer Syntax identification and capability classification.
//!
//! # Specification
//!
//! Maps a DICOM Transfer Syntax UID to a capability class:
//! - Natively supported: reader/writer processes without external codec.
//! - Compressed: pixel data requires a codec.
//! - Lossless: compression preserves all information.
//!
//! ## Invariants
//! - from_uid(x.uid()) == x for every non-Unknown variant.
//! - is_natively_supported() => !is_compressed() || x==DeflatedExplicit.

use super::reader::DicomSliceMetadata;

/// DICOM Transfer Syntax classification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TransferSyntaxKind {
    /// 1.2.840.10008.1.2 - Implicit VR Little Endian.
    ImplicitVrLittleEndian,
    /// 1.2.840.10008.1.2.1 - Explicit VR Little Endian.
    ExplicitVrLittleEndian,
    /// 1.2.840.10008.1.2.2 - Explicit VR Big Endian.
    ExplicitVrBigEndian,
    /// 1.2.840.10008.1.2.4.50 - JPEG Baseline (lossy).
    JpegBaseline,
    /// 1.2.840.10008.1.2.4.70 - JPEG Lossless First Order Prediction.
    JpegLosslessFirstOrderPrediction,
    /// 1.2.840.10008.1.2.4.80 - JPEG-LS Lossless.
    JpegLsLossless,
    /// 1.2.840.10008.1.2.4.81 - JPEG-LS Lossy.
    JpegLsLossy,
    /// 1.2.840.10008.1.2.4.90 - JPEG 2000 Lossless.
    Jpeg2000Lossless,
    /// 1.2.840.10008.1.2.4.91 - JPEG 2000 Lossy.
    Jpeg2000Lossy,
    /// 1.2.840.10008.1.2.5 - RLE Lossless.
    RleLossless,
    /// 1.2.840.10008.1.2.1.99 - Deflated Explicit VR Little Endian.
    DeflatedExplicitVrLittleEndian,
    /// Any UID not matched by the known variants.
    Unknown(String),
}

impl TransferSyntaxKind {
    /// Construct from a Transfer Syntax UID string.
    pub fn from_uid(uid: &str) -> Self {
        match uid.trim() {
            "1.2.840.10008.1.2" => Self::ImplicitVrLittleEndian,
            "1.2.840.10008.1.2.1" => Self::ExplicitVrLittleEndian,
            "1.2.840.10008.1.2.2" => Self::ExplicitVrBigEndian,
            "1.2.840.10008.1.2.4.50" => Self::JpegBaseline,
            "1.2.840.10008.1.2.4.70" => Self::JpegLosslessFirstOrderPrediction,
            "1.2.840.10008.1.2.4.80" => Self::JpegLsLossless,
            "1.2.840.10008.1.2.4.81" => Self::JpegLsLossy,
            "1.2.840.10008.1.2.4.90" => Self::Jpeg2000Lossless,
            "1.2.840.10008.1.2.4.91" => Self::Jpeg2000Lossy,
            "1.2.840.10008.1.2.5" => Self::RleLossless,
            "1.2.840.10008.1.2.1.99" => Self::DeflatedExplicitVrLittleEndian,
            other => Self::Unknown(other.to_string()),
        }
    }

    /// Return the canonical UID string for known variants; returns the stored
    /// string for Unknown.
    pub fn uid(&self) -> &str {
        match self {
            Self::ImplicitVrLittleEndian => "1.2.840.10008.1.2",
            Self::ExplicitVrLittleEndian => "1.2.840.10008.1.2.1",
            Self::ExplicitVrBigEndian => "1.2.840.10008.1.2.2",
            Self::JpegBaseline => "1.2.840.10008.1.2.4.50",
            Self::JpegLosslessFirstOrderPrediction => "1.2.840.10008.1.2.4.70",
            Self::JpegLsLossless => "1.2.840.10008.1.2.4.80",
            Self::JpegLsLossy => "1.2.840.10008.1.2.4.81",
            Self::Jpeg2000Lossless => "1.2.840.10008.1.2.4.90",
            Self::Jpeg2000Lossy => "1.2.840.10008.1.2.4.91",
            Self::RleLossless => "1.2.840.10008.1.2.5",
            Self::DeflatedExplicitVrLittleEndian => "1.2.840.10008.1.2.1.99",
            Self::Unknown(uid) => uid.as_str(),
        }
    }

    /// True when pixel data requires a codec to access.
    pub fn is_compressed(&self) -> bool {
        matches!(
            self,
            Self::JpegBaseline
                | Self::JpegLosslessFirstOrderPrediction
                | Self::JpegLsLossless
                | Self::JpegLsLossy
                | Self::Jpeg2000Lossless
                | Self::Jpeg2000Lossy
                | Self::RleLossless
                | Self::DeflatedExplicitVrLittleEndian
        )
    }

    /// True when no information is destroyed by the compression scheme.
    pub fn is_lossless(&self) -> bool {
        matches!(
            self,
            Self::ImplicitVrLittleEndian
                | Self::ExplicitVrLittleEndian
                | Self::ExplicitVrBigEndian
                | Self::JpegLosslessFirstOrderPrediction
                | Self::JpegLsLossless
                | Self::Jpeg2000Lossless
                | Self::RleLossless
                | Self::DeflatedExplicitVrLittleEndian
        )
    }

    /// True when ritk-io can read/write this syntax without an external codec.
    pub fn is_natively_supported(&self) -> bool {
        matches!(
            self,
            Self::ImplicitVrLittleEndian
                | Self::ExplicitVrLittleEndian
                | Self::ExplicitVrBigEndian
                | Self::DeflatedExplicitVrLittleEndian
        )
    }

    /// Derive from a `DicomSliceMetadata` record.
    ///
    /// Returns `Unknown("")` when the transfer syntax UID is absent.
    pub fn from_metadata(meta: &DicomSliceMetadata) -> Self {
        match meta.transfer_syntax_uid.as_deref() {
            Some(uid) => Self::from_uid(uid),
            None => Self::Unknown(String::new()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::reader::DicomSliceMetadata;
    use super::*;

    fn make_slice_meta(uid: Option<&str>) -> DicomSliceMetadata {
        DicomSliceMetadata {
            transfer_syntax_uid: uid.map(|s| s.to_string()),
            ..DicomSliceMetadata::default()
        }
    }

    #[test]
    fn test_from_uid_implicit_vr_le() {
        assert_eq!(
            TransferSyntaxKind::from_uid("1.2.840.10008.1.2"),
            TransferSyntaxKind::ImplicitVrLittleEndian
        );
    }

    #[test]
    fn test_from_uid_explicit_vr_le() {
        assert_eq!(
            TransferSyntaxKind::from_uid("1.2.840.10008.1.2.1"),
            TransferSyntaxKind::ExplicitVrLittleEndian
        );
    }

    #[test]
    fn test_from_uid_unknown() {
        match TransferSyntaxKind::from_uid("9.9.9.9") {
            TransferSyntaxKind::Unknown(s) => assert_eq!(s, "9.9.9.9"),
            _ => panic!("expected Unknown"),
        }
    }

    #[test]
    fn test_is_compressed_jpeg_baseline() {
        assert!(TransferSyntaxKind::JpegBaseline.is_compressed());
    }

    #[test]
    fn test_is_compressed_explicit_vr_le_false() {
        assert!(!TransferSyntaxKind::ExplicitVrLittleEndian.is_compressed());
    }

    #[test]
    fn test_is_lossless_jpeg_baseline_false() {
        assert!(!TransferSyntaxKind::JpegBaseline.is_lossless());
    }

    #[test]
    fn test_is_natively_supported_deflated() {
        assert!(TransferSyntaxKind::DeflatedExplicitVrLittleEndian.is_natively_supported());
    }

    #[test]
    fn test_from_metadata_none_uid() {
        let meta = make_slice_meta(None);
        match TransferSyntaxKind::from_metadata(&meta) {
            TransferSyntaxKind::Unknown(s) => assert!(s.is_empty()),
            _ => panic!("expected Unknown"),
        }
    }

    #[test]
    fn test_from_metadata_known_uid() {
        let meta = make_slice_meta(Some("1.2.840.10008.1.2.1"));
        assert_eq!(
            TransferSyntaxKind::from_metadata(&meta),
            TransferSyntaxKind::ExplicitVrLittleEndian
        );
    }

    #[test]
    fn test_uid_roundtrip_all_known() {
        let variants = [
            TransferSyntaxKind::ImplicitVrLittleEndian,
            TransferSyntaxKind::ExplicitVrLittleEndian,
            TransferSyntaxKind::ExplicitVrBigEndian,
            TransferSyntaxKind::JpegBaseline,
            TransferSyntaxKind::JpegLosslessFirstOrderPrediction,
            TransferSyntaxKind::JpegLsLossless,
            TransferSyntaxKind::JpegLsLossy,
            TransferSyntaxKind::Jpeg2000Lossless,
            TransferSyntaxKind::Jpeg2000Lossy,
            TransferSyntaxKind::RleLossless,
            TransferSyntaxKind::DeflatedExplicitVrLittleEndian,
        ];
        for v in &variants {
            assert_eq!(
                TransferSyntaxKind::from_uid(v.uid()),
                *v,
                "Roundtrip failed for {:?}",
                v
            );
        }
    }
}
