//! DICOM Transfer Syntax classification.
//!
//! # Invariants
//! - `from_uid(x.uid()) == x` for every non-unknown variant.
//! - `is_native_uncompressed() -> !is_encapsulated()`.
//! - `is_backend_codec_candidate() -> is_encapsulated()`.

/// DICOM Transfer Syntax classification used by RITK decode dispatch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TransferSyntaxKind {
    ImplicitVrLittleEndian,
    ExplicitVrLittleEndian,
    ExplicitVrBigEndian,
    DeflatedExplicitVrLittleEndian,
    JpegBaseline,
    JpegExtended,
    JpegLosslessNonHierarchical,
    JpegLosslessFirstOrderPrediction,
    JpegLsLossless,
    JpegLsLossy,
    Jpeg2000Lossless,
    Jpeg2000Lossy,
    RleLossless,
    JpegXlLossless,
    JpegXlJpegRecompression,
    JpegXl,
    Unknown(String),
}

impl TransferSyntaxKind {
    pub fn from_uid(uid: &str) -> Self {
        match uid.trim() {
            "1.2.840.10008.1.2" => Self::ImplicitVrLittleEndian,
            "1.2.840.10008.1.2.1" => Self::ExplicitVrLittleEndian,
            "1.2.840.10008.1.2.2" => Self::ExplicitVrBigEndian,
            "1.2.840.10008.1.2.1.99" => Self::DeflatedExplicitVrLittleEndian,
            "1.2.840.10008.1.2.4.50" => Self::JpegBaseline,
            "1.2.840.10008.1.2.4.51" => Self::JpegExtended,
            "1.2.840.10008.1.2.4.57" => Self::JpegLosslessNonHierarchical,
            "1.2.840.10008.1.2.4.70" => Self::JpegLosslessFirstOrderPrediction,
            "1.2.840.10008.1.2.4.80" => Self::JpegLsLossless,
            "1.2.840.10008.1.2.4.81" => Self::JpegLsLossy,
            "1.2.840.10008.1.2.4.90" => Self::Jpeg2000Lossless,
            "1.2.840.10008.1.2.4.91" => Self::Jpeg2000Lossy,
            "1.2.840.10008.1.2.5" => Self::RleLossless,
            "1.2.840.10008.1.2.4.110" => Self::JpegXlLossless,
            "1.2.840.10008.1.2.4.111" => Self::JpegXlJpegRecompression,
            "1.2.840.10008.1.2.4.112" => Self::JpegXl,
            other => Self::Unknown(other.to_string()),
        }
    }

    pub fn uid(&self) -> &str {
        match self {
            Self::ImplicitVrLittleEndian => "1.2.840.10008.1.2",
            Self::ExplicitVrLittleEndian => "1.2.840.10008.1.2.1",
            Self::ExplicitVrBigEndian => "1.2.840.10008.1.2.2",
            Self::DeflatedExplicitVrLittleEndian => "1.2.840.10008.1.2.1.99",
            Self::JpegBaseline => "1.2.840.10008.1.2.4.50",
            Self::JpegExtended => "1.2.840.10008.1.2.4.51",
            Self::JpegLosslessNonHierarchical => "1.2.840.10008.1.2.4.57",
            Self::JpegLosslessFirstOrderPrediction => "1.2.840.10008.1.2.4.70",
            Self::JpegLsLossless => "1.2.840.10008.1.2.4.80",
            Self::JpegLsLossy => "1.2.840.10008.1.2.4.81",
            Self::Jpeg2000Lossless => "1.2.840.10008.1.2.4.90",
            Self::Jpeg2000Lossy => "1.2.840.10008.1.2.4.91",
            Self::RleLossless => "1.2.840.10008.1.2.5",
            Self::JpegXlLossless => "1.2.840.10008.1.2.4.110",
            Self::JpegXlJpegRecompression => "1.2.840.10008.1.2.4.111",
            Self::JpegXl => "1.2.840.10008.1.2.4.112",
            Self::Unknown(uid) => uid.as_str(),
        }
    }

    pub fn is_encapsulated(&self) -> bool {
        matches!(
            self,
            Self::JpegBaseline
                | Self::JpegExtended
                | Self::JpegLosslessNonHierarchical
                | Self::JpegLosslessFirstOrderPrediction
                | Self::JpegLsLossless
                | Self::JpegLsLossy
                | Self::Jpeg2000Lossless
                | Self::Jpeg2000Lossy
                | Self::RleLossless
                | Self::JpegXlLossless
                | Self::JpegXlJpegRecompression
                | Self::JpegXl
        )
    }

    pub fn is_lossless(&self) -> bool {
        matches!(
            self,
            Self::ImplicitVrLittleEndian
                | Self::ExplicitVrLittleEndian
                | Self::ExplicitVrBigEndian
                | Self::DeflatedExplicitVrLittleEndian
                | Self::JpegLosslessNonHierarchical
                | Self::JpegLosslessFirstOrderPrediction
                | Self::JpegLsLossless
                | Self::Jpeg2000Lossless
                | Self::RleLossless
                | Self::JpegXlLossless
        )
    }

    pub fn is_native_uncompressed(&self) -> bool {
        matches!(
            self,
            Self::ImplicitVrLittleEndian | Self::ExplicitVrLittleEndian
        )
    }

    pub fn is_big_endian(&self) -> bool {
        matches!(self, Self::ExplicitVrBigEndian)
    }

    pub fn is_backend_codec_candidate(&self) -> bool {
        self.is_encapsulated()
    }

    pub fn is_native_ritk_codec(&self) -> bool {
        matches!(self, Self::RleLossless)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn variants() -> [TransferSyntaxKind; 16] {
        [
            TransferSyntaxKind::ImplicitVrLittleEndian,
            TransferSyntaxKind::ExplicitVrLittleEndian,
            TransferSyntaxKind::ExplicitVrBigEndian,
            TransferSyntaxKind::DeflatedExplicitVrLittleEndian,
            TransferSyntaxKind::JpegBaseline,
            TransferSyntaxKind::JpegExtended,
            TransferSyntaxKind::JpegLosslessNonHierarchical,
            TransferSyntaxKind::JpegLosslessFirstOrderPrediction,
            TransferSyntaxKind::JpegLsLossless,
            TransferSyntaxKind::JpegLsLossy,
            TransferSyntaxKind::Jpeg2000Lossless,
            TransferSyntaxKind::Jpeg2000Lossy,
            TransferSyntaxKind::RleLossless,
            TransferSyntaxKind::JpegXlLossless,
            TransferSyntaxKind::JpegXlJpegRecompression,
            TransferSyntaxKind::JpegXl,
        ]
    }

    #[test]
    fn uid_roundtrip_holds_for_known_transfer_syntaxes() {
        for syntax in variants() {
            assert_eq!(TransferSyntaxKind::from_uid(syntax.uid()), syntax);
        }
    }

    #[test]
    fn native_and_backend_codec_paths_are_disjoint() {
        for syntax in variants() {
            if syntax.is_native_uncompressed() {
                assert!(!syntax.is_encapsulated());
                assert!(!syntax.is_backend_codec_candidate());
                assert!(!syntax.is_big_endian());
            }
            if syntax.is_backend_codec_candidate() {
                assert!(syntax.is_encapsulated());
            }
        }
    }
}
