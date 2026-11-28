//! DICOM Transfer Syntax classification.
//!
//! # Invariants
//! - `from_uid(x.uid()) == x` for every non-unknown variant.
//! - `is_native_uncompressed() -> !is_encapsulated()`.
//! - `is_backend_codec_candidate() -> is_encapsulated()`.
//! - `is_codec_supported() -> is_encapsulated()`.

use arrayvec::ArrayString;
use ritk_codecs::jpeg_2000::{HtJ2kProfile, Jpeg2000Profile};

/// DICOM Transfer Syntax classification used by RITK decode dispatch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TransferSyntaxKind {
    ImplicitVrLittleEndian,
    ExplicitVrLittleEndian,
    ExplicitVrBigEndian,
    DeflatedExplicitVrLittleEndian,
    EncapsulatedUncompressed,
    DeflatedImageFrame,
    JpegBaseline,
    JpegExtended,
    JpegLosslessNonHierarchical,
    JpegLosslessFirstOrderPrediction,
    JpegLsLossless,
    JpegLsLossy,
    Jpeg2000Lossless,
    Jpeg2000Lossy,
    HtJ2kLossless,
    HtJ2kLosslessRpcl,
    HtJ2k,
    RleLossless,
    JpegXlLossless,
    JpegXlJpegRecompression,
    JpegXl,
    Unknown(ArrayString<64>),
}

#[derive(Clone, Copy)]
pub(crate) enum Codec {
    Jpeg,
    JpegLs,
    Jpeg2000(Jpeg2000Profile),
    Rle,
    NativeFrame,
    JpegXl,
}

impl TransferSyntaxKind {
    pub fn from_uid(uid: &str) -> Self {
        match uid.trim().trim_end_matches('\0') {
            "1.2.840.10008.1.2" => Self::ImplicitVrLittleEndian,
            "1.2.840.10008.1.2.1" => Self::ExplicitVrLittleEndian,
            "1.2.840.10008.1.2.2" => Self::ExplicitVrBigEndian,
            "1.2.840.10008.1.2.1.98" => Self::EncapsulatedUncompressed,
            "1.2.840.10008.1.2.8.1" => Self::DeflatedImageFrame,
            "1.2.840.10008.1.2.1.99" => Self::DeflatedExplicitVrLittleEndian,
            "1.2.840.10008.1.2.4.50" => Self::JpegBaseline,
            "1.2.840.10008.1.2.4.51" => Self::JpegExtended,
            "1.2.840.10008.1.2.4.57" => Self::JpegLosslessNonHierarchical,
            "1.2.840.10008.1.2.4.70" => Self::JpegLosslessFirstOrderPrediction,
            "1.2.840.10008.1.2.4.80" => Self::JpegLsLossless,
            "1.2.840.10008.1.2.4.81" => Self::JpegLsLossy,
            "1.2.840.10008.1.2.4.90" => Self::Jpeg2000Lossless,
            "1.2.840.10008.1.2.4.91" => Self::Jpeg2000Lossy,
            "1.2.840.10008.1.2.4.201" => Self::HtJ2kLossless,
            "1.2.840.10008.1.2.4.202" => Self::HtJ2kLosslessRpcl,
            "1.2.840.10008.1.2.4.203" => Self::HtJ2k,
            "1.2.840.10008.1.2.5" => Self::RleLossless,
            "1.2.840.10008.1.2.4.110" => Self::JpegXlLossless,
            "1.2.840.10008.1.2.4.111" => Self::JpegXlJpegRecompression,
            "1.2.840.10008.1.2.4.112" => Self::JpegXl,
            other => Self::Unknown(ArrayString::<64>::try_from(other).unwrap_or_default()),
        }
    }

    pub fn uid(&self) -> &str {
        match self {
            Self::ImplicitVrLittleEndian => "1.2.840.10008.1.2",
            Self::ExplicitVrLittleEndian => "1.2.840.10008.1.2.1",
            Self::ExplicitVrBigEndian => "1.2.840.10008.1.2.2",
            Self::DeflatedExplicitVrLittleEndian => "1.2.840.10008.1.2.1.99",
            Self::EncapsulatedUncompressed => "1.2.840.10008.1.2.1.98",
            Self::DeflatedImageFrame => "1.2.840.10008.1.2.8.1",
            Self::JpegBaseline => "1.2.840.10008.1.2.4.50",
            Self::JpegExtended => "1.2.840.10008.1.2.4.51",
            Self::JpegLosslessNonHierarchical => "1.2.840.10008.1.2.4.57",
            Self::JpegLosslessFirstOrderPrediction => "1.2.840.10008.1.2.4.70",
            Self::JpegLsLossless => "1.2.840.10008.1.2.4.80",
            Self::JpegLsLossy => "1.2.840.10008.1.2.4.81",
            Self::Jpeg2000Lossless => "1.2.840.10008.1.2.4.90",
            Self::Jpeg2000Lossy => "1.2.840.10008.1.2.4.91",
            Self::HtJ2kLossless => "1.2.840.10008.1.2.4.201",
            Self::HtJ2kLosslessRpcl => "1.2.840.10008.1.2.4.202",
            Self::HtJ2k => "1.2.840.10008.1.2.4.203",
            Self::RleLossless => "1.2.840.10008.1.2.5",
            Self::JpegXlLossless => "1.2.840.10008.1.2.4.110",
            Self::JpegXlJpegRecompression => "1.2.840.10008.1.2.4.111",
            Self::JpegXl => "1.2.840.10008.1.2.4.112",
            Self::Unknown(uid) => uid.as_str(),
        }
    }

    /// Part 1 and Part 15 share DICOM frame encapsulation and exact integer readout.
    pub fn is_jpeg2000_family(&self) -> bool {
        matches!(self.codec(), Some(Codec::Jpeg2000(_)))
    }

    /// DICOM PS3.5 §10.18 profile signaled by this transfer syntax.
    pub fn htj2k_profile(&self) -> Option<HtJ2kProfile> {
        match self.codec() {
            Some(Codec::Jpeg2000(Jpeg2000Profile::HighThroughput(profile))) => Some(profile),
            _ => None,
        }
    }

    /// Photometric interpretations delivered as interleaved RGB by the decoder.
    pub fn decodes_to_rgb(&self, photometric: &str) -> bool {
        let pi = photometric.trim();
        pi.eq_ignore_ascii_case("RGB")
            || self.is_jpeg2000_family()
                && (pi.eq_ignore_ascii_case("YBR_RCT") || pi.eq_ignore_ascii_case("YBR_ICT"))
    }

    pub fn is_encapsulated(&self) -> bool {
        self.codec().is_some()
    }

    pub(crate) fn codec(&self) -> Option<Codec> {
        Some(match self {
            Self::JpegBaseline
            | Self::JpegExtended
            | Self::JpegLosslessNonHierarchical
            | Self::JpegLosslessFirstOrderPrediction => Codec::Jpeg,
            Self::JpegLsLossless | Self::JpegLsLossy => Codec::JpegLs,
            Self::Jpeg2000Lossless | Self::Jpeg2000Lossy => {
                Codec::Jpeg2000(Jpeg2000Profile::Part1 {
                    lossless_only: self == &Self::Jpeg2000Lossless,
                })
            }
            Self::HtJ2kLossless => {
                Codec::Jpeg2000(Jpeg2000Profile::HighThroughput(HtJ2kProfile::Lossless))
            }
            Self::HtJ2kLosslessRpcl => {
                Codec::Jpeg2000(Jpeg2000Profile::HighThroughput(HtJ2kProfile::LosslessRpcl))
            }
            Self::HtJ2k => {
                Codec::Jpeg2000(Jpeg2000Profile::HighThroughput(HtJ2kProfile::Unrestricted))
            }
            Self::RleLossless => Codec::Rle,
            Self::EncapsulatedUncompressed | Self::DeflatedImageFrame => Codec::NativeFrame,
            Self::JpegXlLossless | Self::JpegXlJpegRecompression | Self::JpegXl => Codec::JpegXl,
            Self::ImplicitVrLittleEndian
            | Self::ExplicitVrLittleEndian
            | Self::ExplicitVrBigEndian
            | Self::DeflatedExplicitVrLittleEndian
            | Self::Unknown(_) => return None,
        })
    }

    /// Each encapsulated fragment contains a byte-aligned native frame, optionally deflated.
    pub fn has_native_frame_layout(&self) -> bool {
        matches!(
            self,
            Self::EncapsulatedUncompressed | Self::DeflatedImageFrame
        )
    }

    pub fn is_compressed(&self) -> bool {
        self.is_encapsulated() && !matches!(self, Self::EncapsulatedUncompressed)
    }

    pub fn is_lossless(&self) -> bool {
        matches!(
            self,
            Self::ImplicitVrLittleEndian
                | Self::ExplicitVrLittleEndian
                | Self::ExplicitVrBigEndian
                | Self::DeflatedExplicitVrLittleEndian
                | Self::EncapsulatedUncompressed
                | Self::DeflatedImageFrame
                | Self::JpegLosslessNonHierarchical
                | Self::JpegLosslessFirstOrderPrediction
                | Self::JpegLsLossless
                | Self::HtJ2kLossless
                | Self::HtJ2kLosslessRpcl
                | Self::Jpeg2000Lossless
                | Self::RleLossless
                | Self::JpegXlLossless
        )
    }

    pub fn is_native_uncompressed(&self) -> bool {
        matches!(
            self,
            Self::ImplicitVrLittleEndian
                | Self::ExplicitVrLittleEndian
                | Self::ExplicitVrBigEndian
                | Self::DeflatedExplicitVrLittleEndian
        )
    }

    pub fn is_natively_supported(&self) -> bool {
        self.is_native_uncompressed()
    }

    pub fn is_big_endian(&self) -> bool {
        matches!(self, Self::ExplicitVrBigEndian)
    }

    pub fn is_backend_codec_candidate(&self) -> bool {
        self.is_encapsulated()
    }

    /// Returns true if the transfer syntax is a JPEG-LS variant (Lossless or Lossy).
    pub fn is_jpeg_ls(&self) -> bool {
        matches!(self.codec(), Some(Codec::JpegLs))
    }

    pub fn is_codec_supported(&self) -> bool {
        self.is_backend_codec_candidate()
    }

    pub fn is_native_jpeg_codec(&self) -> bool {
        matches!(self.codec(), Some(Codec::Jpeg))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn variants() -> [TransferSyntaxKind; 21] {
        [
            TransferSyntaxKind::ImplicitVrLittleEndian,
            TransferSyntaxKind::ExplicitVrLittleEndian,
            TransferSyntaxKind::ExplicitVrBigEndian,
            TransferSyntaxKind::DeflatedExplicitVrLittleEndian,
            TransferSyntaxKind::EncapsulatedUncompressed,
            TransferSyntaxKind::DeflatedImageFrame,
            TransferSyntaxKind::JpegBaseline,
            TransferSyntaxKind::JpegExtended,
            TransferSyntaxKind::JpegLosslessNonHierarchical,
            TransferSyntaxKind::JpegLosslessFirstOrderPrediction,
            TransferSyntaxKind::JpegLsLossless,
            TransferSyntaxKind::JpegLsLossy,
            TransferSyntaxKind::Jpeg2000Lossless,
            TransferSyntaxKind::Jpeg2000Lossy,
            TransferSyntaxKind::HtJ2kLossless,
            TransferSyntaxKind::HtJ2kLosslessRpcl,
            TransferSyntaxKind::HtJ2k,
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
    fn from_uid_accepts_dicom_ui_padding_byte() {
        assert_eq!(
            TransferSyntaxKind::from_uid("1.2.840.10008.1.2.4.80\0"),
            TransferSyntaxKind::JpegLsLossless
        );
    }

    #[test]
    fn native_and_backend_codec_paths_are_disjoint() {
        for syntax in variants() {
            if syntax.is_native_uncompressed() {
                assert!(!syntax.is_encapsulated());
                assert!(!syntax.is_compressed());
                assert!(!syntax.is_backend_codec_candidate());
                assert!(!syntax.is_codec_supported());
                assert!(syntax.is_natively_supported());
                assert_eq!(
                    syntax.is_big_endian(),
                    syntax == TransferSyntaxKind::ExplicitVrBigEndian
                );
            }
            if syntax.is_backend_codec_candidate() {
                assert!(syntax.is_encapsulated());
                assert_eq!(
                    syntax.is_compressed(),
                    syntax != TransferSyntaxKind::EncapsulatedUncompressed
                );
                assert!(syntax.is_codec_supported());
            }
        }
    }

    #[test]
    fn compatibility_predicates_match_canonical_predicates() {
        for syntax in variants() {
            assert_eq!(
                syntax.is_compressed(),
                syntax.is_encapsulated() && syntax != TransferSyntaxKind::EncapsulatedUncompressed
            );
            assert_eq!(
                syntax.is_codec_supported(),
                syntax.is_backend_codec_candidate()
            );
            assert_eq!(
                syntax.is_natively_supported(),
                syntax.is_native_uncompressed()
            );
        }
    }

    #[test]
    fn jpeg_predicate_tracks_the_jpeg_family() {
        for syntax in variants() {
            match syntax {
                TransferSyntaxKind::JpegBaseline
                | TransferSyntaxKind::JpegExtended
                | TransferSyntaxKind::JpegLosslessNonHierarchical
                | TransferSyntaxKind::JpegLosslessFirstOrderPrediction => {
                    assert!(syntax.is_native_jpeg_codec());
                }
                _ => {
                    assert!(!syntax.is_native_jpeg_codec());
                }
            }
        }
    }
}
