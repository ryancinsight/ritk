//! Transfer Syntax identification and capability classification.
//!
//! # Specification
//!
//! Maps a DICOM Transfer Syntax UID to a capability class:
//! - Natively supported: reader/writer processes without external codec.
//! - Compressed: pixel data encapsulated in fragments requiring a codec.
//! - Lossless: compression preserves all information.
//!
//! ## Invariants
//! - `from_uid(x.uid()) == x` for every non-Unknown variant.
//! - `is_natively_supported()` ⟹ `!is_compressed()` ∧ `!is_big_endian()`.
//! - `is_codec_supported()` ⟹ `is_compressed()`.
//! - `is_natively_supported()` ⟹ `!is_codec_supported()`.

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
    /// 1.2.840.10008.1.2.4.50 - JPEG Baseline (Process 1), lossy 8-bit.
    JpegBaseline,
    /// 1.2.840.10008.1.2.4.51 - JPEG Extended (Process 2 & 4), lossy 12-bit.
    JpegExtended,
    /// 1.2.840.10008.1.2.4.57 - JPEG Lossless, Non-Hierarchical (Process 14).
    JpegLosslessNonHierarchical,
    /// 1.2.840.10008.1.2.4.70 - JPEG Lossless First Order Prediction (Process 14 SV1).
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
    /// 1.2.840.10008.1.2.4.110 - JPEG XL Lossless (ISO 18181-1, lossless).
    JpegXlLossless,
    /// 1.2.840.10008.1.2.4.111 - JPEG XL JPEG Recompression.
    JpegXlJpegRecompression,
    /// 1.2.840.10008.1.2.4.112 - JPEG XL (lossy or lossless).
    JpegXl,
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
            "1.2.840.10008.1.2.4.51" => Self::JpegExtended,
            "1.2.840.10008.1.2.4.57" => Self::JpegLosslessNonHierarchical,
            "1.2.840.10008.1.2.4.70" => Self::JpegLosslessFirstOrderPrediction,
            "1.2.840.10008.1.2.4.80" => Self::JpegLsLossless,
            "1.2.840.10008.1.2.4.81" => Self::JpegLsLossy,
            "1.2.840.10008.1.2.4.90" => Self::Jpeg2000Lossless,
            "1.2.840.10008.1.2.4.91" => Self::Jpeg2000Lossy,
            "1.2.840.10008.1.2.5" => Self::RleLossless,
            "1.2.840.10008.1.2.1.99" => Self::DeflatedExplicitVrLittleEndian,
            "1.2.840.10008.1.2.4.110" => Self::JpegXlLossless,
            "1.2.840.10008.1.2.4.111" => Self::JpegXlJpegRecompression,
            "1.2.840.10008.1.2.4.112" => Self::JpegXl,
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
            Self::JpegExtended => "1.2.840.10008.1.2.4.51",
            Self::JpegLosslessNonHierarchical => "1.2.840.10008.1.2.4.57",
            Self::JpegLosslessFirstOrderPrediction => "1.2.840.10008.1.2.4.70",
            Self::JpegLsLossless => "1.2.840.10008.1.2.4.80",
            Self::JpegLsLossy => "1.2.840.10008.1.2.4.81",
            Self::Jpeg2000Lossless => "1.2.840.10008.1.2.4.90",
            Self::Jpeg2000Lossy => "1.2.840.10008.1.2.4.91",
            Self::RleLossless => "1.2.840.10008.1.2.5",
            Self::DeflatedExplicitVrLittleEndian => "1.2.840.10008.1.2.1.99",
            Self::JpegXlLossless => "1.2.840.10008.1.2.4.110",
            Self::JpegXlJpegRecompression => "1.2.840.10008.1.2.4.111",
            Self::JpegXl => "1.2.840.10008.1.2.4.112",
            Self::Unknown(uid) => uid.as_str(),
        }
    }

    /// True when pixel data is encapsulated in fragments requiring a codec
    /// (DICOM PS3.5 Table A-1 column "Pixel Data encapsulated").
    ///
    /// Note: `DeflatedExplicitVrLittleEndian` compresses the *dataset* byte-stream,
    /// not pixel data fragments; it is therefore NOT included here.
    pub fn is_compressed(&self) -> bool {
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

    /// True when no information is destroyed by the compression scheme.
    pub fn is_lossless(&self) -> bool {
        matches!(
            self,
            Self::ImplicitVrLittleEndian
                | Self::ExplicitVrLittleEndian
                | Self::ExplicitVrBigEndian
                | Self::JpegLosslessNonHierarchical
                | Self::JpegLosslessFirstOrderPrediction
                | Self::JpegLsLossless
                | Self::Jpeg2000Lossless
                | Self::RleLossless
                | Self::DeflatedExplicitVrLittleEndian
                | Self::JpegXlLossless
        )
    }

    /// True when ritk-io can read/write this syntax without an external codec.
    ///
    /// ## Invariant
    /// `is_natively_supported()` ⟹ `!is_compressed()` ∧ `!is_big_endian()`.
    /// `ExplicitVrBigEndian` is excluded: `decode_pixel_bytes` uses LE byte order.
    /// `DeflatedExplicitVrLittleEndian` is excluded: deflate is not handled by
    /// ritk-io's pixel decode path (requires `deflate` feature of
    /// `dicom-transfer-syntax-registry`).
    pub fn is_natively_supported(&self) -> bool {
        matches!(
            self,
            Self::ImplicitVrLittleEndian | Self::ExplicitVrLittleEndian
        )
    }

    /// True when pixel data bytes in this syntax are stored in big-endian byte order.
    ///
    /// Only `ExplicitVrBigEndian` (retired, DICOM PS 3.5 withdrawn 2004) stores
    /// pixels in big-endian order. ritk-io's `decode_pixel_bytes` uses
    /// little-endian decode; calling it on big-endian pixel bytes produces
    /// silently incorrect intensities.
    pub fn is_big_endian(&self) -> bool {
        matches!(self, Self::ExplicitVrBigEndian)
    }

    /// True when pixel data for this syntax can be decoded using the codec registered
    /// via `dicom-pixeldata` / `dicom-transfer-syntax-registry`.
    ///
    /// ## Covered codecs
    ///
    /// | Codec | Feature | Transfer Syntaxes |
    /// |---|---|---|
    /// | `jpeg-decoder` | `jpeg` (enabled via `native`) | Baseline (`.50`), Extended (`.51`), Lossless NH (`.57`), Lossless FOP (`.70`) |
    /// | `dicom-rle` | `rle` (enabled via `native`) | RLE Lossless (`.5`) |
    /// | `jxl-oxide` + `zune-jpegxl` | `jpegxl` | JPEG XL Lossless (`.110`), JPEG XL Recompression (`.111`), JPEG XL (`.112`) |
    ///
    /// ## Not yet supported (require native library features)
    /// - JPEG-LS Lossless/Near-Lossless: enable `charls` feature.
    /// - JPEG 2000 Lossless/Lossy: enable `openjp2` or `openjpeg-sys` feature.
    ///
    /// ## Invariants
    /// - `is_codec_supported()` ⟹ `is_compressed()` — codec path is for encapsulated TS only.
    /// - `is_natively_supported()` ⟹ `!is_codec_supported()` — the two decode paths are disjoint.
    pub fn is_codec_supported(&self) -> bool {
        matches!(
            self,
            Self::JpegBaseline
                | Self::JpegExtended
                | Self::JpegLosslessNonHierarchical
                | Self::JpegLosslessFirstOrderPrediction
                | Self::RleLossless
                | Self::JpegXlLossless
                | Self::JpegXlJpegRecompression
                | Self::JpegXl
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

    /// All 16 known (non-Unknown) variants for exhaustive property tests.
    fn all_known_variants() -> [TransferSyntaxKind; 16] {
        [
            TransferSyntaxKind::ImplicitVrLittleEndian,
            TransferSyntaxKind::ExplicitVrLittleEndian,
            TransferSyntaxKind::ExplicitVrBigEndian,
            TransferSyntaxKind::JpegBaseline,
            TransferSyntaxKind::JpegExtended,
            TransferSyntaxKind::JpegLosslessNonHierarchical,
            TransferSyntaxKind::JpegLosslessFirstOrderPrediction,
            TransferSyntaxKind::JpegLsLossless,
            TransferSyntaxKind::JpegLsLossy,
            TransferSyntaxKind::Jpeg2000Lossless,
            TransferSyntaxKind::Jpeg2000Lossy,
            TransferSyntaxKind::RleLossless,
            TransferSyntaxKind::DeflatedExplicitVrLittleEndian,
            TransferSyntaxKind::JpegXlLossless,
            TransferSyntaxKind::JpegXlJpegRecompression,
            TransferSyntaxKind::JpegXl,
        ]
    }

    fn make_slice_meta(uid: Option<&str>) -> DicomSliceMetadata {
        DicomSliceMetadata {
            transfer_syntax_uid: uid.map(|s| s.to_string()),
            ..DicomSliceMetadata::default()
        }
    }

    // --- UID round-trip ---

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
    fn test_from_uid_jpeg_extended() {
        assert_eq!(
            TransferSyntaxKind::from_uid("1.2.840.10008.1.2.4.51"),
            TransferSyntaxKind::JpegExtended
        );
    }

    #[test]
    fn test_from_uid_jpeg_lossless_non_hierarchical() {
        assert_eq!(
            TransferSyntaxKind::from_uid("1.2.840.10008.1.2.4.57"),
            TransferSyntaxKind::JpegLosslessNonHierarchical
        );
    }

    #[test]
    fn test_from_uid_jpeg_xl_lossless() {
        assert_eq!(
            TransferSyntaxKind::from_uid("1.2.840.10008.1.2.4.110"),
            TransferSyntaxKind::JpegXlLossless
        );
    }

    #[test]
    fn test_from_uid_jpeg_xl_recompression() {
        assert_eq!(
            TransferSyntaxKind::from_uid("1.2.840.10008.1.2.4.111"),
            TransferSyntaxKind::JpegXlJpegRecompression
        );
    }

    #[test]
    fn test_from_uid_jpeg_xl() {
        assert_eq!(
            TransferSyntaxKind::from_uid("1.2.840.10008.1.2.4.112"),
            TransferSyntaxKind::JpegXl
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
    fn test_uid_roundtrip_all_known() {
        for v in &all_known_variants() {
            assert_eq!(
                TransferSyntaxKind::from_uid(v.uid()),
                *v,
                "Roundtrip failed for {v:?}"
            );
        }
    }

    // --- is_compressed ---

    #[test]
    fn test_is_compressed_jpeg_baseline() {
        assert!(TransferSyntaxKind::JpegBaseline.is_compressed());
    }

    #[test]
    fn test_is_compressed_jpeg_extended_true() {
        assert!(
            TransferSyntaxKind::JpegExtended.is_compressed(),
            "JpegExtended must be compressed (encapsulated pixel data)"
        );
    }

    #[test]
    fn test_is_compressed_jpeg_lossless_nh_true() {
        assert!(
            TransferSyntaxKind::JpegLosslessNonHierarchical.is_compressed(),
            "JpegLosslessNonHierarchical must be compressed (encapsulated pixel data)"
        );
    }

    #[test]
    fn test_is_compressed_jpeg_xl_lossless_true() {
        assert!(
            TransferSyntaxKind::JpegXlLossless.is_compressed(),
            "JpegXlLossless must be compressed (encapsulated pixel data)"
        );
    }

    #[test]
    fn test_is_compressed_jpeg_xl_recompression_true() {
        assert!(TransferSyntaxKind::JpegXlJpegRecompression.is_compressed());
    }

    #[test]
    fn test_is_compressed_jpeg_xl_true() {
        assert!(TransferSyntaxKind::JpegXl.is_compressed());
    }

    #[test]
    fn test_is_compressed_explicit_vr_le_false() {
        assert!(!TransferSyntaxKind::ExplicitVrLittleEndian.is_compressed());
    }

    #[test]
    fn test_is_compressed_deflated_false() {
        // Deflated Explicit VR LE compresses the dataset stream, not pixel data fragments.
        // is_compressed() is defined as pixel-data encapsulation only.
        assert!(
            !TransferSyntaxKind::DeflatedExplicitVrLittleEndian.is_compressed(),
            "DeflatedExplicitVrLittleEndian compresses the dataset, not pixel data fragments; \
             is_compressed() must return false"
        );
    }

    // --- is_lossless ---

    #[test]
    fn test_is_lossless_jpeg_baseline_false() {
        assert!(!TransferSyntaxKind::JpegBaseline.is_lossless());
    }

    #[test]
    fn test_is_lossless_jpeg_extended_false() {
        assert!(
            !TransferSyntaxKind::JpegExtended.is_lossless(),
            "JpegExtended (Process 2 & 4) is lossy"
        );
    }

    #[test]
    fn test_is_lossless_jpeg_lossless_nh_true() {
        assert!(
            TransferSyntaxKind::JpegLosslessNonHierarchical.is_lossless(),
            "JpegLosslessNonHierarchical (Process 14) is lossless"
        );
    }

    #[test]
    fn test_is_lossless_jpeg_xl_lossless_true() {
        assert!(
            TransferSyntaxKind::JpegXlLossless.is_lossless(),
            "JpegXlLossless is lossless by definition"
        );
    }

    #[test]
    fn test_is_lossless_jpeg_xl_false() {
        // JpegXl and JpegXlJpegRecompression can be lossy or lossless depending on
        // encoder settings; the TS itself does not guarantee losslessness.
        assert!(
            !TransferSyntaxKind::JpegXl.is_lossless(),
            "JpegXl (generic) is not guaranteed lossless"
        );
        assert!(
            !TransferSyntaxKind::JpegXlJpegRecompression.is_lossless(),
            "JpegXlJpegRecompression is not guaranteed lossless"
        );
    }

    // --- is_natively_supported ---

    #[test]
    fn test_is_natively_supported_deflated_false() {
        assert!(
            !TransferSyntaxKind::DeflatedExplicitVrLittleEndian.is_natively_supported(),
            "DeflatedExplicitVrLittleEndian must not be natively supported \
             since ritk-io's pixel decode path does not handle deflate"
        );
    }

    #[test]
    fn test_big_endian_is_not_natively_supported() {
        assert!(
            !TransferSyntaxKind::ExplicitVrBigEndian.is_natively_supported(),
            "ExplicitVrBigEndian must not be natively supported: decode_pixel_bytes uses LE"
        );
    }

    #[test]
    fn test_implicit_vr_le_is_natively_supported() {
        assert!(TransferSyntaxKind::ImplicitVrLittleEndian.is_natively_supported());
    }

    #[test]
    fn test_explicit_vr_le_is_natively_supported() {
        assert!(TransferSyntaxKind::ExplicitVrLittleEndian.is_natively_supported());
    }

    // --- is_big_endian ---

    #[test]
    fn test_big_endian_is_big_endian_true() {
        assert!(TransferSyntaxKind::ExplicitVrBigEndian.is_big_endian());
    }

    #[test]
    fn test_explicit_vr_le_is_not_big_endian() {
        assert!(!TransferSyntaxKind::ExplicitVrLittleEndian.is_big_endian());
    }

    // --- is_codec_supported ---

    #[test]
    fn test_is_codec_supported_jpeg_baseline_true() {
        assert!(TransferSyntaxKind::JpegBaseline.is_codec_supported());
    }

    #[test]
    fn test_is_codec_supported_jpeg_extended_true() {
        assert!(
            TransferSyntaxKind::JpegExtended.is_codec_supported(),
            "JpegExtended must be codec-supported (pure-Rust jpeg-decoder via `jpeg` feature)"
        );
    }

    #[test]
    fn test_is_codec_supported_jpeg_lossless_nh_true() {
        assert!(
            TransferSyntaxKind::JpegLosslessNonHierarchical.is_codec_supported(),
            "JpegLosslessNonHierarchical must be codec-supported (pure-Rust jpeg-decoder via `jpeg` feature)"
        );
    }

    #[test]
    fn test_is_codec_supported_jpeg_lossless_fop_true() {
        assert!(TransferSyntaxKind::JpegLosslessFirstOrderPrediction.is_codec_supported());
    }

    #[test]
    fn test_is_codec_supported_rle_lossless_true() {
        assert!(TransferSyntaxKind::RleLossless.is_codec_supported());
    }

    #[test]
    fn test_is_codec_supported_jpeg_xl_lossless_true() {
        assert!(
            TransferSyntaxKind::JpegXlLossless.is_codec_supported(),
            "JpegXlLossless must be codec-supported (jxl-oxide decoder + jpegxl feature)"
        );
    }

    #[test]
    fn test_is_codec_supported_jpeg_xl_recompression_true() {
        assert!(
            TransferSyntaxKind::JpegXlJpegRecompression.is_codec_supported(),
            "JpegXlJpegRecompression must be codec-supported (jxl-oxide decoder)"
        );
    }

    #[test]
    fn test_is_codec_supported_jpeg_xl_true() {
        assert!(
            TransferSyntaxKind::JpegXl.is_codec_supported(),
            "JpegXl must be codec-supported (jxl-oxide decoder)"
        );
    }

    #[test]
    fn test_is_codec_supported_jpeg_ls_false() {
        assert!(
            !TransferSyntaxKind::JpegLsLossless.is_codec_supported(),
            "JpegLsLossless must NOT be codec-supported without charls feature"
        );
        assert!(
            !TransferSyntaxKind::JpegLsLossy.is_codec_supported(),
            "JpegLsLossy must NOT be codec-supported without charls feature"
        );
    }

    #[test]
    fn test_is_codec_supported_jpeg2000_false() {
        assert!(
            !TransferSyntaxKind::Jpeg2000Lossless.is_codec_supported(),
            "Jpeg2000Lossless must NOT be codec-supported without openjp2 feature"
        );
        assert!(
            !TransferSyntaxKind::Jpeg2000Lossy.is_codec_supported(),
            "Jpeg2000Lossy must NOT be codec-supported without openjp2 feature"
        );
    }

    #[test]
    fn test_is_codec_supported_deflated_false() {
        assert!(
            !TransferSyntaxKind::DeflatedExplicitVrLittleEndian.is_codec_supported(),
            "DeflatedExplicitVrLittleEndian is not pixel-codec supported; \
             dataset deflation is handled at the open_file level"
        );
    }

    // --- Formal invariants ---

    #[test]
    fn test_natively_supported_implies_not_compressed_and_not_big_endian() {
        // Formal invariant: is_natively_supported() => !is_compressed() && !is_big_endian()
        for v in &all_known_variants() {
            if v.is_natively_supported() {
                assert!(
                    !v.is_compressed(),
                    "{v:?}: is_natively_supported() but is_compressed()"
                );
                assert!(
                    !v.is_big_endian(),
                    "{v:?}: is_natively_supported() but is_big_endian()"
                );
            }
        }
    }

    #[test]
    fn test_codec_supported_implies_compressed() {
        // Formal invariant: is_codec_supported() => is_compressed()
        for v in &all_known_variants() {
            if v.is_codec_supported() {
                assert!(
                    v.is_compressed(),
                    "{v:?}: is_codec_supported() but !is_compressed() — invariant violated"
                );
            }
        }
    }

    #[test]
    fn test_natively_supported_and_codec_supported_are_disjoint() {
        // Formal invariant: is_natively_supported() => !is_codec_supported()
        for v in &all_known_variants() {
            if v.is_natively_supported() {
                assert!(
                    !v.is_codec_supported(),
                    "{v:?}: is_natively_supported() AND is_codec_supported() — paths must be disjoint"
                );
            }
        }
    }

    // --- from_metadata ---

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
}
