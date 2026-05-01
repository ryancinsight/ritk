//! Compatibility re-export for DICOM Transfer Syntax classification.
//!
//! `ritk-dicom` owns the canonical transfer-syntax model. This module preserves
//! the historic `ritk_io::format::dicom::transfer_syntax::TransferSyntaxKind`
//! path for downstream callers while preventing a second enum implementation
//! from drifting out of sync.

pub use ritk_dicom::TransferSyntaxKind;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compatibility_reexport_preserves_uid_roundtrip() {
        let syntax = TransferSyntaxKind::from_uid("1.2.840.10008.1.2.4.50");
        assert_eq!(syntax.uid(), "1.2.840.10008.1.2.4.50");
        assert!(syntax.is_compressed());
        assert!(syntax.is_codec_supported());
    }

    #[test]
    fn compatibility_reexport_preserves_big_endian_rejection_predicate() {
        let syntax = TransferSyntaxKind::from_uid("1.2.840.10008.1.2.2");
        assert!(syntax.is_big_endian());
        assert!(!syntax.is_natively_supported());
    }
}
