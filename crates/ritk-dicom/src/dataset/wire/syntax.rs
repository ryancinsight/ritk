use anyhow::{bail, Result};
use consus_core::ByteOrder;

use crate::TransferSyntaxKind;

#[derive(Debug, Clone, Copy)]
pub(crate) struct Encoding {
    pub(crate) implicit_vr: bool,
    pub(crate) byte_order: ByteOrder,
}

impl Encoding {
    pub(crate) const EXPLICIT_LE: Self = Self {
        implicit_vr: false,
        byte_order: ByteOrder::LittleEndian,
    };
    pub(crate) const IMPLICIT_LE: Self = Self {
        implicit_vr: true,
        byte_order: ByteOrder::LittleEndian,
    };

    pub(crate) fn for_syntax(syntax: &TransferSyntaxKind) -> Result<Self> {
        Ok(match syntax {
            TransferSyntaxKind::ImplicitVrLittleEndian => Self::IMPLICIT_LE,
            TransferSyntaxKind::ExplicitVrBigEndian => Self {
                implicit_vr: false,
                byte_order: ByteOrder::BigEndian,
            },
            TransferSyntaxKind::Unknown(_) => {
                bail!("unsupported DICOM transfer syntax {}", syntax.uid())
            }
            _ => Self::EXPLICIT_LE,
        })
    }
}
