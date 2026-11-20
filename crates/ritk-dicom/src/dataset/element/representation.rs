//! PS3.5 value representations and their wire widths.

use std::fmt;
use std::str::FromStr;

use anyhow::{anyhow, Result};

macro_rules! value_representations {
    ($($vr:ident),+ $(,)?) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub enum VR { $($vr),+ }

        impl VR {
            pub const fn as_str(self) -> &'static str {
                match self { $(Self::$vr => stringify!($vr)),+ }
            }
        }

        impl FromStr for VR {
            type Err = anyhow::Error;

            fn from_str(s: &str) -> Result<Self> {
                match s {
                    $(stringify!($vr) => Ok(Self::$vr)),+,
                    _ => Err(anyhow!("unknown DICOM VR {s:?}")),
                }
            }
        }
    };
}

value_representations!(
    AE, AS, AT, CS, DA, DS, DT, FD, FL, IS, LO, LT, OB, OD, OF, OL, OV, OW, PN, SH, SL, SQ, SS, ST,
    SV, TM, UC, UI, UL, UN, UR, US, UT, UV,
);

impl fmt::Display for VR {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl VR {
    /// Read an explicit representation's two ASCII bytes.
    pub fn from_binary(code: [u8; 2]) -> Option<Self> {
        std::str::from_utf8(&code).ok()?.parse().ok()
    }

    pub(crate) const fn long_length(self) -> bool {
        matches!(
            self,
            Self::OB
                | Self::OD
                | Self::OF
                | Self::OL
                | Self::OV
                | Self::OW
                | Self::SQ
                | Self::SV
                | Self::UC
                | Self::UN
                | Self::UR
                | Self::UT
                | Self::UV
        )
    }

    pub(crate) const fn byte_word(self) -> usize {
        match self {
            Self::OW => 2,
            Self::OL | Self::OF => 4,
            Self::OD | Self::OV => 8,
            _ => 1,
        }
    }

    pub(crate) const fn is_text(self) -> bool {
        matches!(
            self,
            Self::AE
                | Self::AS
                | Self::CS
                | Self::DA
                | Self::DS
                | Self::DT
                | Self::IS
                | Self::LO
                | Self::LT
                | Self::PN
                | Self::SH
                | Self::ST
                | Self::TM
                | Self::UC
                | Self::UI
                | Self::UR
                | Self::UT
        )
    }

    pub(crate) const fn uses_character_set(self) -> bool {
        matches!(
            self,
            Self::LO | Self::LT | Self::PN | Self::SH | Self::ST | Self::UC | Self::UT
        )
    }

    pub(crate) const fn splits_values(self) -> bool {
        self.is_text() && !matches!(self, Self::LT | Self::ST | Self::UR | Self::UT)
    }

    pub(crate) const fn padding(self) -> u8 {
        if self.is_text() && !matches!(self, Self::UI) {
            b' '
        } else {
            0
        }
    }
}
