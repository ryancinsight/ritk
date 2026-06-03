//! DIMSE command element types — VR, value representation, and element structure.

/// VR types used in command sets (Explicit VR LE, PS3.5 Table 6.2-1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommandVr {
    Us,
    Ul,
    Ui,
    Ae,
    As,
    Cs,
    Da,
    Ds,
    Dt,
    Is,
    Lo,
    Lt,
    Pn,
    Sh,
    St,
    Tm,
    Un,
}

impl CommandVr {
    pub fn code(self) -> [u8; 2] {
        match self {
            Self::Us => *b"US",
            Self::Ul => *b"UL",
            Self::Ui => *b"UI",
            Self::Ae => *b"AE",
            Self::As => *b"AS",
            Self::Cs => *b"CS",
            Self::Da => *b"DA",
            Self::Ds => *b"DS",
            Self::Dt => *b"DT",
            Self::Is => *b"IS",
            Self::Lo => *b"LO",
            Self::Lt => *b"LT",
            Self::Pn => *b"PN",
            Self::Sh => *b"SH",
            Self::St => *b"ST",
            Self::Tm => *b"TM",
            Self::Un => *b"UN",
        }
    }

    /// Short-form VRs: 2-byte length. Long-form: 2 reserved + 4-byte length.
    pub fn is_short(self) -> bool {
        matches!(
            self,
            Self::Us
                | Self::Ae
                | Self::As
                | Self::Cs
                | Self::Da
                | Self::Ds
                | Self::Dt
                | Self::Is
                | Self::Lo
                | Self::Pn
                | Self::Sh
                | Self::Tm
                | Self::Ui
        )
    }

    pub fn from_code(code: [u8; 2]) -> Option<Self> {
        match &code {
            b"US" => Some(Self::Us),
            b"UL" => Some(Self::Ul),
            b"UI" => Some(Self::Ui),
            b"AE" => Some(Self::Ae),
            b"AS" => Some(Self::As),
            b"CS" => Some(Self::Cs),
            b"DA" => Some(Self::Da),
            b"DS" => Some(Self::Ds),
            b"DT" => Some(Self::Dt),
            b"IS" => Some(Self::Is),
            b"LO" => Some(Self::Lo),
            b"LT" => Some(Self::Lt),
            b"PN" => Some(Self::Pn),
            b"SH" => Some(Self::Sh),
            b"ST" => Some(Self::St),
            b"TM" => Some(Self::Tm),
            b"UN" => Some(Self::Un),
            _ => None,
        }
    }

    /// Padding byte for even-length conformity. UI → 0x00; others → space.
    pub fn pad_byte(self) -> u8 {
        if self == Self::Ui {
            0x00
        } else {
            b' '
        }
    }
}

/// Value of a command element — small fixed-size values are stored inline
/// without heap allocation (US=2 bytes, UL=4 bytes). Variable-length values
/// (UI, AE, CS, etc.) use heap storage.
#[derive(Debug, Clone)]
pub enum CommandValue {
    /// Inline storage for small fixed-size values.
    /// Field 0 = bytes (only first `field 1` are valid), field 1 = actual length.
    Inline([u8; 4], u8),
    /// Variable-length values on the heap.
    Heap(Vec<u8>),
}

impl CommandValue {
    /// Create an inline US (unsigned short) value — zero allocation.
    pub fn us(v: u16) -> Self {
        let bytes = v.to_le_bytes();
        Self::Inline([bytes[0], bytes[1], 0, 0], 2)
    }

    /// Create a UI (Unique Identifier) value: null-padded to even length.
    pub fn ui(uid: &str) -> Self {
        let len = uid.len();
        let mut b = Vec::with_capacity(len + (len & 1));
        b.extend_from_slice(uid.as_bytes());
        if !len.is_multiple_of(2) {
            b.push(0x00);
        }
        Self::Heap(b)
    }

    /// Create an AE (Application Entity) value: space-padded to even length.
    pub fn ae(s: &str) -> Self {
        let len = s.len();
        let mut b = Vec::with_capacity(len + (len & 1));
        b.extend_from_slice(s.as_bytes());
        if !len.is_multiple_of(2) {
            b.push(b' ');
        }
        Self::Heap(b)
    }

    /// View the value as a byte slice.
    pub fn as_bytes(&self) -> &[u8] {
        match self {
            Self::Inline(bytes, len) => &bytes[..*len as usize],
            Self::Heap(v) => v.as_slice(),
        }
    }

    /// Length of the value in bytes.
    pub fn len(&self) -> usize {
        match self {
            Self::Inline(_, len) => *len as usize,
            Self::Heap(v) => v.len(),
        }
    }

    /// Returns `true` if the value is empty (zero bytes).
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl PartialEq for CommandValue {
    fn eq(&self, other: &Self) -> bool {
        self.as_bytes() == other.as_bytes()
    }
}

/// A single command data element.
#[derive(Debug, Clone, PartialEq)]
pub struct CommandElement {
    pub tag: (u16, u16),
    pub vr: CommandVr,
    pub value: CommandValue,
}
