//! DIMSE message encoding/decoding per DICOM PS 3.7.
//!
//! DIMSE messages are service primitives carried inside P-DATA-TF PDUs.
//! Each DIMSE message has a Command Set (group 0x0000 elements, Explicit VR LE)
//! followed by an optional Data Set.

mod factory;

use anyhow::{bail, Context, Result};

// ── DIMSE Command Field Values ────────────────────────────────────────────────

/// DIMSE command field (Tag 0000,0100). PS3.7 Table E.1-1.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u16)]
pub enum CommandField {
    CEchoRq = 0x0030,
    CEchoRsp = 0x8030,
    CFindRq = 0x0020,
    CFindRsp = 0x8020,
    CStoreRq = 0x0001,
    CStoreRsp = 0x8001,
    CMoveRq = 0x0021,
    CMoveRsp = 0x8021,
    CGetRq = 0x0010,
    CGetRsp = 0x8010,
}

impl CommandField {
    pub(crate) fn from_u16(v: u16) -> Option<Self> {
        match v {
            0x0030 => Some(Self::CEchoRq),
            0x8030 => Some(Self::CEchoRsp),
            0x0020 => Some(Self::CFindRq),
            0x8020 => Some(Self::CFindRsp),
            0x0001 => Some(Self::CStoreRq),
            0x8001 => Some(Self::CStoreRsp),
            0x0021 => Some(Self::CMoveRq),
            0x8021 => Some(Self::CMoveRsp),
            0x0010 => Some(Self::CGetRq),
            0x8010 => Some(Self::CGetRsp),
            _ => None,
        }
    }
}

// ── Status codes ──────────────────────────────────────────────────────────────

/// DIMSE response status codes (Tag 0000,0900). PS3.7 Annex C.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u16)]
pub enum DimseStatus {
    Success = 0x0000,
    Warning = 0x0001,
    CoercedInvalidValue = 0xB000,
    StorageCommitmentDuplicateInstance = 0x0111,
    Pending = 0xFF00,
    PendingWarning = 0xFF01,
    MoveDestinationUnknown = 0xA801,
    RefusedSopClassNotSupported = 0x0122,
    RefusedNotAuthorized = 0x0124,
    FailedUnableToProcess = 0xC000,
    FailedUnableToCalculateMatches = 0xC100,
    FailedUnableToPerformSuboperations = 0xC200,
    FailedSuboperationsTerminatedByFailure = 0xC300,
    Cancel = 0xFE00,
}

// ── Command VR types ──────────────────────────────────────────────────────────

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

// ── Command element ───────────────────────────────────────────────────────────

/// A single command data element.
#[derive(Debug, Clone, PartialEq)]
pub struct CommandElement {
    pub tag: (u16, u16),
    pub vr: CommandVr,
    pub value: Vec<u8>,
}

// ── DIMSE message ─────────────────────────────────────────────────────────────

/// DIMSE message — command set plus optional data set.
#[derive(Debug, Clone, PartialEq)]
pub struct DimseMessage {
    /// Command set elements (group 0x0000).
    pub command_set: Vec<CommandElement>,
    /// Optional data set bytes (no meta-information header).
    pub data_set: Option<Vec<u8>>,
}

// ── Value encoding helpers ────────────────────────────────────────────────────

pub(crate) fn encode_us(v: u16) -> Vec<u8> {
    v.to_le_bytes().to_vec()
}

pub(crate) fn encode_ui(uid: &str) -> Vec<u8> {
    let mut b = uid.as_bytes().to_vec();
    if b.len() % 2 != 0 {
        b.push(0x00);
    }
    b
}

pub(crate) fn encode_ae(s: &str) -> Vec<u8> {
    encode_str_pad(s)
}

pub(crate) fn encode_str_pad(s: &str) -> Vec<u8> {
    let mut b = s.as_bytes().to_vec();
    if b.len() % 2 != 0 {
        b.push(b' ');
    }
    b
}

pub(crate) fn decode_us(bytes: &[u8]) -> Option<u16> {
    (bytes.len() >= 2).then(|| u16::from_le_bytes([bytes[0], bytes[1]]))
}

pub(crate) fn decode_ui(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes)
        .trim_end_matches(|c| c == '\0' || c == ' ')
        .to_owned()
}

pub(crate) fn decode_ae(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).trim_end().to_owned()
}

// ── Command tag constants ─────────────────────────────────────────────────────

pub(crate) const TAG_CMD_GROUP_LENGTH: (u16, u16) = (0x0000, 0x0000);
pub(crate) const TAG_AFFECTED_SOP_CLASS: (u16, u16) = (0x0000, 0x0002);
pub(crate) const TAG_COMMAND_FIELD: (u16, u16) = (0x0000, 0x0100);
pub(crate) const TAG_MESSAGE_ID: (u16, u16) = (0x0000, 0x0110);
pub(crate) const TAG_MESSAGE_ID_RESP: (u16, u16) = (0x0000, 0x0120);
pub(crate) const TAG_CMD_DATA_SET_TYPE: (u16, u16) = (0x0000, 0x0800);
pub(crate) const TAG_STATUS: (u16, u16) = (0x0000, 0x0900);
pub(crate) const TAG_PRIORITY: (u16, u16) = (0x0000, 0x0700);
pub(crate) const TAG_MOVE_DESTINATION: (u16, u16) = (0x0000, 0x0600);
pub(crate) const TAG_AFFECTED_SOP_INSTANCE: (u16, u16) = (0x0000, 0x1000);
pub(crate) const TAG_NUM_SUBOPS: (u16, u16) = (0x0000, 0x1001);

pub(crate) const NO_DATASET: u16 = 0x0101;
pub(crate) const HAS_DATASET: u16 = 0x0001;

// ── Encoding / Decoding ───────────────────────────────────────────────────────

impl DimseMessage {
    /// Write the Explicit VR LE encoding of a single command element into `buf`.
    pub(crate) fn encode_element_into(
        buf: &mut Vec<u8>,
        tag: (u16, u16),
        vr: CommandVr,
        value: &[u8],
    ) {
        buf.extend_from_slice(&tag.0.to_le_bytes());
        buf.extend_from_slice(&tag.1.to_le_bytes());
        buf.extend_from_slice(&vr.code());
        if vr.is_short() {
            buf.extend_from_slice(&(value.len() as u16).to_le_bytes());
        } else {
            buf.extend_from_slice(&0u16.to_le_bytes()); // reserved
            buf.extend_from_slice(&(value.len() as u32).to_le_bytes());
        }
        buf.extend_from_slice(value);
    }

    /// Encode the command set as DICOM Explicit VR Little Endian bytes.
    /// (0000,0000) CommandGroupLength is computed and prepended.
    ///
    /// # Allocation profile
    ///
    /// All encoded element bytes are written directly into a single pre-grown
    /// `Vec<u8>`, avoiding per-element intermediate `Vec` allocations and the
    /// associated resize/copy overhead.
    pub fn encode_command_set(&self) -> Vec<u8> {
        // Pre-compute total body size for a single allocation.
        let body_size: usize = self
            .command_set
            .iter()
            .filter(|e| e.tag != TAG_CMD_GROUP_LENGTH)
            .map(|e| {
                let overhead = if e.vr.is_short() { 8 } else { 12 };
                overhead + e.value.len()
            })
            .sum();

        let mut body = Vec::with_capacity(body_size);
        for elem in &self.command_set {
            if elem.tag == TAG_CMD_GROUP_LENGTH {
                continue;
            }
            Self::encode_element_into(&mut body, elem.tag, elem.vr, &elem.value);
        }

        // Encode the group-length element (always UL — long-form: 12 bytes overhead)
        let mut full = Vec::with_capacity(12 + 4 + body_size);
        Self::encode_element_into(
            &mut full,
            TAG_CMD_GROUP_LENGTH,
            CommandVr::Ul,
            &(body.len() as u32).to_le_bytes(),
        );
        full.extend_from_slice(&body);
        full
    }

    /// Decode a command set from Explicit VR Little Endian bytes.
    /// Returns `DimseMessage` with `data_set` = None (caller attaches data set).
    pub fn decode_command_set(data: &[u8]) -> Result<Self> {
        let mut elements = Vec::new();
        let mut cursor = 0usize;

        while cursor + 6 <= data.len() {
            let group = u16::from_le_bytes([data[cursor], data[cursor + 1]]);
            let element = u16::from_le_bytes([data[cursor + 2], data[cursor + 3]]);
            let vr_code = [data[cursor + 4], data[cursor + 5]];
            let vr = CommandVr::from_code(vr_code)
                .with_context(|| format!("unknown VR {:?} at offset {}", vr_code, cursor))?;
            cursor += 6;

            let value_len = if vr.is_short() {
                if cursor + 2 > data.len() {
                    break;
                }
                u16::from_le_bytes([data[cursor], data[cursor + 1]]) as usize
            } else {
                if cursor + 6 > data.len() {
                    break;
                }
                cursor += 2; // skip reserved
                let len = u32::from_le_bytes([
                    data[cursor],
                    data[cursor + 1],
                    data[cursor + 2],
                    data[cursor + 3],
                ]) as usize;
                cursor += 4;
                len
            };

            if vr.is_short() {
                cursor += 2;
            }

            if cursor + value_len > data.len() {
                bail!(
                    "value length {} at offset {} exceeds buffer ({})",
                    value_len,
                    cursor,
                    data.len()
                );
            }

            elements.push(CommandElement {
                tag: (group, element),
                vr,
                value: data[cursor..cursor + value_len].to_vec(),
            });
            cursor += value_len;
        }

        Ok(Self {
            command_set: elements,
            data_set: None,
        })
    }

    /// CommandField from (0000,0100).
    pub fn command_field(&self) -> Option<CommandField> {
        self.find_element(TAG_COMMAND_FIELD)
            .and_then(|e| decode_us(&e.value))
            .and_then(CommandField::from_u16)
    }

    /// Message ID from (0000,0110).
    pub fn message_id(&self) -> Option<u16> {
        self.find_element(TAG_MESSAGE_ID)
            .and_then(|e| decode_us(&e.value))
    }

    /// Status from (0000,0900).
    pub fn status(&self) -> Option<u16> {
        self.find_element(TAG_STATUS)
            .and_then(|e| decode_us(&e.value))
    }

    /// Affected SOP Class UID from (0000,0002).
    pub fn affected_sop_class_uid(&self) -> Option<String> {
        self.find_element(TAG_AFFECTED_SOP_CLASS)
            .map(|e| decode_ui(&e.value))
    }

    /// Move Destination AE title from (0000,0600).
    pub fn move_destination(&self) -> Option<String> {
        self.find_element(TAG_MOVE_DESTINATION)
            .map(|e| decode_ae(&e.value))
    }

    /// Affected SOP Instance UID from (0000,1000).
    pub fn affected_sop_instance_uid(&self) -> Option<String> {
        self.find_element(TAG_AFFECTED_SOP_INSTANCE)
            .map(|e| decode_ui(&e.value))
    }

    /// Command Data Set Type from (0000,0800).
    pub fn command_data_set_type(&self) -> Option<u16> {
        self.find_element(TAG_CMD_DATA_SET_TYPE)
            .and_then(|e| decode_us(&e.value))
    }

    pub fn find_element(&self, tag: (u16, u16)) -> Option<&CommandElement> {
        self.command_set.iter().find(|e| e.tag == tag)
    }
}

// ── SOP Class UIDs ────────────────────────────────────────────────────────────

pub mod sop_class {
    pub const VERIFICATION: &str = "1.2.840.10008.1.1";
    pub const FIND_STUDY: &str = "1.2.840.10008.5.1.4.1.2.1.1";
    pub const FIND_PATIENT: &str = "1.2.840.10008.5.1.4.1.2.1.3";
    pub const FIND_SERIES: &str = "1.2.840.10008.5.1.4.1.2.2.1";
    pub const FIND_INSTANCE: &str = "1.2.840.10008.5.1.4.1.2.3.1";
    pub const MOVE_STUDY: &str = "1.2.840.10008.5.1.4.1.2.1.2";
    pub const MOVE_PATIENT: &str = "1.2.840.10008.5.1.4.1.2.1.4";
    pub const MOVE_SERIES: &str = "1.2.840.10008.5.1.4.1.2.2.2";
    pub const GET_STUDY: &str = "1.2.840.10008.5.1.4.1.2.1.3";
    pub const STORAGE_COMMITMENT: &str = "1.2.840.10008.1.20.1";
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
