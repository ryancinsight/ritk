//! DIMSE command set encoding and decoding (Implicit VR Little Endian).
//!
//! DIMSE command sets are always encoded using Implicit VR Little Endian
//! regardless of the negotiated transfer syntax (PS3.7 §6.3.1).
//!
//! # Format (IVR-LE element):
//! ```text
//! [group:u16-LE][element:u16-LE][length:u32-LE][value:length bytes]
//! ```
//! The first element MUST be (0000,0000) UL = Command Group Length (byte count
//! of all remaining elements in group 0000).

use arrayvec::ArrayString;

use super::association::NetworkingError;

// ── SOP Class UIDs ────────────────────────────────────────────────────────────

/// Verification SOP Class (C-ECHO) — PS3.4 §A.5.
pub const VERIFICATION_SOP_CLASS: &str = "1.2.840.10008.1.1";
/// Study Root Query/Retrieve — FIND — PS3.4 §C.4.1.
pub const STUDY_ROOT_FIND_SOP_CLASS: &str = "1.2.840.10008.5.1.4.1.2.2.1";
/// Study Root Query/Retrieve — MOVE — PS3.4 §C.4.2.
pub const STUDY_ROOT_MOVE_SOP_CLASS: &str = "1.2.840.10008.5.1.4.1.2.2.2";

// ── Transfer Syntax UIDs ──────────────────────────────────────────────────────

/// Implicit VR Little Endian transfer syntax UID.
pub const IMPLICIT_VR_LE_TS: &str = "1.2.840.10008.1.2";
/// Explicit VR Little Endian transfer syntax UID.
pub const EXPLICIT_VR_LE_TS: &str = "1.2.840.10008.1.2.1";

// ── Command Field values (PS3.7 Table E.1-1) ─────────────────────────────────

/// C-STORE-RQ command field.
pub const C_STORE_RQ: u16 = 0x0001;
/// C-STORE-RSP command field.
pub const C_STORE_RSP: u16 = 0x8001;
/// C-FIND-RQ command field.
pub const C_FIND_RQ: u16 = 0x0020;
/// C-FIND-RSP command field.
pub const C_FIND_RSP: u16 = 0x8020;
/// C-MOVE-RQ command field.
pub const C_MOVE_RQ: u16 = 0x0021;
/// C-MOVE-RSP command field.
pub const C_MOVE_RSP: u16 = 0x8021;
/// C-ECHO-RQ command field.
pub const C_ECHO_RQ: u16 = 0x0030;
/// C-ECHO-RSP command field.
pub const C_ECHO_RSP: u16 = 0x8030;

// ── Command Data Set Type values (PS3.7 §9.3.1) ──────────────────────────────

/// No accompanying dataset.
pub const NO_DATASET: u16 = 0x0101;
/// Accompanying dataset present.
pub const HAS_DATASET: u16 = 0xFEFF;

// ── Priority (PS3.7 §9.3.1) ──────────────────────────────────────────────────

/// MEDIUM priority (0x0002). Used for all commands in this implementation.
pub const PRIORITY_MEDIUM: u16 = 0x0002;

// ── DIMSE status codes ────────────────────────────────────────────────────────

/// Success (0x0000).
pub const STATUS_SUCCESS: u16 = 0x0000;
/// Pending — matches may still be outstanding (C-FIND/C-MOVE).
pub const STATUS_PENDING: u16 = 0xFF00;
/// Pending with warning (C-MOVE sub-operation warning).
pub const STATUS_PENDING_WARN: u16 = 0xFF01;

// ── Value encoding helpers ────────────────────────────────────────────────────

/// Write a UID string as IVR-LE UI value into `buf`: null-padded to even length.
pub fn encode_ui_into(buf: &mut Vec<u8>, uid: &str) {
    buf.extend_from_slice(uid.as_bytes());
    if !uid.len().is_multiple_of(2) {
        buf.push(0x00);
    }
}

/// Write a string (AE/CS/LO/SH) as IVR-LE into `buf`: space-padded to even length.
pub fn encode_str_into(buf: &mut Vec<u8>, s: &str) {
    buf.extend_from_slice(s.as_bytes());
    if !s.len().is_multiple_of(2) {
        buf.push(b' ');
    }
}

#[inline]
fn padded_len(s: &str) -> usize {
    let l = s.len();
    l + (l & 1)
}

/// Encode a string (AE/CS/LO/SH) as IVR-LE: space-padded to even length (PS3.5 §6.2).
pub fn encode_str(s: &str) -> Vec<u8> {
    let mut b = Vec::with_capacity(padded_len(s));
    encode_str_into(&mut b, s);
    b
}

/// Typed DIMSE command-element value that `build_command_pdu` encodes directly
/// into the body buffer, eliminating per-element intermediate `Vec` allocations.
#[derive(Debug, Clone)]
pub enum CommandElementValue<'a> {
    Us(u16),
    Ui(&'a str),
    Str(&'a str),
}

// ── Command PDU construction ──────────────────────────────────────────────────

/// Build a complete DIMSE command PDV body (IVR-LE) from a sequence of typed
/// command element values. Values are encoded directly into the body buffer,
/// eliminating per-element intermediate `Vec` allocations.
///
/// # Example
/// ```ignore
/// let cmd = build_command_pdu(&[
///     (0x0000_0002, CommandElementValue::Ui(VERIFICATION_SOP_CLASS)),
///     (0x0000_0100, CommandElementValue::Us(C_ECHO_RQ)),
///     (0x0000_0110, CommandElementValue::Us(1_u16)),
///     (0x0000_0800, CommandElementValue::Us(NO_DATASET)),
/// ]);
/// ```
pub fn build_command_pdu(elements: &[(u32, CommandElementValue)]) -> Vec<u8> {
    let mut buf = Vec::new();
    build_command_pdu_into(&mut buf, elements);
    buf
}

/// Write a DIMSE command PDV body (IVR-LE) into an existing buffer.
///
/// The buffer is extended with the encoded command set:
/// (0000,0000) UL CommandGroupLength followed by the remaining elements.
pub fn build_command_pdu_into(buf: &mut Vec<u8>, elements: &[(u32, CommandElementValue)]) {
    // First pass: compute total body size for a single allocation.
    let body_size: usize = elements
        .iter()
        .map(|(_, v)| match v {
            CommandElementValue::Us(_) => 8 + 2,
            CommandElementValue::Ui(s) => 8 + padded_len(s),
            CommandElementValue::Str(s) => 8 + padded_len(s),
        })
        .sum();
    buf.reserve(12 + body_size);

    let header_start = buf.len();
    buf.extend_from_slice(&0u16.to_le_bytes()); // group 0000
    buf.extend_from_slice(&0u16.to_le_bytes()); // element 0000
    buf.extend_from_slice(&4u32.to_le_bytes()); // UL value length = 4
    buf.extend_from_slice(&0u32.to_le_bytes()); // placeholder for group length

    for (tag, value) in elements {
        let group = (*tag >> 16) as u16;
        let elem = (*tag & 0xFFFF) as u16;
        buf.extend_from_slice(&group.to_le_bytes());
        buf.extend_from_slice(&elem.to_le_bytes());
        match value {
            CommandElementValue::Us(v) => {
                buf.extend_from_slice(&2u32.to_le_bytes());
                buf.extend_from_slice(&v.to_le_bytes());
            }
            CommandElementValue::Ui(uid) => {
                buf.extend_from_slice(&(padded_len(uid) as u32).to_le_bytes());
                encode_ui_into(buf, uid);
            }
            CommandElementValue::Str(s) => {
                buf.extend_from_slice(&(padded_len(s) as u32).to_le_bytes());
                encode_str_into(buf, s);
            }
        }
    }

    // Fill in group length at the reserved placeholder.
    let body_len = buf.len() - header_start - 12;
    let gl = (body_len as u32).to_le_bytes();
    buf[header_start + 8..header_start + 12].copy_from_slice(&gl);
}

/// Encode a non-command DICOM dataset as Implicit VR Little Endian bytes.
///
/// Used for C-FIND query datasets and C-MOVE query datasets.
/// Does NOT prepend a group-length element (deprecated in modern DICOM).
pub fn build_dataset_ivr_le(elements: &[(u32, &[u8])]) -> Vec<u8> {
    let mut buf = Vec::new();
    for &(tag, value) in elements {
        let group = (tag >> 16) as u16;
        let elem = (tag & 0xFFFF) as u16;
        buf.extend_from_slice(&group.to_le_bytes());
        buf.extend_from_slice(&elem.to_le_bytes());
        buf.extend_from_slice(&(value.len() as u32).to_le_bytes());
        buf.extend_from_slice(value);
    }
    buf
}

// ── Command response parsing ──────────────────────────────────────────────────

/// Parsed fields from a received DIMSE command response PDV.
#[derive(Debug, Default)]
pub struct CommandResponse {
    /// (0000,0100) Command Field.
    pub command_field: u16,
    /// (0000,0900) DIMSE status code.
    pub status: u16,
    /// (0000,0120) Message ID Being Responded To.
    pub message_id_responded: u16,
    /// (0000,0800) Command Data Set Type (0x0101 = no dataset).
    pub data_set_type: u16,
    /// (0000,1020) Number of Remaining Sub-operations (C-MOVE).
    pub number_remaining: Option<u16>,
    /// (0000,1021) Number of Completed Sub-operations (C-MOVE).
    pub number_completed: Option<u16>,
    /// (0000,1022) Number of Failed Sub-operations (C-MOVE).
    pub number_failed: Option<u16>,
    /// (0000,1023) Number of Warning Sub-operations (C-MOVE).
    pub number_warning: Option<u16>,
    /// (0000,1000) Affected SOP Instance UID (C-STORE).
    pub affected_sop_instance_uid: Option<ArrayString<64>>,
}

/// Parse an IVR-LE encoded DIMSE command PDV body into a `CommandResponse`.
///
/// Returns `Err(NetworkingError::ParseError)` if (0000,0100) is absent.
pub fn parse_command_response(data: &[u8]) -> Result<CommandResponse, NetworkingError> {
    let mut resp = CommandResponse {
        data_set_type: NO_DATASET,
        ..Default::default()
    };

    let mut cursor = 0usize;
    while cursor + 8 <= data.len() {
        let group = u16::from_le_bytes([data[cursor], data[cursor + 1]]);
        let element = u16::from_le_bytes([data[cursor + 2], data[cursor + 3]]);
        let len = u32::from_le_bytes([
            data[cursor + 4],
            data[cursor + 5],
            data[cursor + 6],
            data[cursor + 7],
        ]) as usize;
        cursor += 8;

        if cursor + len > data.len() {
            break;
        }
        let value = &data[cursor..cursor + len];
        cursor += len;

        if group == 0x0000 {
            match element {
                0x0100 if len >= 2 => {
                    resp.command_field = u16::from_le_bytes([value[0], value[1]]);
                }
                0x0120 if len >= 2 => {
                    resp.message_id_responded = u16::from_le_bytes([value[0], value[1]]);
                }
                0x0800 if len >= 2 => {
                    resp.data_set_type = u16::from_le_bytes([value[0], value[1]]);
                }
                0x0900 if len >= 2 => {
                    resp.status = u16::from_le_bytes([value[0], value[1]]);
                }
                0x1000 if len > 0 => {
                    resp.affected_sop_instance_uid = Some({
                        let s = std::str::from_utf8(value)
                            .unwrap_or("")
                            .trim_end_matches(['\0', ' ']);
                        let mut arr = ArrayString::new();
                        for ch in s.chars().take(64) {
                            arr.try_push(ch).expect("ArrayString capacity exceeded while building affected SOP instance UID");
                        }
                        arr
                    });
                }
                0x1020 if len >= 2 => {
                    resp.number_remaining = Some(u16::from_le_bytes([value[0], value[1]]));
                }
                0x1021 if len >= 2 => {
                    resp.number_completed = Some(u16::from_le_bytes([value[0], value[1]]));
                }
                0x1022 if len >= 2 => {
                    resp.number_failed = Some(u16::from_le_bytes([value[0], value[1]]));
                }
                0x1023 if len >= 2 => {
                    resp.number_warning = Some(u16::from_le_bytes([value[0], value[1]]));
                }
                _ => {}
            }
        }
    }

    if resp.command_field == 0 {
        return Err(NetworkingError::ParseError(
            "command field (0000,0100) absent in response PDV".to_owned(),
        ));
    }

    Ok(resp)
}

/// Parse an IVR-LE encoded dataset body (non-command) into
/// `Vec<((group, element), value_bytes)>`.
///
/// Used to decode C-FIND response datasets.
pub fn parse_dataset_ivr_le(data: &[u8]) -> Vec<((u16, u16), Vec<u8>)> {
    let mut result = Vec::new();
    let mut cursor = 0usize;
    while cursor + 8 <= data.len() {
        let group = u16::from_le_bytes([data[cursor], data[cursor + 1]]);
        let element = u16::from_le_bytes([data[cursor + 2], data[cursor + 3]]);
        let len = u32::from_le_bytes([
            data[cursor + 4],
            data[cursor + 5],
            data[cursor + 6],
            data[cursor + 7],
        ]) as usize;
        cursor += 8;
        if cursor + len > data.len() {
            break;
        }
        result.push(((group, element), data[cursor..cursor + len].to_vec()));
        cursor += len;
    }
    result
}
