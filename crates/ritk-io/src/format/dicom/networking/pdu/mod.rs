//! DICOM Upper Layer (DUL) PDU codec per PS 3.8.

mod codec;
mod presentation_context;
mod user_info;

use anyhow::{bail, Result};
use arrayvec::ArrayString;

// ── Public re-exports — preserves the original `pdu::*` API ──────────────────

pub use presentation_context::{PresentationContextItemAc, PresentationContextItemRq};
pub use user_info::{
    ApplicationContextItem, AsynchronousOperationsWindowSubItem, ExtendedNegotiation,
    ImplementationClassUidSubItem, ImplementationVersionNameSubItem, MaximumLengthSubItem,
    ScpScuRoleSelectionSubItem, UserIdentity, UserIdentityType, UserInformation,
};

// ── Constants ────────────────────────────────────────────────────────────────

pub const RITK_IMPLEMENTATION_VERSION: &str = "RITK_0_50_71";
pub const RITK_IMPLEMENTATION_CLASS_UID: &str = "1.2.826.0.1.3690043.9.7433.1.1";
pub const DEFAULT_MAXIMUM_LENGTH: u32 = 16384;
pub const APPLICATION_CONTEXT_NAME: &str = "1.2.840.10008.3.1.1.1";

const PDU_ASSOC_RQ: u8 = 0x01;
const PDU_ASSOC_AC: u8 = 0x02;
const PDU_ASSOC_RJ: u8 = 0x03;
const PDU_PDATA: u8 = 0x04;
const PDU_REL_RQ: u8 = 0x05;
const PDU_REL_RP: u8 = 0x06;
const PDU_ABORT: u8 = 0x07;
const IT_APP_CTX: u8 = 0x10;
const IT_PC_RQ: u8 = 0x20;
const IT_PC_AC: u8 = 0x21;
const IT_USER_INFO: u8 = 0x50;
const IT_ABS_SYN: u8 = 0x30;
const IT_XFER_SYN: u8 = 0x40;
const SI_MAX_LEN: u8 = 0x51;
const SI_IMPL_UID: u8 = 0x52;
const SI_IMPL_VER: u8 = 0x55;
const SI_ASYNC: u8 = 0x53;
const SI_ROLE: u8 = 0x54;
const SI_EXT_NEG: u8 = 0x56;
const SI_USER_ID: u8 = 0x58;

// ── Small enums ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum AssociationRejectResult {
    RejectedPermanent = 1,
    RejectedTransient = 2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum AssociationRejectSource {
    Reserved = 0,
    DicomUlServiceProviderAcse = 1,
    DicomUlServiceProviderPresentation = 2,
    DicomUlServiceUser = 3,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommandType {
    DataSet = 0,
    Command = 1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum AbortSource {
    DicomUlServiceUser = 0,
    Reserved = 1,
    DicomUlServiceProviderAcse = 2,
}

// ── PDU structs ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub struct AssociateRqPdu {
    pub protocol_version: u16,
    pub called_ae_title: ArrayString<16>,
    pub calling_ae_title: ArrayString<16>,
    pub application_context_name: ArrayString<64>,
    pub presentation_contexts: Vec<PresentationContextItemRq>,
    pub user_information: UserInformation,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AssociateAcPdu {
    pub protocol_version: u16,
    pub called_ae_title: ArrayString<16>,
    pub calling_ae_title: ArrayString<16>,
    pub application_context_name: ArrayString<64>,
    pub presentation_contexts: Vec<PresentationContextItemAc>,
    pub user_information: UserInformation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AssociateRjPdu {
    pub result: AssociationRejectResult,
    pub source: AssociationRejectSource,
    pub reason: u8,
}

/// Position of a PDV item within a fragmented DIMSE message (PS 3.8 §9.3.5.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FragmentPosition {
    /// This is the last (or only) fragment of the message.
    Last,
    /// More fragments follow.
    More,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MessageControlHeader {
    pub message_type: CommandType,
    pub fragment_position: FragmentPosition,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PresentationDataValueItem {
    pub presentation_context_id: u8,
    pub message_control_header: MessageControlHeader,
    pub data: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PDataTfPdu {
    pub presentation_data_value_items: Vec<PresentationDataValueItem>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReleaseRqPdu;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReleaseRpPdu;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AbortPdu {
    pub source: AbortSource,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Pdu {
    AssociateRq(AssociateRqPdu),
    AssociateAc(AssociateAcPdu),
    AssociateRj(AssociateRjPdu),
    PDataTf(PDataTfPdu),
    ReleaseRq(ReleaseRqPdu),
    ReleaseRp(ReleaseRpPdu),
    Abort(AbortPdu),
}

// ── Low-level write helpers (shared with submodules) ─────────────────────────

fn pad_ae(title: &str) -> [u8; 16] {
    let mut b = [b' '; 16];
    let s = title.as_bytes();
    let n = s.len().min(16);
    b[..n].copy_from_slice(&s[..n]);
    b
}

fn ae_from_bytes(bytes: &[u8]) -> ArrayString<16> {
    let s = std::str::from_utf8(bytes).unwrap_or("").trim_end();
    let mut arr = ArrayString::new();
    for ch in s.chars().take(16) {
        arr.try_push(ch)
            .expect("ArrayString capacity exceeded while building AE title");
    }
    arr
}

pub(crate) fn uid_from_bytes_64(d: &[u8]) -> ArrayString<64> {
    let s = std::str::from_utf8(d)
        .unwrap_or("")
        .trim_end_matches(['\0', ' ']);
    let mut arr = ArrayString::new();
    for ch in s.chars().take(64) {
        arr.try_push(ch)
            .expect("ArrayString capacity exceeded while building UID");
    }
    arr
}

fn w16(buf: &mut Vec<u8>, v: u16) {
    buf.extend_from_slice(&v.to_be_bytes());
}

fn w32(buf: &mut Vec<u8>, v: u32) {
    buf.extend_from_slice(&v.to_be_bytes());
}

fn w_item(buf: &mut Vec<u8>, it: u8, body: &[u8]) {
    buf.push(it);
    buf.push(0x00);
    w16(buf, body.len() as u16);
    buf.extend_from_slice(body);
}

fn r8(d: &[u8], o: &mut usize) -> Result<u8> {
    if *o >= d.len() {
        bail!("EOF at {}", *o);
    }
    let v = d[*o];
    *o += 1;
    Ok(v)
}

fn r16(d: &[u8], o: &mut usize) -> Result<u16> {
    if *o + 2 > d.len() {
        bail!("EOF at {}", *o);
    }
    let v = u16::from_be_bytes([d[*o], d[*o + 1]]);
    *o += 2;
    Ok(v)
}

fn rbytes<'a>(d: &'a [u8], o: &mut usize, n: usize) -> Result<&'a [u8]> {
    if *o + n > d.len() {
        bail!("EOF at {}", *o);
    }
    let s = &d[*o..*o + n];
    *o += n;
    Ok(s)
}

#[cfg(test)]
mod tests_pdu;
