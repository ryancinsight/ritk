//! Presentation context item types and codec for DICOM Upper Layer PDUs (PS 3.8).

use anyhow::Result;
use arrayvec::ArrayString;

use super::{r16, r8, rbytes, uid_from_bytes_64, w_item, IT_ABS_SYN, IT_XFER_SYN};

// â”€â”€ Types â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

#[derive(Debug, Clone, PartialEq)]
pub struct PresentationContextItemRq {
    pub presentation_context_id: u8,
    pub abstract_syntax_uid: ArrayString<64>,
    pub transfer_syntax_uids: Vec<ArrayString<64>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PresentationContextItemAc {
    pub presentation_context_id: u8,
    pub result_reason: u8,
    pub transfer_syntax_uid: ArrayString<64>,
}

// â”€â”€ Encode / Decode â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

pub(crate) fn enc_pc_rq(pc: &PresentationContextItemRq) -> Vec<u8> {
    let mut b = vec![pc.presentation_context_id, 0x00, 0x00, 0x00];
    let mut asb = Vec::with_capacity(64);
    asb.extend_from_slice(pc.abstract_syntax_uid.as_bytes());
    w_item(&mut b, IT_ABS_SYN, &asb);
    for ts in &pc.transfer_syntax_uids {
        let mut tsb = Vec::with_capacity(64);
        tsb.extend_from_slice(ts.as_bytes());
        w_item(&mut b, IT_XFER_SYN, &tsb);
    }
    b
}

pub(crate) fn dec_pc_rq(data: &[u8]) -> Result<PresentationContextItemRq> {
    let mut o = 0usize;
    let id = r8(data, &mut o)?;
    o += 3;
    let mut asyn = ArrayString::new();
    let mut tsyns = Vec::with_capacity(4);
    while o + 4 <= data.len() {
        let it = data[o];
        o += 2;
        let il = r16(data, &mut o)? as usize;
        let d = rbytes(data, &mut o, il)?;
        match it {
            IT_ABS_SYN => asyn = uid_from_bytes_64(d),
            IT_XFER_SYN => tsyns.push(uid_from_bytes_64(d)),
            _ => {}
        }
    }
    Ok(PresentationContextItemRq {
        presentation_context_id: id,
        abstract_syntax_uid: asyn,
        transfer_syntax_uids: tsyns,
    })
}

pub(crate) fn enc_pc_ac(pc: &PresentationContextItemAc) -> Vec<u8> {
    let mut b = vec![pc.presentation_context_id, 0x00, pc.result_reason, 0x00];
    let mut tsb = Vec::with_capacity(64);
    tsb.extend_from_slice(pc.transfer_syntax_uid.as_bytes());
    w_item(&mut b, IT_XFER_SYN, &tsb);
    b
}

pub(crate) fn dec_pc_ac(data: &[u8]) -> Result<PresentationContextItemAc> {
    let mut o = 0usize;
    let id = r8(data, &mut o)?;
    o += 1;
    let rr = r8(data, &mut o)?;
    o += 1;
    let mut ts = ArrayString::new();
    while o + 4 <= data.len() {
        let it = data[o];
        o += 2;
        let il = r16(data, &mut o)? as usize;
        let d = rbytes(data, &mut o, il)?;
        if it == IT_XFER_SYN {
            ts = uid_from_bytes_64(d);
        }
    }
    Ok(PresentationContextItemAc {
        presentation_context_id: id,
        result_reason: rr,
        transfer_syntax_uid: ts,
    })
}
