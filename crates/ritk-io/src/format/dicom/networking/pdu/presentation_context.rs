//! Presentation context item types and codec for DICOM Upper Layer PDUs (PS 3.8).

use anyhow::Result;

use super::{r16, r8, rbytes, w_item, IT_ABS_SYN, IT_XFER_SYN};

// ── Types ────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub struct PresentationContextItemRq {
    pub presentation_context_id: u8,
    pub abstract_syntax_uid: String,
    pub transfer_syntax_uids: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PresentationContextItemAc {
    pub presentation_context_id: u8,
    pub result_reason: u8,
    pub transfer_syntax_uid: String,
}

// ── Encode / Decode ──────────────────────────────────────────────────────────

pub(crate) fn enc_pc_rq(pc: &PresentationContextItemRq) -> Vec<u8> {
    let mut b = vec![pc.presentation_context_id, 0x00, 0x00, 0x00];
    let mut asb = Vec::new();
    asb.extend_from_slice(pc.abstract_syntax_uid.as_bytes());
    w_item(&mut b, IT_ABS_SYN, &asb);
    for ts in &pc.transfer_syntax_uids {
        let mut tsb = Vec::new();
        tsb.extend_from_slice(ts.as_bytes());
        w_item(&mut b, IT_XFER_SYN, &tsb);
    }
    b
}

pub(crate) fn dec_pc_rq(data: &[u8]) -> Result<PresentationContextItemRq> {
    let mut o = 0usize;
    let id = r8(data, &mut o)?;
    o += 3;
    let mut asyn = String::new();
    let mut tsyns = Vec::new();
    while o + 4 <= data.len() {
        let it = data[o];
        o += 2;
        let il = r16(data, &mut o)? as usize;
        let d = rbytes(data, &mut o, il)?;
        match it {
            IT_ABS_SYN => asyn = String::from_utf8_lossy(d).into_owned(),
            IT_XFER_SYN => tsyns.push(String::from_utf8_lossy(d).into_owned()),
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
    let mut tsb = Vec::new();
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
    let mut ts = String::new();
    while o + 4 <= data.len() {
        let it = data[o];
        o += 2;
        let il = r16(data, &mut o)? as usize;
        let d = rbytes(data, &mut o, il)?;
        if it == IT_XFER_SYN {
            ts = String::from_utf8_lossy(d).into_owned();
        }
    }
    Ok(PresentationContextItemAc {
        presentation_context_id: id,
        result_reason: rr,
        transfer_syntax_uid: ts,
    })
}
