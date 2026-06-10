//! PDU encode/decode implementation (PS 3.8 wire format).

use super::*;

use anyhow::{bail, Result};

impl Pdu {
    pub fn encode(&self) -> Vec<u8> {
        match self {
            Pdu::AssociateRq(rq) => Self::enc_assoc_rq(rq),
            Pdu::AssociateAc(ac) => Self::enc_assoc_ac(ac),
            Pdu::AssociateRj(rj) => Self::enc_assoc_rj(rj),
            Pdu::PDataTf(pd) => Self::enc_pdata(pd),
            Pdu::ReleaseRq(_) => Self::enc_simple(PDU_REL_RQ, &[]),
            Pdu::ReleaseRp(_) => Self::enc_simple(PDU_REL_RP, &[]),
            Pdu::Abort(ab) => Self::enc_abort(ab),
        }
    }

    pub fn decode(data: &[u8]) -> Result<Self> {
        if data.len() < 6 {
            bail!("PDU too short");
        }
        let pt = data[0];
        let pl = u32::from_be_bytes([data[2], data[3], data[4], data[5]]) as usize;
        if data.len() < 6 + pl {
            bail!("PDU length mismatch");
        }
        let p = &data[6..6 + pl];
        match pt {
            PDU_ASSOC_RQ => Ok(Self::dec_assoc_rq(p)?),
            PDU_ASSOC_AC => Ok(Self::dec_assoc_ac(p)?),
            PDU_ASSOC_RJ => Ok(Self::dec_assoc_rj(p)?),
            PDU_PDATA => Ok(Self::dec_pdata(p)?),
            PDU_REL_RQ => Ok(Pdu::ReleaseRq(ReleaseRqPdu)),
            PDU_REL_RP => Ok(Pdu::ReleaseRp(ReleaseRpPdu)),
            PDU_ABORT => Ok(Self::dec_abort(p)?),
            _ => bail!("unknown PDU type: 0x{:02x}", pt),
        }
    }

    fn enc_simple(pt: u8, body: &[u8]) -> Vec<u8> {
        let mut b = Vec::with_capacity(6 + body.len());
        b.push(pt);
        b.push(0x00);
        w32(&mut b, body.len() as u32);
        b.extend_from_slice(body);
        b
    }

    fn enc_assoc_rq(rq: &AssociateRqPdu) -> Vec<u8> {
        let mut b = Vec::new();
        w16(&mut b, rq.protocol_version);
        b.push(0x00);
        b.push(0x00);
        b.extend_from_slice(&pad_ae(&rq.called_ae_title));
        b.extend_from_slice(&pad_ae(&rq.calling_ae_title));
        b.extend_from_slice(&[0u8; 32]);
        let mut ac = Vec::new();
        ac.extend_from_slice(rq.application_context_name.as_bytes());
        w_item(&mut b, IT_APP_CTX, &ac);
        for pc in &rq.presentation_contexts {
            w_item(&mut b, IT_PC_RQ, &presentation_context::enc_pc_rq(pc));
        }
        w_item(
            &mut b,
            IT_USER_INFO,
            &user_info::enc_ui(&rq.user_information),
        );
        Self::enc_simple(PDU_ASSOC_RQ, &b)
    }

    fn enc_assoc_ac(ac: &AssociateAcPdu) -> Vec<u8> {
        let mut b = Vec::new();
        w16(&mut b, ac.protocol_version);
        b.push(0x00);
        b.push(0x00);
        b.extend_from_slice(&pad_ae(&ac.called_ae_title));
        b.extend_from_slice(&pad_ae(&ac.calling_ae_title));
        b.extend_from_slice(&[0u8; 32]);
        let mut app = Vec::new();
        app.extend_from_slice(ac.application_context_name.as_bytes());
        w_item(&mut b, IT_APP_CTX, &app);
        for pc in &ac.presentation_contexts {
            w_item(&mut b, IT_PC_AC, &presentation_context::enc_pc_ac(pc));
        }
        w_item(
            &mut b,
            IT_USER_INFO,
            &user_info::enc_ui(&ac.user_information),
        );
        Self::enc_simple(PDU_ASSOC_AC, &b)
    }

    fn enc_assoc_rj(rj: &AssociateRjPdu) -> Vec<u8> {
        Self::enc_simple(
            PDU_ASSOC_RJ,
            &[0x00, rj.result as u8, rj.source as u8, rj.reason],
        )
    }

    fn enc_pdata(pd: &PDataTfPdu) -> Vec<u8> {
        let capacity: usize = pd
            .presentation_data_value_items
            .iter()
            .map(|pdv| 6 + pdv.data.len())
            .sum();
        let mut b = Vec::with_capacity(capacity);
        for pdv in &pd.presentation_data_value_items {
            let mch = (pdv.message_control_header.message_type as u8)
                | (if pdv.message_control_header.fragment_position == FragmentPosition::Last {
                    0x02
                } else {
                    0x00
                });
            w32(&mut b, (1 + 1 + pdv.data.len()) as u32);
            b.push(pdv.presentation_context_id);
            b.push(mch);
            b.extend_from_slice(&pdv.data);
        }
        Self::enc_simple(PDU_PDATA, &b)
    }

    fn enc_abort(ab: &AbortPdu) -> Vec<u8> {
        Self::enc_simple(PDU_ABORT, &[0x00, 0x00, ab.source as u8, 0x00])
    }

    fn dec_assoc_hdr(p: &[u8]) -> Result<(u16, [u8; 16], [u8; 16])> {
        if p.len() < 68 {
            bail!("A-ASSOCIATE payload too short");
        }
        let ver = u16::from_be_bytes([p[0], p[1]]);
        let mut ca = [0u8; 16];
        ca.copy_from_slice(&p[4..20]);
        let mut cg = [0u8; 16];
        cg.copy_from_slice(&p[20..36]);
        Ok((ver, ca, cg))
    }

    fn dec_assoc_rq(p: &[u8]) -> Result<Self> {
        let (ver, ca, cg) = Self::dec_assoc_hdr(p)?;
        let mut off = 68usize;
        let mut app = ArrayString::new();
        let mut pcs = Vec::new();
        let mut ui = UserInformation::default();
        while off + 4 <= p.len() {
            let it = p[off];
            off += 2;
            let il = u16::from_be_bytes([p[off], p[off + 1]]) as usize;
            off += 2;
            let d = &p[off..off + il];
            match it {
                IT_APP_CTX => app = uid_from_bytes_64(d),
                IT_PC_RQ => pcs.push(presentation_context::dec_pc_rq(d)?),
                IT_USER_INFO => ui = user_info::dec_ui(d)?,
                _ => {}
            }
            off += il;
        }
        Ok(Pdu::AssociateRq(AssociateRqPdu {
            protocol_version: ver,
            called_ae_title: ae_from_bytes(&ca),
            calling_ae_title: ae_from_bytes(&cg),
            application_context_name: app,
            presentation_contexts: pcs,
            user_information: ui,
        }))
    }

    fn dec_assoc_ac(p: &[u8]) -> Result<Self> {
        let (ver, ca, cg) = Self::dec_assoc_hdr(p)?;
        let mut off = 68usize;
        let mut app = ArrayString::new();
        let mut pcs = Vec::new();
        let mut ui = UserInformation::default();
        while off + 4 <= p.len() {
            let it = p[off];
            off += 2;
            let il = u16::from_be_bytes([p[off], p[off + 1]]) as usize;
            off += 2;
            let d = &p[off..off + il];
            match it {
                IT_APP_CTX => app = uid_from_bytes_64(d),
                IT_PC_AC => pcs.push(presentation_context::dec_pc_ac(d)?),
                IT_USER_INFO => ui = user_info::dec_ui(d)?,
                _ => {}
            }
            off += il;
        }
        Ok(Pdu::AssociateAc(AssociateAcPdu {
            protocol_version: ver,
            called_ae_title: ae_from_bytes(&ca),
            calling_ae_title: ae_from_bytes(&cg),
            application_context_name: app,
            presentation_contexts: pcs,
            user_information: ui,
        }))
    }

    fn dec_assoc_rj(p: &[u8]) -> Result<Self> {
        if p.len() < 4 {
            bail!("A-ASSOCIATE-RJ too short");
        }
        let r = match p[1] {
            1 => AssociationRejectResult::RejectedPermanent,
            2 => AssociationRejectResult::RejectedTransient,
            v => bail!("unknown result: {}", v),
        };
        let s = match p[2] {
            0 => AssociationRejectSource::Reserved,
            1 => AssociationRejectSource::DicomUlServiceProviderAcse,
            2 => AssociationRejectSource::DicomUlServiceProviderPresentation,
            3 => AssociationRejectSource::DicomUlServiceUser,
            v => bail!("unknown source: {}", v),
        };
        Ok(Pdu::AssociateRj(AssociateRjPdu {
            result: r,
            source: s,
            reason: p[3],
        }))
    }

    fn dec_pdata(p: &[u8]) -> Result<Self> {
        let mut items = Vec::new();
        let mut off = 0usize;
        while off + 4 <= p.len() {
            let pl = u32::from_be_bytes([p[off], p[off + 1], p[off + 2], p[off + 3]]) as usize;
            off += 4;
            if off + pl > p.len() {
                bail!("PDV overflow");
            }
            let cid = p[off];
            let mch = p[off + 1];
            let lf = (mch & 0x02) != 0;
            let mt = if (mch & 0x01) != 0 {
                CommandType::Command
            } else {
                CommandType::DataSet
            };
            items.push(PresentationDataValueItem {
                presentation_context_id: cid,
                message_control_header: MessageControlHeader {
                    message_type: mt,
                    fragment_position: if lf {
                        FragmentPosition::Last
                    } else {
                        FragmentPosition::More
                    },
                },
                data: p[off + 2..off + pl].to_vec(),
            });
            off += pl;
        }
        Ok(Pdu::PDataTf(PDataTfPdu {
            presentation_data_value_items: items,
        }))
    }

    fn dec_abort(p: &[u8]) -> Result<Self> {
        if p.len() < 4 {
            bail!("A-ABORT too short");
        }
        let s = match p[2] {
            0 => AbortSource::DicomUlServiceUser,
            1 => AbortSource::Reserved,
            2 => AbortSource::DicomUlServiceProviderAcse,
            v => bail!("unknown abort source: {}", v),
        };
        Ok(Pdu::Abort(AbortPdu { source: s }))
    }
}
