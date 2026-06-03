//! DICOM Upper Layer (DUL) PDU codec per PS 3.8.
use anyhow::{bail, Result};

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum UserIdentityType {
    Username = 1,
    UsernameAndPassword = 2,
    Kerberos = 3,
    Saml = 4,
    Jwt = 5,
}
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

#[derive(Debug, Clone, PartialEq)]
pub struct ExtendedNegotiation {
    pub sop_class_uid: String,
    pub service_class_application_information: Vec<u8>,
}
#[derive(Debug, Clone, PartialEq)]
pub struct UserIdentity {
    pub identity_type: UserIdentityType,
    pub primary_field: Vec<u8>,
    pub secondary_field: Vec<u8>,
}
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MaximumLengthSubItem {
    pub maximum_length_received: u32,
}
impl Default for MaximumLengthSubItem {
    fn default() -> Self {
        Self {
            maximum_length_received: DEFAULT_MAXIMUM_LENGTH,
        }
    }
}
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ImplementationClassUidSubItem {
    pub implementation_class_uid: String,
}
#[derive(Debug, Clone, PartialEq)]
pub struct ImplementationVersionNameSubItem {
    pub implementation_version_name: String,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AsynchronousOperationsWindowSubItem {
    pub maximum_number_operations_invoked: u16,
    pub maximum_number_operations_performed: u16,
}
#[derive(Debug, Clone, PartialEq)]
pub struct ScpScuRoleSelectionSubItem {
    pub sop_class_uid: String,
    pub scu_role: bool,
    pub scp_role: bool,
}
#[derive(Debug, Clone, PartialEq)]
pub struct ApplicationContextItem {
    pub application_context_name: String,
}
#[derive(Debug, Clone, PartialEq, Default)]
pub struct UserInformation {
    pub maximum_length: MaximumLengthSubItem,
    pub implementation_class_uid: ImplementationClassUidSubItem,
    pub implementation_version_name: Option<ImplementationVersionNameSubItem>,
    pub async_operations_window: Option<AsynchronousOperationsWindowSubItem>,
    pub role_selections: Vec<ScpScuRoleSelectionSubItem>,
    pub extended_negotiations: Vec<ExtendedNegotiation>,
    pub user_identity: Option<UserIdentity>,
}
#[derive(Debug, Clone, PartialEq)]
pub struct AssociateRqPdu {
    pub protocol_version: u16,
    pub called_ae_title: String,
    pub calling_ae_title: String,
    pub application_context_name: String,
    pub presentation_contexts: Vec<PresentationContextItemRq>,
    pub user_information: UserInformation,
}
#[derive(Debug, Clone, PartialEq)]
pub struct AssociateAcPdu {
    pub protocol_version: u16,
    pub called_ae_title: String,
    pub calling_ae_title: String,
    pub application_context_name: String,
    pub presentation_contexts: Vec<PresentationContextItemAc>,
    pub user_information: UserInformation,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AssociateRjPdu {
    pub result: AssociationRejectResult,
    pub source: AssociationRejectSource,
    pub reason: u8,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MessageControlHeader {
    pub message_type: CommandType,
    pub last_fragment: bool,
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

fn pad_ae(title: &str) -> [u8; 16] {
    let mut b = [b' '; 16];
    let s = title.as_bytes();
    let n = s.len().min(16);
    b[..n].copy_from_slice(&s[..n]);
    b
}
fn trim_ae(b: &[u8; 16]) -> String {
    let end = b.iter().rposition(|&c| c != b' ').map_or(0, |i| i + 1);
    String::from_utf8_lossy(&b[..end]).into_owned()
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

fn enc_ui(ui: &UserInformation) -> Vec<u8> {
    let mut b = Vec::new();
    let mut ml = Vec::new();
    w32(&mut ml, ui.maximum_length.maximum_length_received);
    w_item(&mut b, SI_MAX_LEN, &ml);
    let mut ic = Vec::new();
    ic.extend_from_slice(
        ui.implementation_class_uid
            .implementation_class_uid
            .as_bytes(),
    );
    w_item(&mut b, SI_IMPL_UID, &ic);
    if let Some(ref v) = ui.implementation_version_name {
        let mut iv = Vec::new();
        iv.extend_from_slice(v.implementation_version_name.as_bytes());
        w_item(&mut b, SI_IMPL_VER, &iv);
    }
    if let Some(ref aw) = ui.async_operations_window {
        let mut a = Vec::new();
        w16(&mut a, aw.maximum_number_operations_invoked);
        w16(&mut a, aw.maximum_number_operations_performed);
        w_item(&mut b, SI_ASYNC, &a);
    }
    for rs in &ui.role_selections {
        let mut r = Vec::new();
        r.extend_from_slice(rs.sop_class_uid.as_bytes());
        r.push(0x00);
        r.push(rs.scu_role as u8);
        r.push(rs.scp_role as u8);
        w_item(&mut b, SI_ROLE, &r);
    }
    for en in &ui.extended_negotiations {
        let mut e = Vec::new();
        e.extend_from_slice(en.sop_class_uid.as_bytes());
        e.extend_from_slice(&en.service_class_application_information);
        w_item(&mut b, SI_EXT_NEG, &e);
    }
    if let Some(ref uid) = ui.user_identity {
        let mut u = Vec::new();
        u.push(uid.identity_type as u8);
        w16(&mut u, uid.primary_field.len() as u16);
        w16(&mut u, uid.secondary_field.len() as u16);
        u.extend_from_slice(&uid.primary_field);
        u.extend_from_slice(&uid.secondary_field);
        w_item(&mut b, SI_USER_ID, &u);
    }
    b
}
fn dec_ui(data: &[u8]) -> Result<UserInformation> {
    let mut ui = UserInformation::default();
    let mut off = 0usize;
    while off + 4 <= data.len() {
        let it = data[off];
        off += 2;
        let il = u16::from_be_bytes([data[off], data[off + 1]]) as usize;
        off += 2;
        let ie = off + il;
        if ie > data.len() {
            bail!("UI sub-item overflow");
        }
        let d = &data[off..ie];
        match it {
            SI_MAX_LEN if d.len() >= 4 => {
                ui.maximum_length = MaximumLengthSubItem {
                    maximum_length_received: u32::from_be_bytes([d[0], d[1], d[2], d[3]]),
                }
            }
            SI_IMPL_UID => {
                ui.implementation_class_uid = ImplementationClassUidSubItem {
                    implementation_class_uid: String::from_utf8_lossy(d).into_owned(),
                }
            }
            SI_IMPL_VER => {
                ui.implementation_version_name = Some(ImplementationVersionNameSubItem {
                    implementation_version_name: String::from_utf8_lossy(d).into_owned(),
                })
            }
            SI_ASYNC if d.len() >= 4 => {
                ui.async_operations_window = Some(AsynchronousOperationsWindowSubItem {
                    maximum_number_operations_invoked: u16::from_be_bytes([d[0], d[1]]),
                    maximum_number_operations_performed: u16::from_be_bytes([d[2], d[3]]),
                })
            }
            SI_ROLE if d.len() >= 3 => {
                let ue = d.len() - 3;
                ui.role_selections.push(ScpScuRoleSelectionSubItem {
                    sop_class_uid: String::from_utf8_lossy(&d[..ue]).into_owned(),
                    scu_role: d[ue + 1] != 0,
                    scp_role: d[ue + 2] != 0,
                })
            }
            SI_EXT_NEG => {
                let ue = d.iter().position(|&b| b == 0).unwrap_or(d.len());
                ui.extended_negotiations.push(ExtendedNegotiation {
                    sop_class_uid: String::from_utf8_lossy(&d[..ue]).into_owned(),
                    service_class_application_information: d[ue..].to_vec(),
                })
            }
            SI_USER_ID if d.len() >= 5 => {
                let id = d[0];
                let pl = u16::from_be_bytes([d[1], d[2]]) as usize;
                let sl = u16::from_be_bytes([d[3], d[4]]) as usize;
                let pe = 5 + pl;
                let se = pe + sl;
                if se > d.len() {
                    bail!("user identity overflow");
                }
                ui.user_identity = Some(UserIdentity {
                    identity_type: match id {
                        1 => UserIdentityType::Username,
                        2 => UserIdentityType::UsernameAndPassword,
                        3 => UserIdentityType::Kerberos,
                        4 => UserIdentityType::Saml,
                        5 => UserIdentityType::Jwt,
                        _ => bail!("unknown identity type: {}", id),
                    },
                    primary_field: d[5..pe].to_vec(),
                    secondary_field: d[pe..se].to_vec(),
                })
            }
            _ => {}
        }
        off = ie;
    }
    Ok(ui)
}
fn enc_pc_rq(pc: &PresentationContextItemRq) -> Vec<u8> {
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
fn dec_pc_rq(data: &[u8]) -> Result<PresentationContextItemRq> {
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
fn enc_pc_ac(pc: &PresentationContextItemAc) -> Vec<u8> {
    let mut b = vec![pc.presentation_context_id, 0x00, pc.result_reason, 0x00];
    let mut tsb = Vec::new();
    tsb.extend_from_slice(pc.transfer_syntax_uid.as_bytes());
    w_item(&mut b, IT_XFER_SYN, &tsb);
    b
}
fn dec_pc_ac(data: &[u8]) -> Result<PresentationContextItemAc> {
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
            w_item(&mut b, IT_PC_RQ, &enc_pc_rq(pc));
        }
        w_item(&mut b, IT_USER_INFO, &enc_ui(&rq.user_information));
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
            w_item(&mut b, IT_PC_AC, &enc_pc_ac(pc));
        }
        w_item(&mut b, IT_USER_INFO, &enc_ui(&ac.user_information));
        Self::enc_simple(PDU_ASSOC_AC, &b)
    }
    fn enc_assoc_rj(rj: &AssociateRjPdu) -> Vec<u8> {
        Self::enc_simple(
            PDU_ASSOC_RJ,
            &[0x00, rj.result as u8, rj.source as u8, rj.reason],
        )
    }
    fn enc_pdata(pd: &PDataTfPdu) -> Vec<u8> {
        let mut b = Vec::new();
        for pdv in &pd.presentation_data_value_items {
            let mch = (pdv.message_control_header.message_type as u8)
                | (if pdv.message_control_header.last_fragment {
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
        let mut app = String::new();
        let mut pcs = Vec::new();
        let mut ui = UserInformation::default();
        while off + 4 <= p.len() {
            let it = p[off];
            off += 2;
            let il = u16::from_be_bytes([p[off], p[off + 1]]) as usize;
            off += 2;
            let d = &p[off..off + il];
            match it {
                IT_APP_CTX => app = String::from_utf8_lossy(d).into_owned(),
                IT_PC_RQ => pcs.push(dec_pc_rq(d)?),
                IT_USER_INFO => ui = dec_ui(d)?,
                _ => {}
            }
            off += il;
        }
        Ok(Pdu::AssociateRq(AssociateRqPdu {
            protocol_version: ver,
            called_ae_title: trim_ae(&ca),
            calling_ae_title: trim_ae(&cg),
            application_context_name: app,
            presentation_contexts: pcs,
            user_information: ui,
        }))
    }
    fn dec_assoc_ac(p: &[u8]) -> Result<Self> {
        let (ver, ca, cg) = Self::dec_assoc_hdr(p)?;
        let mut off = 68usize;
        let mut app = String::new();
        let mut pcs = Vec::new();
        let mut ui = UserInformation::default();
        while off + 4 <= p.len() {
            let it = p[off];
            off += 2;
            let il = u16::from_be_bytes([p[off], p[off + 1]]) as usize;
            off += 2;
            let d = &p[off..off + il];
            match it {
                IT_APP_CTX => app = String::from_utf8_lossy(d).into_owned(),
                IT_PC_AC => pcs.push(dec_pc_ac(d)?),
                IT_USER_INFO => ui = dec_ui(d)?,
                _ => {}
            }
            off += il;
        }
        Ok(Pdu::AssociateAc(AssociateAcPdu {
            protocol_version: ver,
            called_ae_title: trim_ae(&ca),
            calling_ae_title: trim_ae(&cg),
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
                    last_fragment: lf,
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

#[cfg(test)]
#[path = "tests_pdu.rs"]
mod tests;
