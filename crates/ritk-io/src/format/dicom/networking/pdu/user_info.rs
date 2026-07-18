//! User Information sub-items for DICOM Upper Layer PDUs (PS 3.8).
//!
//! Contains all sub-item types that appear inside the User Information field
//! of A-ASSOCIATE-RQ/AC PDUs, plus their encode/decode functions.

use anyhow::{bail, Result};
use arrayvec::ArrayString;

use super::{
    uid_from_bytes_64, w16, w32, w_item, SI_ASYNC, SI_EXT_NEG, SI_IMPL_UID, SI_IMPL_VER,
    SI_MAX_LEN, SI_ROLE, SI_USER_ID,
};

// â”€â”€ Sub-item constants re-exported for internal use â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

// â”€â”€ Sub-item types â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum UserIdentityType {
    Username = 1,
    UsernameAndPassword = 2,
    Kerberos = 3,
    Saml = 4,
    Jwt = 5,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExtendedNegotiation {
    pub sop_class_uid: ArrayString<64>,
    pub service_class_application_information: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct UserIdentity {
    pub identity_type: UserIdentityType,
    pub primary_field: Vec<u8>,
    pub secondary_field: Vec<u8>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MaximumLengthSubItem {
    pub maximum_length_received: u32,
}

impl Default for MaximumLengthSubItem {
    fn default() -> Self {
        Self {
            maximum_length_received: super::DEFAULT_MAXIMUM_LENGTH,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct ImplementationClassUidSubItem {
    pub implementation_class_uid: ArrayString<64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ImplementationVersionNameSubItem {
    pub implementation_version_name: ArrayString<16>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AsynchronousOperationsWindowSubItem {
    pub maximum_number_operations_invoked: u16,
    pub maximum_number_operations_performed: u16,
}

/// DICOM SCU/SCP role selection for a Presentation Context.
///
/// Encodes the two-bit role negotiation from DICOM PS3.8 Â§D.3.3.4:
/// each bit is independent â€” an association requestor may request both,
/// either, or neither role.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DicomRole {
    /// Neither SCU nor SCP role is requested/supported.
    #[default]
    Neither,
    /// Service Class User role only.
    ScuOnly,
    /// Service Class Provider role only.
    ScpOnly,
    /// Both SCU and SCP roles are supported.
    Both,
}

impl DicomRole {
    /// Construct from the wire-format SCU and SCP octets.
    pub fn from_bits(scu: u8, scp: u8) -> Self {
        match (scu != 0, scp != 0) {
            (false, false) => Self::Neither,
            (true, false) => Self::ScuOnly,
            (false, true) => Self::ScpOnly,
            (true, true) => Self::Both,
        }
    }

    /// Wire-format SCU byte (0 or 1).
    pub fn scu_bit(self) -> u8 {
        matches!(self, Self::ScuOnly | Self::Both) as u8
    }

    /// Wire-format SCP byte (0 or 1).
    pub fn scp_bit(self) -> u8 {
        matches!(self, Self::ScpOnly | Self::Both) as u8
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScpScuRoleSelectionSubItem {
    pub sop_class_uid: ArrayString<64>,
    pub role: DicomRole,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ApplicationContextItem {
    pub application_context_name: ArrayString<64>,
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

// â”€â”€ Helper constructors â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

fn ver_from_bytes(d: &[u8]) -> ArrayString<16> {
    let s = std::str::from_utf8(d).unwrap_or("").trim_end();
    let mut arr = ArrayString::new();
    for ch in s.chars().take(16) {
        arr.try_push(ch)
            .expect("ArrayString capacity exceeded while building implementation version name");
    }
    arr
}

// â”€â”€ Encode / Decode â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

pub(crate) fn enc_ui(ui: &UserInformation) -> Vec<u8> {
    let mut b = Vec::with_capacity(128);
    let mut ml = Vec::with_capacity(4);
    w32(&mut ml, ui.maximum_length.maximum_length_received);
    w_item(&mut b, SI_MAX_LEN, &ml);
    let mut ic = Vec::with_capacity(64);
    ic.extend_from_slice(
        ui.implementation_class_uid
            .implementation_class_uid
            .as_bytes(),
    );
    w_item(&mut b, SI_IMPL_UID, &ic);
    if let Some(ref v) = ui.implementation_version_name {
        let mut iv = Vec::with_capacity(16);
        iv.extend_from_slice(v.implementation_version_name.as_bytes());
        w_item(&mut b, SI_IMPL_VER, &iv);
    }
    if let Some(ref aw) = ui.async_operations_window {
        let mut a = Vec::with_capacity(4);
        w16(&mut a, aw.maximum_number_operations_invoked);
        w16(&mut a, aw.maximum_number_operations_performed);
        w_item(&mut b, SI_ASYNC, &a);
    }
    for rs in &ui.role_selections {
        let mut r = Vec::with_capacity(68);
        r.extend_from_slice(rs.sop_class_uid.as_bytes());
        r.push(0x00);
        r.push(rs.role.scu_bit());
        r.push(rs.role.scp_bit());
        w_item(&mut b, SI_ROLE, &r);
    }
    for en in &ui.extended_negotiations {
        let mut e = Vec::with_capacity(64);
        e.extend_from_slice(en.sop_class_uid.as_bytes());
        e.extend_from_slice(&en.service_class_application_information);
        w_item(&mut b, SI_EXT_NEG, &e);
    }
    if let Some(ref uid) = ui.user_identity {
        let mut u = Vec::with_capacity(32);
        u.push(uid.identity_type as u8);
        w16(&mut u, uid.primary_field.len() as u16);
        w16(&mut u, uid.secondary_field.len() as u16);
        u.extend_from_slice(&uid.primary_field);
        u.extend_from_slice(&uid.secondary_field);
        w_item(&mut b, SI_USER_ID, &u);
    }
    b
}

pub(crate) fn dec_ui(data: &[u8]) -> Result<UserInformation> {
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
                    implementation_class_uid: uid_from_bytes_64(d),
                }
            }
            SI_IMPL_VER => {
                ui.implementation_version_name = Some(ImplementationVersionNameSubItem {
                    implementation_version_name: ver_from_bytes(d),
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
                    sop_class_uid: uid_from_bytes_64(&d[..ue]),
                    role: DicomRole::from_bits(d[ue + 1], d[ue + 2]),
                })
            }
            SI_EXT_NEG => {
                let ue = d.iter().position(|&b| b == 0).unwrap_or(d.len());
                ui.extended_negotiations.push(ExtendedNegotiation {
                    sop_class_uid: uid_from_bytes_64(&d[..ue]),
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
