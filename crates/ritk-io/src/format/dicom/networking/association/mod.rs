//! DICOM Association SCU — association lifecycle per PS 3.8.
//!
//! Establishes a TCP-level association with a remote SCP, exchanges PDU
//! frames, sends/receives DIMSE messages over negotiated presentation
//! contexts, and releases or aborts the association.

mod negotiation;
mod scu;
mod service;
mod transport;

pub use super::context::{transfer_syntax, AssociationConfig, NegotiatedContext};
pub use super::dimse::*;
pub use super::pdu::*;
pub use super::types::{
    AeTitle, DicomAddress, EchoResponse, MoveResponse, NetworkingError, StoreResponse,
};
pub(super) use service::networking_error;

use crate::format::dicom::reader::types::literal_arraystring;
use anyhow::{bail, Context, Result};
use std::collections::VecDeque;
use std::net::TcpStream;

// -- Result types ----------------------------------------------------------

#[derive(Debug, Clone)]
pub struct FindResult {
    pub matches: Vec<Vec<u8>>,
    pub status: u16,
}

impl FindResult {
    /// Decode a string attribute from the first match dataset for `(group, element)`.
    ///
    /// Returns `None` when no match is present in `self.matches` or the tag is absent.
    /// Trailing null (`\x00`) and space padding per DICOM PS3.5 §6.2 are stripped.
    pub fn get_string(&self, group: u16, element: u16) -> Option<String> {
        use super::command::parse_dataset_ivr_le;
        self.matches.first().and_then(|m| {
            parse_dataset_ivr_le(m)
                .into_iter()
                .find(|((g, e), _)| *g == group && *e == element)
                .map(|(_, v)| {
                    std::str::from_utf8(&v)
                        .unwrap_or("")
                        .trim_end_matches(['\0', ' '])
                        .to_owned()
                })
        })
    }
}

#[derive(Debug, Clone)]
pub struct MoveResult {
    pub completed: u16,
    pub failed: u16,
    pub warning: u16,
    pub remaining: u16,
    pub status: u16,
}

// -- Association -----------------------------------------------------------

/// Association lifecycle state, replacing `active: bool`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DicomAssociationState {
    /// No association established yet or has been released/aborted.
    Inactive,
    /// Association is active (A-ASSOCIATE-AC received).
    Active,
}

pub struct Association {
    stream: TcpStream,
    negotiated_contexts: Vec<NegotiatedContext>,
    next_context_id: u8,
    pub remote_max_pdu_length: u32,
    state: DicomAssociationState,
    local_max_pdu_length: u32,
    pending: VecDeque<PresentationDataValueItem>,
}

impl Association {
    /// Establish a DICOM association: TCP connect -> A-ASSOCIATE-RQ -> AC/RJ.
    pub fn connect(config: AssociationConfig) -> Result<Self> {
        let stream = transport::connect(&config.host, config.port, config.timeout)?;
        stream.set_read_timeout(Some(config.timeout))?;
        stream.set_write_timeout(Some(config.timeout))?;

        let mut next_id: u8 = 1;
        let mut pc_items = Vec::with_capacity(config.presentation_contexts.len());
        for rpc in &config.presentation_contexts {
            let ts = negotiation::build_ts_list(&rpc.transfer_syntax_uids);
            pc_items.push(PresentationContextItemRq {
                presentation_context_id: next_id,
                abstract_syntax_uid: rpc.abstract_syntax_uid,
                transfer_syntax_uids: ts,
            });
            next_id = next_id
                .checked_add(2)
                .ok_or_else(|| anyhow::anyhow!("presentation context ID overflow"))?;
        }

        let ui = UserInformation {
            maximum_length: MaximumLengthSubItem {
                maximum_length_received: config.max_pdu_length,
            },
            implementation_class_uid: ImplementationClassUidSubItem {
                implementation_class_uid: literal_arraystring(
                    ritk_dicom::dataset::IMPLEMENTATION_CLASS_UID,
                ),
            },
            implementation_version_name: Some(ImplementationVersionNameSubItem {
                implementation_version_name: literal_arraystring(RITK_IMPLEMENTATION_VERSION),
            }),
            user_identity: config.user_identity.clone(),
            ..Default::default()
        };

        let rq = Pdu::AssociateRq(AssociateRqPdu {
            protocol_version: 1,
            called_ae_title: config.called_ae_title,
            calling_ae_title: config.calling_ae_title,
            application_context_name: literal_arraystring(APPLICATION_CONTEXT_NAME),
            presentation_contexts: pc_items,
            user_information: ui,
        });

        let num_contexts = config.presentation_contexts.len();
        let mut assoc = Self {
            stream,
            negotiated_contexts: Vec::with_capacity(num_contexts),
            next_context_id: next_id,
            remote_max_pdu_length: DEFAULT_MAXIMUM_LENGTH,
            state: DicomAssociationState::Inactive,
            local_max_pdu_length: config.max_pdu_length,
            pending: VecDeque::new(),
        };

        assoc.send_pdu(&rq)?;

        match assoc.recv_pdu()? {
            Pdu::AssociateAc(ac) => {
                let Pdu::AssociateRq(request) = &rq else {
                    unreachable!("locally constructed request")
                };
                for pc in ac
                    .presentation_contexts
                    .iter()
                    .filter(|pc| pc.result_reason == 0)
                {
                    let proposal = request
                        .presentation_contexts
                        .iter()
                        .find(|rq| rq.presentation_context_id == pc.presentation_context_id)
                        .context("remote accepted an unproposed presentation context")?;
                    if !proposal
                        .transfer_syntax_uids
                        .contains(&pc.transfer_syntax_uid)
                        || assoc
                            .negotiated_contexts
                            .iter()
                            .any(|cx| cx.presentation_context_id == pc.presentation_context_id)
                    {
                        bail!("remote accepted an incompatible or duplicate presentation context");
                    }
                    assoc.negotiated_contexts.push(NegotiatedContext {
                        presentation_context_id: pc.presentation_context_id,
                        abstract_syntax_uid: proposal.abstract_syntax_uid,
                        transfer_syntax_uid: pc.transfer_syntax_uid,
                    });
                }
                assoc.remote_max_pdu_length =
                    ac.user_information.maximum_length.maximum_length_received;
                assoc.state = DicomAssociationState::Active;
                Ok(assoc)
            }
            Pdu::AssociateRj(rj) => bail!(
                "association rejected: result={:?} source={:?} reason={}",
                rj.result,
                rj.source,
                rj.reason
            ),
            other => bail!("unexpected PDU in response to A-ASSOCIATE-RQ: {:?}", other),
        }
    }

    pub fn release(mut self) -> Result<()> {
        if self.state != DicomAssociationState::Active {
            return Ok(());
        }
        self.send_pdu(&Pdu::ReleaseRq(ReleaseRqPdu))?;
        match self.recv_pdu()? {
            Pdu::ReleaseRp(_) => {}
            o => bail!("expected A-RELEASE-RP, got {:?}", o),
        }
        self.state = DicomAssociationState::Inactive;
        Ok(())
    }

    pub fn abort(mut self) -> Result<()> {
        if self.state != DicomAssociationState::Active {
            return Ok(());
        }
        let _ = self.send_pdu(&Pdu::Abort(AbortPdu {
            source: AbortSource::DicomUlServiceUser,
        }));
        self.state = DicomAssociationState::Inactive;
        Ok(())
    }

    fn context_for_sop_class(&self, uid: &str, syntaxes: &[&str]) -> Result<u8> {
        self.negotiated_contexts
            .iter()
            .find(|c| {
                c.abstract_syntax_uid.as_str() == uid
                    && syntaxes.contains(&c.transfer_syntax_uid.as_str())
            })
            .map(|c| c.presentation_context_id)
            .ok_or_else(|| NetworkingError::NoPresentationContext(uid.to_owned()).into())
    }

    fn next_message_id(&mut self) -> u16 {
        let id = self.next_context_id as u16;
        self.next_context_id = self.next_context_id.wrapping_add(1);
        if self.next_context_id == 0 {
            self.next_context_id = 1;
        }
        id
    }
}

#[cfg(test)]
#[path = "../tests_association.rs"]
mod tests;

#[cfg(test)]
#[path = "../tests_store.rs"]
mod tests_store;
