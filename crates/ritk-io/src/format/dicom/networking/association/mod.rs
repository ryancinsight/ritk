//! DICOM Association SCU â€” association lifecycle per PS 3.8.
//!
//! Establishes a TCP-level association with a remote SCP, exchanges PDU
//! frames, sends/receives DIMSE messages over negotiated presentation
//! contexts, and releases or aborts the association.

mod helpers;
mod lifecycle;
mod scu;

pub use super::context::{transfer_syntax, AssociationConfig, NegotiatedContext};
pub use super::dimse::*;
pub use super::pdu::*;
pub use super::types::{
    AeTitle, DicomAddress, EchoResponse, MoveResponse, NetworkingError, StoreResponse,
};
pub(crate) use lifecycle::release_client_association;

use crate::format::dicom::reader::types::literal_arraystring;
use anyhow::{bail, Context, Result};
use std::io::{Read, Write};
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
    /// Trailing null (`\x00`) and space padding per DICOM PS3.5 Â§6.2 are stripped.
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
}

impl Association {
    /// Establish a DICOM association: TCP connect -> A-ASSOCIATE-RQ -> AC/RJ.
    pub fn connect(config: AssociationConfig) -> Result<Self> {
        let addr = format!("{}:{}", config.host, config.port);
        let stream = TcpStream::connect_timeout(&addr.parse()?, config.timeout)
            .with_context(|| format!("TCP connect to {}", addr))?;
        stream.set_read_timeout(Some(config.timeout))?;
        stream.set_write_timeout(Some(config.timeout))?;

        let mut next_id: u8 = 1;
        let mut pc_items = Vec::with_capacity(config.presentation_contexts.len());
        for rpc in &config.presentation_contexts {
            let ts = helpers::build_ts_list(&rpc.transfer_syntax_uids);
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
                implementation_class_uid: literal_arraystring(RITK_IMPLEMENTATION_CLASS_UID),
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
        };

        assoc.send_pdu(&rq)?;

        match assoc.recv_pdu()? {
            Pdu::AssociateAc(ac) => {
                let rq_map = helpers::rq_iter_abstracts(&rq);
                for pc in &ac.presentation_contexts {
                    if pc.result_reason == 0 {
                        assoc.negotiated_contexts.push(NegotiatedContext {
                            presentation_context_id: pc.presentation_context_id,
                            abstract_syntax_uid: rq_map
                                .get(&pc.presentation_context_id)
                                .cloned()
                                .unwrap_or_default(),
                            transfer_syntax_uid: pc.transfer_syntax_uid,
                        });
                    }
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

    // -- Internal helpers ----------------------------------------------------

    fn send_pdu(&mut self, pdu: &Pdu) -> Result<()> {
        self.stream
            .write_all(&pdu.encode())
            .with_context(|| "PDU write")?;
        self.stream.flush()?;
        Ok(())
    }

    fn recv_pdu(&mut self) -> Result<Pdu> {
        let mut hdr = [0u8; 6];
        self.stream
            .read_exact(&mut hdr)
            .with_context(|| "PDU header read")?;
        let len = u32::from_be_bytes([hdr[2], hdr[3], hdr[4], hdr[5]]) as usize;
        let mut body = vec![0u8; len];
        self.stream
            .read_exact(&mut body)
            .with_context(|| "PDU body read")?;
        let mut full = Vec::with_capacity(6 + len);
        full.extend_from_slice(&hdr);
        full.extend_from_slice(&body);
        Pdu::decode(&full).with_context(|| "PDU decode")
    }

    fn send_message(&mut self, ctx_id: u8, msg: &DimseMessage) -> Result<()> {
        let max = (self.remote_max_pdu_length.saturating_sub(6) as usize)
            .max(DEFAULT_MAXIMUM_LENGTH as usize - 6)
            .max(1);
        self.fragment_and_send(ctx_id, &msg.encode_command_set(), CommandType::Command, max)?;
        if let Some(ref ds) = msg.data_set {
            self.fragment_and_send(ctx_id, ds, CommandType::DataSet, max)?;
        }
        Ok(())
    }

    fn fragment_and_send(
        &mut self,
        cid: u8,
        data: &[u8],
        ct: CommandType,
        max: usize,
    ) -> Result<()> {
        let mk_pdv = |is_last: bool, d: &[u8]| PresentationDataValueItem {
            presentation_context_id: cid,
            message_control_header: MessageControlHeader {
                message_type: ct,
                fragment_position: if is_last {
                    FragmentPosition::Last
                } else {
                    FragmentPosition::More
                },
            },
            data: d.to_vec(),
        };
        if data.is_empty() {
            self.send_pdu(&Pdu::PDataTf(PDataTfPdu {
                presentation_data_value_items: vec![mk_pdv(true, &[])],
            }))?;
            return Ok(());
        }
        let chunks: Vec<&[u8]> = data.chunks(max).collect();
        for (i, c) in chunks.iter().enumerate() {
            self.send_pdu(&Pdu::PDataTf(PDataTfPdu {
                presentation_data_value_items: vec![mk_pdv(i == chunks.len() - 1, c)],
            }))?;
        }
        Ok(())
    }

    fn recv_message(&mut self) -> Result<(u8, DimseMessage)> {
        let (mut cmd, mut data) = (Vec::with_capacity(256), Vec::with_capacity(4096));
        let mut cid: u8 = 0;
        let mut cmd_last = false;
        loop {
            match self.recv_pdu()? {
                Pdu::PDataTf(pd) => {
                    let last_data = helpers::pdv_last_data(&pd);
                    for pdv in &pd.presentation_data_value_items {
                        cid = pdv.presentation_context_id;
                        match pdv.message_control_header.message_type {
                            CommandType::Command => {
                                cmd.extend_from_slice(&pdv.data);
                                if pdv.message_control_header.fragment_position
                                    == FragmentPosition::Last
                                {
                                    cmd_last = true;
                                }
                            }
                            CommandType::DataSet => data.extend_from_slice(&pdv.data),
                        }
                    }
                    if cmd_last {
                        let mut msg = DimseMessage::decode_command_set(&cmd)?;
                        let has_ds = msg.command_data_set_type().is_some_and(|v| v != 0x0101);
                        if has_ds && !last_data {
                            loop {
                                match self.recv_pdu()? {
                                    Pdu::PDataTf(pd2) => {
                                        for p in pd2.presentation_data_value_items {
                                            if p.message_control_header.message_type
                                                == CommandType::DataSet
                                            {
                                                data.extend_from_slice(&p.data);
                                                if p.message_control_header.fragment_position
                                                    == FragmentPosition::Last
                                                {
                                                    msg.data_set = Some(data);
                                                    return Ok((cid, msg));
                                                }
                                            }
                                        }
                                    }
                                    Pdu::ReleaseRq(_) => {
                                        let _ = self.send_pdu(&Pdu::ReleaseRp(ReleaseRpPdu));
                                        bail!("remote released during data transfer");
                                    }
                                    Pdu::Abort(_) => bail!("remote aborted"),
                                    o => bail!("unexpected PDU during data recv: {:?}", o),
                                }
                            }
                        }
                        msg.data_set = if data.is_empty() { None } else { Some(data) };
                        return Ok((cid, msg));
                    }
                }
                Pdu::ReleaseRq(_) => {
                    let _ = self.send_pdu(&Pdu::ReleaseRp(ReleaseRpPdu));
                    bail!("remote released association");
                }
                Pdu::Abort(_) => bail!("remote aborted"),
                o => bail!("unexpected PDU: {:?}", o),
            }
        }
    }

    fn find_context(&self, uid: &str) -> Option<&NegotiatedContext> {
        self.negotiated_contexts
            .iter()
            .find(|c| c.abstract_syntax_uid.as_str() == uid)
    }

    fn context_for_sop_class(&mut self, uid: &str, _ts: &[&str]) -> Result<u8> {
        self.find_context(uid)
            .map(|c| c.presentation_context_id)
            .ok_or_else(|| anyhow::anyhow!("no negotiated context for {}", uid))
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
