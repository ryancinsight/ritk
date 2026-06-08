//! Association helper functions — request building, context negotiation, PDV fragmentation.

use super::super::context::{transfer_syntax, AssociationConfig, NegotiatedContext};
use super::super::pdu::*;
use super::Association;
use crate::format::dicom::reader::types::literal_arraystring;
use arrayvec::ArrayString;
use std::collections::HashMap;

impl Association {
    /// Build an A-ASSOCIATE-RQ PDU from the given config.
    ///
    /// Exposed as `pub` for testability (verifying PDU structure without TCP).
    pub fn build_associate_rq(config: &AssociationConfig) -> Pdu {
        let mut nid: u8 = 1;
        let pcs: Vec<_> = config
            .presentation_contexts
            .iter()
            .map(|rpc| {
                let mut ts = rpc.transfer_syntax_uids.clone();
                if !ts.iter().any(|t| t == transfer_syntax::IMPLICIT_VR_LE) {
                    ts.push(literal_arraystring(transfer_syntax::IMPLICIT_VR_LE));
                }
                let id = nid;
                nid += 2;
                PresentationContextItemRq {
                    presentation_context_id: id,
                    abstract_syntax_uid: rpc.abstract_syntax_uid,
                    transfer_syntax_uids: ts,
                }
            })
            .collect();

        Pdu::AssociateRq(AssociateRqPdu {
            protocol_version: 1,
            called_ae_title: config.called_ae_title,
            calling_ae_title: config.calling_ae_title,
            application_context_name: literal_arraystring(APPLICATION_CONTEXT_NAME),
            presentation_contexts: pcs,
            user_information: UserInformation {
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
            },
        })
    }

    /// Extract accepted presentation contexts from an A-ASSOCIATE-AC PDU.
    ///
    /// `rq_abstracts` maps presentation-context IDs from the RQ to their
    /// abstract-syntax UIDs; the AC only carries the accepted transfer syntax.
    pub fn negotiated_contexts_from_ac(
        ac: &AssociateAcPdu,
        rq_abstracts: &HashMap<u8, ArrayString<64>>,
    ) -> Vec<NegotiatedContext> {
        ac.presentation_contexts
            .iter()
            .filter(|pc| pc.result_reason == 0)
            .map(|pc| NegotiatedContext {
                presentation_context_id: pc.presentation_context_id,
                abstract_syntax_uid: rq_abstracts
                    .get(&pc.presentation_context_id)
                    .cloned()
                    .unwrap_or_default(),
                transfer_syntax_uid: pc.transfer_syntax_uid,
            })
            .collect()
    }

    /// Fragment `data` into PDV items for a given command/data-set type.
    ///
    /// The `max` parameter is the maximum payload per PDV (excluding the
    /// 6-byte PDV header). Context ID defaults to `1` for test purposes;
    /// real associations set it from the negotiated context.
    pub fn fragment_pdvs(
        data: &[u8],
        ct: CommandType,
        max: usize,
    ) -> Vec<PresentationDataValueItem> {
        let cid = 1u8;
        if data.is_empty() {
            return vec![PresentationDataValueItem {
                presentation_context_id: cid,
                message_control_header: MessageControlHeader {
                    message_type: ct,
                    last_fragment: true,
                },
                data: Vec::new(),
            }];
        }
        let chunks: Vec<&[u8]> = data.chunks(max).collect();
        chunks
            .iter()
            .enumerate()
            .map(|(i, c)| PresentationDataValueItem {
                presentation_context_id: cid,
                message_control_header: MessageControlHeader {
                    message_type: ct,
                    last_fragment: i == chunks.len() - 1,
                },
                data: c.to_vec(),
            })
            .collect()
    }
}

/// Build a map of presentation-context ID → abstract-syntax UID from an RQ PDU.
pub(super) fn rq_iter_abstracts(pdu: &Pdu) -> HashMap<u8, ArrayString<64>> {
    let mut m = HashMap::new();
    if let Pdu::AssociateRq(rq) = pdu {
        for pc in &rq.presentation_contexts {
            m.insert(pc.presentation_context_id, pc.abstract_syntax_uid);
        }
    }
    m
}

/// Return `true` if the P-Data-TF contains a last-fragment data-set PDV.
pub(super) fn pdv_last_data(pd: &PDataTfPdu) -> bool {
    pd.presentation_data_value_items.iter().any(|p| {
        p.message_control_header.message_type == CommandType::DataSet
            && p.message_control_header.last_fragment
    })
}
