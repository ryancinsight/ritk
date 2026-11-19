//! DIMSE SCU operations over an established association.

use super::super::context::transfer_syntax;
use super::super::dimse::*;
use super::Association;
use anyhow::{Context, Result};

impl Association {
    /// Send a C-ECHO-RQ and return the response status.
    pub fn c_echo(&mut self) -> Result<u16> {
        let ctx = self.context_for_sop_class(
            sop_class::VERIFICATION_SOP_CLASS,
            &[
                transfer_syntax::IMPLICIT_VR_LE,
                transfer_syntax::EXPLICIT_VR_LE,
            ],
        )?;
        let msg = DimseMessage::c_echo_rq(self.next_message_id());
        self.send_message(ctx, &msg)?;
        let rsp = self.recv_response(ctx, CommandField::CEchoRsp)?;
        rsp.status().context("missing Status")
    }

    /// Send a C-FIND-RQ and collect all pending responses into a single result.
    pub fn c_find(
        &mut self,
        sop_class_uid: &str,
        identifier: Vec<u8>,
    ) -> Result<super::FindResult> {
        let mut matches = Vec::with_capacity(4);
        let status =
            self.find_responses(sop_class_uid, identifier, |_, data| matches.push(data))?;
        Ok(super::FindResult { matches, status })
    }

    pub(in crate::format::dicom::networking) fn find_responses(
        &mut self,
        sop_class_uid: &str,
        identifier: Vec<u8>,
        mut collect: impl FnMut(u16, Vec<u8>),
    ) -> Result<u16> {
        let ctx = self.context_for_sop_class(sop_class_uid, &[transfer_syntax::IMPLICIT_VR_LE])?;
        let request = DimseMessage::c_find_rq(self.next_message_id(), sop_class_uid, identifier);
        self.send_message(ctx, &request)?;
        loop {
            let response = self.recv_response(ctx, CommandField::CFindRsp)?;
            let status = response.status().context("missing Status")?;
            if status != DimseStatus::Pending as u16 && status != DimseStatus::PendingWarning as u16
            {
                return Ok(status);
            }
            if let Some(data) = response.data_set {
                collect(status, data);
            }
        }
    }

    /// Send a C-STORE-RQ and return the response status.
    pub fn c_store(
        &mut self,
        sop_class_uid: &str,
        sop_instance_uid: &str,
        data_set: Vec<u8>,
    ) -> Result<u16> {
        let response = self.store_response(
            sop_class_uid,
            sop_instance_uid,
            data_set,
            &[
                transfer_syntax::IMPLICIT_VR_LE,
                transfer_syntax::EXPLICIT_VR_LE,
            ],
        )?;
        response.status().context("missing Status")
    }

    pub(in crate::format::dicom::networking) fn store_response(
        &mut self,
        sop_class_uid: &str,
        sop_instance_uid: &str,
        data_set: Vec<u8>,
        syntaxes: &[&str],
    ) -> Result<DimseMessage> {
        let ctx = self.context_for_sop_class(sop_class_uid, syntaxes)?;
        let msg = DimseMessage::c_store_rq(
            self.next_message_id(),
            sop_class_uid,
            sop_instance_uid,
            0x0000,
            data_set,
        );
        self.send_message(ctx, &msg)?;
        self.recv_response(ctx, CommandField::CStoreRsp)
    }

    /// Send a C-MOVE-RQ and collect sub-operation counters.
    pub fn c_move(
        &mut self,
        sop_class_uid: &str,
        move_destination: &str,
        identifier: Vec<u8>,
    ) -> Result<super::MoveResult> {
        let ctx = self.context_for_sop_class(sop_class_uid, &[transfer_syntax::IMPLICIT_VR_LE])?;
        let msg = DimseMessage::c_move_rq(
            self.next_message_id(),
            sop_class_uid,
            move_destination,
            identifier,
        );
        self.send_message(ctx, &msg)?;

        let (mut comp, mut fail, mut warn, mut rem) = (0u16, 0u16, 0u16, 0u16);
        let final_status;
        loop {
            let rsp = self.recv_response(ctx, CommandField::CMoveRsp)?;
            let s = rsp.status().context("missing Status")?;
            let le = |tag: (u16, u16)| -> Option<u16> {
                rsp.find_element(tag)
                    .filter(|e| e.value.len() >= 2)
                    .map(|e| u16::from_le_bytes([e.value.as_bytes()[0], e.value.as_bytes()[1]]))
            };
            if let Some(v) = le((0x0000, 0x1021)) {
                comp = v;
            }
            if let Some(v) = le((0x0000, 0x1022)) {
                fail = v;
            }
            if let Some(v) = le((0x0000, 0x1023)) {
                warn = v;
            }
            if let Some(v) = le((0x0000, 0x1020)) {
                rem = v;
            }
            if s != DimseStatus::Pending as u16 && s != DimseStatus::PendingWarning as u16 {
                final_status = s;
                break;
            }
        }
        Ok(super::MoveResult {
            completed: comp,
            failed: fail,
            warning: warn,
            remaining: rem,
            status: final_status,
        })
    }
}
