//! Bounded Upper Layer transport and DIMSE fragment assembly.

use super::{
    Association, CommandField, CommandType, DicomAssociationState, DimseMessage, FragmentPosition,
    MessageControlHeader, NetworkingError, PDataTfPdu, Pdu, PresentationDataValueItem,
    ReleaseRpPdu, DEFAULT_MAXIMUM_LENGTH,
};
use anyhow::{bail, Context, Result};
use ritk_dicom::ParseBudget;
use std::{
    io::{Read, Write},
    net::{TcpStream, ToSocketAddrs},
    time::{Duration, Instant},
};

const P_DATA_TF: u8 = 0x04;
const PDV_PREFIX: usize = 6;

pub(super) fn connect(host: &str, port: u16, timeout: Duration) -> Result<TcpStream> {
    let host = host
        .strip_prefix('[')
        .and_then(|name| name.strip_suffix(']'))
        .unwrap_or(host);
    let start = Instant::now();
    let endpoints = (host, port)
        .to_socket_addrs()
        .with_context(|| format!("resolve DICOM endpoint {host}:{port}"))?;
    let mut error = None;
    for endpoint in endpoints {
        let remaining = timeout.saturating_sub(start.elapsed());
        if remaining.is_zero() {
            bail!("DICOM TCP connect deadline reached for {host}:{port}");
        }
        match TcpStream::connect_timeout(&endpoint, remaining) {
            Ok(stream) => return Ok(stream),
            Err(failed) => error = Some(failed),
        }
    }
    let error = error.with_context(|| format!("no addresses for DICOM endpoint {host}:{port}"))?;
    Err(NetworkingError::Connection(error).into())
}

impl Association {
    pub(super) fn send_pdu(&mut self, pdu: &Pdu) -> Result<()> {
        self.stream.write_all(&pdu.encode()).context("PDU write")
    }

    pub(super) fn recv_pdu(&mut self) -> Result<Pdu> {
        let mut header = [0u8; 6];
        self.stream
            .read_exact(&mut header)
            .context("PDU header read")?;
        let length = u32::from_be_bytes([header[2], header[3], header[4], header[5]]);
        if header[0] == P_DATA_TF
            && self.local_max_pdu_length != 0
            && length > self.local_max_pdu_length
        {
            bail!(
                "P-DATA body length {length} exceeds advertised limit {}",
                self.local_max_pdu_length
            );
        }
        let size = ParseBudget::DEFAULT
            .checked_bytes(u64::from(length) + 6, "DICOM PDU frame")
            .map_err(|error| anyhow::anyhow!("DICOM PDU frame exceeds parse budget: {error}"))?;
        let mut frame = Vec::new();
        frame
            .try_reserve_exact(size)
            .context("allocate DICOM PDU frame")?;
        frame.extend_from_slice(&header);
        frame.resize(size, 0);
        self.stream
            .read_exact(&mut frame[6..])
            .context("PDU body read")?;
        Pdu::decode(&frame).context("PDU decode")
    }

    pub(super) fn send_message(&mut self, context: u8, message: &DimseMessage) -> Result<()> {
        // PS3.8 Annex D limits the variable field, including the PDV prefix,
        // with zero representing the absence of a negotiated peer ceiling.
        let maximum = if self.remote_max_pdu_length == 0 {
            DEFAULT_MAXIMUM_LENGTH
        } else {
            self.remote_max_pdu_length
        } as usize;
        let payload = maximum.saturating_sub(PDV_PREFIX) & !1;
        if payload == 0 {
            bail!("remote P-DATA length {maximum} cannot hold a fragment");
        }

        self.send_fragments(
            context,
            &message.encode_command_set(),
            CommandType::Command,
            payload,
        )?;
        if let Some(data) = &message.data_set {
            self.send_fragments(context, data, CommandType::DataSet, payload)?;
        }
        Ok(())
    }

    fn send_fragments(
        &mut self,
        context: u8,
        data: &[u8],
        message_type: CommandType,
        payload: usize,
    ) -> Result<()> {
        let empty = data.is_empty().then_some(&[][..]);
        let parts = data.chunks(payload).chain(empty);
        let total = data.len().div_ceil(payload).max(1);
        for (index, part) in parts.enumerate() {
            let position = if index + 1 == total {
                FragmentPosition::Last
            } else {
                FragmentPosition::More
            };
            let item = PresentationDataValueItem {
                presentation_context_id: context,
                message_control_header: MessageControlHeader {
                    message_type,
                    fragment_position: position,
                },
                data: part.to_vec(),
            };
            self.send_pdu(&Pdu::PDataTf(PDataTfPdu {
                presentation_data_value_items: vec![item],
            }))?;
        }
        Ok(())
    }

    fn recv_pdv(&mut self) -> Result<PresentationDataValueItem> {
        while self.pending.is_empty() {
            match self.recv_pdu()? {
                Pdu::PDataTf(data) if !data.presentation_data_value_items.is_empty() => {
                    self.pending.extend(data.presentation_data_value_items)
                }
                Pdu::ReleaseRq(_) => {
                    self.send_pdu(&Pdu::ReleaseRp(ReleaseRpPdu))?;
                    self.state = DicomAssociationState::Inactive;
                    bail!("remote released during a DIMSE transaction");
                }
                Pdu::Abort(_) => {
                    self.state = DicomAssociationState::Inactive;
                    bail!("remote aborted the association");
                }
                _ => bail!("expected P-DATA with at least one value"),
            }
        }
        self.pending
            .pop_front()
            .context("missing presentation data value")
    }

    fn read_fragments(&mut self, context: u8, expected: CommandType) -> Result<Vec<u8>> {
        let mut combined = Vec::new();
        loop {
            let pdv = self.recv_pdv()?;
            if pdv.presentation_context_id != context
                || pdv.message_control_header.message_type != expected
            {
                bail!("unexpected context or fragment type during DIMSE assembly");
            }
            let size = combined
                .len()
                .checked_add(pdv.data.len())
                .context("DIMSE length overflow")?;
            ParseBudget::DEFAULT
                .checked_bytes(size as u64, "DIMSE fragments")
                .map_err(|error| anyhow::anyhow!("DIMSE message exceeds parse budget: {error}"))?;
            combined
                .try_reserve(pdv.data.len())
                .context("reserve DIMSE fragment storage")?;
            combined.extend_from_slice(&pdv.data);
            if pdv.message_control_header.fragment_position == FragmentPosition::Last {
                return Ok(combined);
            }
        }
    }

    pub(super) fn recv_response(
        &mut self,
        context: u8,
        command: CommandField,
    ) -> Result<DimseMessage> {
        let bytes = self.read_fragments(context, CommandType::Command)?;
        let mut response = DimseMessage::decode_command_set(&bytes)?;
        let actual = response
            .command_field()
            .context("missing response CommandField")?;
        if actual != command {
            bail!("expected {command:?}, received {actual:?}");
        }
        if response
            .command_data_set_type()
            .is_some_and(|value| value != 0x0101)
        {
            response.data_set = Some(self.read_fragments(context, CommandType::DataSet)?);
        }
        Ok(response)
    }
}
