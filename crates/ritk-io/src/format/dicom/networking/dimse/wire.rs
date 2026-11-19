//! Implicit VR Little Endian command sets (DICOM PS3.7 §6.3.1).

use anyhow::{bail, Context, Result};

use super::{CommandElement, CommandValue, CommandVr, DimseMessage, TAG_CMD_GROUP_LENGTH};

const HEADER: usize = 8;

impl DimseMessage {
    /// Encode the group-zero command with 32-bit value lengths.
    ///
    /// # Panics
    /// Panics if a command cannot be represented by the wire's 32-bit group length.
    #[must_use]
    pub fn encode_command_set(&self) -> Vec<u8> {
        let values = self
            .command_set
            .iter()
            .filter(|entry| entry.tag != TAG_CMD_GROUP_LENGTH);
        let length: usize = values.clone().map(|entry| HEADER + entry.value.len()).sum();
        let length_field = u32::try_from(length).expect("DIMSE command body fits wire length");
        let mut encoded = Vec::with_capacity(HEADER + 4 + length);
        append_element(
            &mut encoded,
            TAG_CMD_GROUP_LENGTH,
            &length_field.to_le_bytes(),
        );
        if values.clone().is_sorted_by_key(|entry| entry.tag) {
            for entry in values {
                append_element(&mut encoded, entry.tag, entry.value.as_bytes());
            }
        } else {
            let mut ordered = values.collect::<Vec<_>>();
            ordered.sort_unstable_by_key(|entry| entry.tag);
            for entry in ordered {
                append_element(&mut encoded, entry.tag, entry.value.as_bytes());
            }
        }
        encoded
    }

    /// Decode a command set before attaching its separate data set.
    ///
    /// # Errors
    /// Rejects truncated or non-group-zero fields, and lengths that exceed the
    /// supplied command set or violate the required even wire alignment.
    pub fn decode_command_set(mut bytes: &[u8]) -> Result<Self> {
        let mut command_set = Vec::with_capacity(16);
        while !bytes.is_empty() {
            let header = bytes
                .get(..HEADER)
                .context("truncated DIMSE command header")?;
            let tag = (
                u16::from_le_bytes([header[0], header[1]]),
                u16::from_le_bytes([header[2], header[3]]),
            );
            if tag.0 != 0 {
                bail!("DIMSE command has nonzero group {:04X}", tag.0);
            }
            let length = u32::from_le_bytes([header[4], header[5], header[6], header[7]]) as usize;
            if !length.is_multiple_of(2) {
                bail!("DIMSE command value has odd length {length}");
            }
            let (value, remaining) = bytes[HEADER..]
                .split_at_checked(length)
                .context("truncated DIMSE command value")?;
            command_set.push(CommandElement {
                tag,
                vr: representation(tag.1),
                value: inline_or_heap(value),
            });
            bytes = remaining;
        }
        Ok(Self {
            command_set,
            data_set: None,
        })
    }
}

fn append_element(out: &mut Vec<u8>, (group, element): (u16, u16), value: &[u8]) {
    out.extend_from_slice(&group.to_le_bytes());
    out.extend_from_slice(&element.to_le_bytes());
    out.extend_from_slice(&(value.len() as u32).to_le_bytes());
    out.extend_from_slice(value);
}

fn inline_or_heap(value: &[u8]) -> CommandValue {
    let mut storage = [0; 4];
    if let Some(prefix) = storage.get_mut(..value.len()) {
        prefix.copy_from_slice(value);
        CommandValue::Inline(storage, value.len() as u8)
    } else {
        CommandValue::Heap(value.to_vec())
    }
}

// The representation follows the command elements in PS3.7 Annex E.
// Unused and retired commands retain their bytes as unclassified values.
fn representation(element: u16) -> CommandVr {
    match element {
        0x0000 | 0x0001 => CommandVr::Ul,
        0x0002 | 0x0003 | 0x1000 | 0x1001 => CommandVr::Ui,
        0x0600 | 0x1030 => CommandVr::Ae,
        0x0902 => CommandVr::Lo,
        0x0100
        | 0x0110
        | 0x0120
        | 0x0700
        | 0x0800
        | 0x0900
        | 0x0903
        | 0x1002
        | 0x1008
        | 0x1020..=0x1023
        | 0x1031 => CommandVr::Us,
        _ => CommandVr::Un,
    }
}
