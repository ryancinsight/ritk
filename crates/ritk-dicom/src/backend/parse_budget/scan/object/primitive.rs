use anyhow::{bail, Context, Result};
use bytes::Bytes;
use consus_core::{decode_each, ByteOrder, EndianScalar};
use smallvec::SmallVec;

use crate::dataset::{PrimitiveValue, VR};
use crate::{CharacterSet, DicomTag};

pub(super) fn numbers<T: EndianScalar>(bytes: &[u8], order: ByteOrder) -> Result<SmallVec<[T; 2]>> {
    let mut values = SmallVec::new();
    decode_each(bytes, order, |v| values.push(v)).context("partial DICOM binary scalar")?;
    Ok(values)
}

pub(super) fn decode(
    vr: VR,
    bytes: Bytes,
    order: ByteOrder,
    charset: &CharacterSet,
) -> Result<PrimitiveValue> {
    if bytes.is_empty() {
        return Ok(PrimitiveValue::Empty);
    }
    if vr.is_text() {
        let ascii = CharacterSet::default();
        let selected = if vr.uses_character_set() {
            charset
        } else {
            &ascii
        };
        let text = selected.decode(&bytes)?;
        let text = text.trim_end_matches([' ', '\0']);
        return Ok(if vr.splits_values() && text.contains('\\') {
            PrimitiveValue::Strs(text.split('\\').map(str::to_owned).collect())
        } else {
            PrimitiveValue::Str(text.to_owned())
        });
    }
    Ok(match vr {
        VR::AT => {
            let words = numbers::<u16>(&bytes, order)?;
            if !words.len().is_multiple_of(2) {
                bail!("partial DICOM attribute tag");
            }
            PrimitiveValue::Tags(
                words
                    .chunks_exact(2)
                    .map(|v| DicomTag::new(v[0], v[1]))
                    .collect(),
            )
        }
        VR::US => PrimitiveValue::U16(numbers(&bytes, order)?),
        VR::SS => PrimitiveValue::I16(numbers(&bytes, order)?),
        VR::UL => PrimitiveValue::U32(numbers(&bytes, order)?),
        VR::SL => PrimitiveValue::I32(numbers(&bytes, order)?),
        VR::UV => PrimitiveValue::U64(numbers(&bytes, order)?),
        VR::SV => PrimitiveValue::I64(numbers(&bytes, order)?),
        VR::FL => PrimitiveValue::F32(numbers(&bytes, order)?),
        VR::FD => PrimitiveValue::F64(numbers(&bytes, order)?),
        _ => {
            let width = vr.byte_word();
            if width > 1 && !bytes.len().is_multiple_of(width) {
                bail!("partial DICOM VR {vr} word");
            }
            let bytes = if width > 1 && order == ByteOrder::BigEndian {
                let normalized: Vec<_> = bytes
                    .chunks_exact(width)
                    .flat_map(|word| word.iter().rev().copied())
                    .collect();
                Bytes::from(normalized)
            } else {
                bytes
            };
            PrimitiveValue::Bytes(bytes)
        }
    })
}
