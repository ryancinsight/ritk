//! Implicit VR lookup from the numeric PS3.6 registry.
//!
//! The pinned edition, checksum, and regeneration procedure are in
//! scripts/dicom-dictionary.py. Repeating-tag patterns preserve all fixed bits
//! (including the element number) and prefer the most specific match.

use crate::dataset::VR;
use crate::DicomTag;

const EXACT: &[u8] = include_bytes!("exact.bin");
const REPEATING: &[u8] = include_bytes!("patterns.bin");
const EXACT_WIDTH: usize = 6;
const PATTERN_WIDTH: usize = 10;

fn code(tag: DicomTag) -> Option<[u8; 2]> {
    let raw = (u32::from(tag.group) << 16) | u32::from(tag.element);
    let needle = raw.to_be_bytes();
    let (mut lower, mut upper) = (0, EXACT.len() / EXACT_WIDTH);
    while lower < upper {
        let middle = (lower + upper) / 2;
        let entry = &EXACT[middle * EXACT_WIDTH..][..EXACT_WIDTH];
        match entry[..4].cmp(&needle) {
            std::cmp::Ordering::Less => lower = middle + 1,
            std::cmp::Ordering::Greater => upper = middle,
            std::cmp::Ordering::Equal => return Some([entry[4], entry[5]]),
        }
    }

    REPEATING.chunks_exact(PATTERN_WIDTH).find_map(|entry| {
        let mask = u32::from_be_bytes([entry[0], entry[1], entry[2], entry[3]]);
        let value = u32::from_be_bytes([entry[4], entry[5], entry[6], entry[7]]);
        (raw & mask == value).then_some([entry[8], entry[9]])
    })
}

pub(crate) fn vr(tag: DicomTag, signed_pixels: bool) -> VR {
    if tag.element == 0 {
        return VR::UL;
    }
    if tag.group % 2 == 1 {
        return if (0x0010..=0x00ff).contains(&tag.element) {
            VR::LO
        } else {
            VR::UN
        };
    }
    match code(tag) {
        Some([b'X', b'S'] | [b'L', b'W']) if signed_pixels => VR::SS,
        Some([b'X', b'S'] | [b'L', b'W']) => VR::US,
        Some([b'O', b'X'] | [b'L', b'X']) => VR::OW,
        Some(header) => std::str::from_utf8(&header)
            .ok()
            .and_then(|text| text.parse().ok())
            .unwrap_or(VR::UN),
        None => VR::UN,
    }
}
