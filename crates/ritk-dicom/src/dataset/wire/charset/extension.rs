//! PS3.5 ISO 2022 designations for text values and person names.

use anyhow::{anyhow, bail, Result};
use smallvec::SmallVec;

use super::{decode, encode, repertoire, Repertoire};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Designation {
    code: &'static str,
    sequence: &'static [u8],
    slot: Slot,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Slot {
    G0,
    G1,
    Japanese,
    JapaneseSupplement,
    Korean,
    Chinese,
}

const ASCII: Designation = Designation {
    code: "ISO 2022 IR 6",
    sequence: b"\x1b(B",
    slot: Slot::G0,
};
const SETS: &[Designation] = &[
    ASCII,
    Designation {
        code: "ISO 2022 IR 13",
        sequence: b"\x1b)I",
        slot: Slot::G1,
    },
    Designation {
        code: "ISO 2022 IR 87",
        sequence: b"\x1b$B",
        slot: Slot::Japanese,
    },
    Designation {
        code: "ISO 2022 IR 159",
        sequence: b"\x1b$(D",
        slot: Slot::JapaneseSupplement,
    },
    Designation {
        code: "ISO 2022 IR 100",
        sequence: b"\x1b-A",
        slot: Slot::G1,
    },
    Designation {
        code: "ISO 2022 IR 101",
        sequence: b"\x1b-B",
        slot: Slot::G1,
    },
    Designation {
        code: "ISO 2022 IR 109",
        sequence: b"\x1b-C",
        slot: Slot::G1,
    },
    Designation {
        code: "ISO 2022 IR 110",
        sequence: b"\x1b-D",
        slot: Slot::G1,
    },
    Designation {
        code: "ISO 2022 IR 144",
        sequence: b"\x1b-L",
        slot: Slot::G1,
    },
    Designation {
        code: "ISO 2022 IR 127",
        sequence: b"\x1b-G",
        slot: Slot::G1,
    },
    Designation {
        code: "ISO 2022 IR 126",
        sequence: b"\x1b-F",
        slot: Slot::G1,
    },
    Designation {
        code: "ISO 2022 IR 138",
        sequence: b"\x1b-H",
        slot: Slot::G1,
    },
    Designation {
        code: "ISO 2022 IR 148",
        sequence: b"\x1b-M",
        slot: Slot::G1,
    },
    Designation {
        code: "ISO 2022 IR 149",
        sequence: b"\x1b$)C",
        slot: Slot::Korean,
    },
    Designation {
        code: "ISO 2022 IR 58",
        sequence: b"\x1b$)A",
        slot: Slot::Chinese,
    },
    Designation {
        code: "ISO 2022 IR 166",
        sequence: b"\x1b-T",
        slot: Slot::G1,
    },
];

#[derive(Debug, Clone, Default)]
pub(super) struct Designations(SmallVec<[Designation; 3]>);

impl Designations {
    pub(super) fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub(super) fn insert(&mut self, code: &str) -> Result<()> {
        let entry = SETS
            .iter()
            .find(|entry| entry.code == code)
            .ok_or_else(|| anyhow!("invalid DICOM code-extension repertoire {code:?}"))?;
        if !self.0.contains(entry) {
            self.0.push(*entry);
        }
        Ok(())
    }

    pub(super) fn decode(&self, first: Repertoire, mut bytes: &[u8]) -> Result<String> {
        let mut output = String::new();
        let mut g0 = ASCII;
        let mut g1 = None;
        while let Some((&byte, tail)) = bytes.split_first() {
            if byte == 0x1b {
                let entry = SETS
                    .iter()
                    .find(|entry| bytes.starts_with(entry.sequence))
                    .ok_or_else(|| anyhow!("unknown DICOM ISO 2022 escape sequence"))?;
                if *entry != ASCII && !self.0.contains(entry) {
                    bail!(
                        "DICOM escape designates undeclared repertoire {}",
                        entry.code
                    );
                }
                match entry.slot {
                    Slot::G0 | Slot::Japanese | Slot::JapaneseSupplement => g0 = *entry,
                    _ => g1 = Some(*entry),
                }
                bytes = &bytes[entry.sequence.len()..];
            } else if byte >= 0x80 {
                let (reader, wide) = g1.map_or(Ok((first, false)), |entry| {
                    Ok::<_, anyhow::Error>((
                        repertoire(entry.code)?,
                        matches!(entry.slot, Slot::Korean | Slot::Chinese),
                    ))
                })?;
                let width = if wide { 2 } else { 1 };
                let encoded = bytes
                    .get(..width)
                    .ok_or_else(|| anyhow!("truncated DICOM character"))?;
                output.push_str(&decode(reader, encoded)?);
                bytes = &bytes[width..];
            } else if g0 != ASCII && (0x21..=0x7e).contains(&byte) {
                let second = *tail
                    .first()
                    .filter(|b| (0x21..=0x7e).contains(*b))
                    .ok_or_else(|| anyhow!("truncated DICOM Japanese character"))?;
                let pair = [byte | 0x80, second | 0x80];
                let triple = [0x8f, pair[0], pair[1]];
                let encoded: &[u8] = if g0.slot == Slot::Japanese {
                    &pair
                } else {
                    &triple
                };
                output.push_str(&decode(repertoire(g0.code)?, encoded)?);
                bytes = &tail[1..];
            } else {
                output.push(char::from(byte));
                if byte == b'\\'
                    || byte == b'^'
                    || byte == b'='
                    || [b'\r', b'\n', 0x0c].contains(&byte)
                {
                    g0 = ASCII;
                    g1 = None;
                }
                bytes = tail;
            }
        }
        Ok(output)
    }

    pub(super) fn encode(&self, first: Repertoire, text: &str) -> Result<Vec<u8>> {
        let mut output = Vec::with_capacity(text.len());
        let mut g0 = ASCII;
        let mut g1 = None;
        for ch in text.chars() {
            if ch.is_ascii() {
                if g0 != ASCII {
                    output.extend_from_slice(ASCII.sequence);
                    g0 = ASCII;
                }
                output.push(u8::try_from(u32::from(ch))?);
                if matches!(ch, '\\' | '^' | '=' | '\r' | '\n' | '\x0c') {
                    g1 = None;
                }
                continue;
            }
            let mut buffer = [0; 4];
            let scalar = ch.encode_utf8(&mut buffer);
            if g1.is_none()
                && let Ok(bytes) = encode(first, scalar)
            {
                if bytes.len() == 1 && bytes[0] >= 0x80 {
                    output.extend_from_slice(&bytes);
                    continue;
                }
            }
            let mut written = false;
            for &entry in &self.0 {
                let reader = repertoire(entry.code)?;
                let Ok(bytes) = encode(reader, scalar) else {
                    continue;
                };
                match entry.slot {
                    Slot::Japanese if bytes.len() == 2 && bytes[0] >= 0xa1 && bytes[1] >= 0xa1 => {
                        if g0 != entry {
                            output.extend_from_slice(entry.sequence);
                            g0 = entry;
                        }
                        output.extend(bytes.iter().map(|b| b & 0x7f));
                    }
                    Slot::JapaneseSupplement if bytes.len() == 3 && bytes[0] == 0x8f => {
                        if g0 != entry {
                            output.extend_from_slice(entry.sequence);
                            g0 = entry;
                        }
                        output.extend(bytes[1..].iter().map(|b| b & 0x7f));
                    }
                    Slot::G1 if bytes.len() == 1 && bytes[0] >= 0x80 => {
                        if g1 != Some(entry) {
                            output.extend_from_slice(entry.sequence);
                            g1 = Some(entry);
                        }
                        output.extend_from_slice(&bytes);
                    }
                    Slot::Korean | Slot::Chinese
                        if bytes.len() == 2 && bytes[0] >= 0xa1 && bytes[1] >= 0xa1 =>
                    {
                        if g1 != Some(entry) {
                            output.extend_from_slice(entry.sequence);
                            g1 = Some(entry);
                        }
                        output.extend_from_slice(&bytes);
                    }
                    _ => continue,
                }
                written = true;
                break;
            }
            if !written {
                bail!("no designated DICOM repertoire can encode {ch:?}");
            }
        }
        if g0 != ASCII {
            output.extend_from_slice(ASCII.sequence);
        }
        Ok(output)
    }
}
