//! PackBits run-length decoding used by DICOM RLE Lossless.
//!
//! # Contract
//! `packbits_decode(packbits_encode(S), S.len()) = S` for every byte slice `S`.

use anyhow::{bail, Result};

pub fn packbits_decode(input: &[u8], expected_len: usize) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(expected_len);
    let mut pos = 0usize;
    while pos < input.len() && out.len() < expected_len {
        let header = input[pos] as i8;
        pos += 1;
        if header >= 0 {
            let count = header as usize + 1;
            let end = pos + count;
            if end > input.len() {
                bail!(
                    "PackBits literal run length {} at {} exceeds input length {}",
                    count,
                    pos,
                    input.len()
                );
            }
            out.extend_from_slice(&input[pos..end]);
            pos = end;
        } else if header != i8::MIN {
            let count = (-(header as i16)) as usize + 1;
            if pos >= input.len() {
                bail!("PackBits repeat run at {} has no data byte", pos);
            }
            let byte = input[pos];
            pos += 1;
            out.resize(out.len() + count, byte);
        }
    }
    if out.len() < expected_len {
        bail!(
            "PackBits decoded {} bytes but expected {}",
            out.len(),
            expected_len
        );
    }
    out.truncate(expected_len);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn literal_only_roundtrip(bytes in proptest::collection::vec(any::<u8>(), 1..=128)) {
            let mut encoded = Vec::with_capacity(bytes.len() + 1);
            encoded.push((bytes.len() - 1) as u8);
            encoded.extend_from_slice(&bytes);
            let decoded = packbits_decode(&encoded, bytes.len()).unwrap();
            prop_assert_eq!(decoded, bytes);
        }
    }
}
