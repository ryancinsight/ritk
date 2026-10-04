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

/// Longest run or literal run one PackBits header can express.
///
/// A header byte carries a count biased by one, and the repeat form is signed,
/// so 128 is the ceiling for both forms (`127 + 1` literal, `-127 + 1` repeat).
const MAX_RUN: usize = 128;

/// PackBits run-length encoding, the write half of the contract above.
///
/// A run of three or more identical bytes becomes a repeat header plus one data
/// byte; everything else accumulates into literal runs. Three is the threshold
/// rather than two because a two-byte repeat costs a header and a data byte to
/// replace two literal bytes, so it encodes no smaller while making the stream
/// harder to read.
///
/// Every emitted literal run is non-empty: a run of three or more identical
/// bytes is always taken by the repeat branch before the literal scan starts, so
/// the scan's "stop at the next run of three" condition cannot fire at offset
/// zero.
pub fn packbits_encode(input: &[u8]) -> Vec<u8> {
    // Worst case every byte is its own literal run: one header byte per run.
    let mut out = Vec::with_capacity(input.len() + input.len() / 64 + 1);
    let mut pos = 0usize;
    while pos < input.len() {
        let byte = input[pos];
        let mut run = 1usize;
        while pos + run < input.len() && input[pos + run] == byte && run < MAX_RUN {
            run += 1;
        }
        if run >= 3 {
            // Repeat header: `-(count - 1)`, biased so a count of 128 is -127.
            out.push((1i32 - run as i32) as i8 as u8);
            out.push(byte);
            pos += run;
            continue;
        }
        let start = pos;
        let mut literal = 0usize;
        while pos < input.len() && literal < MAX_RUN {
            let starts_run = pos + 2 < input.len()
                && input[pos] == input[pos + 1]
                && input[pos + 1] == input[pos + 2];
            if starts_run {
                break;
            }
            pos += 1;
            literal += 1;
        }
        out.push((literal - 1) as u8);
        out.extend_from_slice(&input[start..start + literal]);
    }
    out
}

#[cfg(test)]
mod tests {
    #![expect(clippy::unwrap_used, reason = "ratchet RITK-UNWRAP-1")]
    use super::*;
    use proptest::prelude::*;

    /// A header byte biases its count by one, so a literal run of 128 and a
    /// repeat run of 128 both sit at the encoders' `MAX_RUN` boundary.
    const LONG_RUN: usize = MAX_RUN + 71;

    #[test]
    fn literal_run_header_encodes_one_byte_of_data() {
        let input = b"abcde";
        assert_eq!(packbits_encode(input), [0x04, b'a', b'b', b'c', b'd', b'e']);
    }

    #[test]
    fn repeat_run_header_encodes_one_byte_of_data() {
        // Three identical bytes is the first length the repeat form wins at;
        // the trailing pair is a two-byte run, so it stays in a literal run.
        let input = b"aaabb";
        assert_eq!(packbits_encode(input), [0xfe, b'a', 0x01, b'b', b'b']);
    }

    #[test]
    fn two_identical_bytes_stay_literal() {
        // A two-byte repeat costs a header and a data byte to replace two
        // literal bytes, so the encoder keeps them in the literal run.
        let input = b"aab";
        assert_eq!(packbits_encode(input), [0x02, b'a', b'a', b'b']);
    }

    #[test]
    fn runs_at_the_max_run_length_split_across_headers() {
        let mut input = vec![0xA5; LONG_RUN];
        input.extend_from_slice(b"tail");
        let encoded = packbits_encode(&input);

        // A 199-byte run of one byte is a 128-byte repeat plus a 71-byte one.
        assert_eq!(encoded[0], 0x81, "first repeat header biases 128 to -127");
        assert_eq!(encoded[1], 0xA5);
        assert_eq!(encoded[2], 0xBA, "second repeat header biases 71 to -70");
        assert_eq!(encoded[3], 0xA5);
        assert_eq!(packbits_decode(&encoded, input.len()).unwrap(), input);
    }

    #[test]
    fn literal_run_at_the_max_run_length_is_one_header() {
        let input: Vec<u8> = (0..MAX_RUN).map(|i| i as u8).collect();
        let encoded = packbits_encode(&input);
        assert_eq!(encoded[0], 0x7F, "literal header biases 128 to 127");
        assert_eq!(packbits_decode(&encoded, input.len()).unwrap(), input);
    }

    #[test]
    fn empty_input_encodes_to_nothing() {
        assert!(packbits_encode(&[]).is_empty());
    }

    proptest! {
        /// The module contract: `packbits_decode(packbits_encode(S), S.len()) = S`.
        #[test]
        fn encode_decode_roundtrip(bytes in proptest::collection::vec(any::<u8>(), 0..=512)) {
            let encoded = packbits_encode(&bytes);
            let decoded = packbits_decode(&encoded, bytes.len()).unwrap();
            prop_assert_eq!(decoded, bytes);
        }

        /// Repetitive inputs are what the repeat form exists for, so they are
        /// exercised separately: a random byte string almost never emits one.
        #[test]
        fn roundtrip_with_runs(
            bytes in proptest::collection::vec(any::<u8>(), 0..=256),
            run_byte in any::<u8>(),
            run_len in 1usize..=400,
        ) {
            let mut input = bytes.clone();
            input.extend(std::iter::repeat_n(run_byte, run_len));
            let encoded = packbits_encode(&input);
            let decoded = packbits_decode(&encoded, input.len()).unwrap();
            prop_assert_eq!(decoded, input);
        }
    }
}
