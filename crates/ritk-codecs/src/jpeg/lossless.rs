//! JPEG lossless (SOF3) encoding.
//!
//! # What SOF3 is
//!
//! SOF3 carries no DCT and no quantisation: each sample is predicted from its
//! already-reconstructed neighbours and the *difference* is entropy-coded. That
//! makes it the only JPEG mode in which a decoded sample is bit-exact, which is
//! why DICOM pairs it with the `JpegLossless*` transfer syntaxes -- and why a
//! lossy mode would be the wrong thing to reach for when a study is archived.
//!
//! # Prediction
//!
//! Each sample is predicted from one of two reconstructed neighbours the standard
//! calls `Ra` (the sample to the left, same row) and `Rb` (the sample directly
//! above). The scan header's `Ss` field selects:
//!
//! - `Ss = 1` -- start with `Ra`, switch to `Rb` as soon as the first
//!   difference has a nonzero category. *First-order prediction*, the
//!   `JpegLosslessFirstOrderPrediction` transfer syntax.
//! - `Ss = 0` -- use `Rb` for the whole scan. *Non-hierarchical prediction*,
//!   the `JpegLosslessNonHierarchical` transfer syntax.
//!
//! Neither neighbour exists for the first sample of the first row, so both
//! predictors are the standard's `1 << (precision - 1)` seed there.
//!
//! # Why the Huffman table is built from the image
//!
//! The Annex K default tables are *incomplete*: neither carries a code for
//! magnitude categories 11 through 16, and those categories are reachable. The
//! first sample of a scan is predicted from the seed above, so a 16-bit image
//! whose first sample sits far from 32768 yields a 16-bit difference on the very
//! first code -- a fixed-table encoder cannot encode that image at all.
//!
//! A SOF3 stream carries its own `DHT` segments and decoders read them (this
//! workspace's decoder installs whatever tables the stream declares), so the
//! table is built from the image's own category histogram. Every category
//! becomes codable, and the codes come out shorter than a uniform assignment for
//! any image whose differences are mostly small -- which every real image is.

use anyhow::{bail, Context, Result};

/// Longest code a JPEG Huffman table may use.
const MAX_CODE_BITS: u8 = 16;

/// Number of magnitude categories: `SSSS = 0` plus `1..=16`.
const CATEGORY_COUNT: usize = 17;

/// Magnitude categories that get a short code. Four bits leaves room for 16
/// codes of that length; the remaining categories take five.
const SHORT_CODE_SLOTS: usize = 9;

/// The predictor a scan header selects through `Ss`.
///
/// `Ss` is a *static* selector in T.81, not a starting mode that switches: the
/// standard's selection table maps 1 to `Ra` (the sample to the left), 2 to `Rb`
/// (the sample above), 3 to the upper-left sample, and so on. This workspace's
/// decoder implements that table literally, so the encoder mirrors it rather
/// than implementing a switching rule the decoder does not have.
///
/// | Variant | `Ss` | DICOM transfer syntax |
/// |---|---|---|
/// | [`Left`](Self::Left) | 1 | `JpegLosslessFirstOrderPrediction` |
/// | [`Above`](Self::Above) | 2 | -- |
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JpegLosslessPrediction {
    /// `Ss = 1`: predict from `Ra`, the reconstructed sample to the left.
    Left,
    /// `Ss = 2`: predict from `Rb`, the reconstructed sample above.
    Above,
}

impl JpegLosslessPrediction {
    const fn marker_class(self) -> u8 {
        match self {
            Self::Above => 2,
            Self::Left => 1,
        }
    }
}

/// A canonical Huffman code table, in both the forms the format needs: `BITS`
/// and `HUFFVAL` for a `DHT` segment, and `(length, code)` per symbol here.
struct HuffmanTable {
    bits: [u8; MAX_CODE_BITS as usize],
    values: Vec<u8>,
    codes: Vec<(u8, u16)>,
}

impl HuffmanTable {
    /// Builds a canonical table from per-symbol code lengths.
    ///
    /// A symbol with length `0` is absent from the table, which is how JPEG
    /// encodes "this symbol never occurs".
    fn from_lengths(lengths: &[u8; CATEGORY_COUNT]) -> Self {
        let mut bits = [0u8; MAX_CODE_BITS as usize];
        let mut values = Vec::with_capacity(CATEGORY_COUNT);
        for (symbol, &length) in lengths.iter().enumerate() {
            if length == 0 {
                continue;
            }
            debug_assert!(
                length <= MAX_CODE_BITS,
                "code length {length} exceeds the maximum"
            );
            bits[usize::from(length) - 1] += 1;
            values.push(symbol as u8);
        }
        assert!(
            bits.iter().sum::<u8>() as usize == values.len(),
            "BITS and HUFFVAL disagree: {} codes for {} symbols",
            bits.iter().sum::<u8>(),
            values.len()
        );

        // Canonical assignment: codes run in symbol order within each length, and
        // each length starts one place past the previous block shifted left, so
        // every longer code extends some shorter one.
        let mut codes = vec![(0u8, 0u16); 256];
        let mut code: u16 = 0;
        let mut value_index = 0usize;
        for length in 1..=MAX_CODE_BITS {
            for _ in 0..bits[usize::from(length) - 1] {
                codes[usize::from(values[value_index])] = (length, code);
                code += 1;
                value_index += 1;
            }
            if length < MAX_CODE_BITS {
                code <<= 1;
            }
        }
        Self {
            bits,
            values,
            codes,
        }
    }

    /// Builds a table from a category histogram.
    ///
    /// The `SHORT_CODE_SLOTS` most frequent categories get four-bit codes and the
    /// rest five, which is a valid canonical assignment -- `9/16 + 8/32 < 1` --
    /// rather than a full Huffman tree. Ties break by category index so the same
    /// image always produces the same stream.
    fn from_histogram(histogram: &[u32; CATEGORY_COUNT]) -> Self {
        let mut order: Vec<usize> = (0..CATEGORY_COUNT)
            .filter(|&symbol| histogram[symbol] > 0)
            .collect();
        if order.is_empty() {
            order.push(0);
        }
        order.sort_by(|&left, &right| {
            histogram[right]
                .cmp(&histogram[left])
                .then_with(|| left.cmp(&right))
        });

        let mut lengths = [0u8; CATEGORY_COUNT];
        for (rank, &symbol) in order.iter().enumerate() {
            lengths[symbol] = if rank < SHORT_CODE_SLOTS { 4 } else { 5 };
        }
        Self::from_lengths(&lengths)
    }
}

/// Bit-oriented writer with the byte stuffing entropy-coded data requires.
struct EntropyWriter {
    out: Vec<u8>,
    current: u8,
    filled: u8,
}

impl EntropyWriter {
    fn new() -> Self {
        Self {
            out: Vec::new(),
            current: 0,
            filled: 0,
        }
    }

    fn write_bits(&mut self, value: u16, length: u8) {
        debug_assert!(
            length <= MAX_CODE_BITS,
            "code length {length} exceeds the maximum"
        );
        for index in (0..length).rev() {
            let bit = ((value >> index) & 1) as u8;
            self.current = (self.current << 1) | bit;
            self.filled += 1;
            if self.filled == 8 {
                self.flush_byte();
            }
        }
    }

    fn flush_byte(&mut self) {
        let byte = self.current;
        self.out.push(byte);
        // A 0xFF in entropy-coded data is followed by a stuffed 0x00 so a decoder
        // does not read it as a marker. Without this the stream becomes
        // undecodable the moment a run of 1 bits happens to line up.
        if byte == 0xFF {
            self.out.push(0x00);
        }
        self.current = 0;
        self.filled = 0;
    }

    /// Pads the final partial byte with 1 bits, as the standard requires.
    fn finish(mut self) -> Vec<u8> {
        if self.filled > 0 {
            let pad = 8 - self.filled;
            self.current = (self.current << pad) | (((1u16 << pad) - 1) as u8);
            self.filled = 8;
            self.flush_byte();
        }
        self.out
    }
}

/// The standard's SSSS: the bit length of the difference's magnitude, with zero
/// mapped to zero.
fn magnitude_category(difference: i32) -> u8 {
    if difference == 0 {
        return 0;
    }
    let magnitude = difference.unsigned_abs();
    (32 - magnitude.leading_zeros()) as u8
}

fn push_segment_length(out: &mut Vec<u8>, value: u16) {
    out.extend_from_slice(&value.to_be_bytes());
}

fn push_dht(out: &mut Vec<u8>, table: &HuffmanTable) {
    let length = 2 + 1 + MAX_CODE_BITS as usize + table.values.len();
    out.push(0xFF);
    out.push(0xC4);
    push_segment_length(out, length as u16);
    // Class 0 (DC), table id 0. The same table is selected for both classes by
    // the scan header, so one segment carries every category.
    out.push(0x00);
    out.extend_from_slice(&table.bits);
    out.extend_from_slice(&table.values);
}

/// The differences a scan produces, with the predictor state that produced them.
///
/// Prediction is deterministic and depends only on already-reconstructed
/// samples, so this can be run twice over the same input with identical results.
fn scan_differences(
    samples: &[u16],
    rows: usize,
    cols: usize,
    precision: u8,
    prediction: JpegLosslessPrediction,
) -> Vec<i32> {
    let seed = 1_i32 << (precision - 1);
    let mut previous = vec![0i32; cols];
    let mut differences = Vec::with_capacity(samples.len());

    for row in 0..rows {
        let base = row * cols;
        for column in 0..cols {
            let sample = i32::from(samples[base + column]);
            // Mirrors the decoder's branch order exactly: the very first sample
            // seeds, the rest of the first row predicts from the left, the first
            // column of every later row predicts from above, and only then does
            // `Ss` choose.
            let predicted = if row == 0 && column == 0 {
                seed
            } else if row == 0 {
                i32::from(samples[base + column - 1])
            } else if column == 0 {
                previous[0]
            } else {
                match prediction {
                    JpegLosslessPrediction::Left => i32::from(samples[base + column - 1]),
                    JpegLosslessPrediction::Above => previous[column],
                }
            };
            differences.push(sample - predicted);
            previous[column] = sample;
        }
    }
    differences
}

/// Encodes `samples` as a grayscale SOF3 codestream.
///
/// `samples` is row-major `[rows, cols]` at `precision` bits (2..=16), and the
/// result is a complete JPEG stream: SOI, SOF3, the `DHT` derived from this
/// image, SOS, the entropy-coded scan, EOI.
///
/// # Errors
///
/// Returns an error when the sample count does not match `rows * cols`, when
/// `precision` is outside 2..=16, or when any dimension is zero.
pub fn encode_grayscale_jpeg_lossless(
    samples: &[u16],
    rows: usize,
    cols: usize,
    precision: u8,
    prediction: JpegLosslessPrediction,
) -> Result<Vec<u8>> {
    if !(2..=16).contains(&precision) {
        bail!("JPEG lossless precision {precision} is outside the standard's 2..=16");
    }
    if rows == 0 || cols == 0 {
        bail!("JPEG lossless dimensions {cols}x{rows} must both be nonzero");
    }
    let expected = rows
        .checked_mul(cols)
        .context("JPEG lossless frame size overflow")?;
    if samples.len() != expected {
        bail!(
            "JPEG lossless encoder received {} samples for a {cols}x{rows} frame",
            samples.len()
        );
    }

    // Pass 1: differences, then the histogram that sizes the table.
    let differences = scan_differences(samples, rows, cols, precision, prediction);
    let mut histogram = [0u32; CATEGORY_COUNT];
    for &difference in &differences {
        histogram[usize::from(magnitude_category(difference))] += 1;
    }
    let table = HuffmanTable::from_histogram(&histogram);

    let mut out = Vec::with_capacity(expected * 2 + 64);
    out.extend_from_slice(&[0xFF, 0xD8]); // SOI

    // SOF3: lossless, Huffman. One component, no sampling, no quantisation.
    out.extend_from_slice(&[0xFF, 0xC3]);
    push_segment_length(&mut out, 11);
    out.push(precision);
    push_segment_length(&mut out, rows as u16);
    push_segment_length(&mut out, cols as u16);
    out.push(1); // components
    out.push(1); // component id
    out.push(0x11); // 1x1 sampling
    out.push(0); // no quantisation table

    push_dht(&mut out, &table);

    // SOS. Td/Ta both select table 0; Ss carries the predictor class; Se is 0;
    // Al is the point transform, precision - 1.
    out.extend_from_slice(&[0xFF, 0xDA]);
    push_segment_length(&mut out, 8);
    out.push(1); // components in scan
    out.push(1); // component selector
    out.push(0x00); // Td = 0, Ta = 0
    out.push(prediction.marker_class());
    out.push(0); // Se
                 // Ah/Al carries the point transform (Pt) in its low nibble, and Pt is a
                 // *subtraction* from the frame precision: a reader reconstructs at
                 //  and rejects anything above that bound. Writing
                 //  here -- which reads like the sample width -- would tell a
                 // decoder the stream is 1-bit and every sample exceeds the bound. No point
                 // transform is what an encoder that stores full-precision samples means.
    out.push(0); // Ah = 0, Al = 0: no point transform

    // Pass 2: emit. The table is complete over all 17 categories, so no symbol
    // can miss.
    let mut writer = EntropyWriter::new();
    for &difference in &differences {
        let category = magnitude_category(difference);
        let (length, code) = table.codes[usize::from(category)];
        assert!(
            length > 0,
            "category {category} has no code, so the image is not fully codable"
        );
        writer.write_bits(code, length);
        if category > 0 {
            write_magnitude(&mut writer, difference, category);
        }
    }

    out.extend_from_slice(&writer.finish());
    out.extend_from_slice(&[0xFF, 0xD9]); // EOI
    Ok(out)
}

/// Writes the `category` low bits of `difference`.
///
/// A negative difference is stored one's-complement style as `diff - 1`, except
/// for the one value where that would not fit: `-32768` is written as all ones
/// for its width, which the standard special-cases explicitly.
fn write_magnitude(writer: &mut EntropyWriter, difference: i32, category: u8) {
    if category == 16 {
        // T.81 Table H.2 gives category 16 the single value -32768 and codes it
        // with no magnitude bits at all. Writing sixteen here would desynchronise
        // every sample after it.
        return;
    }
    let encoded = if difference > 0 {
        difference as u16
    } else {
        (difference - 1) as u16
    };
    writer.write_bits(encoded & ((1u16 << category) - 1), category);
}

#[cfg(test)]
#[path = "tests_lossless.rs"]
mod tests;
