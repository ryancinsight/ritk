//! JPEG marker and frame header parsing.
//!
//! Parses the JPEG bitstream up to and including the first SOS marker,
//! collecting all tables and metadata needed for entropy decoding.
//! Entropy data begins immediately after the SOS segment.
//!
//! # Specification
//! ITU-T T.81 §B.1–§B.3.

use std::fmt;
use std::num::NonZeroU8;

use anyhow::{bail, Context, Result};

use super::huffman::HuffmanTable;

// ─── Marker Constants ─────────────────────────────────────────────────────────

pub(crate) const SOI: u16 = 0xFFD8;
pub(crate) const EOI: u16 = 0xFFD9;
pub(crate) const SOF0: u16 = 0xFFC0; // Baseline DCT
pub(crate) const SOF1: u16 = 0xFFC1; // Extended sequential DCT
pub(crate) const SOF3: u16 = 0xFFC3; // Lossless Huffman
pub(crate) const DHT: u16 = 0xFFC4;
pub(crate) const DQT: u16 = 0xFFDB;
pub(crate) const SOS: u16 = 0xFFDA;
pub(crate) const DRI: u16 = 0xFFDD;

// ─── Data Structures ──────────────────────────────────────────────────────────

/// Quantization table precision (T.81 §B.2.4.1, Pq field).
///
/// `Bits8` (Pq = 0) means 8-bit quantization values; `Bits16` (Pq = 1) means
/// 16-bit values. Baseline DCT (SOF0) requires `Bits8`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub(crate) enum QuantPrecision {
    Bits8 = 0,
    Bits16 = 1,
}

impl TryFrom<u8> for QuantPrecision {
    type Error = u8;

    fn try_from(v: u8) -> Result<Self, u8> {
        match v {
            0 => Ok(Self::Bits8),
            1 => Ok(Self::Bits16),
            other => Err(other),
        }
    }
}

/// JPEG quantization table (T.81 §B.2.4.1).
#[derive(Debug, Clone)]
pub(crate) struct QuantTable {
    pub(crate) precision: QuantPrecision,
    pub(crate) values: [u16; 64], // zigzag order
}

/// Index of a quantization or Huffman table slot (T.81: Tq, Td, Ta).
///
/// JPEG encodes these in a nibble, so the wire form admits 0-15 while a frame
/// holds four slots of each kind. Validating at the parse boundary makes the
/// out-of-range value a typed error there rather than a panic at the table
/// lookup, and lets `Self::index` be the only place the conversion happens.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub(crate) struct TableId(u8);

impl TableId {
    /// Slot count for each table kind (T.81 §B.2.4.1, §B.2.4.2).
    pub(crate) const COUNT: usize = 4;

    /// Position of this slot in a `[Option<_>; TableId::COUNT]` array.
    ///
    /// In range by construction, so the caller indexes without a check.
    #[inline]
    pub(crate) const fn index(self) -> usize {
        self.0 as usize
    }
}

impl TryFrom<u8> for TableId {
    type Error = anyhow::Error;

    fn try_from(v: u8) -> Result<Self> {
        if (v as usize) < Self::COUNT {
            Ok(Self(v))
        } else {
            bail!(
                "JPEG table id {v} is out of range; a frame holds {} slots",
                Self::COUNT
            )
        }
    }
}

impl fmt::Display for TableId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

/// Component sampling factor (T.81 §B.2.2, Hi and Vi).
///
/// The spec bounds these to 1-4. Zero is the dangerous value: it reaches
/// `max_samp / factor` in the scan decoder as a division by zero, so it is
/// rejected here rather than downstream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub(crate) struct SamplingFactor(NonZeroU8);

impl SamplingFactor {
    /// Largest sampling factor T.81 §B.2.2 admits.
    const MAX: u8 = 4;

    /// The factor as a divisor and multiplier, never zero.
    #[inline]
    pub(crate) const fn get(self) -> usize {
        self.0.get() as usize
    }
}

impl TryFrom<u8> for SamplingFactor {
    type Error = anyhow::Error;

    fn try_from(v: u8) -> Result<Self> {
        let factor = NonZeroU8::new(v).with_context(|| {
            "JPEG sampling factor is zero; T.81 §B.2.2 requires 1-4".to_string()
        })?;
        if v > Self::MAX {
            bail!(
                "JPEG sampling factor {v} exceeds the T.81 §B.2.2 maximum of {}",
                Self::MAX
            );
        }
        Ok(Self(factor))
    }
}

/// Per-component frame header entry (T.81 §B.2.2).
#[derive(Debug, Clone)]
pub(crate) struct FrameComponent {
    pub(crate) id: u8,
    pub(crate) h_samp: SamplingFactor,
    pub(crate) v_samp: SamplingFactor,
    pub(crate) quant_id: TableId,
}

/// SOFn frame header (T.81 §B.2.2).
#[derive(Debug, Clone)]
pub(crate) struct SofFrame {
    pub(crate) sof_marker: u16,
    pub(crate) precision: u8,
    pub(crate) height: u16,
    pub(crate) width: u16,
    pub(crate) components: Vec<FrameComponent>,
}

/// Per-component scan header entry (T.81 §B.2.3).
#[derive(Debug, Clone)]
pub(crate) struct ScanComponent {
    pub(crate) id: u8,
    pub(crate) dc_table_id: TableId,
    pub(crate) ac_table_id: TableId,
}

/// SOS scan header (T.81 §B.2.3).
#[derive(Debug, Clone)]
pub(crate) struct SosHeader {
    pub(crate) components: Vec<ScanComponent>,
    /// Ss: start of spectral selection (0 for DC; 1–7 = predictor for lossless).
    pub(crate) ss: u8,
    /// Se: end of spectral selection.
    pub(crate) se: u8,
    /// Ah: successive approximation bit position high.
    pub(crate) ah: u8,
    /// Al: successive approximation bit position low / point transform (lossless).
    pub(crate) al: u8,
}

/// Fully parsed JPEG frame up to the first SOS, with all tables.
#[derive(Debug)]
pub(crate) struct JpegFrameData {
    pub(crate) sof: SofFrame,
    pub(crate) quant: [Option<QuantTable>; TableId::COUNT],
    pub(crate) dc_huff: [Option<HuffmanTable>; TableId::COUNT],
    pub(crate) ac_huff: [Option<HuffmanTable>; TableId::COUNT],
    pub(crate) sos: SosHeader,
    /// Byte offset in the original fragment where entropy data begins.
    pub(crate) scan_data_start: usize,
}

// ─── Parser ───────────────────────────────────────────────────────────────────

/// Bounds-checked forward reader over a JPEG fragment.
///
/// This parser reads bytes it does not control: fragments arrive inside DICOM
/// pixel data and may be truncated mid-segment or malformed outright. Every
/// bounds check lives here, so the parse body is panic-free by construction
/// rather than by per-site review, and a short read is a typed error naming
/// the offset.
struct Cursor<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    #[inline]
    const fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    /// Current byte offset, which is also the scan-data start once the parse
    /// stops at SOS.
    #[inline]
    const fn pos(&self) -> usize {
        self.pos
    }

    #[inline]
    const fn is_empty(&self) -> bool {
        self.pos >= self.data.len()
    }

    /// Byte at the cursor without advancing.
    fn peek(&self) -> Result<u8> {
        self.data
            .get(self.pos)
            .copied()
            .with_context(|| format!("JPEG stream truncated at offset {}", self.pos))
    }

    fn u8(&mut self) -> Result<u8> {
        let byte = self.peek()?;
        self.pos += 1;
        Ok(byte)
    }

    fn u16(&mut self) -> Result<u16> {
        Ok(u16::from_be_bytes(self.array::<2>()?))
    }

    /// Read `N` bytes as a fixed-size array, so the caller keeps the length in
    /// the type instead of re-checking it.
    fn array<const N: usize>(&mut self) -> Result<[u8; N]> {
        let bytes: [u8; N] = self
            .data
            .get(self.pos..self.pos.saturating_add(N))
            .and_then(|s| s.try_into().ok())
            .with_context(|| {
                format!(
                    "JPEG stream truncated reading {N} bytes at offset {}",
                    self.pos
                )
            })?;
        self.pos += N;
        Ok(bytes)
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let bytes = self
            .data
            .get(self.pos..self.pos.saturating_add(n))
            .with_context(|| {
                format!(
                    "JPEG stream truncated reading {n} bytes at offset {}",
                    self.pos
                )
            })?;
        self.pos += n;
        Ok(bytes)
    }

    /// Advance by `n`, refusing to move past the end so a bogus segment length
    /// cannot wrap the offset or park the cursor beyond the buffer.
    fn skip(&mut self, n: usize) -> Result<()> {
        let next = self
            .pos
            .checked_add(n)
            .filter(|&p| p <= self.data.len())
            .with_context(|| {
                format!(
                    "JPEG segment length {n} at offset {} runs past the {}-byte fragment",
                    self.pos,
                    self.data.len()
                )
            })?;
        self.pos = next;
        Ok(())
    }
}

/// Parse a JPEG bitstream and return `JpegFrameData` plus the scan start offset.
pub(crate) fn parse_jpeg(data: &[u8]) -> Result<JpegFrameData> {
    let mut cur = Cursor::new(data);

    if cur
        .u16()
        .context("JPEG fragment is too short to hold a SOI marker")?
        != SOI
    {
        bail!("JPEG fragment does not begin with SOI marker");
    }

    let mut sof: Option<SofFrame> = None;
    let mut quant: [Option<QuantTable>; TableId::COUNT] = [None, None, None, None];
    let mut dc_huff: [Option<HuffmanTable>; TableId::COUNT] = [None, None, None, None];
    let mut ac_huff: [Option<HuffmanTable>; TableId::COUNT] = [None, None, None, None];

    loop {
        // Markers always start with 0xFF; skip fill bytes.
        let first = cur.peek().context("JPEG stream ended before SOS marker")?;
        if first != 0xFF {
            bail!(
                "expected JPEG marker at offset {}, got 0x{first:02X}",
                cur.pos()
            );
        }
        while !cur.is_empty() && cur.peek()? == 0xFF {
            cur.skip(1)?;
        }
        let marker = 0xFF00u16 | u16::from(cur.u8().context("JPEG stream ended in marker prefix")?);

        match marker {
            EOI => bail!("JPEG stream ended (EOI) before SOS"),
            0xFF00 => {} // padding — skip
            // SOI inside segment: skip
            SOI => {}
            // APPn (0xFFE0–0xFFEF) and COM (0xFFFE): skip segment
            m if (0xFFE0..=0xFFEF).contains(&m) || m == 0xFFFE => {
                skip_segment(&mut cur, m)?;
            }
            // DRI: restart interval (2 bytes payload, ignore value)
            DRI => {
                let _len = cur.u16()?;
                cur.skip(2)?; // the DRI value itself
            }
            DQT => {
                let end = segment_end(&mut cur)?;
                while cur.pos() < end {
                    let pq_tq = cur.u8()?;
                    let precision = QuantPrecision::try_from(pq_tq >> 4).map_err(|v| {
                        anyhow::anyhow!("DQT precision {v} is invalid; expected 0 or 1")
                    })?;
                    let id = TableId::try_from(pq_tq & 0x0F).context("DQT")?;
                    let mut values = [0u16; 64];
                    if precision == QuantPrecision::Bits8 {
                        for (value, byte) in values.iter_mut().zip(cur.array::<64>()?) {
                            *value = u16::from(byte);
                        }
                    } else {
                        for value in &mut values {
                            *value = cur.u16()?;
                        }
                    }
                    quant[id.index()] = Some(QuantTable { precision, values });
                }
            }
            DHT => {
                let end = segment_end(&mut cur)?;
                while cur.pos() < end {
                    let tc_th = cur.u8()?;
                    let table_class = tc_th >> 4; // 0=DC/lossless, 1=AC
                    let id = TableId::try_from(tc_th & 0x0F).context("DHT")?;
                    let bits = cur.array::<16>()?;
                    let n_syms: usize = bits.iter().map(|&b| b as usize).sum();
                    let huffval = cur.take(n_syms)?;
                    let table = HuffmanTable::from_bits_huffval(&bits, huffval)?;
                    if table_class == 0 {
                        dc_huff[id.index()] = Some(table);
                    } else {
                        ac_huff[id.index()] = Some(table);
                    }
                }
            }
            SOF0 | SOF1 | SOF3 => {
                let _len = cur.u16()?;
                let precision = cur.u8()?;
                let height = cur.u16()?;
                let width = cur.u16()?;
                let ncomp = cur.u8()? as usize;
                if ncomp > TableId::COUNT {
                    bail!("SOF: too many components: {ncomp}");
                }
                let mut components = Vec::with_capacity(ncomp);
                for _ in 0..ncomp {
                    let [id, samp, quant_id] = cur.array::<3>()?;
                    components.push(FrameComponent {
                        id,
                        h_samp: SamplingFactor::try_from(samp >> 4)
                            .with_context(|| format!("SOF component {id} horizontal"))?,
                        v_samp: SamplingFactor::try_from(samp & 0x0F)
                            .with_context(|| format!("SOF component {id} vertical"))?,
                        quant_id: TableId::try_from(quant_id)
                            .with_context(|| format!("SOF component {id} quantization"))?,
                    });
                }
                sof = Some(SofFrame {
                    sof_marker: marker,
                    precision,
                    height,
                    width,
                    components,
                });
            }
            SOS => {
                let _len = cur.u16()?;
                let ncomp = cur.u8()? as usize;
                let mut scan_comps = Vec::with_capacity(ncomp);
                for _ in 0..ncomp {
                    let [id, tables] = cur.array::<2>()?;
                    scan_comps.push(ScanComponent {
                        id,
                        dc_table_id: TableId::try_from(tables >> 4)
                            .with_context(|| format!("SOS component {id} DC"))?,
                        ac_table_id: TableId::try_from(tables & 0x0F)
                            .with_context(|| format!("SOS component {id} AC"))?,
                    });
                }
                let [ss, se, ah_al] = cur.array::<3>()?;
                let sos = SosHeader {
                    components: scan_comps,
                    ss,
                    se,
                    ah: ah_al >> 4,
                    al: ah_al & 0x0F,
                };
                let frame = sof.context("SOS before SOF in JPEG stream")?;
                return Ok(JpegFrameData {
                    sof: frame,
                    quant,
                    dc_huff,
                    ac_huff,
                    sos,
                    scan_data_start: cur.pos(),
                });
            }
            other => skip_segment(&mut cur, other)?,
        }
    }
}

/// Consume a segment's length field and return the offset one past its end.
///
/// The length counts itself, so a value below two cannot describe a segment and
/// would leave the table loops spinning on an end that precedes the cursor.
fn segment_end(cur: &mut Cursor<'_>) -> Result<usize> {
    let start = cur.pos();
    let len = cur.u16()? as usize;
    if len < 2 {
        bail!("JPEG segment at offset {start} declares length {len}; the field counts itself");
    }
    let end = start + len;
    if end > cur.data.len() {
        bail!(
            "JPEG segment at offset {start} declares length {len}, past the {}-byte fragment",
            cur.data.len()
        );
    }
    Ok(end)
}

/// Skip a marker segment the parser does not interpret.
fn skip_segment(cur: &mut Cursor<'_>, marker: u16) -> Result<()> {
    let len = cur.u16()? as usize;
    let payload = len
        .checked_sub(2)
        .with_context(|| format!("malformed JPEG marker 0x{marker:04X} with length {len}"))?;
    cur.skip(payload)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The lossless 8-bit hand-crafted fixture must parse to a 1×1 SOF3 frame.
    #[test]
    fn parse_lossless_8bit_fixture() {
        let data = crate::jpeg::scan_lossless::tests::lossless_8bit_fixture();
        let frame = parse_jpeg(&data).unwrap();
        assert_eq!(frame.sof.sof_marker, SOF3);
        assert_eq!(frame.sof.precision, 8);
        assert_eq!(frame.sof.width, 1);
        assert_eq!(frame.sof.height, 1);
        assert_eq!(frame.sof.components.len(), 1);
        assert_eq!(frame.sos.ss, 1); // predictor Ra
        assert_eq!(frame.sos.al, 0); // no point transform
                                     // DC table 0 must be present, AC table not needed for lossless
        assert!(frame.dc_huff[0].is_some());
    }

    /// The lossless 16-bit fixture must parse to a 1×1 SOF3 frame with precision 16.
    #[test]
    fn parse_lossless_16bit_fixture() {
        let data = crate::jpeg::scan_lossless::tests::lossless_16bit_fixture();
        let frame = parse_jpeg(&data).unwrap();
        assert_eq!(frame.sof.sof_marker, SOF3);
        assert_eq!(frame.sof.precision, 16);
        assert_eq!(frame.sof.width, 1);
        assert_eq!(frame.sof.height, 1);
    }

    /// Truncation is decided by the header boundary, never by a panic.
    ///
    /// Fragments arrive inside DICOM pixel data, where a short read is the
    /// ordinary failure: a truncated transfer, or a frame boundary mistaken for
    /// a fragment boundary. Sweeping every prefix cuts inside each field the
    /// parser reads, which is what indexing by offset got wrong.
    ///
    /// `parse_jpeg` stops at the end of the SOS header, so the exact contract
    /// is a step function at `scan_data_start`: every shorter prefix errors,
    /// every longer one yields the same header. Asserting the step rather than
    /// "does not panic" makes an over-permissive parser fail here too.
    #[test]
    fn truncation_errors_below_the_header_and_parses_identically_above_it() {
        for fixture in [
            crate::jpeg::scan_lossless::tests::lossless_8bit_fixture(),
            crate::jpeg::scan_lossless::tests::lossless_16bit_fixture(),
        ] {
            let full = parse_jpeg(&fixture).expect("the intact fixture parses");
            let header_end = full.scan_data_start;

            for cut in 0..header_end {
                assert!(
                    parse_jpeg(&fixture[..cut]).is_err(),
                    "prefix of {cut} bytes ends inside the header, but parsed"
                );
            }
            for cut in header_end..=fixture.len() {
                let partial = parse_jpeg(&fixture[..cut])
                    .unwrap_or_else(|e| panic!("prefix of {cut} bytes holds the header: {e}"));
                assert_eq!(partial.scan_data_start, header_end);
                assert_eq!(partial.sof.width, full.sof.width);
                assert_eq!(partial.sof.height, full.sof.height);
                assert_eq!(partial.sof.precision, full.sof.precision);
                assert_eq!(partial.sof.components.len(), full.sof.components.len());
                assert_eq!(partial.sos.ss, full.sos.ss);
                assert_eq!(partial.sos.al, full.sos.al);
            }
        }
    }

    /// Single-byte corruption either fails or yields a structurally valid frame.
    ///
    /// The nibble fields are the hazard: a flipped byte turns a table id into
    /// 15 or a sampling factor into 0, both of which the scan decoder consumes
    /// as an array index and a divisor. Substituting the extremes at every
    /// offset covers each field the header defines without needing a corpus.
    ///
    /// Both halves of the assertion matter. Reaching this code at all requires
    /// no panic, and a frame that comes back must satisfy the invariants the
    /// decoder relies on — so a parser that admitted the bad value rather than
    /// rejecting it fails here instead of downstream.
    #[test]
    fn single_byte_corruption_never_yields_an_invalid_frame() {
        for fixture in [
            crate::jpeg::scan_lossless::tests::lossless_8bit_fixture(),
            crate::jpeg::scan_lossless::tests::lossless_16bit_fixture(),
        ] {
            for offset in 0..fixture.len() {
                for byte in [0x00u8, 0x0F, 0xF0, 0xFF] {
                    let mut corrupt = fixture.clone();
                    corrupt[offset] = byte;
                    let Ok(frame) = parse_jpeg(&corrupt) else {
                        continue;
                    };
                    for fc in &frame.sof.components {
                        assert!(fc.quant_id.index() < TableId::COUNT);
                        assert!((1..=4).contains(&fc.h_samp.get()));
                        assert!((1..=4).contains(&fc.v_samp.get()));
                    }
                    for sc in &frame.sos.components {
                        assert!(sc.dc_table_id.index() < TableId::COUNT);
                        assert!(sc.ac_table_id.index() < TableId::COUNT);
                    }
                    assert!(
                        frame.scan_data_start <= corrupt.len(),
                        "byte {byte:#04X} at {offset} put scan data past the fragment"
                    );
                }
            }
        }
    }

    /// A table id outside the four slots is rejected at the parse boundary.
    ///
    /// JPEG carries these in a nibble, so a corrupt byte reaches 15 while the
    /// frame holds four slots. The scan decoder indexes the slot array with
    /// this value, so admitting it here is an out-of-bounds panic there.
    #[test]
    fn out_of_range_table_ids_are_rejected() {
        for id in 0..=3u8 {
            assert!(TableId::try_from(id).is_ok(), "id {id} names a real slot");
        }
        for id in 4..=15u8 {
            assert!(TableId::try_from(id).is_err(), "id {id} has no slot");
        }
    }

    /// A zero sampling factor is rejected before it divides in the scan decoder.
    #[test]
    fn zero_and_oversized_sampling_factors_are_rejected() {
        assert!(SamplingFactor::try_from(0).is_err());
        for factor in 1..=4u8 {
            assert_eq!(
                SamplingFactor::try_from(factor)
                    .expect("T.81 admits 1-4")
                    .get(),
                factor as usize
            );
        }
        assert!(SamplingFactor::try_from(5).is_err());
    }

    /// A component's declared table ids must survive the parse unchanged.
    ///
    /// Guards the newtype conversions against silently clamping rather than
    /// rejecting, which would decode against the wrong table.
    #[test]
    fn component_table_ids_round_trip_through_the_parse() {
        let data = crate::jpeg::scan_lossless::tests::lossless_8bit_fixture();
        let frame = parse_jpeg(&data).expect("fixture parses");
        assert_eq!(frame.sof.components[0].quant_id.index(), 0);
        assert_eq!(frame.sos.components[0].dc_table_id.index(), 0);
        assert_eq!(frame.sof.components[0].h_samp.get(), 1);
        assert_eq!(frame.sof.components[0].v_samp.get(), 1);
    }
}
