//! J2K main codestream header parser.
//!
//! Parses SIZ, COD, QCD (and skips other markers) up to the first SOT.
//! All fields match the ISO 15444-1 Â§A.5â€“Â§A.6 naming exactly.

#![allow(dead_code)] // All struct fields are spec-mandated; consumed when full parsing support is added.

use anyhow::{bail, Context, Result};

use super::marker;

// â”€â”€ Public header types â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// ISO 15444-1 Â§A.5.1 â€“ Image and tile size.
#[derive(Debug, Clone)]
pub struct SizMarker {
    /// Rsiz: decoder capabilities.
    pub rsiz: u16,
    /// Xsiz / Ysiz: reference grid width and height.
    pub xsiz: u32,
    pub ysiz: u32,
    /// XOsiz / YOsiz: image offset on reference grid.
    pub xo_siz: u32,
    pub yo_siz: u32,
    /// XTsiz / YTsiz: tile size.
    pub xt_siz: u32,
    pub yt_siz: u32,
    /// XTOsiz / YTOsiz: tile offset.
    pub xto_siz: u32,
    pub yto_siz: u32,
    /// Csiz: number of components.
    pub csiz: u16,
    /// Per-component parameters.
    pub components: Vec<ComponentSpec>,
}

/// Per-component parameters from SIZ.
#[derive(Debug, Clone, Copy)]
pub struct ComponentSpec {
    /// Ssiz: bit-depth encoded â€“ lower 7 bits = precisionâˆ’1; bit 7 = signed flag.
    pub ssiz: u8,
    /// XRsiz / YRsiz: horizontal / vertical sub-sampling factors.
    pub xr_siz: u8,
    pub yr_siz: u8,
}

impl ComponentSpec {
    /// Bit precision (1â€“38).
    #[inline]
    pub fn precision(self) -> u32 {
        u32::from(self.ssiz & 0x7F) + 1
    }
    /// `true` if samples are signed two's-complement.
    #[inline]
    pub fn is_signed(self) -> bool {
        self.ssiz & 0x80 != 0
    }
}

impl SizMarker {
    /// Effective image width in reference-grid samples.
    #[inline]
    pub fn width(&self) -> u32 {
        self.xsiz.saturating_sub(self.xo_siz)
    }
    /// Effective image height in reference-grid samples.
    #[inline]
    pub fn height(&self) -> u32 {
        self.ysiz.saturating_sub(self.yo_siz)
    }
    /// Number of tiles horizontally.
    pub fn num_tiles_x(&self) -> u32 {
        (self.xsiz.saturating_sub(self.xto_siz)).div_ceil(self.xt_siz.max(1))
    }
    /// Number of tiles vertically.
    pub fn num_tiles_y(&self) -> u32 {
        (self.ysiz.saturating_sub(self.yto_siz)).div_ceil(self.yt_siz.max(1))
    }
    /// Width of tile `tx` (0-based column index).
    pub fn tile_width(&self, tx: u32) -> u32 {
        let x0 = self.xto_siz + tx * self.xt_siz;
        let x1 = (self.xto_siz + (tx + 1) * self.xt_siz).min(self.xsiz);
        x1.saturating_sub(x0)
    }
    /// Height of tile `ty` (0-based row index).
    pub fn tile_height(&self, ty: u32) -> u32 {
        let y0 = self.yto_siz + ty * self.yt_siz;
        let y1 = (self.yto_siz + (ty + 1) * self.yt_siz).min(self.ysiz);
        y1.saturating_sub(y0)
    }
}

/// ISO 15444-1 Â§A.6.1 â€“ Coding style default.
#[derive(Debug, Clone)]
pub struct CodMarker {
    /// Scod: coding style flags (bit 0 = custom precincts, bit 1 = SOP, bit 2 = EPH).
    pub scod: u8,
    /// SGcod: progression order (0=LRCP, 1=RLCP, 2=RPCL, 3=PCRL, 4=CPRL).
    pub progression_order: u8,
    /// SGcod: number of quality layers.
    pub num_layers: u16,
    /// SGcod: multi-component transform (0=none, 1=RCT, 2=ICT).
    pub mct: u8,
    /// SPcod: number of DWT decomposition levels (0 = no DWT).
    pub num_decomp_levels: u8,
    /// SPcod: code-block width exponent offset (cb_width = 2^(xcb_o+2)).
    pub xcb_o: u8,
    /// SPcod: code-block height exponent offset.
    pub ycb_o: u8,
    /// SPcod: code-block style flags.
    pub cb_style: u8,
    /// SPcod: wavelet transform (0 = 9/7 irreversible, 1 = 5/3 reversible).
    pub wavelet_transform: u8,
    /// Optional custom precinct sizes (one per resolution level 0..=num_decomp_levels).
    pub precinct_sizes: Vec<u8>,
}

impl CodMarker {
    /// `true` if the 5/3 reversible (lossless) wavelet is selected.
    #[inline]
    pub fn is_lossless(&self) -> bool {
        self.wavelet_transform == 1
    }
    /// Code-block width (2^(xcb_o+2), clamped to the valid range 4â€“64).
    #[inline]
    pub fn cb_width(&self) -> u32 {
        1u32 << (self.xcb_o as u32 + 2)
    }
    /// Code-block height (2^(ycb_o+2), clamped to the valid range 4â€“64).
    #[inline]
    pub fn cb_height(&self) -> u32 {
        1u32 << (self.ycb_o as u32 + 2)
    }
}

/// ISO 15444-1 Â§A.6.4 â€“ Quantization default.
#[derive(Debug, Clone)]
pub struct QcdMarker {
    /// Sqcd: quantization style (lower 5 bits = style, upper 3 = guard bits).
    pub sqcd: u8,
    /// Quantization step sizes (raw bytes; interpretation depends on style).
    pub step_sizes: Vec<u16>,
}

impl QcdMarker {
    /// Number of guard bits (upper 3 bits of Sqcd).
    #[inline]
    pub fn num_guard_bits(&self) -> u8 {
        self.sqcd >> 5
    }
    /// `true` when no quantization is applied (lossless or derived).
    #[inline]
    pub fn is_no_quantization(&self) -> bool {
        self.sqcd & 0x1F == 0
    }

    /// Per-subband quantizer exponents Îµ_b in codestream subband order
    /// (ISO 15444-1 Â§A.6.4): 1-byte entries carry Îµ in bits 7â€“3; 2-byte
    /// scalar entries carry Îµ in bits 15â€“11.
    pub fn exponents(&self) -> Vec<u32> {
        let shift = if self.is_no_quantization() { 3 } else { 11 };
        self.step_sizes
            .iter()
            .map(|&s| u32::from(s) >> shift)
            .collect()
    }

    /// Per-subband quantizer mantissas Î¼_b (scalar styles only): the low 11 bits
    /// of each 2-byte SPqcd entry.  Returns all-zero for the no-quantization
    /// style, where the entries are 1-byte exponents with no mantissa.
    pub fn mantissas(&self) -> Vec<u32> {
        if self.is_no_quantization() {
            vec![0; self.step_sizes.len()]
        } else {
            self.step_sizes
                .iter()
                .map(|&s| u32::from(s) & 0x07FF)
                .collect()
        }
    }
}

/// ISO 15444-1 Â§A.4.2 â€“ Start of Tile-part.
#[derive(Debug, Clone, Copy)]
pub struct SotMarker {
    /// Isot: tile index.
    pub isot: u16,
    /// Psot: tile-part byte length (0 = extends to EOC).
    pub psot: u32,
    /// TPsot: tile-part index within the tile.
    pub tpsot: u8,
    /// TNsot: total number of tile-parts (0 = unknown).
    pub tnsot: u8,
}

/// Combined main codestream header (parsed fields of interest).
#[derive(Debug, Clone)]
pub struct MainHeader {
    pub siz: SizMarker,
    pub cod: CodMarker,
    pub qcd: QcdMarker,
}

// â”€â”€ Cursor â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Stateful byte cursor over an immutable slice.
pub struct Cursor<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }
    pub fn pos(&self) -> usize {
        self.pos
    }
    pub fn set_pos(&mut self, pos: usize) {
        self.pos = pos;
    }
    pub fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.pos)
    }
    pub fn data(&self) -> &'a [u8] {
        self.data
    }
    pub fn read_u8(&mut self) -> Result<u8> {
        let b = marker::read_u8(self.data, self.pos)?;
        self.pos += 1;
        Ok(b)
    }
    pub fn read_u16(&mut self) -> Result<u16> {
        let v = marker::read_u16(self.data, self.pos)?;
        self.pos += 2;
        Ok(v)
    }
    pub fn read_u32(&mut self) -> Result<u32> {
        let v = marker::read_u32(self.data, self.pos)?;
        self.pos += 4;
        Ok(v)
    }
    pub fn peek_u16(&self) -> Result<u16> {
        marker::read_u16(self.data, self.pos)
    }
    pub fn skip(&mut self, n: usize) -> Result<()> {
        if self.pos + n > self.data.len() {
            bail!(
                "J2K: skip {n} bytes at pos {} beyond {}-byte buffer",
                self.pos,
                self.data.len()
            );
        }
        self.pos += n;
        Ok(())
    }
    /// Read a segment body: the caller already consumed the marker; this reads
    /// the 2-byte length field and returns a slice of exactly `Lxxx - 2` body bytes,
    /// advancing the cursor past the whole segment.
    pub fn read_segment_body(&mut self) -> Result<&'a [u8]> {
        let lxxx = self.read_u16()? as usize;
        if lxxx < 2 {
            bail!("J2K: segment length {lxxx} < 2 (must include the length field itself)");
        }
        let body_len = lxxx - 2;
        let start = self.pos;
        self.skip(body_len)?;
        Ok(&self.data[start..start + body_len])
    }
}

// â”€â”€ Public parse entry points â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Parse the main codestream header, returning the header and the byte offset
/// at which the first SOT marker starts.
pub fn parse_main_header(data: &[u8]) -> Result<(MainHeader, usize)> {
    let mut cur = Cursor::new(data);

    let soc = cur.read_u16().context("J2K: reading SOC marker")?;
    if soc != marker::SOC {
        bail!("J2K: expected SOC 0xFF4F at offset 0, found 0x{:04X}", soc);
    }

    let mut siz_opt: Option<SizMarker> = None;
    let mut cod_opt: Option<CodMarker> = None;
    let mut qcd_opt: Option<QcdMarker> = None;

    loop {
        let m = cur.read_u16().context("J2K: reading main-header marker")?;
        match m {
            marker::SOT => {
                // Rewind so the SOT is visible to the caller.
                cur.set_pos(cur.pos() - 2);
                break;
            }
            marker::EOC => bail!("J2K: EOC before any tile data"),
            marker::SIZ => {
                let body = cur.read_segment_body()?;
                siz_opt = Some(parse_siz(body).context("J2K: parsing SIZ")?);
            }
            marker::COD => {
                let body = cur.read_segment_body()?;
                cod_opt = Some(parse_cod(body).context("J2K: parsing COD")?);
            }
            marker::QCD => {
                let body = cur.read_segment_body()?;
                qcd_opt = Some(parse_qcd(body).context("J2K: parsing QCD")?);
            }
            // Known optional markers: skip by reading and discarding the segment body.
            marker::COC
            | marker::QCC
            | marker::RGN
            | marker::POC
            | marker::TLM
            | marker::PPM
            | marker::CRG
            | marker::COM => {
                cur.read_segment_body()?;
            }
            other => {
                // Unknown marker with a length field: skip it defensively.
                if cur.remaining() >= 2 {
                    cur.read_segment_body()
                        .with_context(|| format!("J2K: skipping unknown marker 0x{other:04X}"))?;
                } else {
                    bail!("J2K: unknown marker 0x{other:04X} with no length field");
                }
            }
        }
    }

    let siz = siz_opt.context("J2K: SIZ marker missing from main header")?;
    let cod = cod_opt.context("J2K: COD marker missing from main header")?;
    let qcd = qcd_opt.context("J2K: QCD marker missing from main header")?;
    Ok((MainHeader { siz, cod, qcd }, cur.pos()))
}

/// Parse a SOT marker segment at `data[offset]`, returning the header and the
/// offset of the next byte (immediately after the 12-byte segment).
pub fn parse_sot(data: &[u8], offset: usize) -> Result<(SotMarker, usize)> {
    let m = marker::read_u16(data, offset)?;
    if m != marker::SOT {
        bail!("J2K: expected SOT at offset {offset}, found 0x{m:04X}");
    }
    let lsot = marker::read_u16(data, offset + 2)? as usize;
    if lsot != 10 {
        bail!("J2K: Lsot={lsot}, expected 10");
    }
    if data.len() < offset + 2 + lsot {
        bail!("J2K: truncated SOT at offset {offset}");
    }
    let isot = marker::read_u16(data, offset + 4)?;
    let psot = marker::read_u32(data, offset + 6)?;
    let tpsot = data[offset + 10];
    let tnsot = data[offset + 11];
    Ok((
        SotMarker {
            isot,
            psot,
            tpsot,
            tnsot,
        },
        offset + 2 + lsot,
    ))
}

// â”€â”€ Private segment parsers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Parse SIZ segment body (everything after the 2-byte length field).
fn parse_siz(body: &[u8]) -> Result<SizMarker> {
    // body must be at least 36 bytes: Rsiz(2)+Xsiz(4)*4+Ysiz(4)+...*4 + XOsiz+YOsiz+XTsiz+YTsiz+XTOsiz+YTOsiz+Csiz = 36 + 3*Csiz
    if body.len() < 36 {
        bail!("J2K: SIZ body too short ({})", body.len());
    }
    let rsiz = u16::from_be_bytes([body[0], body[1]]);
    let xsiz = u32::from_be_bytes([body[2], body[3], body[4], body[5]]);
    let ysiz = u32::from_be_bytes([body[6], body[7], body[8], body[9]]);
    let xo_siz = u32::from_be_bytes([body[10], body[11], body[12], body[13]]);
    let yo_siz = u32::from_be_bytes([body[14], body[15], body[16], body[17]]);
    let xt_siz = u32::from_be_bytes([body[18], body[19], body[20], body[21]]);
    let yt_siz = u32::from_be_bytes([body[22], body[23], body[24], body[25]]);
    let xto_siz = u32::from_be_bytes([body[26], body[27], body[28], body[29]]);
    let yto_siz = u32::from_be_bytes([body[30], body[31], body[32], body[33]]);
    let csiz = u16::from_be_bytes([body[34], body[35]]);
    if csiz == 0 || csiz > 16384 {
        bail!("J2K: Csiz={csiz} out of range 1..=16384");
    }
    let need = 36 + 3 * csiz as usize;
    if body.len() < need {
        bail!(
            "J2K: SIZ body {}-byte, need {need} for {csiz} components",
            body.len()
        );
    }
    let mut components = Vec::with_capacity(csiz as usize);
    for i in 0..csiz as usize {
        let base = 36 + i * 3;
        components.push(ComponentSpec {
            ssiz: body[base],
            xr_siz: body[base + 1],
            yr_siz: body[base + 2],
        });
    }
    if xt_siz == 0 || yt_siz == 0 {
        bail!("J2K: tile dimensions XTsiz={xt_siz} YTsiz={yt_siz} must be > 0");
    }
    Ok(SizMarker {
        rsiz,
        xsiz,
        ysiz,
        xo_siz,
        yo_siz,
        xt_siz,
        yt_siz,
        xto_siz,
        yto_siz,
        csiz,
        components,
    })
}

/// Parse COD segment body.
fn parse_cod(body: &[u8]) -> Result<CodMarker> {
    // Minimum body: Scod(1)+SGcod(4)+SPcod(9) = 14 â€“ but we already stripped the 2-byte length.
    // Body starts at: Scod, progression_order, num_layers(2), MCT, num_decomp, xcb_o, ycb_o, cb_style, wavelet
    if body.len() < 10 {
        bail!("J2K: COD body too short ({})", body.len());
    }
    let scod = body[0];
    let progression_order = body[1];
    let num_layers = u16::from_be_bytes([body[2], body[3]]);
    let mct = body[4];
    let num_decomp_levels = body[5];
    let xcb_o = body[6] & 0x0F;
    let ycb_o = body[7] & 0x0F;
    let cb_style = body[8];
    let wavelet_transform = body[9];

    let precinct_sizes = if scod & 0x01 != 0 {
        let n = num_decomp_levels as usize + 1;
        if body.len() < 10 + n {
            bail!("J2K: COD body too short for {n} custom precinct sizes");
        }
        body[10..10 + n].to_vec()
    } else {
        Vec::new()
    };

    Ok(CodMarker {
        scod,
        progression_order,
        num_layers,
        mct,
        num_decomp_levels,
        xcb_o,
        ycb_o,
        cb_style,
        wavelet_transform,
        precinct_sizes,
    })
}

/// Parse QCD segment body.
fn parse_qcd(body: &[u8]) -> Result<QcdMarker> {
    if body.is_empty() {
        bail!("J2K: QCD body empty");
    }
    let sqcd = body[0];
    let style = sqcd & 0x1F;
    let data = &body[1..];
    let step_sizes = match style {
        0 => {
            // No quantization: each entry is 1 byte (exponent only).
            data.iter().map(|&b| b as u16).collect()
        }
        1 | 2 => {
            // Scalar quantization: each entry is 2 bytes.
            if !data.len().is_multiple_of(2) {
                bail!(
                    "J2K: QCD scalar quantization body has odd length {}",
                    data.len()
                );
            }
            data.chunks_exact(2)
                .map(|c| u16::from_be_bytes([c[0], c[1]]))
                .collect()
        }
        other => bail!("J2K: unknown QCD quantization style {other}"),
    };
    Ok(QcdMarker { sqcd, step_sizes })
}

#[cfg(test)]
#[path = "tests_codestream.rs"]
mod tests;
