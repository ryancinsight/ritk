//! J2K main codestream header parser.
//!
//! Parses SIZ, COD, QCD (and skips other markers) up to the first SOT.
//! All fields match the ISO 15444-1 §A.5–§A.6 naming exactly.
//! Tile-grid validation and bounds follow
//! [ITU-T T.800 (11/2015)](https://www.itu.int/rec/T-REC-T.800-201511-S),
//! Annex B.3, Equations B-3 through B-11.

#![expect(dead_code, reason = "ratchet RITK-LINT-1")]

use anyhow::{bail, Context, Result};

use super::marker;

/// Most wavelet decomposition levels a COD segment may declare.
///
/// ISO/IEC 15444-1 Table A.11. The value drives `1 << k` shifts in the subband
/// geometry, so it is a capability rather than a hint.
pub(crate) const MAX_DECOMPOSITION_LEVELS: u8 = 32;

// ── Public header types ───────────────────────────────────────────────────────

/// ISO 15444-1 §A.5.1 – Image and tile size.
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
    /// Ssiz: bit-depth encoded – lower 7 bits = precision−1; bit 7 = signed flag.
    pub ssiz: u8,
    /// XRsiz / YRsiz: horizontal / vertical sub-sampling factors.
    pub xr_siz: u8,
    pub yr_siz: u8,
}

impl ComponentSpec {
    /// Bit precision (1–38).
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
    pub fn num_tiles_x(&self) -> u64 {
        u64::from(self.xsiz - self.xto_siz).div_ceil(u64::from(self.xt_siz))
    }
    /// Number of tiles vertically.
    pub fn num_tiles_y(&self) -> u64 {
        u64::from(self.ysiz - self.yto_siz).div_ceil(u64::from(self.yt_siz))
    }

    /// Number of tiles in the image.
    ///
    /// The SOT `Isot` field permits tile indices 0 through 65,534, so a
    /// conforming codestream contains at most 65,535 tiles.
    pub fn num_tiles(&self) -> Result<u32> {
        let count = self
            .num_tiles_x()
            .checked_mul(self.num_tiles_y())
            .filter(|&n| n <= u64::from(u16::MAX))
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "J2K: tile grid {}x{} exceeds the 65535-tile SOT index range",
                    self.num_tiles_x(),
                    self.num_tiles_y()
                )
            })?;
        u32::try_from(count).context("J2K: tile count does not fit u32")
    }

    /// Image-domain bounds for the tile identified by `Isot`.
    ///
    /// The max/min intersection is the normative T.800 B-7 through B-10
    /// definition. Computing in `u64` prevents the marker's `u32` tile size
    /// and index fields from wrapping before the image-domain clamp.
    pub fn tile_bounds(&self, isot: u16) -> Result<TileBounds> {
        let tile_count = self.num_tiles()?;
        let tile_index = u32::from(isot);
        if tile_index >= tile_count {
            bail!(
                "J2K: SOT tile index Isot={tile_index} is outside 0..{}",
                tile_count - 1
            );
        }

        let tiles_x = self.num_tiles_x();
        let tx = u64::from(tile_index) % tiles_x;
        let ty = u64::from(tile_index) / tiles_x;
        let tile_width = u64::from(self.xt_siz);
        let tile_height = u64::from(self.yt_siz);

        let x0 = (u64::from(self.xto_siz) + tx * tile_width).max(u64::from(self.xo_siz));
        let y0 = (u64::from(self.yto_siz) + ty * tile_height).max(u64::from(self.yo_siz));
        let x1 = (u64::from(self.xto_siz) + (tx + 1) * tile_width).min(u64::from(self.xsiz));
        let y1 = (u64::from(self.yto_siz) + (ty + 1) * tile_height).min(u64::from(self.ysiz));

        Ok(TileBounds {
            x0: usize::try_from(x0 - u64::from(self.xo_siz))
                .context("J2K: tile x origin does not fit usize")?,
            y0: usize::try_from(y0 - u64::from(self.yo_siz))
                .context("J2K: tile y origin does not fit usize")?,
            width: usize::try_from(x1 - x0).context("J2K: tile width does not fit usize")?,
            height: usize::try_from(y1 - y0).context("J2K: tile height does not fit usize")?,
        })
    }
}

/// One tile's half-open bounds after intersection with the image area.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TileBounds {
    /// Horizontal offset from the image area's left edge.
    pub x0: usize,
    /// Vertical offset from the image area's top edge.
    pub y0: usize,
    /// Tile width inside the image area.
    pub width: usize,
    /// Tile height inside the image area.
    pub height: usize,
}

/// ISO 15444-1 §A.6.1 – Coding style default.
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
    /// Code-block width (2^(xcb_o+2), clamped to the valid range 4–64).
    #[inline]
    pub fn cb_width(&self) -> u32 {
        1u32 << (self.xcb_o as u32 + 2)
    }
    /// Code-block height (2^(ycb_o+2), clamped to the valid range 4–64).
    #[inline]
    pub fn cb_height(&self) -> u32 {
        1u32 << (self.ycb_o as u32 + 2)
    }
}

/// ISO 15444-1 §A.6.4 – Quantization default.
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

    /// Per-subband quantizer exponents ε_b in codestream subband order
    /// (ISO 15444-1 §A.6.4): 1-byte entries carry ε in bits 7–3; 2-byte
    /// scalar entries carry ε in bits 15–11.
    pub fn exponents(&self) -> Vec<u32> {
        let shift = if self.is_no_quantization() { 3 } else { 11 };
        self.step_sizes
            .iter()
            .map(|&s| u32::from(s) >> shift)
            .collect()
    }

    /// Per-subband quantizer mantissas μ_b (scalar styles only): the low 11 bits
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

/// ISO 15444-1 §A.4.2 – Start of Tile-part.
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

// ── Cursor ────────────────────────────────────────────────────────────────────

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

// ── Public parse entry points ─────────────────────────────────────────────────

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
            marker::COC | marker::QCC | marker::RGN | marker::POC => bail!(
                "J2K: main-header coding override 0x{m:04X} is unsupported by the native decoder"
            ),
            marker::PPM => {
                bail!("J2K: packed main-header packet data (PPM) is unsupported")
            }
            // These markers do not alter packet coding or progression.
            marker::TLM | marker::CRG | marker::COM => {
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
    if isot == u16::MAX {
        bail!("J2K: Isot=65535 is reserved; the maximum tile index is 65534");
    }
    if psot != 0 && psot < 14 {
        bail!("J2K: Psot={psot} is invalid; expected 0 or at least 14 bytes");
    }
    if tpsot == u8::MAX {
        bail!("J2K: TPsot=255 is reserved; the maximum tile-part index is 254");
    }
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

// ── Private segment parsers ───────────────────────────────────────────────────

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
    if xsiz <= xo_siz || ysiz <= yo_siz {
        bail!(
            "J2K: image extent Xsiz={xsiz} Ysiz={ysiz} must exceed \
             XOsiz={xo_siz} YOsiz={yo_siz}"
        );
    }
    if xto_siz > xo_siz || yto_siz > yo_siz {
        bail!(
            "J2K: tile origin XTOsiz={xto_siz} YTOsiz={yto_siz} must not exceed \
             image origin XOsiz={xo_siz} YOsiz={yo_siz}"
        );
    }
    if u64::from(xto_siz) + u64::from(xt_siz) <= u64::from(xo_siz)
        || u64::from(yto_siz) + u64::from(yt_siz) <= u64::from(yo_siz)
    {
        bail!(
            "J2K: first tile XTOsiz={xto_siz} YTOsiz={yto_siz} \
             XTsiz={xt_siz} YTsiz={yt_siz} does not intersect the image origin"
        );
    }
    for (index, component) in components.iter().enumerate() {
        if component.precision() > 38 {
            bail!(
                "J2K: component {index} precision {} exceeds 38 bits",
                component.precision()
            );
        }
        if component.xr_siz == 0 || component.yr_siz == 0 {
            bail!(
                "J2K: component {index} sampling XRsiz={} YRsiz={} must be in 1..=255",
                component.xr_siz,
                component.yr_siz
            );
        }
    }

    let siz = SizMarker {
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
    };
    siz.num_tiles()?;
    Ok(siz)
}

/// Parse COD segment body.
fn parse_cod(body: &[u8]) -> Result<CodMarker> {
    // Minimum body: Scod(1)+SGcod(4)+SPcod(9) = 14 – but we already stripped the 2-byte length.
    // Body starts at: Scod, progression_order, num_layers(2), MCT, num_decomp, xcb_o, ycb_o, cb_style, wavelet
    if body.len() < 10 {
        bail!("J2K: COD body too short ({})", body.len());
    }
    let scod = body[0];
    let progression_order = body[1];
    let num_layers = u16::from_be_bytes([body[2], body[3]]);
    let mct = body[4];
    // Table A.11 bounds SPcod's decomposition levels to 0-32. The value feeds
    // `1usize << k` in the subband geometry, so an unbounded byte shifts past
    // the width of `usize` long before the transform's own check is reached.
    let num_decomp_levels = body[5];
    if num_decomp_levels > MAX_DECOMPOSITION_LEVELS {
        bail!(
            "J2K: COD declares {num_decomp_levels} decomposition levels; Table A.11 allows 0 to \
             {MAX_DECOMPOSITION_LEVELS}"
        );
    }
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
