//! Validate frame geometry, tile-part boundaries and DICOM PS3.5 §10.18 profiles
//! before selecting a packet reader. Entropy bytes are skipped only by Psot.
use super::codestream::{parse_cod, parse_qcd, parse_siz, parse_sot, MainHeader, SizMarker};
use super::image::{marker_segment_end, terminal_eoc_position};
use super::marker;
use crate::PixelLayout;
use anyhow::{bail, ensure, Context, Result};

/// High-Throughput JPEG 2000 transfer syntax declared by a DICOM instance.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HtJ2kProfile {
    Lossless,
    LosslessRpcl,
    Unrestricted,
}

/// Constraints specified by the DICOM transfer syntax around a raw codestream.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Jpeg2000Profile {
    /// ISO/IEC 15444-1 under the lossless-only (.90) or unrestricted (.91) syntax.
    Part1 { lossless_only: bool },
    /// ISO/IEC 15444-15 under one of the three HTJ2K transfer syntaxes.
    HighThroughput(HtJ2kProfile),
}

pub(super) struct Organization {
    pub header: MainHeader,
    pub scalar_packets: bool,
    pub tiles: Vec<TileSequence>,
}

pub(super) fn size_marker(data: &[u8]) -> Result<SizMarker> {
    ensure!(
        marker::read_u16(data, 0)? == marker::SOC && marker::read_u16(data, 2)? == marker::SIZ,
        "JPEG 2000 requires SOC followed by SIZ"
    );
    let end = marker_segment_end(data, 2, marker::SIZ, data.len())?;
    parse_siz(&data[6..end])
}

#[derive(Default)]
pub(super) struct TileSequence {
    ranges: Vec<std::ops::Range<usize>>,
    next: u16,
    total: Option<u8>,
}

impl TileSequence {
    pub(super) fn bytes<'a>(&self, source: &'a [u8]) -> std::borrow::Cow<'a, [u8]> {
        if let [range] = self.ranges.as_slice() {
            return std::borrow::Cow::Borrowed(&source[range.clone()]);
        }
        let mut joined = Vec::with_capacity(self.ranges.iter().map(|r| r.len()).sum());
        for r in &self.ranges {
            joined.extend_from_slice(&source[r.clone()]);
        }
        std::borrow::Cow::Owned(joined)
    }
}

pub(super) fn inspect(
    data: &[u8],
    layout: PixelLayout,
    syntax: Jpeg2000Profile,
) -> Result<Organization> {
    let (profile, lossless_only) = match syntax {
        Jpeg2000Profile::Part1 { lossless_only } => (None, lossless_only),
        Jpeg2000Profile::HighThroughput(profile) => {
            (Some(profile), profile != HtJ2kProfile::Unrestricted)
        }
    };
    ensure!(
        profile.is_none() || layout.bits_allocated >= 8,
        "HTJ2K requires at least eight allocated bits"
    );
    let siz = size_marker(data)?;
    ensure!(
        siz.width() as usize == layout.cols && siz.height() as usize == layout.rows,
        "J2K: image dimensions {}×{} do not match layout {}×{}",
        siz.width(),
        siz.height(),
        layout.cols,
        layout.rows
    );
    ensure!(
        usize::from(siz.csiz) == layout.samples_per_pixel,
        "J2K: Csiz={} does not match layout samples_per_pixel={}",
        siz.csiz,
        layout.samples_per_pixel
    );
    ensure!(
        matches!(siz.csiz, 1 | 3),
        "DICOM JPEG 2000 requires one or three components"
    );
    ensure!(
        siz.components
            .iter()
            .all(|c| c.xr_siz == 1 && c.yr_siz == 1),
        "DICOM JPEG 2000 component uses unsupported sampling; requires 1x1 sampling"
    );
    let high_throughput = siz.rsiz & 0x4000 != 0;
    ensure!(
        high_throughput == profile.is_some(),
        "JPEG 2000 SIZ capabilities disagree with transfer syntax"
    );
    let end = terminal_eoc_position(data)?;
    let mut pos = marker_segment_end(data, 2, marker::SIZ, end)?;
    let mut tile_end = None;
    let mut sequences = Vec::<TileSequence>::new();
    let mut native = siz.xo_siz == 0 && siz.yo_siz == 0;
    let mut cap = false;
    let mut cap_seen = false;
    let mut tlm = super::tile_lengths::TileLengths::default();
    let mut part_lengths = Vec::new();
    let mut default_qcd = None;
    let mut default_cod = None;
    let mut tile_index = 0usize;
    while pos < end {
        let code = marker::read_u16(data, pos)?;
        let limit = tile_end.unwrap_or(end);
        if code == marker::SOT {
            ensure!(tile_end.is_none(), "SOT before SOD");
            ensure!(
                default_cod.is_some() && default_qcd.is_some(),
                "incomplete JPEG 2000 main COD or QCD before SOT"
            );
            if sequences.is_empty() {
                sequences.resize_with(usize::try_from(siz.num_tiles()?)?, TileSequence::default);
            }
            let (part, after) = parse_sot(data, pos)?;
            let entry = sequences
                .get_mut(usize::from(part.isot))
                .with_context(|| format!("SOT tile index Isot={} outside tile grid", part.isot))?;
            ensure!(
                u16::from(part.tpsot) == entry.next,
                "out-of-order or duplicate tile-part in tile {}",
                part.isot
            );
            entry.next += 1;
            if part.tnsot != 0 {
                ensure!(
                    entry.total.is_none_or(|n| n == part.tnsot) && part.tpsot < part.tnsot,
                    "inconsistent TNsot"
                );
                entry.total = Some(part.tnsot);
            }
            tile_index = usize::from(part.isot);
            let stop = if part.psot == 0 {
                end
            } else {
                pos.checked_add(usize::try_from(part.psot)?)
                    .context("Psot overflow")?
            };
            ensure!(
                stop <= end && after + 2 <= stop,
                "J2K: tile-part Psot={} outside bounds or beyond terminal EOC",
                part.psot
            );
            part_lengths.push((part.isot, stop - pos));
            tile_end = Some(stop);
            pos = after;
            continue;
        }
        if code == marker::SOD {
            ensure!(tile_end.is_some(), "SOD outside tile-part");
            ensure!(
                pos + 2 <= limit,
                "tile-header marker at byte {pos} extends beyond the tile-part end"
            );
            sequences[tile_index].ranges.push(pos + 2..limit);
            pos = tile_end.take().context("SOD outside tile-part")?;
            continue;
        }
        if pos + 2 > limit || matches!(code, marker::SOC | marker::SIZ | marker::EOC) {
            bail!("unexpected marker 0x{code:04X} at byte {pos}")
        }
        let next = marker_segment_end(data, pos, code, limit)?;
        let body = &data[pos + 4..next];
        match code {
            marker::COD => {
                let c = parse_cod(body).with_context(|| format!("invalid COD at byte {pos}"))?;
                ensure!(
                    c.progression_order <= 4
                        && c.num_layers > 0
                        && c.wavelet_transform <= 1
                        && c.mct <= 1
                        && (c.mct == 0 || siz.csiz == 3),
                    "invalid COD progression, layers, MCT or wavelet transform"
                );
                if lossless_only {
                    ensure!(
                        c.wavelet_transform == 1,
                        "lossless transfer syntax requires reversible wavelet"
                    );
                }
                if profile == Some(HtJ2kProfile::LosslessRpcl) {
                    ensure!(
                        c.progression_order == 2,
                        "HTJ2K .202 requires RPCL progression"
                    );
                    ensure!(
                        u64::from(siz.width().min(siz.height()))
                            .div_ceil(1u64 << c.num_decomp_levels)
                            <= 64,
                        "HTJ2K .202 base resolution exceeds 64"
                    );
                }
                native &= tile_end.is_none()
                    && c.scod == 0
                    && c.xcb_o == 4
                    && c.ycb_o == 4
                    && ((c.cb_style == 0 && !high_throughput)
                        || (c.cb_style == 0x40
                            && high_throughput
                            && c.wavelet_transform == 1
                            && c.num_layers == 1));
                let alignment = 1u64 << c.num_decomp_levels;
                native &= (siz.num_tiles_x() == 1
                    || u64::from(siz.xt_siz).is_multiple_of(alignment))
                    && (siz.num_tiles_y() == 1 || u64::from(siz.yt_siz).is_multiple_of(alignment))
                    && siz.width().max(siz.height()) <= 32768;
                if tile_end.is_none() {
                    ensure!(default_cod.is_none(), "duplicate main COD");
                    default_cod = Some(c);
                }
            }
            marker::QCD | marker::QCC => {
                if code == marker::QCD && tile_end.is_none() {
                    ensure!(default_qcd.is_none(), "duplicate main QCD");
                    default_qcd = Some(parse_qcd(body)?);
                }
                let index = usize::from(code == marker::QCC);
                let style = body.get(index).context("empty quantization marker")? & 31;
                if lossless_only {
                    ensure!(
                        style == 0,
                        "lossless transfer syntax requires reversible quantization"
                    );
                }
                native &= code == marker::QCD && tile_end.is_none() && style != 1;
            }
            0xff50 => {
                let flags = marker::read_u32(body, 0)?;
                ensure!(
                    tile_end.is_none()
                        && !cap_seen
                        && body.len() == 4 + 2 * flags.count_ones() as usize,
                    "invalid JPEG 2000 CAP"
                );
                cap_seen = true;
                cap = flags & (1 << 17) != 0;
            }
            marker::TLM => {
                ensure!(
                    tile_end.is_none() && body.len() >= 2,
                    "invalid tile length marker"
                );
                tlm.add(body)?;
            }
            marker::COC => {
                ensure!(
                    body.len() >= 7
                        && u16::from(body[0]) < siz.csiz
                        && body[2] <= super::codestream::MAX_DECOMPOSITION_LEVELS,
                    "invalid COC component or coding style"
                );
                if lossless_only {
                    ensure!(
                        body[6] == 1,
                        "lossless transfer syntax requires reversible component wavelet"
                    );
                }
                if profile == Some(HtJ2kProfile::LosslessRpcl) {
                    ensure!(
                        u64::from(siz.width().min(siz.height())).div_ceil(1u64 << body[2]) <= 64,
                        "HTJ2K .202 component base resolution exceeds 64"
                    );
                }
                native = false;
            }
            marker::POC => {
                ensure!(
                    !body.is_empty() && body.len().is_multiple_of(7),
                    "invalid JPEG 2000 POC entries"
                );
                if profile == Some(HtJ2kProfile::LosslessRpcl) {
                    ensure!(
                        body.chunks_exact(7).all(|r| r[6] == 2),
                        "HTJ2K .202 requires RPCL in POC"
                    );
                }
                native = false;
            }
            marker::PPM | marker::PPT | marker::RGN => native = false,
            marker::COM | marker::CRG | marker::PLT | 0xff57 => {}
            other => {
                ensure!(
                    tile_end.is_none(),
                    "unsupported tile-header marker 0x{other:04X}"
                );
                native = false;
            }
        }
        pos = next;
    }
    ensure!(
        !sequences.is_empty() && tile_end.is_none(),
        "incomplete JPEG 2000 main or tile header"
    );
    for (i, s) in sequences.iter().enumerate() {
        ensure!(
            s.next > 0 && s.total.is_none_or(|n| u16::from(n) == s.next),
            "EOC reached before tile {i} of {} or its declared TNsot={} tile-parts",
            sequences.len(),
            s.total.unwrap_or(0)
        );
    }
    ensure!(
        cap == high_throughput,
        "HTJ2K CAP and SIZ signaling disagree"
    );
    tlm.verify(&part_lengths, profile == Some(HtJ2kProfile::LosslessRpcl))?;
    let cod = default_cod.context("missing main COD")?;
    let qcd = default_qcd.context("missing main QCD")?;
    if native && siz.csiz == 3 {
        for (i, tile) in sequences.iter().enumerate() {
            let b = siz.tile_bounds(u16::try_from(i)?)?;
            super::packet::reader::validate_component_packets(
                &tile.bytes(data),
                b.width,
                b.height,
                &cod,
                3,
            )?;
        }
    }
    Ok(Organization {
        scalar_packets: native && siz.csiz == 1,
        header: MainHeader { siz, cod, qcd },
        tiles: sequences,
    })
}
