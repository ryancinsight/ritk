//! JPEG 2000 tier-2 packet encoder and decoder (ISO 15444-1 Annex B).

pub mod reader;
pub mod writer;

#[cfg(test)]
mod tests;

#[allow(unused_imports)]
pub use reader::{decode_tile_part, BitReader, TileCodingParams, TileComponentSamples};
#[allow(unused_imports)]
pub use writer::encode_tile_part;
#[allow(unused_imports)]
pub(crate) use writer::BitWriter;

use crate::jpeg_2000::subband::Subband;
use crate::jpeg_2000::tag_tree::TagTree;

/// Wavelet transform family selected for a tile (ISO 15444-1 Â§A.6.1, COD
/// `SPcod` wavelet field).  `Reversible` is the integer 5/3 (lossless);
/// `Irreversible` is the floating-point 9/7 (lossy, scalar-quantized).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum WaveletTransform {
    /// 5/3 integer lifting â€” bit-exact, no quantization.
    Reversible,
    /// 9/7 floating-point lifting â€” lossy, dead-zone scalar quantization.
    Irreversible }

// â”€â”€ Code-block partitioning â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Nominal code-block size (COD `xcb = ycb = 4` â†’ 2^(4+2) = 64), shared by the
/// encoder's COD emission and both tier-2 directions.
pub(crate) const CBLK_SIZE: usize = 64;

/// One code-block: its subband, grid position, and rectangle within the band.
#[derive(Clone, Copy, Debug)]
pub(crate) struct CblkRef {
    /// Index into the subband list.
    pub(crate) band: usize,
    /// Grid position within the band's code-block grid.
    pub(crate) gx: usize,
    pub(crate) gy: usize,
    /// Rectangle within the subband (band-local coordinates).
    pub(crate) x0: usize,
    pub(crate) y0: usize,
    pub(crate) w: usize,
    pub(crate) h: usize }

/// Per-band code-block grid dimensions (`ceil(dim / CBLK_SIZE)`).
pub(crate) fn cblk_grid(band_w: usize, band_h: usize) -> (usize, usize) {
    (band_w.div_ceil(CBLK_SIZE), band_h.div_ceil(CBLK_SIZE))
}

/// Enumerate the code-blocks of one subband in raster order.
pub(crate) fn band_cblks(band_idx: usize, band: &Subband) -> Vec<CblkRef> {
    if band.w == 0 || band.h == 0 {
        return Vec::new();
    }
    let (gw, gh) = cblk_grid(band.w, band.h);
    let mut out = Vec::with_capacity(gw * gh);
    for gy in 0..gh {
        for gx in 0..gw {
            let x0 = gx * CBLK_SIZE;
            let y0 = gy * CBLK_SIZE;
            out.push(CblkRef {
                band: band_idx,
                gx,
                gy,
                x0,
                y0,
                w: (band.w - x0).min(CBLK_SIZE),
                h: (band.h - y0).min(CBLK_SIZE) });
        }
    }
    out
}

pub(crate) struct BandTrees {
    pub(crate) incl: TagTree,
    pub(crate) msbs: TagTree }

pub(crate) fn band_trees(bands: &[Subband]) -> Vec<Option<BandTrees>> {
    bands
        .iter()
        .map(|b| {
            if b.w == 0 || b.h == 0 {
                None
            } else {
                let (gw, gh) = cblk_grid(b.w, b.h);
                Some(BandTrees {
                    incl: TagTree::new(gw, gh),
                    msbs: TagTree::new(gw, gh) })
            }
        })
        .collect()
}

// â”€â”€ Lblock byte-count encoding â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Extra length bits beyond the stored `Lblock` for a packet contributing
/// `ncp` passes: `âŒŠlogâ‚‚ ncpâŒ‹` (ISO 15444-1 Â§B.10.7.1).
pub(crate) fn lblock_extra_bits(ncp: u32) -> u8 {
    if ncp == 0 {
        return 0;
    }
    (u32::BITS - ncp.leading_zeros() - 1) as u8
}
