//! Subband geometry for the Mallat DWT layout (ISO 15444-1 Â§B.5).
//!
//! For a tile of `width Ã— height` with `N` decomposition levels the codestream
//! carries `3N + 1` subbands in resolution order:
//! `LL_N, HL_N, LH_N, HH_N, HL_{Nâˆ’1}, â€¦, HH_1` â€” the same order used for the
//! QCD `SPqcd` entries and for the LRCP packet sequence (resolution `r = 0` is
//! `LL_N`; resolution `r â‰¥ 1` carries the three subbands of level `N âˆ’ r + 1`).

use super::ebcot::SubbandOrientation;
use super::wavelet::ceil_div_pow2;

/// One subband: its rectangle in the Mallat coefficient plane, the ZC-context
/// orientation, and the reversible-transform log2 gain.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Subband {
    /// ZC context-table selector (Table D.1 H/V roles).
    pub orient: SubbandOrientation,
    /// Top-left corner in the Mallat plane.
    pub x0: usize,
    pub y0: usize,
    /// Subband dimensions (may be 0 for degenerate tile sizes).
    pub w: usize,
    pub h: usize,
    /// log2 coefficient gain of the 5/3 reversible transform
    /// (LL: 0, HL/LH: 1, HH: 2) â€” ISO 15444-1 Â§E.1.1.
    pub gain: u32 }

/// Build the subband list for `width Ã— height` with `num_levels` decomposition
/// levels, in codestream order (see module docs). For `num_levels == 0` the
/// single entry is the LLâ‚€ band covering the whole tile.
pub(crate) fn subband_layout(width: usize, height: usize, num_levels: u8) -> Vec<Subband> {
    let n = u32::from(num_levels);
    let mut bands = Vec::with_capacity(3 * num_levels as usize + 1);

    // LL_N: top-left ceil(w/2^N) Ã— ceil(h/2^N).
    bands.push(Subband {
        orient: SubbandOrientation::LlOrLh,
        x0: 0,
        y0: 0,
        w: ceil_div_pow2(width, n),
        h: ceil_div_pow2(height, n),
        gain: 0 });

    // Levels from coarsest (N) to finest (1): HL, LH, HH.
    for lvl in (1..=n).rev() {
        let prev_w = ceil_div_pow2(width, lvl - 1);
        let prev_h = ceil_div_pow2(height, lvl - 1);
        let sn_x = prev_w.div_ceil(2);
        let sn_y = prev_h.div_ceil(2);
        let dn_x = prev_w - sn_x;
        let dn_y = prev_h - sn_y;

        // HL: horizontally high-pass (right half), vertically low-pass â€”
        // ZC contexts use the H/V-swapped table.
        bands.push(Subband {
            orient: SubbandOrientation::Hl,
            x0: sn_x,
            y0: 0,
            w: dn_x,
            h: sn_y,
            gain: 1 });
        // LH: horizontally low-pass, vertically high-pass (bottom half) â€”
        // shares the LL/LH ZC table.
        bands.push(Subband {
            orient: SubbandOrientation::LlOrLh,
            x0: 0,
            y0: sn_y,
            w: sn_x,
            h: dn_y,
            gain: 1 });
        // HH: high-pass in both directions.
        bands.push(Subband {
            orient: SubbandOrientation::Hh,
            x0: sn_x,
            y0: sn_y,
            w: dn_x,
            h: dn_y,
            gain: 2 });
    }

    bands
}

/// Subband index range `[start, end)` belonging to resolution `r` (LRCP packet
/// order): resolution 0 is `[0, 1)`; resolution `r â‰¥ 1` is `[3r âˆ’ 2, 3r + 1)`.
#[inline]
pub(crate) fn resolution_band_range(r: usize) -> (usize, usize) {
    if r == 0 {
        (0, 1)
    } else {
        (3 * r - 2, 3 * r + 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_zero_levels_is_whole_tile() {
        let bands = subband_layout(7, 5, 0);
        assert_eq!(bands.len(), 1);
        assert_eq!((bands[0].w, bands[0].h), (7, 5));
        assert_eq!(bands[0].gain, 0);
    }

    #[test]
    fn layout_two_levels_dimensions_tile_exactly() {
        // 13Ã—9, 2 levels: level-1 split: sn=(7,5) dn=(6,4); level-2 on 7Ã—5:
        // sn=(4,3) dn=(3,2).
        let bands = subband_layout(13, 9, 2);
        assert_eq!(bands.len(), 7);
        // LL_2
        assert_eq!(
            (bands[0].x0, bands[0].y0, bands[0].w, bands[0].h),
            (0, 0, 4, 3)
        );
        // Level 2: HL (3Ã—3 at x=4), LH (4Ã—2 at y=3), HH (3Ã—2).
        assert_eq!(
            (bands[1].x0, bands[1].y0, bands[1].w, bands[1].h),
            (4, 0, 3, 3)
        );
        assert_eq!(
            (bands[2].x0, bands[2].y0, bands[2].w, bands[2].h),
            (0, 3, 4, 2)
        );
        assert_eq!(
            (bands[3].x0, bands[3].y0, bands[3].w, bands[3].h),
            (4, 3, 3, 2)
        );
        // Level 1: HL (6Ã—5 at x=7), LH (7Ã—4 at y=5), HH (6Ã—4).
        assert_eq!(
            (bands[4].x0, bands[4].y0, bands[4].w, bands[4].h),
            (7, 0, 6, 5)
        );
        assert_eq!(
            (bands[5].x0, bands[5].y0, bands[5].w, bands[5].h),
            (0, 5, 7, 4)
        );
        assert_eq!(
            (bands[6].x0, bands[6].y0, bands[6].w, bands[6].h),
            (7, 5, 6, 4)
        );
        // Coefficient counts tile the plane exactly.
        let total: usize = bands.iter().map(|b| b.w * b.h).sum();
        assert_eq!(total, 13 * 9);
    }

    #[test]
    fn resolution_ranges_cover_band_list() {
        let levels = 3usize;
        let mut covered = Vec::new();
        for r in 0..=levels {
            let (s, e) = resolution_band_range(r);
            covered.extend(s..e);
        }
        assert_eq!(covered, (0..3 * levels + 1).collect::<Vec<_>>());
    }
}
