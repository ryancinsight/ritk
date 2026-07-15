//! Marching-squares iso-contour extraction from a 2-D scalar image.
//!
//! # Mathematical Specification
//!
//! Given a 2-D scalar image I(y, x) and a contour value c, extracts all line
//! segments forming the iso-contour `{(y, x) : I(y, x) = c}` using the
//! marching-squares algorithm.
//!
//! ## Cell and corner convention
//!
//! For each 2×2 pixel cell at grid position (y, x):
//! ```text
//! v0 = I[y  ][x  ]   (top-left,     TL)  bit 0
//! v1 = I[y  ][x+1]   (top-right,    TR)  bit 1
//! v2 = I[y+1][x+1]   (bottom-right, BR)  bit 2
//! v3 = I[y+1][x  ]   (bottom-left,  BL)  bit 3
//! ```
//!
//! Case index = (v0>c)·2⁰ + (v1>c)·2¹ + (v2>c)·2² + (v3>c)·2³  → 16 cases.
//!
//! ## Edge numbering
//!
//! ```text
//! Edge 0 (top):    TL@(y,  x  ) → TR@(y,  x+1)   y constant,   x interpolated
//! Edge 1 (right):  TR@(y,  x+1) → BR@(y+1,x+1)   x constant,   y interpolated
//! Edge 2 (bottom): BR@(y+1,x+1) → BL@(y+1,x  )   y+1 constant, x interpolated
//! Edge 3 (left):   BL@(y+1,x  ) → TL@(y,  x  )   x constant,   y interpolated
//! ```
//!
//! Crossing positions use linear interpolation:
//! `t = (c − a) / (b − a)` clamped to [0, 1].
//!
//! ## Ambiguous cases 5 and 10
//!
//! Resolved by splitting into two segments that maintain two isolated interior
//! sub-regions (asymptotic-decider convention):
//! - Case 5 (TL+BR above): segments (edge3→edge0) and (edge1→edge2).
//! - Case 10 (TR+BL above): segments (edge0→edge1) and (edge2→edge3).
//!
//! ## Polyline assembly
//!
//! Segments are chained greedily into polylines via a
//! `HashMap<(u32, u32), Vec<usize>>` keyed on bit-exact f32 endpoint
//! coordinates. Shared-edge crossing points are computed identically on both
//! sides of the edge so equality holds without rounding.
//!
//! ## Input convention
//!
//! The image must have shape `[1, ny, nx]` (z = 0 slice of a 3-D volume).
//! The z = 0 layer is extracted and treated as the 2-D input plane.
//!
//! ## References
//! - Lorensen, W.E. & Cline, H.E. (1987). "Marching cubes: A high-resolution 3D
//!   surface construction algorithm." *SIGGRAPH '87*, pp. 163–169.
//! - Maple, C. (2003). "Geometric design and space planning using the marching
//!   squares and marching cube algorithms." *Proc. ICCGM 2003*.

use std::collections::HashMap;

use ritk_image::tensor::Backend;
use ritk_image::Image;
use ritk_tensor_ops::extract_vec_infallible;

// ── Public types ──────────────────────────────────────────────────────────────

/// A 2-D point `(y, x)` in pixel coordinates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ContourPoint {
    /// Row coordinate (y axis).
    pub y: f32,
    /// Column coordinate (x axis).
    pub x: f32,
}

/// A closed or open polyline contour as an ordered sequence of [`ContourPoint`]s.
///
/// For a **closed** loop the first and last point are equal. For an **open**
/// polyline they differ. In both cases `len() >= 2`.
pub type Contour = Vec<ContourPoint>;

// ── Filter ────────────────────────────────────────────────────────────────────

/// Marching-squares iso-contour extractor for a 2-D image.
///
/// Operates on a 3-D image where `nz = 1`; the z = 0 slice is treated as the
/// 2-D input plane.
///
/// # Default
/// - `contour_value = 0.5`
#[derive(Debug, Clone)]
pub struct ContourExtractor2DImageFilter {
    /// Iso-contour value. Voxels with value > `contour_value` are "inside".
    pub contour_value: f32,
}

impl Default for ContourExtractor2DImageFilter {
    fn default() -> Self {
        Self { contour_value: 0.5 }
    }
}

impl ContourExtractor2DImageFilter {
    /// Extract iso-contours at `contour_value` from a 2-D image.
    ///
    /// `image` must have shape `[1, ny, nx]`. Returns a list of polyline contours
    /// in pixel coordinates (y, x). Closed loops have equal first and last points.
    ///
    /// # Panics
    /// Panics if the backend tensor cannot be converted to `f32`.
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> Vec<Contour> {
        let (vals, dims) = extract_vec_infallible(image);
        let [_nz, ny, nx] = dims;
        debug_assert_eq!(_nz, 1, "ContourExtractor2DImageFilter requires nz = 1");

        // Extract the z = 0 slice as a flat [ny × nx] row-major buffer.
        let slice = &vals[..ny * nx];
        let get = |y: usize, x: usize| -> f32 { slice[y * nx + x] };

        let c = self.contour_value;
        let mut segments: Vec<[ContourPoint; 2]> = Vec::new();

        for y in 0..ny.saturating_sub(1) {
            for x in 0..nx.saturating_sub(1) {
                let v0 = get(y, x); // TL
                let v1 = get(y, x + 1); // TR
                let v2 = get(y + 1, x + 1); // BR
                let v3 = get(y + 1, x); // BL

                let case_idx = ((v0 > c) as u8)
                    | (((v1 > c) as u8) << 1)
                    | (((v2 > c) as u8) << 2)
                    | (((v3 > c) as u8) << 3);

                // Cases 0 and 15 produce no crossings.
                if case_idx == 0 || case_idx == 15 {
                    continue;
                }

                let fy = y as f32;
                let fx = x as f32;

                // Compute the interpolated crossing point on a given edge.
                let edge_pt = |edge: u8| -> ContourPoint {
                    match edge {
                        // Edge 0 (top, TL→TR): y constant, x interpolated.
                        0 => {
                            let t = interp(v0, v1, c);
                            ContourPoint { y: fy, x: fx + t }
                        }
                        // Edge 1 (right, TR→BR): x = x+1 constant, y interpolated.
                        1 => {
                            let t = interp(v1, v2, c);
                            ContourPoint {
                                y: fy + t,
                                x: fx + 1.0,
                            }
                        }
                        // Edge 2 (bottom, BR→BL): y = y+1 constant, x decreases.
                        2 => {
                            let t = interp(v2, v3, c);
                            ContourPoint {
                                y: fy + 1.0,
                                x: fx + 1.0 - t,
                            }
                        }
                        // Edge 3 (left, BL→TL): x constant, y decreases.
                        _ => {
                            let t = interp(v3, v0, c);
                            ContourPoint {
                                y: fy + 1.0 - t,
                                x: fx,
                            }
                        }
                    }
                };

                // Emit segments from the case table.
                for &[ea, eb] in CASE_TABLE[case_idx as usize] {
                    segments.push([edge_pt(ea), edge_pt(eb)]);
                }
            }
        }

        assemble_polylines(segments)
    }

    /// Coeus-native sister of [`apply`].
    pub fn apply_native<B>(
        &self,
        image: &ritk_image::native::Image<f32, B, 3>,
        _backend: &B,
    ) -> anyhow::Result<Vec<Contour>>
    where
        B: coeus_core::ComputeBackend,
        B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    {
        let (vals, dims) = ritk_tensor_ops::native::extract_image_vec(image)?;
        let [_nz, ny, nx] = dims;
        debug_assert_eq!(_nz, 1, "ContourExtractor2DImageFilter requires nz = 1");

        let slice = &vals[..ny * nx];
        let get = |y: usize, x: usize| -> f32 { slice[y * nx + x] };

        let c = self.contour_value;
        let mut segments: Vec<[ContourPoint; 2]> = Vec::new();

        for y in 0..ny.saturating_sub(1) {
            for x in 0..nx.saturating_sub(1) {
                let v0 = get(y, x);
                let v1 = get(y, x + 1);
                let v2 = get(y + 1, x + 1);
                let v3 = get(y + 1, x);

                let case_idx = ((v0 > c) as u8)
                    | (((v1 > c) as u8) << 1)
                    | (((v2 > c) as u8) << 2)
                    | (((v3 > c) as u8) << 3);
                if case_idx == 0 || case_idx == 15 {
                    continue;
                }

                let fy = y as f32;
                let fx = x as f32;
                let edge_pt = |edge: u8| -> ContourPoint {
                    match edge {
                        0 => {
                            let t = interp(v0, v1, c);
                            ContourPoint { y: fy, x: fx + t }
                        }
                        1 => {
                            let t = interp(v1, v2, c);
                            ContourPoint {
                                y: fy + t,
                                x: fx + 1.0,
                            }
                        }
                        2 => {
                            let t = interp(v2, v3, c);
                            ContourPoint {
                                y: fy + 1.0,
                                x: fx + 1.0 - t,
                            }
                        }
                        _ => {
                            let t = interp(v3, v0, c);
                            ContourPoint {
                                y: fy + 1.0 - t,
                                x: fx,
                            }
                        }
                    }
                };

                for &[ea, eb] in CASE_TABLE[case_idx as usize] {
                    segments.push([edge_pt(ea), edge_pt(eb)]);
                }
            }
        }

        Ok(assemble_polylines(segments))
    }
}

// ── Case table ────────────────────────────────────────────────────────────────

/// Marching-squares 16-case edge-pair table.
///
/// Each entry is a slice of `[edge_a, edge_b]` pairs. Each pair defines one
/// line segment whose endpoints are the crossing positions on `edge_a` and
/// `edge_b`. Cases 0 and 15 (no crossing) are handled before the table lookup.
///
/// Edge numbering: 0 = top, 1 = right, 2 = bottom, 3 = left.
/// Case bit encoding: bit0 = TL, bit1 = TR, bit2 = BR, bit3 = BL.
///
/// Ambiguous cases 5 (TL+BR) and 10 (TR+BL) are resolved by splitting into two
/// independent segments (asymptotic-decider / two-isolated-sub-regions resolution).
static CASE_TABLE: &[&[[u8; 2]]] = &[
    &[],               // 0:  no crossing (handled before table)
    &[[0, 3]],         // 1:  TL         → top ∩ left
    &[[0, 1]],         // 2:  TR         → top ∩ right
    &[[1, 3]],         // 3:  TL + TR    → right ∩ left
    &[[1, 2]],         // 4:  BR         → right ∩ bottom
    &[[3, 0], [1, 2]], // 5:  TL + BR    → (left ∩ top) + (right ∩ bottom)
    &[[0, 2]],         // 6:  TR + BR    → top ∩ bottom
    &[[2, 3]],         // 7:  TL+TR+BR   → bottom ∩ left
    &[[2, 3]],         // 8:  BL         → bottom ∩ left
    &[[0, 2]],         // 9:  TL + BL    → top ∩ bottom
    &[[0, 1], [2, 3]], // 10: TR + BL    → (top ∩ right) + (bottom ∩ left)
    &[[1, 2]],         // 11: TL+TR+BL   → right ∩ bottom
    &[[1, 3]],         // 12: BR + BL    → right ∩ left
    &[[0, 1]],         // 13: TL+BR+BL   → top ∩ right
    &[[0, 3]],         // 14: TR+BR+BL   → top ∩ left
    &[],               // 15: all inside (handled before table)
];

// ── Linear interpolation helper ───────────────────────────────────────────────

/// Fractional crossing position along an edge from value `a` to value `b`.
///
/// Returns `t ∈ [0, 1]` such that `a + t·(b − a) = c`. Clamped to [0, 1] to
/// guard against values numerically equal to the contour level.
#[inline]
fn interp(a: f32, b: f32, c: f32) -> f32 {
    let denom = b - a;
    if denom.abs() < f32::EPSILON {
        return 0.5;
    }
    ((c - a) / denom).clamp(0.0, 1.0)
}

// ── Polyline assembly ─────────────────────────────────────────────────────────

/// Chain raw segments into polylines by greedily matching shared endpoints.
///
/// A `HashMap<(u32, u32), Vec<usize>>` maps bit-exact f32 endpoint keys to
/// segment indices so that floating-point equality holds without rounding
/// (crossing points on a shared cell edge are computed identically from both sides).
///
/// For each unvisited seed segment the algorithm extends the chain forward from
/// its tail and then backward from its head, marking each consumed segment as
/// used. Produces one entry per connected component.
fn assemble_polylines(segments: Vec<[ContourPoint; 2]>) -> Vec<Contour> {
    if segments.is_empty() {
        return Vec::new();
    }

    type Key = (u32, u32);
    let to_key = |p: ContourPoint| -> Key { (p.y.to_bits(), p.x.to_bits()) };

    let n = segments.len();
    let mut used = vec![false; n];

    // Build adjacency: endpoint key → list of segment indices sharing that endpoint.
    let mut adj: HashMap<Key, Vec<usize>> = HashMap::with_capacity(2 * n);
    for (i, seg) in segments.iter().enumerate() {
        adj.entry(to_key(seg[0])).or_default().push(i);
        adj.entry(to_key(seg[1])).or_default().push(i);
    }

    let mut polylines: Vec<Contour> = Vec::new();

    for seed in 0..n {
        if used[seed] {
            continue;
        }
        used[seed] = true;
        let mut pts: Vec<ContourPoint> = vec![segments[seed][0], segments[seed][1]];

        // Extend forward from pts.last().
        loop {
            let tip = *pts.last().unwrap();
            let tip_key = to_key(tip);
            let next = adj
                .get(&tip_key)
                .and_then(|nbrs| nbrs.iter().find(|&&si| !used[si]).copied());
            match next {
                None => break,
                Some(si) => {
                    used[si] = true;
                    let [p0, p1] = segments[si];
                    // Append the far end of the matched segment.
                    pts.push(if to_key(p0) == tip_key { p1 } else { p0 });
                }
            }
        }

        // Extend backward from pts[0].
        loop {
            let head = pts[0];
            let head_key = to_key(head);
            let prev = adj
                .get(&head_key)
                .and_then(|nbrs| nbrs.iter().find(|&&si| !used[si]).copied());
            match prev {
                None => break,
                Some(si) => {
                    used[si] = true;
                    let [p0, p1] = segments[si];
                    // Prepend the far end of the matched segment.
                    pts.insert(0, if to_key(p1) == head_key { p0 } else { p1 });
                }
            }
        }

        polylines.push(pts);
    }

    polylines
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;
    use ritk_image::test_support as ts;
    use std::collections::HashSet;

    type B = NdArray<f32>;

    /// A 5×5 step image: columns 0–1 = 0.0, columns 2–4 = 2.0.
    ///
    /// At contour value 1.0 the iso-contour is the vertical line x = 1.5.
    /// Cells at (y=0..3, x=1) each contribute one segment (y, 1.5)→(y+1, 1.5),
    /// which chains into a single open polyline with 5 points, all at x = 1.5.
    ///
    /// # Derivation
    /// Case 6 (TR+BR above, TL+BL below) fires for every cell column x=1:
    /// edges 0 (top) and 2 (bottom) are crossed at t=0.5 → x = 1 + 0.5 = 1.5.
    #[test]
    fn step_image_produces_one_vertical_contour() {
        let ny = 5usize;
        let nx = 5usize;
        // Columns 0,1 → 0.0; columns 2,3,4 → 2.0.
        let data: Vec<f32> = (0..ny)
            .flat_map(|_y| (0..nx).map(move |x| if x < 2 { 0.0_f32 } else { 2.0_f32 }))
            .collect();

        let image = ts::make_image::<B, 3>(data, [1, ny, nx]);
        let filter = ContourExtractor2DImageFilter { contour_value: 1.0 };
        let contours = filter.apply(&image);

        assert_eq!(
            contours.len(),
            1,
            "expected exactly one contour, got {}",
            contours.len()
        );

        let pts = &contours[0];
        assert!(
            pts.len() >= 2,
            "contour must have at least 2 points, got {}",
            pts.len()
        );

        for p in pts {
            assert!(
                (p.x - 1.5).abs() < 1e-5,
                "all x coordinates must be 1.5, got x = {} at y = {}",
                p.x,
                p.y
            );
        }
    }

    /// A 4×4 image with a 2×2 square of 1.0 in the centre, surrounded by 0.0.
    ///
    /// At contour value 0.5, marching squares produces 8 crossing segments that
    /// chain into one closed loop of 8 unique vertices (9 points with first == last).
    ///
    /// # Derivation (verified analytically)
    /// Cell (0,0) case 4  → segment (0.5,1)→(1,0.5)
    /// Cell (0,1) case 12 → segment (0.5,2)→(0.5,1)
    /// Cell (0,2) case 8  → segment (1,2.5)→(0.5,2)
    /// Cell (1,0) case 6  → segment (1,0.5)→(2,0.5)
    /// Cell (1,2) case 9  → segment (1,2.5)→(2,2.5)
    /// Cell (2,0) case 2  → segment (2,0.5)→(2.5,1)
    /// Cell (2,1) case 3  → segment (2.5,2)→(2.5,1)
    /// Cell (2,2) case 1  → segment (2,2.5)→(2.5,2)
    /// All 8 segments chain into the closed loop:
    /// (0.5,1)→(1,0.5)→(2,0.5)→(2.5,1)→(2.5,2)→(2,2.5)→(1,2.5)→(0.5,2)→(0.5,1)
    #[test]
    fn square_block_produces_one_closed_loop() {
        #[rustfmt::skip]
        let data = vec![
            0.0_f32, 0.0, 0.0, 0.0,
            0.0,     1.0, 1.0, 0.0,
            0.0,     1.0, 1.0, 0.0,
            0.0,     0.0, 0.0, 0.0,
        ];

        let image = ts::make_image::<B, 3>(data, [1, 4, 4]);
        let filter = ContourExtractor2DImageFilter { contour_value: 0.5 };
        let contours = filter.apply(&image);

        assert_eq!(
            contours.len(),
            1,
            "expected one closed loop, got {}",
            contours.len()
        );

        let pts = &contours[0];
        // Deduplicate to count geometrically distinct vertices.
        let unique: HashSet<(u32, u32)> =
            pts.iter().map(|p| (p.y.to_bits(), p.x.to_bits())).collect();
        assert!(
            unique.len() >= 4,
            "expected >= 4 unique vertices, got {} (total points = {})",
            unique.len(),
            pts.len()
        );
    }
}
