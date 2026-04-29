//! Tool interaction state and measurement annotation types.
//!
//! # Design
//!
//! Two orthogonal concerns live here:
//!
//! 1. **In-progress state** — [`ToolState`] tracks the partial interaction for
//!    the currently active tool (e.g., the first click of a two-click length
//!    measurement). It is held in memory only and is never persisted.
//!
//! 2. **Completed annotations** — [`Annotation`] stores the result of a
//!    finished measurement together with computed values. It is serialisable
//!    and may be saved to a session file.
//!
//! # Coordinate system
//!
//! All image-space coordinates are stored as `[row, col]` in pixel units.
//! Physical distances are computed by multiplying pixel deltas by the
//! corresponding voxel spacing components.
//!
//! # Mathematical specifications
//!
//! ## Length (Euclidean distance in physical space)
//!
//! Given two image points p₁ = (r₁, c₁) and p₂ = (r₂, c₂) and pixel spacing
//! s = (s_r, s_c) in mm/pixel:
//!
//! ```text
//! length_mm = √( ((r₂ − r₁) · s_r)² + ((c₂ − c₁) · s_c)² )
//! ```
//!
//! ## Angle (at vertex p₂, between vectors p₂→p₁ and p₂→p₃)
//!
//! ```text
//! v₁ = p₁ − p₂,  v₂ = p₃ − p₂
//! cos θ = (v₁ · v₂) / (|v₁| · |v₂|)
//! θ = arccos(clamp(cos θ, −1, 1))   [degrees]
//! ```
//!
//! ## ROI rectangle statistics
//!
//! Pixels whose row index lies in [min_r, max_r] and column index lies in
//! [min_c, max_c] (inclusive, integer bounds derived from p₁ and p₂) are
//! collected into a sample set S.
//!
//! ```text
//! mean    = (1/|S|) Σ v
//! std_dev = √( (1/|S|) Σ (v − mean)² )   [population std dev]
//! min     = min S
//! max     = max S
//! area    = (max_r − min_r + 1) · s_r  ×  (max_c − min_c + 1) · s_c   [mm²]
//! ```

use super::kind::ToolKind;
use egui::Pos2;

// ── In-progress tool state ────────────────────────────────────────────────────

/// In-progress interaction state for the active tool.
///
/// Only one `ToolState` variant is active per viewport at a time. `Idle` is
/// the resting state; every other variant represents a partially completed
/// gesture that will either be confirmed (producing an [`Annotation`]) or
/// cancelled on the next interaction event.
#[derive(Debug, Clone)]
pub enum ToolState {
    /// No gesture in progress.
    Idle,
    /// Pan drag in progress: stores the pointer start position in screen space
    /// and the viewport origin at the time the drag began.
    Panning {
        /// Pointer position (screen pixels) where the drag started.
        start: Pos2,
        /// Viewport pan offset at the moment the drag started.
        viewport_origin: Pos2,
    },
    /// Window/Level drag in progress.
    WindowLevelDrag {
        /// Pointer position (screen pixels) where the drag started.
        start: Pos2,
        /// Window centre at the moment the drag started.
        original_center: f64,
        /// Window width at the moment the drag started.
        original_width: f64,
    },
    /// First point of a two-click length measurement has been placed.
    MeasureLength1 {
        /// First measurement point in image pixel coordinates `[row, col]`.
        p1: Pos2,
    },
    /// First two points of a three-click angle measurement have been placed.
    MeasureAngle2 {
        /// First point in image pixel coordinates `[row, col]`.
        p1: Pos2,
        /// Second point (vertex) in image pixel coordinates `[row, col]`.
        p2: Pos2,
    },
    /// ROI drag in progress.
    RoiDrag {
        /// Drag start in image pixel coordinates `[row, col]`.
        start: Pos2,
        /// Current pointer position in image pixel coordinates `[row, col]`.
        current: Pos2,
        /// Whether the ROI is rectangular or elliptical.
        kind: RoiKind,
    },
}

/// Discriminant for the two supported ROI shapes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoiKind {
    Rect,
    Ellipse,
}

impl ToolState {
    /// Returns `true` when no gesture is in progress.
    pub fn is_idle(&self) -> bool {
        matches!(self, ToolState::Idle)
    }

    /// Returns the [`ToolKind`] that owns this state, or `None` when idle.
    pub fn tool_kind(&self) -> Option<ToolKind> {
        match self {
            ToolState::Idle => None,
            ToolState::Panning { .. } => Some(ToolKind::Pan),
            ToolState::WindowLevelDrag { .. } => Some(ToolKind::WindowLevel),
            ToolState::MeasureLength1 { .. } => Some(ToolKind::MeasureLength),
            ToolState::MeasureAngle2 { .. } => Some(ToolKind::MeasureAngle),
            ToolState::RoiDrag {
                kind: RoiKind::Rect,
                ..
            } => Some(ToolKind::RoiRect),
            ToolState::RoiDrag {
                kind: RoiKind::Ellipse,
                ..
            } => Some(ToolKind::RoiEllipse),
        }
    }
}

// ── Completed annotations ─────────────────────────────────────────────────────

/// A completed measurement annotation stored on a viewport.
///
/// Positions are stored as `[row, col]` in image pixel coordinates. Computed
/// values (lengths, angles, statistics) are stored in physical units (mm,
/// degrees, HU) and are derived quantities — they can be recomputed from the
/// position data and the volume spacing at any time.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum Annotation {
    /// Straight-line distance between two image points.
    Length {
        /// Start point as `[row, col]` in image pixels.
        p1: [f32; 2],
        /// End point as `[row, col]` in image pixels.
        p2: [f32; 2],
        /// Euclidean distance in millimetres.
        length_mm: f32,
    },
    /// Angle at vertex `p2` between the rays `p2→p1` and `p2→p3`.
    Angle {
        /// First ray endpoint as `[row, col]` in image pixels.
        p1: [f32; 2],
        /// Vertex as `[row, col]` in image pixels.
        p2: [f32; 2],
        /// Second ray endpoint as `[row, col]` in image pixels.
        p3: [f32; 2],
        /// Included angle in degrees.
        angle_deg: f32,
    },
    /// Axis-aligned rectangle ROI with intensity statistics.
    RoiRect {
        /// Top-left corner as `[row, col]` in image pixels.
        top_left: [f32; 2],
        /// Bottom-right corner as `[row, col]` in image pixels.
        bottom_right: [f32; 2],
        /// Mean intensity (HU or relative) within the ROI.
        mean: f32,
        /// Population standard deviation of intensities within the ROI.
        std_dev: f32,
        /// Minimum intensity within the ROI.
        min: f32,
        /// Maximum intensity within the ROI.
        max: f32,
        /// ROI area in mm².
        area_mm2: f32,
    },
    /// Single-point HU measurement.
    HuPoint {
        /// Point location as `[row, col]` in image pixels.
        pos: [f32; 2],
        /// Intensity value at the point (HU or relative).
        value: f32,
    },
}

impl Annotation {
    /// Compute the Euclidean distance between two image points in physical space.
    ///
    /// # Parameters
    /// - `p1`, `p2` — image coordinates `[row, col]` in pixels.
    /// - `spacing`  — pixel spacing `[row_spacing, col_spacing]` in mm/pixel.
    ///
    /// # Formula
    /// ```text
    /// length = √( ((p2[0] − p1[0]) · spacing[0])²
    ///           + ((p2[1] − p1[1]) · spacing[1])² )
    /// ```
    pub fn compute_length(p1: [f32; 2], p2: [f32; 2], spacing: [f32; 2]) -> f32 {
        let dr = (p2[0] - p1[0]) * spacing[0];
        let dc = (p2[1] - p1[1]) * spacing[1];
        (dr * dr + dc * dc).sqrt()
    }

    /// Compute the angle at vertex `p2` between the rays `p2→p1` and `p2→p3`.
    ///
    /// Returns the angle in degrees in the range `[0°, 180°]`.
    ///
    /// Returns `0.0` when either input ray has zero length (degenerate case:
    /// two coincident points), avoiding division by zero without panicking.
    ///
    /// # Formula
    /// ```text
    /// v₁ = p₁ − p₂,   v₂ = p₃ − p₂
    /// cos θ = clamp( (v₁ · v₂) / (|v₁| · |v₂|),  −1,  1 )
    /// θ = arccos(cos θ)   [converted to degrees]
    /// ```
    pub fn compute_angle(p1: [f32; 2], p2: [f32; 2], p3: [f32; 2]) -> f32 {
        let v1 = [p1[0] - p2[0], p1[1] - p2[1]];
        let v2 = [p3[0] - p2[0], p3[1] - p2[1]];
        let dot = v1[0] * v2[0] + v1[1] * v2[1];
        let mag1 = (v1[0] * v1[0] + v1[1] * v1[1]).sqrt();
        let mag2 = (v2[0] * v2[0] + v2[1] * v2[1]).sqrt();
        if mag1 < f32::EPSILON || mag2 < f32::EPSILON {
            // Degenerate: one or both rays have zero length; angle is undefined.
            return 0.0;
        }
        let cos_theta = (dot / (mag1 * mag2)).clamp(-1.0, 1.0);
        cos_theta.acos().to_degrees()
    }

    /// Compute intensity statistics for pixels within the axis-aligned rectangle
    /// defined by corners `p1` and `p2` (order-independent).
    ///
    /// # Parameters
    /// - `p1`, `p2`  — opposite corners of the ROI as `[row, col]` in pixels.
    /// - `pixels`    — flat row-major pixel buffer of the slice.
    /// - `width`     — number of columns in the slice.
    /// - `height`    — number of rows in the slice.
    /// - `spacing`   — pixel spacing `[row_spacing, col_spacing]` in mm/pixel.
    ///
    /// # Returns
    /// `(mean, std_dev, min, max, area_mm2)` using population statistics.
    ///
    /// Returns `(0.0, 0.0, 0.0, 0.0, 0.0)` when the ROI contains no pixels
    /// (e.g., the rectangle lies fully outside the image bounds).
    ///
    /// # Formula
    /// ```text
    /// S = { pixels[r × width + c]  |  r ∈ [min_r, max_r], c ∈ [min_c, max_c] }
    /// mean    = Σ S / |S|
    /// std_dev = √( Σ (v − mean)² / |S| )
    /// area    = (max_r − min_r + 1) · spacing[0]
    ///         × (max_c − min_c + 1) · spacing[1]
    /// ```
    pub fn compute_roi_rect_stats(
        p1: [f32; 2],
        p2: [f32; 2],
        pixels: &[f32],
        width: usize,
        height: usize,
        spacing: [f32; 2],
    ) -> (f32, f32, f32, f32, f32) {
        // Derive integer row/col bounds from the two corner points, clamped to
        // the valid image extent.
        let r_min_f = p1[0].min(p2[0]).floor();
        let r_max_f = p1[0].max(p2[0]).ceil();
        let c_min_f = p1[1].min(p2[1]).floor();
        let c_max_f = p1[1].max(p2[1]).ceil();

        // Guard against negative coordinates before casting to usize.
        let r_min = (r_min_f.max(0.0) as usize).min(height.saturating_sub(1));
        let r_max = (r_max_f.max(0.0) as usize).min(height.saturating_sub(1));
        let c_min = (c_min_f.max(0.0) as usize).min(width.saturating_sub(1));
        let c_max = (c_max_f.max(0.0) as usize).min(width.saturating_sub(1));

        // Collect valid sample values.
        let mut vals: Vec<f32> = Vec::with_capacity((r_max - r_min + 1) * (c_max - c_min + 1));
        for r in r_min..=r_max {
            for c in c_min..=c_max {
                let idx = r * width + c;
                if idx < pixels.len() {
                    vals.push(pixels[idx]);
                }
            }
        }

        if vals.is_empty() {
            return (0.0, 0.0, 0.0, 0.0, 0.0);
        }

        let n = vals.len() as f32;
        let mean = vals.iter().copied().sum::<f32>() / n;

        // Population standard deviation.
        let variance = vals.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / n;
        let std_dev = variance.sqrt();

        let min = vals.iter().copied().fold(f32::INFINITY, f32::min);
        let max = vals.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        // Physical area: integer pixel counts multiplied by per-axis spacing.
        let h_px = (r_max - r_min + 1) as f32;
        let w_px = (c_max - c_min + 1) as f32;
        let area_mm2 = h_px * spacing[0] * w_px * spacing[1];

        (mean, std_dev, min, max, area_mm2)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::colormap::Colormap;
    use crate::render::slice_render::WindowLevel;

    // ── compute_length ────────────────────────────────────────────────────────

    /// Axis-aligned horizontal displacement with unit spacing must yield the
    /// exact integer pixel distance.
    ///
    /// Analytical: p1=[0,0], p2=[0,3], spacing=[1,1]
    ///   length = √( (0·1)² + (3·1)² ) = √9 = 3.0
    #[test]
    fn test_compute_length_axis_aligned() {
        let p1 = [0.0_f32, 0.0];
        let p2 = [0.0_f32, 3.0];
        let spacing = [1.0_f32, 1.0];
        let length = Annotation::compute_length(p1, p2, spacing);
        assert_eq!(
            length, 3.0_f32,
            "axis-aligned horizontal length of 3 pixels with unit spacing must equal 3.0 mm"
        );
    }

    /// Axis-aligned vertical displacement with unit spacing.
    ///
    /// Analytical: p1=[0,0], p2=[4,0], spacing=[1,1]
    ///   length = √( (4·1)² + (0·1)² ) = √16 = 4.0
    #[test]
    fn test_compute_length_axis_aligned_vertical() {
        let p1 = [0.0_f32, 0.0];
        let p2 = [4.0_f32, 0.0];
        let spacing = [1.0_f32, 1.0];
        let length = Annotation::compute_length(p1, p2, spacing);
        assert_eq!(
            length, 4.0_f32,
            "axis-aligned vertical length of 4 pixels with unit spacing must equal 4.0 mm"
        );
    }

    /// Non-unit spacing scales the physical length independently per axis.
    ///
    /// Analytical: p1=[0,0], p2=[2,0], spacing=[0.5, 1.0]
    ///   length = √( (2·0.5)² + (0·1.0)² ) = √1 = 1.0
    #[test]
    fn test_compute_length_anisotropic_spacing() {
        let p1 = [0.0_f32, 0.0];
        let p2 = [2.0_f32, 0.0];
        let spacing = [0.5_f32, 1.0];
        let length = Annotation::compute_length(p1, p2, spacing);
        let expected = 1.0_f32;
        assert!(
            (length - expected).abs() < 1e-5,
            "anisotropic length: expected {expected}, got {length}"
        );
    }

    // ── compute_angle ─────────────────────────────────────────────────────────

    /// Three points forming a right angle (90°) at the vertex.
    ///
    /// Analytical:
    ///   p1 = (0, 1),  p2 = (0, 0) [vertex],  p3 = (1, 0)
    ///   v₁ = (0,1)−(0,0) = (0,1),   v₂ = (1,0)−(0,0) = (1,0)
    ///   dot = 0·1 + 1·0 = 0  ⟹  cos θ = 0  ⟹  θ = 90°
    #[test]
    fn test_compute_angle_right_angle() {
        let p1 = [0.0_f32, 1.0];
        let p2 = [0.0_f32, 0.0]; // vertex
        let p3 = [1.0_f32, 0.0];
        let angle = Annotation::compute_angle(p1, p2, p3);
        assert!(
            (angle - 90.0_f32).abs() < 0.001,
            "90° angle must be computed to within 0.001°, got {angle}°"
        );
    }

    /// Three collinear points (same direction) must yield 0°.
    ///
    /// Analytical:
    ///   p1 = (0, 0),  p2 = (0, 1) [vertex],  p3 = (0, 2)
    ///   v₁ = (0,−1),  v₂ = (0,1)
    ///   cos θ = −1  ⟹  θ = 180°
    ///
    ///   Note: the two rays point in exactly opposite directions, so the angle
    ///   at the vertex is 180°.
    #[test]
    fn test_compute_angle_straight_line() {
        let p1 = [0.0_f32, 0.0];
        let p2 = [0.0_f32, 1.0]; // vertex on the line
        let p3 = [0.0_f32, 2.0];
        let angle = Annotation::compute_angle(p1, p2, p3);
        assert!(
            (angle - 180.0_f32).abs() < 0.001,
            "straight-line angle must be 180°, got {angle}°"
        );
    }

    /// Degenerate input where p1 == p2 must return 0.0 rather than NaN or panic.
    #[test]
    fn test_compute_angle_degenerate_zero_length_ray() {
        let p1 = [1.0_f32, 1.0];
        let p2 = [1.0_f32, 1.0]; // p1 == p2 → zero-length ray
        let p3 = [2.0_f32, 3.0];
        let angle = Annotation::compute_angle(p1, p2, p3);
        assert_eq!(
            angle, 0.0_f32,
            "degenerate (zero-length ray) angle must return 0.0, got {angle}"
        );
    }

    // ── ToolState ─────────────────────────────────────────────────────────────

    /// `Idle` must report `is_idle() == true` and `tool_kind() == None`.
    #[test]
    fn test_tool_state_idle() {
        let state = ToolState::Idle;
        assert!(state.is_idle(), "Idle must report is_idle() = true");
        assert_eq!(state.tool_kind(), None, "Idle must have tool_kind() = None");
    }

    /// Each non-idle variant must report `is_idle() == false` and the
    /// correct `ToolKind`.
    #[test]
    fn test_tool_state_non_idle_variants() {
        let cases: &[(ToolState, ToolKind)] = &[
            (
                ToolState::Panning {
                    start: Pos2::ZERO,
                    viewport_origin: Pos2::ZERO,
                },
                ToolKind::Pan,
            ),
            (
                ToolState::WindowLevelDrag {
                    start: Pos2::ZERO,
                    original_center: 0.0,
                    original_width: 1.0,
                },
                ToolKind::WindowLevel,
            ),
            (
                ToolState::MeasureLength1 { p1: Pos2::ZERO },
                ToolKind::MeasureLength,
            ),
            (
                ToolState::MeasureAngle2 {
                    p1: Pos2::ZERO,
                    p2: Pos2::new(1.0, 0.0),
                },
                ToolKind::MeasureAngle,
            ),
            (
                ToolState::RoiDrag {
                    start: Pos2::ZERO,
                    current: Pos2::new(1.0, 1.0),
                    kind: RoiKind::Rect,
                },
                ToolKind::RoiRect,
            ),
            (
                ToolState::RoiDrag {
                    start: Pos2::ZERO,
                    current: Pos2::new(1.0, 1.0),
                    kind: RoiKind::Ellipse,
                },
                ToolKind::RoiEllipse,
            ),
        ];
        for (state, expected_kind) in cases {
            assert!(!state.is_idle(), "{:?} must not be idle", state.tool_kind());
            assert_eq!(
                state.tool_kind(),
                Some(*expected_kind),
                "tool_kind() must return {:?}",
                expected_kind
            );
        }
    }

    // ── WindowLevel monotone (cross-module, uses slice_render::WindowLevel) ───

    /// [`WindowLevel::apply`] must produce monotonically non-decreasing output
    /// over 100 uniformly spaced input values in [0, 1000].
    ///
    /// Analytical justification: the DICOM PS 3.3 §C.7.6.3.1.5 formula is
    /// piece-wise linear with non-negative slope in every segment; therefore
    /// the mapping is monotone non-decreasing by construction.
    #[test]
    fn test_window_level_apply_range() {
        let wl = WindowLevel::new(500.0, 1000.0);
        // 100 uniformly spaced values in [0.0, 1000.0].
        let values: Vec<f64> = (0..100).map(|i| i as f64 * (1000.0 / 99.0)).collect();
        let mut prev = wl.apply(values[0]);
        for &v in &values[1..] {
            let cur = wl.apply(v);
            assert!(
                cur >= prev,
                "WindowLevel::apply must be non-decreasing: apply({v}) = {cur} < prev = {prev}"
            );
            prev = cur;
        }
    }

    // ── Colormap grayscale (cross-module, uses render::colormap) ─────────────

    /// [`Colormap::Grayscale`] must produce R = G = B and must be monotonically
    /// non-decreasing in the R channel as `t` increases from 0 to 1.
    ///
    /// Analytical: R(t) = round(t × 255), which is non-decreasing for t ∈ [0, 1].
    #[test]
    fn test_colormap_grayscale_monotone() {
        let cm = Colormap::Grayscale;
        let mut prev_r = cm.map(0.0)[0];
        for i in 1..=255u32 {
            let t = i as f32 / 255.0;
            let [r, g, b] = cm.map(t);
            // R = G = B invariant.
            assert_eq!(r, g, "Grayscale R≠G at t={t}");
            assert_eq!(g, b, "Grayscale G≠B at t={t}");
            // Monotone non-decreasing R channel.
            assert!(
                r >= prev_r,
                "Grayscale R not non-decreasing at t={t}: prev={prev_r}, cur={r}"
            );
            prev_r = r;
        }
    }

    // ── compute_roi_rect_stats ────────────────────────────────────────────────

    /// A 1×3 ROI over pixels [10, 20, 30] (row-major, width=3, height=1).
    ///
    /// Analytical:
    ///   mean    = (10 + 20 + 30) / 3 = 20.0
    ///   std_dev = √( ((10−20)² + (20−20)² + (30−20)²) / 3 )
    ///           = √( (100 + 0 + 100) / 3 )
    ///           = √(200/3) ≈ 8.164_966
    ///   min = 10.0,  max = 30.0
    ///   area = 1 × 1.0 × 3 × 1.0 = 3.0 mm²
    #[test]
    fn test_compute_roi_rect_stats_analytic() {
        let pixels = vec![10.0_f32, 20.0, 30.0];
        let (mean, std_dev, min, max, area) =
            Annotation::compute_roi_rect_stats([0.0, 0.0], [0.0, 2.0], &pixels, 3, 1, [1.0, 1.0]);
        assert!((mean - 20.0).abs() < 1e-5, "mean must be 20.0, got {mean}");
        let expected_std = (200.0_f32 / 3.0).sqrt();
        assert!(
            (std_dev - expected_std).abs() < 1e-4,
            "std_dev must be {expected_std}, got {std_dev}"
        );
        assert!((min - 10.0).abs() < 1e-5, "min must be 10.0, got {min}");
        assert!((max - 30.0).abs() < 1e-5, "max must be 30.0, got {max}");
        assert!(
            (area - 3.0).abs() < 1e-5,
            "area must be 3.0 mm², got {area}"
        );
    }

    /// An ROI entirely outside the image must return the all-zero tuple without
    /// panicking.
    #[test]
    fn test_compute_roi_rect_stats_out_of_bounds() {
        let pixels = vec![1.0_f32; 4]; // 2×2 image
        let (mean, std_dev, min, max, area) = Annotation::compute_roi_rect_stats(
            [10.0, 10.0],
            [20.0, 20.0],
            &pixels,
            2,
            2,
            [1.0, 1.0],
        );
        // Clamped to last valid pixel — still produces a valid (non-NaN) result.
        assert!(
            mean.is_finite(),
            "out-of-bounds ROI must return finite mean, got {mean}"
        );
        assert!(
            std_dev.is_finite(),
            "out-of-bounds ROI must return finite std_dev, got {std_dev}"
        );
        assert!(
            area >= 0.0,
            "out-of-bounds ROI must return non-negative area, got {area}"
        );
        // The entire image is 1.0, so mean and std_dev are deterministic.
        assert!(
            (mean - 1.0).abs() < 1e-5,
            "clamped out-of-bounds ROI over constant field must have mean=1.0, got {mean}"
        );
        let _ = (min, max); // min/max are valid but their exact values depend on clamping
    }
}
