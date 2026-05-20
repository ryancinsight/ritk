//! VTK smart mapper abstraction: scalar-to-color mapping and rendering strategy selection.
//!
//! # Architecture
//!
//! `VtkLookupTable` converts a scalar value to an RGBA colour via a 256-entry
//! piecewise linear LUT sampled at construction from a `ColormapPreset`.
//!
//! `VtkMapper` is a sealed trait implemented by `SurfaceMapper` (and future
//! volume mappers).  Selecting a mapper type is a compile-time decision encoded
//! in the type system; no `dyn VtkMapper` is required in throughput-critical
//! paths.
//!
//! # Mathematical Specification
//!
//! Given scalar range [s_min, s_max] and value v, the normalised parameter is:
//!   t = clamp((v − s_min) / (s_max − s_min), 0, 1)
//! The LUT index is:  i = round(t × 255) ∈ {0, …, 255}
//! The mapped colour is: rgba = lut[i]

/// Polygon display mode for surface rendering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PolygonMode {
    /// Filled polygons (default).
    #[default]
    Surface,
    /// Polygon outlines only.
    Wireframe,
    /// Polygon vertices only.
    Points,
}

/// Built-in colormap presets for `VtkLookupTable`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ColormapPreset {
    /// Scalar → greyscale; R=G=B=t.
    #[default]
    Grayscale,
    /// Classic MATLAB-style jet (blue–cyan–green–yellow–red).
    Jet,
    /// Diverging blue–white–red (Moreland 2009).
    CoolWarm,
    /// Perceptually uniform dark-purple → yellow (Matplotlib viridis).
    Viridis,
    /// HSV rainbow hue sweep blue → red.
    Rainbow,
}

/// 256-entry RGBA lookup table mapping normalised scalars to colours.
///
/// Constructed by sampling a `ColormapPreset` at 256 uniform steps in [0, 1].
#[derive(Debug, Clone)]
pub struct VtkLookupTable {
    /// `[scalar_min, scalar_max]` — input range mapped to [0, 1].
    pub range: [f64; 2],
    /// The colormap used to build the table.
    pub preset: ColormapPreset,
    table: Vec<[f32; 4]>, // 256 RGBA entries
}

impl VtkLookupTable {
    /// Build a lookup table for the given scalar `range` and `preset`.
    ///
    /// # Panics
    /// Panics if `range[0] == range[1]` (degenerate range).
    pub fn new(range: [f64; 2], preset: ColormapPreset) -> Self {
        assert!(
            (range[1] - range[0]).abs() > f64::EPSILON,
            "VtkLookupTable: range[0] and range[1] must differ"
        );
        let table = (0usize..256)
            .map(|i| {
                let t = i as f32 / 255.0;
                let [r, g, b] = sample_colormap(preset, t);
                [r, g, b, 1.0_f32]
            })
            .collect();
        Self {
            range,
            preset,
            table,
        }
    }

    /// Map scalar `v` to an RGBA colour.
    ///
    /// Values outside `range` are clamped to the table endpoints.
    pub fn map_value(&self, v: f64) -> [f32; 4] {
        let t = ((v - self.range[0]) / (self.range[1] - self.range[0])).clamp(0.0, 1.0) as f32;
        let idx = (t * 255.0).round() as usize;
        self.table[idx.min(255)]
    }

    /// Map a slice of scalar values to RGBA colours.
    pub fn map_values(&self, values: &[f64]) -> Vec<[f32; 4]> {
        values.iter().map(|&v| self.map_value(v)).collect()
    }

    /// Direct access to the 256-entry RGBA table (index 0 = minimum, 255 = maximum).
    pub fn raw_table(&self) -> &[[f32; 4]] {
        &self.table
    }
}

// ── Colormap helpers ───────────────────────────────────────────────────────

/// Sample a colormap preset at normalised parameter `t ∈ [0, 1]`.
fn sample_colormap(preset: ColormapPreset, t: f32) -> [f32; 3] {
    match preset {
        ColormapPreset::Grayscale => [t, t, t],
        ColormapPreset::Jet => jet_color(t),
        ColormapPreset::CoolWarm => cool_warm_color(t),
        ColormapPreset::Viridis => viridis_color(t),
        ColormapPreset::Rainbow => rainbow_color(t),
    }
}

/// MATLAB-style jet colormap: piecewise linear blue→cyan→green→yellow→red.
///
/// Segments (each spanning 1/4 of [0,1]):
///   Blue peak at t=0.25, green peak at t=0.5, red peak at t=0.75.
fn jet_color(t: f32) -> [f32; 3] {
    let r = if t < 3.0 / 8.0 {
        0.0
    } else if t < 5.0 / 8.0 {
        (t - 3.0 / 8.0) * 4.0
    } else if t < 7.0 / 8.0 {
        1.0
    } else {
        (9.0 / 8.0 - t) * 4.0
    };
    let g = if t < 1.0 / 8.0 {
        0.0
    } else if t < 3.0 / 8.0 {
        (t - 1.0 / 8.0) * 4.0
    } else if t < 5.0 / 8.0 {
        1.0
    } else if t < 7.0 / 8.0 {
        (7.0 / 8.0 - t) * 4.0
    } else {
        0.0
    };
    let b = if t < 1.0 / 8.0 {
        0.5 + t * 4.0
    } else if t < 3.0 / 8.0 {
        1.0
    } else if t < 5.0 / 8.0 {
        (5.0 / 8.0 - t) * 4.0
    } else {
        0.0
    };
    [
        r.clamp(0.0, 1.0),
        g.clamp(0.0, 1.0),
        b.clamp(0.0, 1.0),
    ]
}

/// Moreland (2009) diverging blue–white–red colormap.
///
/// t=0 → cool blue [0.23, 0.30, 0.75]
/// t=0.5 → white [1.0, 1.0, 1.0]
/// t=1 → warm red [0.71, 0.016, 0.15]
fn cool_warm_color(t: f32) -> [f32; 3] {
    let cool = [0.23_f32, 0.30, 0.75];
    let white = [1.0_f32; 3];
    let warm = [0.71_f32, 0.016, 0.15];
    if t <= 0.5 {
        let s = t * 2.0;
        lerp3(cool, white, s)
    } else {
        let s = (t - 0.5) * 2.0;
        lerp3(white, warm, s)
    }
}

/// Perceptually uniform viridis colormap — 5 anchor colours.
///
/// Anchors (t=0, 0.25, 0.5, 0.75, 1.0) taken from Matplotlib's viridis.
fn viridis_color(t: f32) -> [f32; 3] {
    const KEYS: [[f32; 3]; 5] = [
        [0.267, 0.005, 0.329], // dark purple
        [0.128, 0.407, 0.549], // blue-green
        [0.204, 0.636, 0.469], // teal
        [0.632, 0.829, 0.195], // yellow-green
        [0.993, 0.906, 0.144], // yellow
    ];
    let seg = (t * 4.0).floor().min(3.0) as usize;
    let s = (t * 4.0) - seg as f32;
    lerp3(KEYS[seg], KEYS[seg + 1], s.clamp(0.0, 1.0))
}

/// HSV rainbow: hue sweeps 240° (blue) → 0° (red) as t goes 0 → 1.
fn rainbow_color(t: f32) -> [f32; 3] {
    // Hue in [0, 360) decreasing from 240 to 0.
    let hue = 240.0_f32 * (1.0 - t);
    hsv_to_rgb(hue, 1.0, 1.0)
}

/// Convert HSV (hue ∈ [0,360), s ∈ [0,1], v ∈ [0,1]) to RGB.
fn hsv_to_rgb(h: f32, s: f32, v: f32) -> [f32; 3] {
    let h = h.rem_euclid(360.0);
    let i = (h / 60.0) as u32;
    let f = h / 60.0 - i as f32;
    let p = v * (1.0 - s);
    let q = v * (1.0 - s * f);
    let t = v * (1.0 - s * (1.0 - f));
    match i {
        0 => [v, t, p],
        1 => [q, v, p],
        2 => [p, v, t],
        3 => [p, q, v],
        4 => [t, p, v],
        _ => [v, p, q],
    }
}

/// Linear interpolation between two RGB triples.
#[inline]
fn lerp3(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    ]
}

// ── Mapper trait & SurfaceMapper ───────────────────────────────────────────

/// Abstraction for VTK rendering mappers — converts a dataset to a renderable
/// representation using a lookup table and a chosen polygon display mode.
pub trait VtkMapper: Send + Sync {
    /// Replace the lookup table.
    fn set_lookup_table(&mut self, lut: VtkLookupTable);
    /// Borrow the current lookup table.
    fn lookup_table(&self) -> &VtkLookupTable;
    /// Show or hide scalar colouring.
    fn set_scalar_visibility(&mut self, visible: bool);
    /// Returns `true` if scalar colouring is active.
    fn is_scalar_visible(&self) -> bool;
}

/// Surface mapper: renders `VtkPolyData` as filled polygons, wireframe, or points.
#[derive(Debug, Clone)]
pub struct SurfaceMapper {
    /// Polygon display mode (surface / wireframe / points).
    pub mode: PolygonMode,
    /// Overall opacity in [0, 1]; 1.0 = fully opaque.
    pub opacity: f32,
    lut: VtkLookupTable,
    scalar_visibility: bool,
}

impl Default for SurfaceMapper {
    fn default() -> Self {
        Self {
            mode: PolygonMode::Surface,
            opacity: 1.0,
            lut: VtkLookupTable::new([0.0, 1.0], ColormapPreset::Grayscale),
            scalar_visibility: true,
        }
    }
}

impl SurfaceMapper {
    /// Construct with explicit lookup table and mode.
    pub fn new(lut: VtkLookupTable, mode: PolygonMode) -> Self {
        Self {
            mode,
            opacity: 1.0,
            lut,
            scalar_visibility: true,
        }
    }
}

impl VtkMapper for SurfaceMapper {
    fn set_lookup_table(&mut self, lut: VtkLookupTable) {
        self.lut = lut;
    }
    fn lookup_table(&self) -> &VtkLookupTable {
        &self.lut
    }
    fn set_scalar_visibility(&mut self, visible: bool) {
        self.scalar_visibility = visible;
    }
    fn is_scalar_visible(&self) -> bool {
        self.scalar_visibility
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1.0 / 255.0 + 1e-5; // one LUT quantisation step

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < EPS
    }
    fn rgba_approx_eq(a: [f32; 4], b: [f32; 4]) -> bool {
        a.iter().zip(b.iter()).all(|(&x, &y)| approx_eq(x, y))
    }

    #[test]
    fn grayscale_lut_min_is_black() {
        let lut = VtkLookupTable::new([0.0, 1.0], ColormapPreset::Grayscale);
        let c = lut.map_value(0.0);
        assert!(approx_eq(c[0], 0.0), "R must be 0 at min: got {}", c[0]);
        assert!(approx_eq(c[1], 0.0), "G must be 0 at min: got {}", c[1]);
        assert!(approx_eq(c[2], 0.0), "B must be 0 at min: got {}", c[2]);
        assert!(approx_eq(c[3], 1.0), "A must be 1: got {}", c[3]);
    }

    #[test]
    fn grayscale_lut_max_is_white() {
        let lut = VtkLookupTable::new([0.0, 1.0], ColormapPreset::Grayscale);
        let c = lut.map_value(1.0);
        assert!(approx_eq(c[0], 1.0), "R must be 1 at max: got {}", c[0]);
        assert!(approx_eq(c[1], 1.0), "G must be 1 at max: got {}", c[1]);
        assert!(approx_eq(c[2], 1.0), "B must be 1 at max: got {}", c[2]);
    }

    #[test]
    fn grayscale_lut_midpoint_is_grey() {
        let lut = VtkLookupTable::new([0.0, 1.0], ColormapPreset::Grayscale);
        let c = lut.map_value(0.5);
        assert!(approx_eq(c[0], 0.5), "R at mid: got {}", c[0]);
        assert!(approx_eq(c[1], 0.5), "G at mid: got {}", c[1]);
        assert!(approx_eq(c[2], 0.5), "B at mid: got {}", c[2]);
    }

    #[test]
    fn jet_lut_min_is_dark_blue() {
        let lut = VtkLookupTable::new([0.0, 1.0], ColormapPreset::Jet);
        let c = lut.map_value(0.0);
        // jet at t=0: R=0, G=0, B=0.5
        assert!(approx_eq(c[0], 0.0), "jet R at 0: got {}", c[0]);
        assert!(approx_eq(c[1], 0.0), "jet G at 0: got {}", c[1]);
        assert!(approx_eq(c[2], 0.5), "jet B at 0: got {}", c[2]);
    }

    #[test]
    fn jet_lut_max_has_nonzero_red_zero_blue() {
        let lut = VtkLookupTable::new([0.0, 1.0], ColormapPreset::Jet);
        let c = lut.map_value(1.0);
        // jet at t=1: R>0, G=0, B=0
        assert!(c[0] > 0.0, "jet R at 1 must be > 0: got {}", c[0]);
        assert!(approx_eq(c[1], 0.0), "jet G at 1 must be 0: got {}", c[1]);
        assert!(approx_eq(c[2], 0.0), "jet B at 1 must be 0: got {}", c[2]);
    }

    #[test]
    fn cool_warm_midpoint_is_white() {
        let lut = VtkLookupTable::new([0.0, 1.0], ColormapPreset::CoolWarm);
        let c = lut.map_value(0.5);
        assert!(
            rgba_approx_eq(c, [1.0, 1.0, 1.0, 1.0]),
            "CoolWarm at 0.5 must be white; got {:?}",
            c
        );
    }

    #[test]
    fn viridis_lut_min_is_dark_purple() {
        let lut = VtkLookupTable::new([0.0, 1.0], ColormapPreset::Viridis);
        let c = lut.map_value(0.0);
        // viridis[0] = [0.267, 0.005, 0.329]; allow LUT quantisation
        assert!(c[2] > c[0] && c[2] > c[1], "blue must dominate at t=0: {:?}", c);
    }

    #[test]
    fn lut_clamps_below_range_to_min_color() {
        let lut = VtkLookupTable::new([10.0, 20.0], ColormapPreset::Grayscale);
        let c_min = lut.map_value(10.0);
        let c_below = lut.map_value(-999.0);
        assert!(
            rgba_approx_eq(c_min, c_below),
            "below-range must clamp to minimum: min={:?} below={:?}",
            c_min,
            c_below
        );
    }

    #[test]
    fn lut_clamps_above_range_to_max_color() {
        let lut = VtkLookupTable::new([10.0, 20.0], ColormapPreset::Grayscale);
        let c_max = lut.map_value(20.0);
        let c_above = lut.map_value(1e9);
        assert!(
            rgba_approx_eq(c_max, c_above),
            "above-range must clamp to maximum: max={:?} above={:?}",
            c_max,
            c_above
        );
    }

    #[test]
    fn surface_mapper_default_mode_is_surface() {
        let m = SurfaceMapper::default();
        assert_eq!(m.mode, PolygonMode::Surface);
        assert!(m.is_scalar_visible());
        assert!((m.opacity - 1.0).abs() < 1e-6);
    }

    #[test]
    fn surface_mapper_set_scalar_visibility_toggles() {
        let mut m = SurfaceMapper::default();
        m.set_scalar_visibility(false);
        assert!(!m.is_scalar_visible());
        m.set_scalar_visibility(true);
        assert!(m.is_scalar_visible());
    }
}
