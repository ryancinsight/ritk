//! VTK smart mapper abstraction: scalar-to-color mapping and rendering strategy selection.
//!
//! # Architecture
//!
//! `VtkLookupTable` converts a scalar value to an RGBA colour via an Iris-owned
//! 256-entry lookup table sampled at construction from a [`NamedColorMap`].
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
//! The mapped colour is: rgba = lut\[i\]

use iris::color::LookupTable;

use super::NamedColorMap;

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

/// 256-entry RGBA lookup table mapping normalised scalars to colours.
///
/// Constructed by sampling an Iris [`NamedColorMap`] at 256 uniform steps in
/// `[0, 1]`.
#[derive(Debug, Clone)]
pub struct VtkLookupTable {
    /// `[scalar_min, scalar_max]` — input range mapped to [0, 1].
    pub range: [f64; 2],
    /// The colormap used to build the table.
    pub preset: NamedColorMap,
    table: [[f32; 4]; 256],
}

impl VtkLookupTable {
    /// Build a lookup table for the given scalar `range` and `preset`.
    ///
    /// # Panics
    /// Panics if `range[0] == range[1]` (degenerate range).
    pub fn new(range: [f64; 2], preset: NamedColorMap) -> Self {
        assert!(
            (range[1] - range[0]).abs() > f64::EPSILON,
            "VtkLookupTable: range[0] and range[1] must differ"
        );
        let iris_table = LookupTable::<NamedColorMap, 256>::from_map(preset);
        let table = core::array::from_fn(|index| *iris_table.entries()[index].channels());
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

// ── Mapper trait & SurfaceMapper ───────────────────────────────────────────

/// Controls whether scalar colouring is active on a mapper.
///
/// Replaces bare `bool` to eliminate call-site boolean blindness.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ScalarVisibility {
    /// Scalar colouring is disabled; the actor colour is used.
    #[default]
    Hidden,
    /// Scalar values drive colour via the lookup table.
    Visible,
}

/// Abstraction for VTK rendering mappers — converts a dataset to a renderable
/// representation using a lookup table and a chosen polygon display mode.
pub trait VtkMapper: Send + Sync {
    /// Replace the lookup table.
    fn set_lookup_table(&mut self, lut: VtkLookupTable);
    /// Borrow the current lookup table.
    fn lookup_table(&self) -> &VtkLookupTable;
    /// Show or hide scalar colouring.
    fn set_scalar_visibility(&mut self, visible: ScalarVisibility);
    /// Returns the current scalar visibility state.
    fn scalar_visibility(&self) -> ScalarVisibility;
}

/// Surface mapper: renders `VtkPolyData` as filled polygons, wireframe, or points.
#[derive(Debug, Clone)]
pub struct SurfaceMapper {
    /// Polygon display mode (surface / wireframe / points).
    pub mode: PolygonMode,
    /// Overall opacity in [0, 1]; 1.0 = fully opaque.
    pub opacity: f32,
    lut: VtkLookupTable,
    scalar_visibility: ScalarVisibility,
}

impl Default for SurfaceMapper {
    fn default() -> Self {
        Self {
            mode: PolygonMode::Surface,
            opacity: 1.0,
            lut: VtkLookupTable::new([0.0, 1.0], NamedColorMap::Grayscale),
            scalar_visibility: ScalarVisibility::Visible,
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
            scalar_visibility: ScalarVisibility::Visible,
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
    fn set_scalar_visibility(&mut self, visible: ScalarVisibility) {
        self.scalar_visibility = visible;
    }
    fn scalar_visibility(&self) -> ScalarVisibility {
        self.scalar_visibility
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use iris::color::{ColorMap, Normalized};

    // One uniform LUT step plus four rounding errors in the affine map.
    const EPS: f32 = 1.0 / 255.0 + 4.0 * f32::EPSILON;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < EPS
    }
    fn rgba_approx_eq(a: [f32; 4], b: [f32; 4]) -> bool {
        a.iter().zip(b.iter()).all(|(&x, &y)| approx_eq(x, y))
    }

    #[test]
    fn lookup_tables_match_every_iris_map_at_all_sample_nodes() {
        for map in NamedColorMap::ALL {
            let lut = VtkLookupTable::new([0.0, 1.0], map);
            for index in 0_u8..=u8::MAX {
                let coordinate = Normalized::from_u8(index);
                assert_eq!(
                    lut.raw_table()[usize::from(index)].map(f32::to_bits),
                    map.sample(coordinate).channels().map(f32::to_bits),
                    "map={map:?}, index={index}"
                );
            }
        }
    }

    #[test]
    fn grayscale_lut_min_is_black() {
        let lut = VtkLookupTable::new([0.0, 1.0], NamedColorMap::Grayscale);
        let c = lut.map_value(0.0);
        assert!(approx_eq(c[0], 0.0), "R must be 0 at min: got {}", c[0]);
        assert!(approx_eq(c[1], 0.0), "G must be 0 at min: got {}", c[1]);
        assert!(approx_eq(c[2], 0.0), "B must be 0 at min: got {}", c[2]);
        assert!(approx_eq(c[3], 1.0), "A must be 1: got {}", c[3]);
    }

    #[test]
    fn grayscale_lut_max_is_white() {
        let lut = VtkLookupTable::new([0.0, 1.0], NamedColorMap::Grayscale);
        let c = lut.map_value(1.0);
        assert!(approx_eq(c[0], 1.0), "R must be 1 at max: got {}", c[0]);
        assert!(approx_eq(c[1], 1.0), "G must be 1 at max: got {}", c[1]);
        assert!(approx_eq(c[2], 1.0), "B must be 1 at max: got {}", c[2]);
    }

    #[test]
    fn grayscale_lut_midpoint_is_grey() {
        let lut = VtkLookupTable::new([0.0, 1.0], NamedColorMap::Grayscale);
        let c = lut.map_value(0.5);
        assert!(approx_eq(c[0], 0.5), "R at mid: got {}", c[0]);
        assert!(approx_eq(c[1], 0.5), "G at mid: got {}", c[1]);
        assert!(approx_eq(c[2], 0.5), "B at mid: got {}", c[2]);
    }

    #[test]
    fn jet_lut_min_is_dark_blue() {
        let lut = VtkLookupTable::new([0.0, 1.0], NamedColorMap::Jet);
        let c = lut.map_value(0.0);
        // jet at t=0: R=0, G=0, B=0.5
        assert!(approx_eq(c[0], 0.0), "jet R at 0: got {}", c[0]);
        assert!(approx_eq(c[1], 0.0), "jet G at 0: got {}", c[1]);
        assert!(approx_eq(c[2], 0.5), "jet B at 0: got {}", c[2]);
    }

    #[test]
    fn jet_lut_max_has_nonzero_red_zero_blue() {
        let lut = VtkLookupTable::new([0.0, 1.0], NamedColorMap::Jet);
        let c = lut.map_value(1.0);
        // jet at t=1: R>0, G=0, B=0
        assert!(c[0] > 0.0, "jet R at 1 must be > 0: got {}", c[0]);
        assert!(approx_eq(c[1], 0.0), "jet G at 1 must be 0: got {}", c[1]);
        assert!(approx_eq(c[2], 0.0), "jet B at 1 must be 0: got {}", c[2]);
    }

    #[test]
    fn cool_warm_midpoint_is_white() {
        let lut = VtkLookupTable::new([0.0, 1.0], NamedColorMap::CoolWarm);
        let c = lut.map_value(0.5);
        assert!(
            rgba_approx_eq(c, [1.0, 1.0, 1.0, 1.0]),
            "CoolWarm at 0.5 must be white; got {:?}",
            c
        );
    }

    #[test]
    fn viridis_lut_min_is_dark_purple() {
        let lut = VtkLookupTable::new([0.0, 1.0], NamedColorMap::Viridis);
        let c = lut.map_value(0.0);
        // viridis[0] = [0.267, 0.005, 0.329]; allow LUT quantisation
        assert!(
            c[2] > c[0] && c[2] > c[1],
            "blue must dominate at t=0: {:?}",
            c
        );
    }

    #[test]
    fn lut_clamps_below_range_to_min_color() {
        let lut = VtkLookupTable::new([10.0, 20.0], NamedColorMap::Grayscale);
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
        let lut = VtkLookupTable::new([10.0, 20.0], NamedColorMap::Grayscale);
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
        assert_eq!(m.scalar_visibility(), ScalarVisibility::Visible);
        assert!((m.opacity - 1.0).abs() < 1e-6);
    }

    #[test]
    fn surface_mapper_set_scalar_visibility_toggles() {
        let mut m = SurfaceMapper::default();
        m.set_scalar_visibility(ScalarVisibility::Hidden);
        assert_eq!(m.scalar_visibility(), ScalarVisibility::Hidden);
        m.set_scalar_visibility(ScalarVisibility::Visible);
        assert_eq!(m.scalar_visibility(), ScalarVisibility::Visible);
    }
}
