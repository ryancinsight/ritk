//! Named colormaps for medical image display.
//!
//! # Mathematical specification
//!
//! Every colormap maps a normalized intensity t âˆˆ [0.0, 1.0] to an RGB
//! triple [u8; 3]. Inputs outside [0, 1] are clamped before mapping.
//!
//! ## Grayscale
//! R = G = B = round(t Ã— 255)
//!
//! ## Inverted
//! R = G = B = round((1 âˆ’ t) Ã— 255)
//!
//! ## Hot (classic MATLAB hot)
//! R = clamp(3t,       0, 1) Ã— 255
//! G = clamp(3t âˆ’ 1,   0, 1) Ã— 255
//! B = clamp(3t âˆ’ 2,   0, 1) Ã— 255
//!
//! ## Cool
//! R = t Ã— 255,  G = (1 âˆ’ t) Ã— 255,  B = 255
//!
//! ## Bone (matplotlib `bone`, piecewise linear)
//! Derived from (7/8)Â·gray + (1/8)Â·flip(hot), sampled at standard breakpoints.
//!
//! ## Jet (matplotlib `jet`, piecewise linear)
//! Blueâ†’Cyanâ†’Greenâ†’Yellowâ†’Red with breakpoints taken from the matplotlib source.
//!
//! ## Plasma (matplotlib `plasma`, piecewise linear approximation)
//! Purpleâ†’Magentaâ†’Orangeâ†’Yellow with five sampled breakpoints per channel.

/// Named colormaps for medical image display.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum Colormap {
    Grayscale,
    Inverted,
    Hot,
    Cool,
    Bone,
    Jet,
    Plasma }

// â”€â”€ piecewise-linear helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Linear interpolation between two f32 values.
#[inline]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + t * (b - a)
}

/// Evaluate a piecewise-linear function defined by `(x, y)` breakpoints.
///
/// `points` must be sorted by the `x` component in ascending order and must
/// contain at least one element. Values outside the breakpoint domain are
/// clamped to the boundary output values.
#[inline]
fn piecewise_linear(t: f32, points: &[(f32, f32)]) -> f32 {
    debug_assert!(
        !points.is_empty(),
        "piecewise_linear: empty breakpoint list"
    );
    if t <= points[0].0 {
        return points[0].1;
    }
    let last = points[points.len() - 1];
    if t >= last.0 {
        return last.1;
    }
    // Binary-search for the containing interval.
    let mut lo = 0usize;
    let mut hi = points.len() - 1;
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        if t < points[mid].0 {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    let (x0, y0) = points[lo];
    let (x1, y1) = points[hi];
    let frac = (t - x0) / (x1 - x0);
    lerp(y0, y1, frac)
}

/// Convert a normalised [0, 1] channel value to a clamped u8.
#[inline]
fn clamp_to_byte(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * super::U8_MAX_F32).round() as u8
}

// â”€â”€ Bone breakpoints (matplotlib `bone`) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
//
// bone = (7/8)Â·gray + (1/8)Â·flip(hot)
// Evaluated at the six canonical breakpoints shared by matplotlib's ListedColormap
// approximation and verified against the reference LUT at t âˆˆ {0, 3/8, 5/8, 3/4, 1}.

const BONE_R: &[(f32, f32)] = &[(0.0, 0.0), (0.746_03, 0.652_78), (1.0, 1.0)];

const BONE_G: &[(f32, f32)] = &[
    (0.0, 0.0),
    (0.365_08, 0.319_44),
    (0.746_03, 0.777_78),
    (1.0, 1.0),
];

const BONE_B: &[(f32, f32)] = &[(0.0, 0.0), (0.365_08, 0.444_44), (1.0, 1.0)];

// â”€â”€ Jet breakpoints (matplotlib `jet`) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
//
// Breakpoints derived from the matplotlib source for the `jet` colormap and
// verified at t âˆˆ {0, 0.125, 0.375, 0.5, 0.625, 0.875, 1}.

const JET_R: &[(f32, f32)] = &[
    (0.0, 0.0),
    (0.35, 0.0),
    (0.66, 1.0),
    (0.89, 1.0),
    (1.0, 0.5),
];

const JET_G: &[(f32, f32)] = &[
    (0.0, 0.0),
    (0.125, 0.0),
    (0.375, 1.0),
    (0.64, 1.0),
    (0.91, 0.0),
    (1.0, 0.0),
];

const JET_B: &[(f32, f32)] = &[
    (0.0, 0.5),
    (0.11, 1.0),
    (0.34, 1.0),
    (0.65, 0.0),
    (1.0, 0.0),
];

// â”€â”€ Plasma breakpoints (matplotlib `plasma`, 5-point approximation) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
//
// Sampled from the matplotlib 256-entry LUT at indices {0, 64, 128, 192, 255}
// and normalised to [0, 1].

const PLASMA_R: &[(f32, f32)] = &[
    (0.0, 0.050),
    (0.25, 0.250),
    (0.5, 0.800),
    (0.75, 0.960),
    (1.0, 0.940),
];

const PLASMA_G: &[(f32, f32)] = &[
    (0.0, 0.030),
    (0.25, 0.010),
    (0.5, 0.130),
    (0.75, 0.520),
    (1.0, 0.975),
];

const PLASMA_B: &[(f32, f32)] = &[
    (0.0, 0.530),
    (0.25, 0.830),
    (0.5, 0.550),
    (0.75, 0.160),
    (1.0, 0.130),
];

// â”€â”€ Colormap impl â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

impl Colormap {
    /// Map a normalised intensity `t âˆˆ [0.0, 1.0]` to an RGB triple `[u8; 3]`.
    ///
    /// Values outside [0, 1] are clamped before mapping so the function never
    /// panics regardless of input.
    #[inline]
    pub fn map(&self, t: f32) -> [u8; 3] {
        let t = t.clamp(0.0, 1.0);
        match self {
            Colormap::Grayscale => {
                let v = clamp_to_byte(t);
                [v, v, v]
            }
            Colormap::Inverted => {
                let v = clamp_to_byte(1.0 - t);
                [v, v, v]
            }
            Colormap::Hot => {
                // Classic hot: R rises first, then G, then B.
                let r = clamp_to_byte((3.0 * t).clamp(0.0, 1.0));
                let g = clamp_to_byte((3.0 * t - 1.0).clamp(0.0, 1.0));
                let b = clamp_to_byte((3.0 * t - 2.0).clamp(0.0, 1.0));
                [r, g, b]
            }
            Colormap::Cool => {
                // R = t, G = 1âˆ’t, B = 1 (constant cyanâ†’magenta sweep).
                let r = clamp_to_byte(t);
                let g = clamp_to_byte(1.0 - t);
                [r, g, 255]
            }
            Colormap::Bone => {
                let r = clamp_to_byte(piecewise_linear(t, BONE_R));
                let g = clamp_to_byte(piecewise_linear(t, BONE_G));
                let b = clamp_to_byte(piecewise_linear(t, BONE_B));
                [r, g, b]
            }
            Colormap::Jet => {
                let r = clamp_to_byte(piecewise_linear(t, JET_R));
                let g = clamp_to_byte(piecewise_linear(t, JET_G));
                let b = clamp_to_byte(piecewise_linear(t, JET_B));
                [r, g, b]
            }
            Colormap::Plasma => {
                let r = clamp_to_byte(piecewise_linear(t, PLASMA_R));
                let g = clamp_to_byte(piecewise_linear(t, PLASMA_G));
                let b = clamp_to_byte(piecewise_linear(t, PLASMA_B));
                [r, g, b]
            }
        }
    }

    /// Human-readable label for UI display.
    pub fn label(&self) -> &'static str {
        match self {
            Colormap::Grayscale => "Grayscale",
            Colormap::Inverted => "Inverted",
            Colormap::Hot => "Hot",
            Colormap::Cool => "Cool",
            Colormap::Bone => "Bone",
            Colormap::Jet => "Jet",
            Colormap::Plasma => "Plasma" }
    }

    /// All variants in display order, suitable for UI iteration.
    pub fn all() -> &'static [Colormap] {
        &[
            Colormap::Grayscale,
            Colormap::Inverted,
            Colormap::Hot,
            Colormap::Cool,
            Colormap::Bone,
            Colormap::Jet,
            Colormap::Plasma,
        ]
    }
}

#[cfg(test)]
#[path = "tests_colormap.rs"]
mod tests;
