//! H-transform grayscale morphological filters (h-maxima, h-minima, h-convex,
//! h-concave) for 3-D images.
//!
//! # Mathematical Specification
//!
//! Let `f` be the input image, `h > 0` a contrast height, and `R^δ` / `R^ε` the
//! morphological reconstruction by dilation / erosion (geodesic reconstruction,
//! Vincent 1993).
//!
//! - **H-maxima** suppresses all regional maxima whose dynamic (contrast to the
//!   surrounding) is below `h`:
//!   `HMAX_h(f) = R^δ_f(f − h)`
//! - **H-minima** suppresses regional minima of dynamic below `h`:
//!   `HMIN_h(f) = R^ε_f(f + h)`
//! - **H-convex** extracts the suppressed bright dynamic:
//!   `HCONVEX_h(f) = f − HMAX_h(f)`
//! - **H-concave** extracts the suppressed dark dynamic:
//!   `HCONCAVE_h(f) = HMIN_h(f) − f`
//!
//! Because the reconstruction marker satisfies `f − h ≤ f` (dilation) and
//! `f + h ≥ f` (erosion) by construction, the reconstruction preconditions hold
//! for every `h ≥ 0`.
//!
//! # ITK / SimpleITK Parity
//!
//! | Filter            | ITK class                  | SimpleITK |
//! |-------------------|----------------------------|-----------|
//! | `HMaximaFilter`   | `HMaximaImageFilter`       | `HMaxima` |
//! | `HMinimaFilter`   | `HMinimaImageFilter`       | `HMinima` |
//! | `HConvexFilter`   | `HConvexImageFilter`       | `HConvex` |
//! | `HConcaveFilter`  | `HConcaveImageFilter`      | `HConcave`|
//!
//! All four delegate to the bit-exact
//! [`crate::morphology::label_morphology::MorphologicalReconstruction`]; the
//! h-maxima / h-minima outputs are float-exact to their SimpleITK counterparts.
//!
//! # References
//! - Soille, P. (2003). *Morphological Image Analysis*, 2nd ed. Springer,
//!   §6.3 (h-transforms, dynamics).
//! - Vincent, L. (1993). Morphological grayscale reconstruction in image
//!   analysis. *IEEE Trans. Image Process.* 2(2):176–201.

use crate::morphology::label_morphology::{MorphologicalReconstruction, ReconstructionMode};
use crate::morphology::Connectivity;
use burn::tensor::backend::Backend;
use ritk_image::Image;
use ritk_tensor_ops::{extract_vec, rebuild};

/// Reconstruct an h-extrema image: shift the input by `±height` to form the
/// marker, then reconstruct it under the original image.
///
/// `mode = Dilation` with `−height` yields the h-maxima image; `mode = Erosion`
/// with `+height` yields the h-minima image.
fn reconstruct_h_extrema<B: Backend>(
    image: &Image<B, 3>,
    height: f32,
    mode: ReconstructionMode,
    connectivity: Connectivity,
) -> anyhow::Result<Image<B, 3>> {
    let (vals, dims) = extract_vec(image)?;
    let shift = match mode {
        ReconstructionMode::Dilation => -height,
        ReconstructionMode::Erosion => height,
    };
    let marker_vals: Vec<f32> = vals.iter().map(|&v| v + shift).collect();
    let marker = rebuild(marker_vals, dims, image);
    MorphologicalReconstruction::new(mode)
        .with_connectivity(connectivity)
        .apply(&marker, image)
}

/// Pointwise difference `a − b` of two co-shaped images (helper for h-convex /
/// h-concave). Spatial metadata is taken from `a`.
fn difference<B: Backend>(a: &Image<B, 3>, b: &Image<B, 3>) -> anyhow::Result<Image<B, 3>> {
    let (av, dims) = extract_vec(a)?;
    let (bv, _) = extract_vec(b)?;
    let out: Vec<f32> = av.iter().zip(bv.iter()).map(|(&x, &y)| x - y).collect();
    Ok(rebuild(out, dims, a))
}

// ── HMaximaFilter ─────────────────────────────────────────────────────────────

/// H-maxima filter: `HMAX_h(f) = R^δ_f(f − h)`.
///
/// Suppresses every bright regional maximum whose contrast to the surrounding
/// is below `height`, leaving the rest of the intensity surface unchanged.
#[derive(Debug, Clone)]
pub struct HMaximaFilter {
    height: f32,
    connectivity: Connectivity,
}

impl HMaximaFilter {
    /// Create an h-maxima filter with the given contrast height.
    pub fn new(height: f32) -> Self {
        Self {
            height,
            connectivity: Connectivity::Face6,
        }
    }

    /// Set the structuring-element adjacency (defaults to [`Connectivity::Face6`],
    /// matching ITK's `FullyConnectedOff`).
    pub fn with_connectivity(mut self, connectivity: Connectivity) -> Self {
        self.connectivity = connectivity;
        self
    }

    /// Apply the h-maxima transform.
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> anyhow::Result<Image<B, 3>> {
        reconstruct_h_extrema(
            image,
            self.height,
            ReconstructionMode::Dilation,
            self.connectivity,
        )
    }
}

// ── HMinimaFilter ─────────────────────────────────────────────────────────────

/// H-minima filter: `HMIN_h(f) = R^ε_f(f + h)`.
///
/// Suppresses every dark regional minimum whose contrast is below `height`.
#[derive(Debug, Clone)]
pub struct HMinimaFilter {
    height: f32,
    connectivity: Connectivity,
}

impl HMinimaFilter {
    /// Create an h-minima filter with the given contrast height.
    pub fn new(height: f32) -> Self {
        Self {
            height,
            connectivity: Connectivity::Face6,
        }
    }

    /// Set the structuring-element adjacency (defaults to [`Connectivity::Face6`]).
    pub fn with_connectivity(mut self, connectivity: Connectivity) -> Self {
        self.connectivity = connectivity;
        self
    }

    /// Apply the h-minima transform.
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> anyhow::Result<Image<B, 3>> {
        reconstruct_h_extrema(
            image,
            self.height,
            ReconstructionMode::Erosion,
            self.connectivity,
        )
    }
}

// ── HConvexFilter ─────────────────────────────────────────────────────────────

/// H-convex filter: `HCONVEX_h(f) = f − HMAX_h(f)`.
///
/// Extracts the bright dynamic suppressed by the h-maxima transform — non-zero
/// only on regional maxima of contrast ≥ `height`.
#[derive(Debug, Clone)]
pub struct HConvexFilter {
    height: f32,
    connectivity: Connectivity,
}

impl HConvexFilter {
    /// Create an h-convex filter with the given contrast height.
    pub fn new(height: f32) -> Self {
        Self {
            height,
            connectivity: Connectivity::Face6,
        }
    }

    /// Set the structuring-element adjacency (defaults to [`Connectivity::Face6`]).
    pub fn with_connectivity(mut self, connectivity: Connectivity) -> Self {
        self.connectivity = connectivity;
        self
    }

    /// Apply the h-convex transform.
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> anyhow::Result<Image<B, 3>> {
        let hmax = reconstruct_h_extrema(
            image,
            self.height,
            ReconstructionMode::Dilation,
            self.connectivity,
        )?;
        difference(image, &hmax)
    }
}

// ── HConcaveFilter ────────────────────────────────────────────────────────────

/// H-concave filter: `HCONCAVE_h(f) = HMIN_h(f) − f`.
///
/// Extracts the dark dynamic suppressed by the h-minima transform.
#[derive(Debug, Clone)]
pub struct HConcaveFilter {
    height: f32,
    connectivity: Connectivity,
}

impl HConcaveFilter {
    /// Create an h-concave filter with the given contrast height.
    pub fn new(height: f32) -> Self {
        Self {
            height,
            connectivity: Connectivity::Face6,
        }
    }

    /// Set the structuring-element adjacency (defaults to [`Connectivity::Face6`]).
    pub fn with_connectivity(mut self, connectivity: Connectivity) -> Self {
        self.connectivity = connectivity;
        self
    }

    /// Apply the h-concave transform.
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> anyhow::Result<Image<B, 3>> {
        let hmin = reconstruct_h_extrema(
            image,
            self.height,
            ReconstructionMode::Erosion,
            self.connectivity,
        )?;
        difference(&hmin, image)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "tests_h_transform.rs"]
mod tests_h_transform;
