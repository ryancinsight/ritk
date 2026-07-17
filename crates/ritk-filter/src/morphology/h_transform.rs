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
use ritk_image::tensor::Backend;
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
    validate_height_and_samples(height, &vals)?;
    let shift = match mode {
        ReconstructionMode::Dilation => -height,
        ReconstructionMode::Erosion => height,
    };
    let marker_vals = shifted_marker(&vals, shift)?;
    let marker = rebuild(marker_vals, dims, image);
    MorphologicalReconstruction::new(mode)
        .with_connectivity(connectivity)
        .apply(&marker, image)
}

fn reconstruct_h_extrema_native<B>(
    image: &ritk_image::native::Image<f32, B, 3>,
    height: f32,
    mode: ReconstructionMode,
    connectivity: Connectivity,
    backend: &B) -> anyhow::Result<ritk_image::native::Image<f32, B, 3>>
where
    B: coeus_core::ComputeBackend,
    B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
{
    let values = image.data_slice()?;
    validate_height_and_samples(height, values)?;
    let shift = match mode {
        ReconstructionMode::Dilation => -height,
        ReconstructionMode::Erosion => height,
    };
    let marker = ritk_image::native::Image::from_flat_on(
        shifted_marker(values, shift)?,
        image.shape(),
        *image.origin(),
        *image.spacing(),
        *image.direction(),
        backend,
    )?;
    MorphologicalReconstruction::new(mode)
        .with_connectivity(connectivity)
        .apply_native(&marker, image, backend)
}

fn shifted_marker(values: &[f32], shift: f32) -> anyhow::Result<Vec<f32>> {
    values
        .iter()
        .copied()
        .enumerate()
        .map(|(index, value)| {
            let marker = value + shift;
            anyhow::ensure!(
                marker.is_finite(),
                "h-transform marker at flat index {index} must remain finite after shift, got {marker}"
            );
            Ok(marker)
        })
        .collect()
}

fn validate_height_and_samples(height: f32, values: &[f32]) -> anyhow::Result<()> {
    anyhow::ensure!(
        height.is_finite() && height >= 0.0,
        "h-transform height must be finite and nonnegative, got {height}"
    );
    if let Some((index, value)) = values
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        anyhow::bail!("h-transform sample at flat index {index} must be finite, got {value}");
    }
    Ok(())
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

    /// Apply the h-minima transform to a Coeus-native image.
    ///
    /// # Errors
    ///
    /// Returns an error for a non-finite/negative height, a non-finite sample,
    /// inaccessible backend storage, or native output construction failure.
    pub fn apply_native<B>(
        &self,
        image: &ritk_image::native::Image<f32, B, 3>,
        backend: &B) -> anyhow::Result<ritk_image::native::Image<f32, B, 3>>
    where
        B: coeus_core::ComputeBackend,
        B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    {
        reconstruct_h_extrema_native(
            image,
            self.height,
            ReconstructionMode::Erosion,
            self.connectivity,
            backend,
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
