//! Diffusion-family filters: Peronaâ€“Malik anisotropic, curvature anisotropic, and coherence-enhancing diffusion.
use crate::errors::{RitkPyError, RitkResult};
use crate::image::{image_from_py, into_py_image, PyImage};
use pyo3::prelude::*;
use ritk_filter::diffusion::{
    CoherenceConfig, ConductanceFunction, CurvatureConfig, DiffusionConfig,
    GradientAnisotropicDiffusionFilter, GradientDiffusionConfig,
};
use ritk_filter::edge::GaussianSigma;
use ritk_filter::{
    AntiAliasBinaryImageFilter, BinaryMinMaxCurvatureFlowConfig,
    BinaryMinMaxCurvatureFlowImageFilter, CoherenceEnhancingDiffusionFilter,
    CurvatureAnisotropicDiffusionFilter, CurvatureFlowConfig, CurvatureFlowImageFilter,
    MinMaxCurvatureFlowConfig, MinMaxCurvatureFlowImageFilter, ScalarChanAndVeseDenseLevelSet,
};

/// Conductance function kind for anisotropic diffusion, replacing `exponential: bool`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PyConductanceKind {
    /// Quadratic conductance: c(s) = 1/(1+(s/K)Â²)
    Quadratic,
    /// Exponential conductance: c(s) = exp(-(s/K)Â²)
    Exponential,
}

impl From<PyConductanceKind> for ConductanceFunction {
    fn from(kind: PyConductanceKind) -> Self {
        match kind {
            PyConductanceKind::Quadratic => ConductanceFunction::Quadratic,
            PyConductanceKind::Exponential => ConductanceFunction::Exponential,
        }
    }
}

impl<'a> From<Option<&'a str>> for PyConductanceKind {
    fn from(s: Option<&'a str>) -> Self {
        match s.map(|v| v.to_lowercase()).as_deref() {
            Some("quadratic") | Some("q") => PyConductanceKind::Quadratic,
            _ => PyConductanceKind::Exponential,
        }
    }
}

/// Apply Perona-Malik anisotropic diffusion for edge-preserving smoothing.
///
/// Reduces noise while preserving edges via the PDE:
/// âˆ‚I/âˆ‚t = div(c(|âˆ‡I|) Â· âˆ‡I)
///
/// The default `"exponential"` kind dispatches to the ITK-exact
/// `GradientAnisotropicDiffusionImageFilter` (face-gradient conductance with the
/// per-iteration average-gradient-magnitude `K` rescaling), so it matches
/// SimpleITK's `GradientAnisotropicDiffusion`. The `"quadratic"` kind uses the
/// crate's simpler Perona-Malik conductance `1/(1+(s/K)Â²)` (no SimpleITK
/// equivalent).
///
/// Args:
/// image: Input PyImage.
/// iterations: Number of explicit Euler time steps (default 20).
/// conductance: Edge-stopping parameter K (default 3.0; larger = more smoothing).
/// time_step: Euler step size Î”t (default 0.0625; must be â‰¤ 1/6 for 3-D stability).
/// conductance_kind: Conductance function kind â€” "exponential" (default) or "quadratic".
///
/// Returns:
/// Smoothed PyImage with identical shape and spatial metadata.
///
/// Raises:
/// RuntimeError: on tensor extraction failure.
#[pyfunction]
#[pyo3(signature = (image, iterations=20, conductance=3.0, time_step=0.0625, conductance_kind="exponential"))]
pub fn anisotropic_diffusion(
    py: Python<'_>,
    image: &PyImage,
    iterations: usize,
    conductance: f64,
    time_step: f64,
    conductance_kind: Option<&str>,
) -> RitkResult<PyImage> {
    let kind = PyConductanceKind::from(conductance_kind);
    let image = image_from_py(image);
    py.allow_threads(|| match kind {
        // ITK-exact gradient anisotropic diffusion (matches SimpleITK).
        PyConductanceKind::Exponential => {
            GradientAnisotropicDiffusionFilter::new(GradientDiffusionConfig {
                num_iterations: iterations,
                time_step: time_step as f32,
                conductance: conductance as f32,
            })
            .apply(&image)
            .map_err(|e| RitkPyError::runtime(e.to_string()))
        }
        // Crate-specific Perona-Malik with quadratic conductance.
        PyConductanceKind::Quadratic => DiffusionConfig {
            num_iterations: iterations,
            conductance: conductance as f32,
            time_step: time_step as f32,
            function: ConductanceFunction::Quadratic,
        }
        .apply(&image)
        .map_err(|e| RitkPyError::runtime(e.to_string())),
    })
    .map(into_py_image)
}

/// Apply curvature anisotropic diffusion (ITK `CurvatureAnisotropicDiffusion`).
///
/// Modified Curvature Diffusion Equation (Whitaker & Xue 2001):
/// âˆ‚I/âˆ‚t = |âˆ‡I| Â· âˆ‡Â·( c(|âˆ‡I|) Â· âˆ‡I/|âˆ‡I| )
///
/// Args:
///     image: Input PyImage.
///     iterations: Number of explicit Euler time steps (default 20).
///     time_step: Euler Î”t (default 0.0625; ITK's stable step is smaller for
///         fine spacing â€” keep â‰² 0.044 for ~0.35 mm CT to avoid instability).
///     conductance: Conductance parameter K (default 3.0). Larger K â†’ more
///         isotropic smoothing.
///
/// Returns:
///     Smoothed PyImage with identical shape and spatial metadata.
///
/// Raises:
///     RuntimeError: on tensor extraction failure.
#[pyfunction]
#[pyo3(signature = (image, iterations=20, time_step=0.0625, conductance=3.0))]
pub fn curvature_anisotropic_diffusion(
    py: Python<'_>,
    image: &PyImage,
    iterations: usize,
    time_step: f64,
    conductance: f64,
) -> RitkResult<PyImage> {
    let image = image_from_py(image);
    py.allow_threads(|| {
        let filter = CurvatureAnisotropicDiffusionFilter::new(CurvatureConfig {
            num_iterations: iterations,
            time_step: time_step as f32,
            conductance: conductance as f32,
        });
        filter
            .apply(&image)
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(into_py_image)
}

/// Apply pure mean-curvature flow: âˆ‚I/âˆ‚t = Îº for `iterations` explicit-Euler
/// steps. ITK Parity: CurvatureFlowImageFilter (`sitk.CurvatureFlow`).
#[pyfunction]
#[pyo3(signature = (image, time_step=0.0625, iterations=5))]
pub fn curvature_flow(
    py: Python<'_>,
    image: &PyImage,
    time_step: f64,
    iterations: usize,
) -> RitkResult<PyImage> {
    let image = image_from_py(image);
    py.allow_threads(|| {
        CurvatureFlowImageFilter::new(CurvatureFlowConfig {
            num_iterations: iterations,
            time_step: time_step as f32,
        })
        .apply(&image)
        .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(into_py_image)
}

/// Apply min/max curvature flow for `iterations` steps, gating the
/// curvature-flow speed by a stencil-radius min/max threshold to suppress
/// smoothing of features finer than the stencil. ITK Parity:
/// MinMaxCurvatureFlowImageFilter (`sitk.MinMaxCurvatureFlow`).
#[pyfunction]
#[pyo3(signature = (image, time_step=0.05, iterations=5, stencil_radius=2))]
pub fn min_max_curvature_flow(
    py: Python<'_>,
    image: &PyImage,
    time_step: f64,
    iterations: usize,
    stencil_radius: usize,
) -> RitkResult<PyImage> {
    let image = image_from_py(image);
    py.allow_threads(|| {
        MinMaxCurvatureFlowImageFilter::new(MinMaxCurvatureFlowConfig {
            num_iterations: iterations,
            time_step: time_step as f32,
            stencil_radius,
        })
        .apply(&image)
        .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(into_py_image)
}

/// Apply binary min/max curvature flow: curvature flow gated by comparing the
/// stencil-radius sphere average to a fixed `threshold`. ITK Parity:
/// BinaryMinMaxCurvatureFlowImageFilter (`sitk.BinaryMinMaxCurvatureFlow`).
#[pyfunction]
#[pyo3(signature = (image, time_step=0.05, iterations=5, stencil_radius=2, threshold=0.0))]
pub fn binary_min_max_curvature_flow(
    py: Python<'_>,
    image: &PyImage,
    time_step: f64,
    iterations: usize,
    stencil_radius: usize,
    threshold: f64,
) -> RitkResult<PyImage> {
    let image = image_from_py(image);
    py.allow_threads(|| {
        BinaryMinMaxCurvatureFlowImageFilter::new(BinaryMinMaxCurvatureFlowConfig {
            num_iterations: iterations,
            time_step: time_step as f32,
            stencil_radius,
            threshold,
        })
        .apply(&image)
        .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(into_py_image)
}

/// Apply coherence-enhancing diffusion (Weickert 1999).
///
/// Anisotropic diffusion that smooths along coherent structures (edges,
/// ridges) while preserving them across the structure orientation. Uses the
/// structure tensor to drive diffusion.
///
/// Args:
///     image: Input PyImage.
///     sigma: Gaussian sigma for structure tensor smoothing (integration scale, default 3.0).
///     contrast: Contrast parameter C (default 1e-10).
///     alpha: Smoothing parameter in flat regions (default 0.001).
///     time_step: Euler step Î”t (default 0.0625).
///     iterations: Number of iterations (default 10).
///
/// Returns:
///     Filtered PyImage with identical shape and spatial metadata.
///
/// Raises:
///     RuntimeError: on internal computation failure.
#[pyfunction]
#[pyo3(signature = (image, sigma=3.0, contrast=1e-10, alpha=0.001, time_step=0.0625, iterations=10))]
pub fn coherence_enhancing_diffusion(
    py: Python<'_>,
    image: &PyImage,
    sigma: f64,
    contrast: f64,
    alpha: f64,
    time_step: f64,
    iterations: usize,
) -> PyImage {
    let image = image_from_py(image);
    let result = py.allow_threads(|| {
        let config = CoherenceConfig {
            sigma: GaussianSigma::new_unchecked(sigma),
            contrast,
            alpha,
            time_step,
            n_iterations: iterations,
        };
        let filter = CoherenceEnhancingDiffusionFilter::new(config);
        filter.apply(&image)
    });
    into_py_image(result)
}

// â”€â”€ AntiAliasBinary â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Narrow-band level-set smoothing of binary image boundaries,
/// matching `SimpleITK.AntiAliasBinaryImageFilter` (bit-exact).
///
/// Evolves a signed-distance level set from the binary image boundary via the
/// ITK SparseField mean-curvature solver until the RMS change falls below
/// `max_rms_error` or `number_of_iterations` steps complete. Returns the
/// floating-point level set: **positive inside** the (foreground) object,
/// negative outside, with the zero crossing at the anti-aliased sub-voxel
/// boundary â€” the per-voxel sign is locked to the input binary.
///
/// Args:
///     image:                Binary float32 PyImage (foreground = max value).
///     max_rms_error:        Convergence threshold on per-voxel RMS change (ITK default 0.07).
///     number_of_iterations: Maximum SparseField evolution steps (ITK default 1000).
///
/// Returns:
///     Level-set PyImage (positive inside the object, negative outside).
#[pyfunction]
#[pyo3(signature = (image, max_rms_error=0.07_f32, number_of_iterations=1000_usize))]
pub fn anti_alias_binary(
    py: Python<'_>,
    image: &PyImage,
    max_rms_error: f32,
    number_of_iterations: usize,
) -> PyImage {
    let arc = image_from_py(image);
    let result = py.allow_threads(|| {
        AntiAliasBinaryImageFilter {
            max_rms_error,
            number_of_iterations,
        }
        .apply(&arc)
    });
    into_py_image(result)
}

// â”€â”€ ScalarChanAndVeseDenseLevelSet â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Dense Chan-Vese level set segmentation, bit-exact to
/// `SimpleITK.ScalarChanAndVeseDenseLevelSet`.
///
/// Minimises the Chan-Vese energy (region-based active contour without edges)
/// from a user-supplied signed-distance level set with per-iteration Maurer
/// reinitialization. Returns the **binary segmentation** (`1` where Ï† < 0).
///
/// Args:
///     initial_level_set: Ï†â‚€ image; negative = inside region (float32 3-D PyImage).
///     feature_image:     uâ‚€ for the Chan-Vese energy data term.
///     number_of_iterations: dense PDE steps (default 20).
///     lambda1:           Inside-region data weight (default 1.0).
///     lambda2:           Outside-region data weight (default 1.0).
///     curvature_weight:  Curvature (length) penalty Î¼ (default 1.0).
///     area_weight:       Area penalty subtracted from the data term (default 0.0).
///     epsilon:           Heaviside/Dirac regularisation width (default 1.0).
///
/// Returns:
///     Binary segmentation PyImage (`1.0` inside, `0.0` outside).
#[pyfunction]
#[pyo3(signature = (initial_level_set, feature_image, number_of_iterations=20,
                    lambda1=1.0_f32, lambda2=1.0_f32, mu=1.0_f32,
                    nu=0.0_f32, epsilon=1.0_f32))]
#[allow(clippy::too_many_arguments)]
pub fn scalar_chan_and_vese_dense_level_set(
    py: Python<'_>,
    initial_level_set: &PyImage,
    feature_image: &PyImage,
    number_of_iterations: usize,
    lambda1: f32,
    lambda2: f32,
    mu: f32,
    nu: f32,
    epsilon: f32,
) -> RitkResult<PyImage> {
    let arc_init = image_from_py(initial_level_set);
    let arc_feat = image_from_py(feature_image);
    let result = py.allow_threads(|| {
        ScalarChanAndVeseDenseLevelSet {
            number_of_iterations,
            lambda1,
            lambda2,
            mu,
            nu,
            epsilon,
        }
        .apply(&arc_init, &arc_feat)
    });
    result
        .map(into_py_image)
        .map_err(|e| crate::errors::RitkPyError::runtime(e.to_string()))
}
