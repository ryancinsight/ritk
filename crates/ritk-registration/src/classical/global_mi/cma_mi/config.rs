// ─── Configuration ────────────────────────────────────────────────────────────

use crate::metric::{MutualInformationVariant, NormalizationMethod};
use crate::optimizer::StopReason;

use super::super::config::GlobalMiConfig;

// ── Per-level cascade configuration ──────────────────────────────────────────

/// Configuration for a single pyramid level in the CMA-ES multi-scale cascade.
///
/// When [`CmaMiConfig::pyramid_schedule`] is non-empty, each element
/// defines one coarse-to-fine pass.  The best parameter vector found at
/// level *k* is used as the initial mean for level *k+1*, with this level's
/// `cma_sigma0` controlling the search radius at the finer scale.
#[derive(Debug, Clone)]
pub struct CmaMiLevelConfig {
    /// Isotropic shrink factor for this level.  Overridden by `shrink_per_axis`
    /// when `Some`.  Typical values: 16 (very coarse), 8 (coarse), 4 (medium).
    pub shrink: usize,

    /// Per-axis shrink factors `[sz, sy, sx]`.  Overrides `shrink` when `Some`.
    /// Useful for thin-slab CT volumes (e.g. `[1, 8, 8]` preserves z-slices).
    pub shrink_per_axis: Option<[usize; 3]>,

    /// Gaussian pre-smoothing sigma (mm) applied before downsampling.
    /// Should be ≥ shrink/2 to satisfy the Nyquist criterion.
    pub sigma_mm: f64,

    /// CMA-ES initial step-size σ₀ at this level, in normalised parameter
    /// units.  Set larger (e.g. 0.8) at coarse levels for wide exploration and
    /// smaller (e.g. 0.1) at fine levels for local refinement.
    pub cma_sigma0: f64,

    /// Maximum CMA-ES generations at this level.
    pub max_generations: usize,

    /// CMA-ES population size λ (0 = auto: 4 + ⌊3·ln n⌋).
    pub lambda: usize,

    /// IPOP restart count for this level (0 = disabled).
    pub ipop_restarts: usize,
}

impl CmaMiLevelConfig {
    /// Convenience constructor: create a level with the given shrink, smoothing,
    /// σ₀, and generation budget.  All other fields use sensible defaults.
    pub fn new(shrink: usize, sigma_mm: f64, cma_sigma0: f64, max_generations: usize) -> Self {
        Self {
            shrink,
            shrink_per_axis: None,
            sigma_mm,
            cma_sigma0,
            max_generations,
            lambda: 0,
            ipop_restarts: 0,
        }
    }
}

// ── Main config ───────────────────────────────────────────────────────────────

/// Configuration for the CMA-ES → RSGD cascade registration pipeline.
///
/// Tune `coarse_shrink` first: a factor of 8 on a typical 256³ brain image
/// yields 32³ pyramid levels (~32k voxels), keeping each MI evaluation
/// under ~1 ms on CPU.  Increase `cma_config.max_generations` if the search
/// stalls before convergence.
///
/// For multi-scale search, populate `pyramid_schedule` with a sequence of
/// [`CmaMiLevelConfig`] entries from coarsest to finest. When non-empty,
/// `pyramid_schedule` overrides `coarse_shrink` / `coarse_sigma_mm`.
#[derive(Debug, Clone)]
pub struct CmaMiConfig {
    /// CMA-ES solver settings (used for the single-level path; each level in
    /// `pyramid_schedule` overrides sigma0, max_generations, lambda, and
    /// ipop_restarts but inherits seed, sigma_tol, ftol, and record_history).
    pub cma_config: crate::optimizer::CmaEsConfig,

    /// Isotropic shrink factor applied to both images for the CMA-ES search
    /// level. Used only when `pyramid_schedule` is empty. Default: **8**.
    pub coarse_shrink: usize,

    /// Gaussian smoothing sigma applied before downsampling, in physical units
    /// (mm). Used only when `pyramid_schedule` is empty. Default: **4.0 mm**.
    pub coarse_sigma_mm: f64,

    /// Number of histogram bins for MI estimation. Default: **32**.
    pub num_mi_bins: usize,

    /// Fraction of voxels randomly sampled per MI evaluation ∈ (0, 1].
    /// Sparse sampling dramatically reduces wall time at the cost of
    /// gradient variance. Default: **0.15** (15%).
    pub sampling_percentage: f32,

    /// Half-range for translation parameters in mm. The CMA-ES searches
    /// `t ∈ [−range, +range]` after normalisation. Default: **60.0 mm**.
    pub translation_range_mm: f64,

    /// Half-range for rotation parameters in radians. Default: **π/4**.
    pub rotation_range_rad: f64,

    /// Optional RSGD fine-refinement configuration. When `Some`, a
    /// [`GlobalMiRegistration`] run is started from the CMA-ES solution.
    /// Default: **None** (CMA-ES result used directly).
    pub rsgd_refine: Option<GlobalMiConfig>,

    /// When `true` and `initial_translation` is `None`, automatically compute
    /// a center-of-mass translation to pre-align the images. Default: **true**.
    ///
    /// Note: CoM initialisation is unreliable for CT↔MRI T1 because HU
    /// densities are bone-dominated while T1 reflects tissue water content.
    /// Set `false` for cross-modal registration.
    pub use_com_init: bool,

    /// Per-axis shrink factors `[sz, sy, sx]` for the single-level CMA-ES
    /// pyramid. Overrides `coarse_shrink` when `Some`. Used only when
    /// `pyramid_schedule` is empty. Default: **None** (isotropic).
    pub shrink_per_axis: Option<[usize; 3]>,

    /// Number of IPOP-CMA-ES restarts for the single-level path. When > 0,
    /// `register_rigid` uses `CmaEsOptimizer::run_ipop`. Each restart doubles
    /// the population.  Default: **0** (no restarts).
    pub ipop_restarts: usize,

    /// Mutual information variant used during CMA-ES evaluation.
    ///
    /// - `Mattes` (default) — cubic B-spline Parzen windows; well-tested.
    /// - `Normalized(JointEntropy)` — NMI = (H(X)+H(Y))/H(X,Y); more robust
    ///   to overlap changes during rotation; recommended for brain CT↔MRI.
    /// - `Standard` — Viola–Wells; simpler but less accurate at sparse sampling.
    ///
    /// Default: `Mattes` (single-level path); `Normalized(JointEntropy)` in
    /// `brain_rigid_default()` and `brain_rigid_multiscale()`.
    pub mi_variant: MutualInformationVariant,

    /// Multi-scale CMA-ES cascade schedule (coarse → fine).
    ///
    /// When non-empty, each element defines one pyramid level.  The best
    /// parameter vector found at level *k* seeds level *k+1* at its σ₀.
    /// All levels share `num_mi_bins`, `sampling_percentage`,
    /// `translation_range_mm`, `rotation_range_rad`, `mi_variant`, and the
    /// seed / σ_tol / ftol from `cma_config`.
    ///
    /// When empty (default), the single-level path using `coarse_shrink` and
    /// `cma_config` is used.
    pub pyramid_schedule: Vec<CmaMiLevelConfig>,
}

impl Default for CmaMiConfig {
    fn default() -> Self {
        Self {
            cma_config: crate::optimizer::CmaEsConfig {
                sigma0: 0.3,
                lambda: 0,
                max_generations: 400,
                sigma_tol: 1e-8,
                // The MI objective is −MI(x) which is always negative.
                // ftol = 1e-12 would fire immediately since −MI < 1e-12 always.
                // Setting NEG_INFINITY disables this criterion; the search
                // terminates via max_generations or sigma_tol instead.
                ftol: f64::NEG_INFINITY,
                seed: 0xcafe_babe_dead_beef,
                parallel_population: true,
                record_history: false,
            },
            coarse_shrink: 8,
            coarse_sigma_mm: 4.0,
            num_mi_bins: 32,
            sampling_percentage: 0.15,
            translation_range_mm: 60.0,
            rotation_range_rad: std::f64::consts::FRAC_PI_4,
            rsgd_refine: None,
            use_com_init: true,
            shrink_per_axis: None,
            ipop_restarts: 0,
            mi_variant: MutualInformationVariant::Mattes,
            pyramid_schedule: Vec::new(),
        }
    }
}

impl CmaMiConfig {
    /// Pre-tuned configuration for brain CT↔MRI T1 rigid registration without
    /// brain extraction or masking.
    ///
    /// Key design choices:
    /// - `coarse_shrink = 8`: yields ~4×64×64 pyramid levels on typical brain
    ///   images; each MI evaluation ≈ 5 ms in release.
    /// - `sampling_percentage = 0.30`: ~4,900 samples at shrink=8 (4.8/bin).
    /// - `sigma0 = 0.7`: covers ~42 mm from the 60 mm search range.
    /// - `use_com_init = false`: CoM is unreliable for CT/MRI (HU vs T1).
    /// - `mi_variant = Normalized(AverageEntropy)`: 2·MI/(H(X)+H(Y)) is immune
    ///   to the OOB zero-pad artefact that inflates JointEntropy NMI when the
    ///   transform maps many voxels outside the moving image field of view.
    pub fn brain_rigid_default() -> Self {
        Self {
            cma_config: crate::optimizer::CmaEsConfig {
                sigma0: 0.7,
                lambda: 0,
                max_generations: 200,
                sigma_tol: 1e-8,
                ftol: f64::NEG_INFINITY,
                seed: 0xcafe_babe_dead_beef,
                parallel_population: true,
                record_history: false,
            },
            coarse_shrink: 8,
            coarse_sigma_mm: 4.0,
            num_mi_bins: 32,
            sampling_percentage: 0.30,
            translation_range_mm: 60.0,
            rotation_range_rad: std::f64::consts::FRAC_PI_4,
            rsgd_refine: None,
            use_com_init: false,
            shrink_per_axis: None,
            ipop_restarts: 0,
            mi_variant: MutualInformationVariant::Normalized(NormalizationMethod::AverageEntropy),
            pyramid_schedule: Vec::new(),
        }
    }

    /// Pre-tuned configuration for fast exploratory CT↔MRI registration.
    ///
    /// Uses a very coarse pyramid (shrink=16) and wider search ranges for rapid
    /// landscape exploration. Best used as a first pass before refining with
    /// `brain_rigid_default()`.  Each MI evaluation ≈ 1 ms in release.
    pub fn fast_exploratory() -> Self {
        Self {
            cma_config: crate::optimizer::CmaEsConfig {
                sigma0: 0.5,
                lambda: 0,
                max_generations: 100,
                sigma_tol: 1e-6,
                ftol: f64::NEG_INFINITY,
                seed: 0xcafe_babe_dead_beef,
                parallel_population: true,
                record_history: false,
            },
            coarse_shrink: 16,
            coarse_sigma_mm: 8.0,
            num_mi_bins: 16,
            sampling_percentage: 0.25,
            translation_range_mm: 100.0,
            rotation_range_rad: std::f64::consts::FRAC_PI_2,
            rsgd_refine: None,
            use_com_init: false,
            shrink_per_axis: None,
            ipop_restarts: 0,
            mi_variant: MutualInformationVariant::Mattes,
            pyramid_schedule: Vec::new(),
        }
    }

    /// Pre-tuned configuration for thin-slab CT volumes with few z-slices.
    ///
    /// Uses anisotropic shrink factors `[1, 8, 8]` to preserve z-resolution
    /// while downsampling the xy plane. Suitable for CT volumes with < 64 slices
    /// at coarse spacing (e.g., RIRE patient data with ~30 slices at 4mm).
    pub fn thin_slab_ct_default() -> Self {
        Self {
            shrink_per_axis: Some([1, 8, 8]),
            coarse_shrink: 8, // fallback if shrink_per_axis is ignored
            ..Self::brain_rigid_default()
        }
    }

    /// Pre-tuned three-level coarse-to-fine CMA-ES cascade for brain CT↔MRI T1.
    ///
    /// Runs three sequential passes:
    ///
    /// | Level | Shrink | σ_mm | σ₀  | Max gen | Purpose                     |
    /// |-------|--------|------|-----|---------|-----------------------------|
    /// |   0   |   16   |  8.0 | 0.8 |   100   | Wide exploration, very fast |
    /// |   1   |    8   |  4.0 | 0.3 |   200   | Refine to rough basin       |
    /// |   2   |    4   |  2.0 | 0.1 |   100   | Fine convergence            |
    ///
    /// The best parameters from each level seed the next.  Total wall time is
    /// roughly 3× the single `brain_rigid_default()` run but typically produces
    /// better TRE by escaping coarse-scale local maxima.
    ///
    /// Uses NMI (AverageEntropy = 2·MI/(H(X)+H(Y))) which is immune to the
    /// out-of-bounds (OOB) zero-pad artefact: when the transform maps most fixed
    /// voxels outside the moving image, the OOB samples return 0.0 which inflates
    /// the JointEntropy NMI (H(X)+H(Y))/H(X,Y) by concentrating the joint
    /// histogram in one column.  AverageEntropy NMI = 2·MI/(H(X)+H(Y)) correctly
    /// assigns zero score when MI=0 (full OOB or random mapping), making it a
    /// much cleaner objective for CMA-ES cold-start search.
    pub fn brain_rigid_multiscale() -> Self {
        Self {
            // cma_config provides the template: seed, sigma_tol, ftol, record_history.
            // Per-level sigma0, max_generations, lambda, ipop_restarts come from
            // pyramid_schedule entries.
            cma_config: crate::optimizer::CmaEsConfig {
                sigma0: 0.8, // overridden per level
                lambda: 0,
                max_generations: 100, // overridden per level
                sigma_tol: 1e-8,
                ftol: f64::NEG_INFINITY,
                seed: 0xcafe_babe_dead_beef,
                parallel_population: true,
                record_history: false,
            },
            coarse_shrink: 8,     // unused when pyramid_schedule is non-empty
            coarse_sigma_mm: 4.0, // unused when pyramid_schedule is non-empty
            num_mi_bins: 32,
            sampling_percentage: 0.25,
            translation_range_mm: 60.0,
            rotation_range_rad: std::f64::consts::FRAC_PI_4,
            rsgd_refine: None,
            use_com_init: false,
            shrink_per_axis: None,
            ipop_restarts: 0,
            // AverageEntropy NMI = 2·MI/(H(X)+H(Y)) is immune to the OOB zero-pad
            // artefact that inflates JointEntropy NMI at large displacements.
            mi_variant: MutualInformationVariant::Normalized(NormalizationMethod::AverageEntropy),
            pyramid_schedule: vec![
                CmaMiLevelConfig::new(16, 8.0, 0.8, 100),
                CmaMiLevelConfig::new(8, 4.0, 0.3, 200),
                CmaMiLevelConfig::new(4, 2.0, 0.1, 100),
            ],
        }
    }

    /// Pre-tuned three-level coarse-to-fine CMA-ES cascade for **thin-slab** CT volumes
    /// (typically < 50 z-slices at >= 2 mm z-spacing, e.g. RIRE 29-slice CT).
    ///
    /// Uses **anisotropic** per-axis shrink to preserve all z-slices at every pyramid
    /// level.  Isotropic shrink destroys z-information for thin CT slabs and produces
    /// spurious MI maxima.
    ///
    /// | Level | shrink_per_axis | s_mm | s0  | Max gen | CT voxels (RIRE)  |
    /// |-------|-----------------|------|-----|---------|-------------------|
    /// |   0   | [1, 16, 16]     |  8.0 | 0.8 |   100   | 29x32x32 ~30 K    |
    /// |   1   | [1,  8,  8]     |  4.0 | 0.3 |   200   | 29x64x64 ~119 K   |
    /// |   2   | [1,  4,  4]     |  2.0 | 0.1 |   100   | 29x128x128 ~476 K |
    ///
    /// Otherwise identical to `brain_rigid_multiscale()` (NMI, 32 bins, 25% sampling,
    /// +/-60 mm / +/-pi/4 search, CoM init disabled).
    pub fn brain_rigid_multiscale_thin_slab() -> Self {
        Self {
            pyramid_schedule: vec![
                CmaMiLevelConfig {
                    shrink_per_axis: Some([1, 16, 16]),
                    // 1 IPOP restart doubles population → better escape from false MI maxima
                    // at the very coarse 29×32×32 pyramid level where MI is noisy.
                    ipop_restarts: 1,
                    ..CmaMiLevelConfig::new(16, 8.0, 0.8, 150)
                },
                CmaMiLevelConfig {
                    shrink_per_axis: Some([1, 8, 8]),
                    ..CmaMiLevelConfig::new(8, 4.0, 0.3, 200)
                },
                CmaMiLevelConfig {
                    shrink_per_axis: Some([1, 4, 4]),
                    ..CmaMiLevelConfig::new(4, 2.0, 0.15, 100)
                },
            ],
            ..Self::brain_rigid_multiscale()
        }
    }
}

// ─── Result ─────────────────────────────────────────────────────────────────────────────

/// Result produced by [`CmaMiRegistration::register_rigid`](super::CmaMiRegistration::register_rigid).
#[derive(Debug, Clone)]
pub struct CmaMiResult {
    /// 4×4 homogeneous matrix of the final transform (row-major, f64).
    pub matrix: [f64; 16],

    /// Final MI value (positive; negated from the CMA-ES loss).
    /// Reflects the CMA-ES coarse-level MI, not the full-resolution value
    /// even when RSGD refinement is applied.
    pub final_mi: f64,

    /// Number of CMA-ES generations executed (last level for cascade mode).
    pub cma_generations: usize,

    /// Reason the CMA-ES loop terminated (last level for cascade mode).
    pub cma_stop_reason: StopReason,

    /// CMA-ES final step-size σ (last level for cascade mode).
    pub cma_final_sigma: f64,

    /// Total RSGD iterations across all resolution levels (0 if no refinement).
    pub rsgd_iterations: usize,

    /// Per-iteration loss history from RSGD refinement (empty if no refinement).
    pub rsgd_loss_history: Vec<f64>,

    /// Normalised CMA-ES best parameter vector `[α_n, β_n, γ_n, tz_n, ty_n, tx_n]`.
    /// Each component is in `[−1, 1]`; multiply by `rotation_range_rad` (first 3) or
    /// `translation_range_mm` (last 3) to recover physical units.
    /// Populated from the last cascade level in multi-scale mode.
    pub cma_best_params: Vec<f64>,
}
