//! Grayscale morphological filters for 3-D images.
//!
//! # Mathematical Specification
//!
//! Grayscale morphology extends binary morphology to scalar-valued images by
//! replacing set union/intersection with max/min over a structuring element B:
//!
//! - **Erosion**: `(E_B f)(x) = min_{b ∈ B} f(x + b)`
//! - **Dilation**: `(D_B f)(x) = max_{b ∈ B} f(x - b)`
//!
//! The structuring element used here is a cubic neighbourhood of half-width
//! `radius`, i.e. B = { b ∈ ℤ³ : |b_i| ≤ r for all i }. Boundary handling
//! uses replicate (clamp) padding.
//!
//! # Derived Operations
//!
//! - **Opening**: `O_B = D_B ∘ E_B` — removes bright features smaller than B.
//! - **Closing**: `C_B = E_B ∘ D_B` — removes dark features smaller than B.
//!
//! # Complexity
//!
//! O(N · (2r+1)³) where N is the total voxel count and r is the radius.
//!
//! # References
//!
//! - Serra, J. (1982). *Image Analysis and Mathematical Morphology*. Academic Press.
//! - Soille, P. (2003). *Morphological Image Analysis*, 2nd ed. Springer.

pub mod binary_pruning;
pub mod binary_thinning;
pub mod erode_object;
pub mod grayscale_dilation;
pub mod grayscale_erosion;

pub use binary_pruning::BinaryPruningFilter;
pub use binary_thinning::BinaryThinningFilter;
pub use erode_object::ErodeObjectMorphologyFilter;
pub use grayscale_dilation::GrayscaleDilation;
pub use grayscale_erosion::GrayscaleErosion;

pub mod hit_or_miss;
pub mod label_morphology;
pub mod top_hat;

pub use hit_or_miss::HitOrMissTransform;
pub use label_morphology::{
    LabelClosing, LabelDilation, LabelErosion, LabelOpening, MorphologicalReconstruction,
    ReconstructionMode,
};
pub use top_hat::{BlackTopHatFilter, WhiteTopHatFilter};

pub mod binary_closing;
pub mod binary_dilate;
pub mod binary_erode;
pub mod binary_fillhole;
pub mod binary_opening;
pub mod native;

pub use binary_closing::BinaryMorphologicalClosing;
pub use binary_dilate::BinaryDilateFilter;
pub use binary_erode::BinaryErodeFilter;
pub use binary_fillhole::BinaryFillholeFilter;
pub use binary_opening::BinaryMorphologicalOpening;

pub mod grayscale_closing;
pub mod grayscale_fillhole;
pub mod grayscale_gradient;
pub mod grayscale_grind_peak;
pub mod grayscale_opening;

pub use grayscale_closing::GrayscaleClosingFilter;
pub use grayscale_fillhole::GrayscaleFillholeFilter;
pub use grayscale_gradient::GrayscaleMorphologicalGradientFilter;
pub use grayscale_grind_peak::GrayscaleGrindPeakFilter;
pub use grayscale_opening::GrayscaleOpeningFilter;

pub mod grayscale_geodesic;
pub use grayscale_geodesic::{GrayscaleGeodesicDilationFilter, GrayscaleGeodesicErosionFilter};

pub mod h_transform;
pub use h_transform::{HConcaveFilter, HConvexFilter, HMaximaFilter, HMinimaFilter};

pub mod regional_extrema;
pub use regional_extrema::{
    RegionalMaximaFilter, RegionalMinimaFilter, ValuedRegionalMaximaFilter,
    ValuedRegionalMinimaFilter,
};

pub mod reconstruction_opening_closing;
pub use reconstruction_opening_closing::{
    ClosingByReconstructionFilter, OpeningByReconstructionFilter,
};

pub mod morphological_laplace;
pub use morphological_laplace::MorphologicalLaplacian;

pub mod binary_contour;
pub mod connectivity;
pub mod label_contour;
pub mod voting_binary;
pub mod voting_hole_filling;

pub use binary_contour::BinaryContourImageFilter;
pub use connectivity::Connectivity;
pub use label_contour::LabelContourImageFilter;
pub use voting_binary::VotingBinaryImageFilter;
pub use voting_hole_filling::VotingBinaryHoleFillingImageFilter;

pub mod iterate_structure;
pub use iterate_structure::{iterate_structure, iterate_structure_with_origin, BoolStructure};

pub mod types;
pub use types::ForegroundValue;

thread_local! {
    #[cfg_attr(
        all(windows, target_env = "gnu"),
        expect(
            clippy::missing_const_for_thread_local,
            reason = "clippy 1.97 false positive on the windows-gnu thread_local expansion: the initializer is already a const block"
        )
    )]
    static SCRATCH: std::cell::RefCell<(
        Vec<f32>,
        Vec<f32>,
        std::collections::VecDeque<usize>,
    )> = const { std::cell::RefCell::new((Vec::new(), Vec::new(), std::collections::VecDeque::new())) };
}

#[cfg(test)]
#[path = "tests_native_grayscale.rs"]
mod tests_native_grayscale;

/// True if voxel `(iz, iy, ix)` lies on the image border, **ignoring degenerate
/// (size-1) axes**.
///
/// A naive `iz == 0 || iz == nz-1 || …` test marks *every* voxel of a `z = 1`
/// (2-D) volume as border, because `iz == 0` is always true — which silently
/// turns border-seeded reconstructions (fill-hole, grind-peak) into the
/// identity on 2-D images, diverging from ITK/SimpleITK. Excluding size-1 axes
/// makes the border the frame of the genuinely-present dimensions (the 2-D frame
/// for a `z = 1` slab). See the `z = 1` degenerate-axis trap.
#[inline]
pub(crate) fn on_image_border(iz: usize, iy: usize, ix: usize, dims: [usize; 3]) -> bool {
    let [nz, ny, nx] = dims;
    (nz > 1 && (iz == 0 || iz == nz - 1))
        || (ny > 1 && (iy == 0 || iy == ny - 1))
        || (nx > 1 && (ix == 0 || ix == nx - 1))
}

/// Replicate-pad a flat `Z×Y×X` volume by `r` voxels on every face (edge-clamp).
///
/// ITK's composed grayscale opening/closing pads the input by the SE radius
/// before the erode/dilate pair and crops afterward (the "safe border"). Without
/// it, the second operation of the pair reads edge-clamped intermediate values
/// instead of the true padded ones, so the border band (within `r` of an edge)
/// diverges from `sitk.GrayscaleMorphological{Opening,Closing}`. Replicating a
/// degenerate (size-1) axis is harmless — the duplicated planes are identical,
/// so the min/max over them is unchanged.
pub(crate) fn pad_replicate_3d(data: &[f32], dims: [usize; 3], r: usize) -> (Vec<f32>, [usize; 3]) {
    if r == 0 {
        return (data.to_vec(), dims);
    }
    let [nz, ny, nx] = dims;
    let pdims = [nz + 2 * r, ny + 2 * r, nx + 2 * r];
    let [pz, py, px] = pdims;
    let mut out = vec![0.0_f32; pz * py * px];
    let clamp = |v: isize, n: usize| v.clamp(0, n as isize - 1) as usize;
    for z in 0..pz {
        let sz = clamp(z as isize - r as isize, nz);
        for y in 0..py {
            let sy = clamp(y as isize - r as isize, ny);
            for x in 0..px {
                let sx = clamp(x as isize - r as isize, nx);
                out[z * py * px + y * px + x] = data[sz * ny * nx + sy * nx + sx];
            }
        }
    }
    (out, pdims)
}

/// Crop the central `r`-voxel border off a padded `Z×Y×X` volume (inverse of
/// [`pad_replicate_3d`]).
pub(crate) fn crop_border_3d(data: &[f32], pdims: [usize; 3], r: usize) -> (Vec<f32>, [usize; 3]) {
    if r == 0 {
        return (data.to_vec(), pdims);
    }
    let [pz, py, px] = pdims;
    let dims = [pz - 2 * r, py - 2 * r, px - 2 * r];
    let [nz, ny, nx] = dims;
    let mut out = Vec::with_capacity(nz * ny * nx);
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                out.push(data[(z + r) * py * px + (y + r) * px + (x + r)]);
            }
        }
    }
    (out, dims)
}

// ── Shared morphological primitive ───────────────────────────────────────────────────────────

/// Which extremum a flat-box morphological scan computes.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Extremum {
    /// Grayscale erosion (minimum over the structuring element).
    Min,
    /// Grayscale dilation (maximum over the structuring element).
    Max,
}

/// Flat-box grayscale erosion/dilation via **separable 1-D sliding windows**,
/// parallelised over independent z-slices on all three passes via `moirai`.
///
/// The min/max of a cubic `(2r+1)³` box is separable — `max` over the box equals
/// `max_z(max_y(max_x))` — so three independent 1-D passes (X, then Y, then Z)
/// produce a result **bit-identical** to the naive O(N·(2r+1)³) cube scan while
/// running in **O(N)** total, independent of `r`. Each 1-D pass is a monotonic-
/// deque sliding-window extremum over the clamp-truncated window
/// `[max(0,i−r), min(n−1,i+r)]`, which equals the edge-clamped box because a
/// clamped out-of-bounds neighbour only re-reads an in-window edge voxel.
///
/// All three passes are parallelised:
/// - **X-pass**: `nz` z-slice chunks (each `ny×nx`); per-thread scratch `nx`.
/// - **Y-pass**: `nz` z-slice chunks; writes to a fresh buffer while reading the
///   X-processed source immutably (disjoint allocations — borrow-safe).
/// - **Z-pass**: transposed to `[n_cols, nz]` layout (Z-columns contiguous), then
///   `n_cols` independent `nz`-element chunks processed in parallel, then
///   transposed back to `[nz, ny, nx]`.
///
/// Scratch buffers are allocated once per thread via `thread_local!` storage and
/// grown-on-demand via `Vec::resize`, eliminating per-z-slice (X/Y passes) and
/// per-z-column (Z-pass) allocations (P-6).
///
/// Output is **bit-identical** to the serial version — the passes are
/// embarrassingly parallel with no data sharing within a pass.
// Clippy 1.97.0 reports the const initializer; 1.97.1 reports an `expect`
// for the same lint as unfulfilled.
#[allow(clippy::missing_const_for_thread_local, reason = "ratchet RITK-LINT-1")]
    for i in 0..n {
        let hi = (i + radius).min(n - 1);
        while next <= hi {
            let v = line[next];
            while let Some(&b) = deque.back() {
                if dominates(v, line[b]) {
                    deque.pop_back();
                } else {
                    break;
                }
            }
            deque.push_back(next);
            next += 1;
        }
        let lo = i.saturating_sub(radius);
        while let Some(&f) = deque.front() {
            if f < lo {
                deque.pop_front();
            } else {
                break;
            }
        }
        out[i] = line[*deque.front().expect("window non-empty: lo <= i <= hi")];
    }
}
