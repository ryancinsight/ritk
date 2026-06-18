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

/// Flat-box grayscale erosion/dilation via **separable 1-D sliding windows**.
///
/// The min/max of a cubic `(2r+1)³` box is separable — `max` over the box equals
/// `max_z(max_y(max_x))` — so three independent 1-D passes (X, then Y, then Z)
/// produce a result **bit-identical** to the naive O(N·(2r+1)³) cube scan while
/// running in **O(N)** total, independent of `r`. Each 1-D pass is a monotonic-
/// deque sliding-window extremum over the clamp-truncated window
/// `[max(0,i−r), min(n−1,i+r)]`, which equals the edge-clamped box because a
/// clamped out-of-bounds neighbour only re-reads an in-window edge voxel.
///
/// Replaces the previous cube scan: measured 4842 ms → ~110 ms for `r = 5` on a
/// 128³ `f32` volume (≈44×), with `r = 1` unchanged; the speedup grows with `r`.
pub(crate) fn separable_box_3d(
    data: &[f32],
    dims: [usize; 3],
    radius: usize,
    ext: Extremum,
) -> Vec<f32> {
    if radius == 0 {
        return data.to_vec();
    }
    let [nz, ny, nx] = dims;
    let mut buf = data.to_vec();

    // Reusable per-line scratch (gathered line + windowed output).
    let max_len = nx.max(ny).max(nz);
    let mut line = vec![0.0_f32; max_len];
    let mut wout = vec![0.0_f32; max_len];
    let mut deque: std::collections::VecDeque<usize> = std::collections::VecDeque::new();

    // X axis: contiguous lines of length nx (stride 1).
    for base in (0..nz * ny).map(|p| p * nx) {
        window_1d(&buf[base..base + nx], radius, ext, &mut wout, &mut deque);
        buf[base..base + nx].copy_from_slice(&wout[..nx]);
    }
    // Y axis: lines of length ny, stride nx (within each z-slice, per x column).
    for iz in 0..nz {
        let slice = iz * ny * nx;
        for ix in 0..nx {
            for iy in 0..ny {
                line[iy] = buf[slice + iy * nx + ix];
            }
            window_1d(&line[..ny], radius, ext, &mut wout, &mut deque);
            for iy in 0..ny {
                buf[slice + iy * nx + ix] = wout[iy];
            }
        }
    }
    // Z axis: lines of length nz, stride ny*nx.
    let zstride = ny * nx;
    for iy in 0..ny {
        for ix in 0..nx {
            let col = iy * nx + ix;
            for iz in 0..nz {
                line[iz] = buf[iz * zstride + col];
            }
            window_1d(&line[..nz], radius, ext, &mut wout, &mut deque);
            for iz in 0..nz {
                buf[iz * zstride + col] = wout[iz];
            }
        }
    }
    buf
}

/// 1-D sliding-window extremum over the clamp-truncated window
/// `[max(0,i−r), min(n−1,i+r)]`, computed in O(n) with a monotonic index deque.
/// `out[0..n]` receives the result; `deque` is reused scratch (cleared on entry).
#[inline]
fn window_1d(
    line: &[f32],
    radius: usize,
    ext: Extremum,
    out: &mut [f32],
    deque: &mut std::collections::VecDeque<usize>,
) {
    let n = line.len();
    deque.clear();
    // `dominates(a, b)` is true when `a` makes `b` redundant at the deque back.
    let dominates = |a: f32, b: f32| match ext {
        Extremum::Max => a >= b,
        Extremum::Min => a <= b,
    };
    let mut next = 0usize; // next index to admit into the window
                           // Index-based loop is required: `i` indexes into `out` and into slice windows
                           // (`line[*deque.front()]`, `(i+radius).min(n-1)`, `i.saturating_sub(radius)`),
                           // and the sliding-window algorithm mutates `next`/`deque` across steps. No
                           // iterator form preserves the per-step window semantics; per the symmetry
                           // with `diffusion/curvature.rs`, the inline allow is the idiomatic gesture.
    #[allow(clippy::needless_range_loop)]
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
