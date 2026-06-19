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
/// Output is **bit-identical** to the serial version — the passes are
/// embarrassingly parallel with no data sharing within a pass.
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
    let n_total = nz * ny * nx;

    // ── Pass 1: X-axis (contiguous nx-element rows) ──────────────────────────
    // nz z-slices × ny rows each; chunk = one z-slice (ny*nx elements).
    // Per-thread scratch: tmp[nx], wout_t[nx], deque_t — allocated once per slice.
    let mut buf = data.to_vec();
    moirai::for_each_chunk_mut_enumerated_with::<moirai::Adaptive, _, _>(
        &mut buf,
        ny * nx,
        |_iz, slice| {
            let mut wout_t = vec![0.0f32; nx];
            let mut deque_t = std::collections::VecDeque::new();
            let mut tmp = vec![0.0f32; nx];
            for iy in 0..ny {
                let base = iy * nx;
                tmp.copy_from_slice(&slice[base..base + nx]);
                window_1d(&tmp, radius, ext, &mut wout_t, &mut deque_t);
                slice[base..base + nx].copy_from_slice(&wout_t[..nx]);
            }
        },
    );

    // ── Pass 2: Y-axis (strided ny-element columns within each z-slice) ──────
    // nz z-slices; each thread processes nx Y-columns for its slice.
    // buf_x is the X-processed source, captured immutably; writes go to buf_y
    // (a separate allocation). `buf_x: &[f32]` is Sync; `buf_y` is mutably
    // borrowed by moirai — no aliasing.
    let buf_x = buf;
    let mut buf_y = vec![0.0f32; n_total];
    moirai::for_each_chunk_mut_enumerated_with::<moirai::Adaptive, _, _>(
        &mut buf_y,
        ny * nx,
        |iz, out_slice| {
            let src = &buf_x[iz * ny * nx..(iz + 1) * ny * nx];
            let mut col = vec![0.0f32; ny];
            let mut wout_t = vec![0.0f32; ny];
            let mut deque_t = std::collections::VecDeque::new();
            for ix in 0..nx {
                for iy in 0..ny {
                    col[iy] = src[iy * nx + ix];
                }
                window_1d(&col[..ny], radius, ext, &mut wout_t, &mut deque_t);
                for iy in 0..ny {
                    out_slice[iy * nx + ix] = wout_t[iy];
                }
            }
        },
    );

    // ── Pass 3: Z-axis (strided nz-element columns; transpose for contiguity) ─
    // Transpose buf_y from [nz, ny, nx] to [n_cols, nz] layout so Z-columns
    // are contiguous, run parallel window_1d over nz-element chunks, then
    // scatter back to [nz, ny, nx].
    let n_cols = ny * nx;
    let mut buf_zt = vec![0.0f32; nz * n_cols];
    // Forward transpose: buf_zt[col*nz + iz] = buf_y[iz*n_cols + col]
    for iz in 0..nz {
        for col in 0..n_cols {
            buf_zt[col * nz + iz] = buf_y[iz * n_cols + col];
        }
    }
    // Parallel window_1d over Z-columns (each nz contiguous elements).
    // z_col: &mut [f32] coerces to &[f32] for the read argument; the immutable
    // reborrow ends before copy_from_slice takes the mutable reborrow.
    moirai::for_each_chunk_mut_enumerated_with::<moirai::Adaptive, _, _>(
        &mut buf_zt,
        nz,
        |_col, z_col| {
            let mut wout_t = vec![0.0f32; nz];
            let mut deque_t = std::collections::VecDeque::new();
            window_1d(z_col, radius, ext, &mut wout_t, &mut deque_t);
            z_col.copy_from_slice(&wout_t[..nz]);
        },
    );
    // Inverse transpose: out[iz*n_cols + col] = buf_zt[col*nz + iz]
    let mut out = vec![0.0f32; n_total];
    for col in 0..n_cols {
        for iz in 0..nz {
            out[iz * n_cols + col] = buf_zt[col * nz + iz];
        }
    }
    out
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
