//! Sliding-window MP-PCA driver: Casorati assembly, eigendecomposition,
//! truncated reconstruction, and overlap averaging.

use super::threshold::{marchenko_pastur_boundary, NoiseBoundary};
use super::{MpEstimator, MpPcaError, PatchExtent};
use leto::Array2;
use leto_ops::{symmetric_eigen_jacobi, RealScalar};

/// Minimum volume count the Marchenko-Pastur boundary can split.
const MIN_VOLUMES: usize = 2;

/// MP-PCA denoiser over a series of co-registered 3-D volumes.
///
/// See the [module documentation](super) for the estimator.
///
/// # Examples
///
/// ```
/// use ritk_filter::mppca::MpPcaDenoiser;
///
/// // A rank-one series: every voxel scales the same volume profile.
/// let shape = [3, 3, 3];
/// let profile = [1.0_f64, 2.0, 3.0, 4.0];
/// let volumes: Vec<Vec<f64>> = profile
///     .iter()
///     .map(|&p| (0..27).map(|v| p * (1.0 + v as f64)).collect())
///     .collect();
/// let views: Vec<&[f64]> = volumes.iter().map(Vec::as_slice).collect();
///
/// let output = MpPcaDenoiser::default().denoise(shape, &views)?;
/// for (clean, denoised) in volumes.iter().zip(output.volumes()) {
///     for (a, b) in clean.iter().zip(denoised) {
///         assert!((a - b).abs() <= 1e-9 * a.abs());
///     }
/// }
/// # Ok::<(), ritk_filter::mppca::MpPcaError>(())
/// ```
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MpPcaDenoiser {
    extent: Option<PatchExtent>,
    estimator: MpEstimator,
}

impl MpPcaDenoiser {
    /// Use an explicit window extent instead of the derived one.
    #[must_use]
    pub fn with_extent(self, extent: PatchExtent) -> Self {
        Self {
            extent: Some(extent),
            ..self
        }
    }

    /// Select the Marchenko-Pastur ratio estimator.
    #[must_use]
    pub fn with_estimator(self, estimator: MpEstimator) -> Self {
        Self { estimator, ..self }
    }

    /// The window extent used for a series of `volumes` volumes: the explicit
    /// extent, or [`PatchExtent::for_volume_count`] when none was set.
    #[must_use]
    pub fn extent_for(&self, volumes: usize) -> PatchExtent {
        self.extent
            .unwrap_or_else(|| PatchExtent::for_volume_count(volumes))
    }

    /// Denoise a series of volumes sharing `shape` (row-major, last axis
    /// fastest).
    ///
    /// Windows are reconstructed in parallel; the result is bitwise identical
    /// to a sequential sweep (see the [module documentation](super)).
    ///
    /// # Errors
    ///
    /// - [`MpPcaError::TooFewVolumes`] for fewer than two volumes.
    /// - [`MpPcaError::VolumeLength`] when a volume's length disagrees with `shape`.
    /// - [`MpPcaError::NonFinite`] for a NaN or infinite sample.
    /// - [`MpPcaError::PatchExceedsImage`] when the window does not fit `shape`.
    /// - [`MpPcaError::Eigen`] when a window's eigendecomposition fails; the
    ///   first failing window in centre order is reported.
    pub fn denoise<T: RealScalar>(
        &self,
        shape: [usize; 3],
        volumes: &[&[T]],
    ) -> Result<MpPcaOutput<T>, MpPcaError> {
        let depth = volumes.len();
        if depth < MIN_VOLUMES {
            return Err(MpPcaError::TooFewVolumes {
                count: depth,
                minimum: MIN_VOLUMES,
            });
        }
        let voxel_count: usize = shape.iter().product();
        validate_series(shape, voxel_count, volumes)?;
        let extent = self.extent_for(depth);
        extent.fit(shape)?;
        let batch = window_batch::<T>(extent.voxels(), depth);
        self.sweep::<moirai::Parallel, T>(shape, volumes, extent, batch)
    }

    /// The window sweep under execution policy `P`, staging `batch` windows
    /// per parallel region, over a validated series.
    ///
    /// Each region reconstructs its windows into per-window slots, every
    /// worker reusing one [`WindowWorkspace`]; the slots are then added to the
    /// overlap sums serially in centre order. Every output voxel therefore
    /// sums its windows in ascending centre order — the order of a sequential
    /// sweep — whatever `P`, the worker count, or `batch`, so the result is
    /// bitwise independent of all three.
    pub(super) fn sweep<P: moirai::ExecutionPolicy, T: RealScalar>(
        &self,
        shape: [usize; 3],
        volumes: &[&[T]],
        extent: PatchExtent,
        batch: usize,
    ) -> Result<MpPcaOutput<T>, MpPcaError> {
        let depth = volumes.len();
        let voxel_count: usize = shape.iter().product();
        let window_voxels = extent.voxels();
        let estimator = self.estimator;
        let mut slots: Vec<WindowSlot<T>> = (0..batch.clamp(1, voxel_count))
            .map(|_| WindowSlot::new(window_voxels, depth))
            .collect();
        // Voxel-major overlap sums, so each window row adds contiguously.
        let mut sum = vec![T::ZERO; voxel_count * depth];
        let mut coverage = vec![0_usize; voxel_count];
        let mut sigma = vec![T::ZERO; voxel_count];
        let mut components = vec![0_usize; voxel_count];

        let staged = slots.len();
        for first in (0..voxel_count).step_by(staged) {
            let active = &mut slots[..staged.min(voxel_count - first)];
            for (offset, slot) in active.iter_mut().enumerate() {
                slot.center = first + offset;
            }
            moirai::for_each_chunk_mut_with_state::<P, _, _, _, _>(
                active,
                1,
                || WindowWorkspace::new(window_voxels, depth),
                |workspace, run| {
                    for slot in run {
                        let origin = extent.origin(unravel(slot.center, shape), shape);
                        workspace.gather(origin, extent.extent(), shape, volumes);
                        slot.outcome = workspace.reconstruct(estimator, &mut slot.reconstruction);
                    }
                },
            );
            for slot in active.iter_mut() {
                let boundary = std::mem::replace(&mut slot.outcome, Ok(NoiseBoundary::EMPTY))?;
                let origin = extent.origin(unravel(slot.center, shape), shape);
                accumulate(
                    &mut sum,
                    &mut coverage,
                    &slot.reconstruction,
                    origin,
                    extent.extent(),
                    shape,
                );
                sigma[slot.center] = boundary.variance.sqrt();
                components[slot.center] = boundary.signal_components;
            }
        }

        let volumes = (0..depth)
            .map(|d| {
                coverage
                    .iter()
                    .enumerate()
                    .map(|(voxel, &count)| sum[voxel * depth + d] / T::from_usize(count))
                    .collect()
            })
            .collect();
        Ok(MpPcaOutput {
            volumes,
            sigma,
            components,
        })
    }
}

/// Bytes of window reconstructions staged per parallel region.
///
/// Bounds the staging memory independently of the image size while giving
/// every worker many windows: at the `dwidenoise` geometry (`V = 64`,
/// `D = 60`, `f64`) one slot is 30 KiB, so the budget stages 546 windows —
/// over 20 per worker on a 24-thread host, against which the static
/// contiguous partition's imbalance of at most one window is small.
const WINDOW_BATCH_BYTES: usize = 16 << 20;

/// Windows per parallel region: [`WINDOW_BATCH_BYTES`] of reconstructions,
/// at least one.
fn window_batch<T>(window_voxels: usize, depth: usize) -> usize {
    let slot_bytes = (window_voxels * depth * size_of::<T>()).max(1);
    (WINDOW_BATCH_BYTES / slot_bytes).max(1)
}

/// Add one window's `V × D` reconstruction to the voxel-major overlap sums.
///
/// A window row along the last axis is `extent[2]` consecutive voxels in both
/// the image and the reconstruction, so each row adds as one contiguous run
/// of `extent[2] · D` values.
fn accumulate<T: RealScalar>(
    sum: &mut [T],
    coverage: &mut [usize],
    reconstruction: &[T],
    origin: [usize; 3],
    extent: [usize; 3],
    shape: [usize; 3],
) {
    let depth = sum.len() / coverage.len();
    let run = extent[2] * depth;
    let mut rows = reconstruction.chunks_exact(run);
    for z in origin[0]..origin[0] + extent[0] {
        for y in origin[1]..origin[1] + extent[1] {
            let first = (z * shape[1] + y) * shape[2] + origin[2];
            let row = rows
                .next()
                .expect("invariant: a reconstruction holds extent[0] · extent[1] rows");
            for (total, &value) in sum[first * depth..first * depth + run].iter_mut().zip(row) {
                *total += value;
            }
            for count in &mut coverage[first..first + extent[2]] {
                *count += 1;
            }
        }
    }
}

/// One window staged between the parallel reconstruction and the serial
/// overlap sum.
struct WindowSlot<T> {
    center: usize,
    /// `V × D` reconstruction; row `r` is the window's `r`-th voxel in
    /// `z, y, x` order.
    reconstruction: Vec<T>,
    outcome: Result<NoiseBoundary<T>, MpPcaError>,
}

impl<T: RealScalar> WindowSlot<T> {
    fn new(voxels: usize, depth: usize) -> Self {
        Self {
            center: 0,
            reconstruction: vec![T::ZERO; voxels * depth],
            outcome: Ok(NoiseBoundary::EMPTY),
        }
    }
}

/// Denoised series with its per-voxel noise and rank maps.
#[derive(Debug, Clone, PartialEq)]
pub struct MpPcaOutput<T> {
    volumes: Vec<Vec<T>>,
    sigma: Vec<T>,
    components: Vec<usize>,
}

impl<T> MpPcaOutput<T> {
    /// Denoised volumes, in input order, each in the input layout.
    #[must_use]
    pub fn volumes(&self) -> &[Vec<T>] {
        &self.volumes
    }

    /// Take ownership of the denoised volumes.
    #[must_use]
    pub fn into_volumes(self) -> Vec<Vec<T>> {
        self.volumes
    }

    /// Noise standard deviation `σ̂` estimated from each voxel's own window.
    #[must_use]
    pub fn noise_sigma(&self) -> &[T] {
        &self.sigma
    }

    /// Signal component count `P̂` of each voxel's own window.
    #[must_use]
    pub fn signal_components(&self) -> &[usize] {
        &self.components
    }
}

/// Reject a series whose volumes disagree with `shape` or carry non-finite samples.
fn validate_series<T: RealScalar>(
    shape: [usize; 3],
    voxel_count: usize,
    volumes: &[&[T]],
) -> Result<(), MpPcaError> {
    for (volume, samples) in volumes.iter().enumerate() {
        if samples.len() != voxel_count {
            return Err(MpPcaError::VolumeLength {
                volume,
                len: samples.len(),
                expected: voxel_count,
                shape,
            });
        }
        if let Some(sample) = samples.iter().position(|value| !value.is_finite()) {
            return Err(MpPcaError::NonFinite { volume, sample });
        }
    }
    Ok(())
}

/// Row-major multi-index of a flat voxel index.
fn unravel(index: usize, shape: [usize; 3]) -> [usize; 3] {
    let plane = shape[1] * shape[2];
    [index / plane, (index % plane) / shape[2], index % shape[2]]
}

/// One worker's reusable window storage: the Casorati matrix, its voxel
/// indices, the Gram matrix, and the spectrum buffers, so a worker allocates
/// only inside the eigensolver.
struct WindowWorkspace<T> {
    voxels: usize,
    depth: usize,
    indices: Vec<usize>,
    casorati: Vec<T>,
    gram: Vec<T>,
    descending: Vec<T>,
    trailing: Vec<T>,
}

impl<T: RealScalar> WindowWorkspace<T> {
    fn new(voxels: usize, depth: usize) -> Self {
        let m = voxels.min(depth);
        Self {
            voxels,
            depth,
            indices: Vec::with_capacity(voxels),
            casorati: vec![T::ZERO; voxels * depth],
            gram: Vec::with_capacity(m * m),
            descending: Vec::with_capacity(m),
            trailing: Vec::with_capacity(m + 1),
        }
    }

    /// Fill the `V × D` Casorati matrix (row = voxel, column = volume).
    fn gather(
        &mut self,
        origin: [usize; 3],
        extent: [usize; 3],
        shape: [usize; 3],
        volumes: &[&[T]],
    ) {
        self.indices.clear();
        for z in origin[0]..origin[0] + extent[0] {
            for y in origin[1]..origin[1] + extent[1] {
                let row_start = (z * shape[1] + y) * shape[2];
                self.indices
                    .extend(row_start + origin[2]..row_start + origin[2] + extent[2]);
            }
        }
        for (slot, &voxel) in self.indices.iter().enumerate() {
            let row = &mut self.casorati[slot * self.depth..(slot + 1) * self.depth];
            for (value, volume) in row.iter_mut().zip(volumes) {
                *value = volume[voxel];
            }
        }
    }

    /// Project the Casorati matrix onto its signal subspace, writing the
    /// `V × D` reconstruction into `out`.
    ///
    /// The Gram matrix is formed on the smaller dimension `m` and divided by
    /// the larger `n` (Veraart et al. 2016, Eq. 2). Projecting onto its top
    /// `P̂` eigenvectors equals the rank-`P̂` truncated SVD of the Casorati
    /// matrix, whichever side the Gram matrix was formed on.
    fn reconstruct(
        &mut self,
        estimator: MpEstimator,
        out: &mut [T],
    ) -> Result<NoiseBoundary<T>, MpPcaError> {
        let (voxels, depth) = (self.voxels, self.depth);
        let volumes_smaller = depth <= voxels;
        let (m, n) = if volumes_smaller {
            (depth, voxels)
        } else {
            (voxels, depth)
        };
        let y = &self.casorati;
        let at = |row: usize, col: usize| y[row * depth + col];

        let scale = T::from_usize(n);
        let mut gram = std::mem::take(&mut self.gram);
        gram.clear();
        gram.resize(m * m, T::ZERO);
        for i in 0..m {
            for j in i..m {
                let mut dot = T::ZERO;
                if volumes_smaller {
                    for r in 0..voxels {
                        dot += at(r, i) * at(r, j);
                    }
                } else {
                    for c in 0..depth {
                        dot += at(i, c) * at(j, c);
                    }
                }
                let value = dot / scale;
                gram[i * m + j] = value;
                gram[j * m + i] = value;
            }
        }

        let matrix = Array2::from_shape_vec([m, m], gram).map_err(MpPcaError::Eigen)?;
        let eigen = symmetric_eigen_jacobi(&matrix.view());
        self.gram = matrix.into_vec();
        let eigen = eigen.map_err(MpPcaError::Eigen)?;
        self.descending.clear();
        self.descending
            .extend(eigen.eigenvalues.iter().rev().copied());
        let boundary =
            marchenko_pastur_boundary(&self.descending, n, estimator, &mut self.trailing);
        let vectors = eigen
            .eigenvectors
            .as_slice()
            .expect("invariant: leto builds eigenvectors in contiguous row-major storage");
        // Eigenvalues ascend, so the signal eigenvectors are the last P̂ columns.
        let signal = m - boundary.signal_components..m;

        out.fill(T::ZERO);
        if volumes_smaller {
            // Ŷ = Y·U·Uᵀ over volume-space eigenvectors u_k.
            for r in 0..voxels {
                for k in signal.clone() {
                    let mut coefficient = T::ZERO;
                    for i in 0..depth {
                        coefficient += at(r, i) * vectors[i * m + k];
                    }
                    for i in 0..depth {
                        out[r * depth + i] += coefficient * vectors[i * m + k];
                    }
                }
            }
        } else {
            // Ŷ = W·Wᵀ·Y over voxel-space eigenvectors w_k.
            for c in 0..depth {
                for k in signal.clone() {
                    let mut coefficient = T::ZERO;
                    for r in 0..voxels {
                        coefficient += vectors[r * m + k] * at(r, c);
                    }
                    for r in 0..voxels {
                        out[r * depth + c] += coefficient * vectors[r * m + k];
                    }
                }
            }
        }
        Ok(boundary)
    }
}
