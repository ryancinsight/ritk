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
    /// # Errors
    ///
    /// - [`MpPcaError::TooFewVolumes`] for fewer than two volumes.
    /// - [`MpPcaError::VolumeLength`] when a volume's length disagrees with `shape`.
    /// - [`MpPcaError::NonFinite`] for a NaN or infinite sample.
    /// - [`MpPcaError::PatchExceedsImage`] when the window does not fit `shape`.
    /// - [`MpPcaError::Eigen`] when a window's eigendecomposition fails.
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

        let window_voxels = extent.voxels();
        let mut window = WindowWorkspace::new(window_voxels, depth);
        let mut sum = vec![T::ZERO; voxel_count * depth];
        let mut coverage = vec![0_usize; voxel_count];
        let mut sigma = vec![T::ZERO; voxel_count];
        let mut components = vec![0_usize; voxel_count];

        for center in 0..voxel_count {
            let origin = extent.origin(unravel(center, shape), shape);
            window.gather(origin, extent.extent(), shape, volumes);
            let boundary = window.reconstruct(self.estimator)?;
            for (slot, &voxel) in window.indices.iter().enumerate() {
                coverage[voxel] += 1;
                let row = &window.reconstruction[slot * depth..(slot + 1) * depth];
                for (d, &value) in row.iter().enumerate() {
                    sum[d * voxel_count + voxel] += value;
                }
            }
            sigma[center] = boundary.variance.sqrt();
            components[center] = boundary.signal_components;
        }

        let volumes = sum
            .chunks_exact(voxel_count)
            .map(|volume| {
                volume
                    .iter()
                    .zip(&coverage)
                    .map(|(&total, &count)| total / T::from_usize(count))
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

/// Reused per-window storage: the Casorati matrix, its voxel indices, and the
/// reconstruction, so the sweep allocates only inside the eigensolver.
struct WindowWorkspace<T> {
    voxels: usize,
    depth: usize,
    indices: Vec<usize>,
    casorati: Vec<T>,
    reconstruction: Vec<T>,
}

impl<T: RealScalar> WindowWorkspace<T> {
    fn new(voxels: usize, depth: usize) -> Self {
        Self {
            voxels,
            depth,
            indices: Vec::with_capacity(voxels),
            casorati: vec![T::ZERO; voxels * depth],
            reconstruction: vec![T::ZERO; voxels * depth],
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

    /// Project the Casorati matrix onto its signal subspace.
    ///
    /// The Gram matrix is formed on the smaller dimension `m` and divided by
    /// the larger `n` (Veraart et al. 2016, Eq. 2). Projecting onto its top
    /// `P̂` eigenvectors equals the rank-`P̂` truncated SVD of the Casorati
    /// matrix, whichever side the Gram matrix was formed on.
    fn reconstruct(&mut self, estimator: MpEstimator) -> Result<NoiseBoundary<T>, MpPcaError> {
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
        let mut gram = vec![T::ZERO; m * m];
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
        let eigen = symmetric_eigen_jacobi(&matrix.view()).map_err(MpPcaError::Eigen)?;
        let descending: Vec<T> = eigen.eigenvalues.iter().rev().copied().collect();
        let boundary = marchenko_pastur_boundary(&descending, n, estimator);
        let vectors = eigen
            .eigenvectors
            .as_slice()
            .expect("invariant: leto builds eigenvectors in contiguous row-major storage");
        // Eigenvalues ascend, so the signal eigenvectors are the last P̂ columns.
        let signal = m - boundary.signal_components..m;

        self.reconstruction.fill(T::ZERO);
        let out = &mut self.reconstruction;
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
