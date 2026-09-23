//! Typed failures of MP-PCA denoising.

/// Reasons an MP-PCA denoising request is rejected.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum MpPcaError {
    /// Fewer volumes than the Marchenko-Pastur boundary needs.
    ///
    /// With one volume the Casorati matrix has a single eigenvalue, which is
    /// signal and noise at once; the law needs at least two.
    #[error("MP-PCA needs at least {minimum} volumes; the series has {count}")]
    TooFewVolumes {
        /// Volumes supplied.
        count: usize,
        /// Minimum the estimator accepts.
        minimum: usize,
    },
    /// A patch extent is zero on some axis or holds fewer than two voxels.
    #[error("patch extent {extent:?} must be at least 1 on every axis and hold at least 2 voxels")]
    InvalidPatch {
        /// The rejected extent.
        extent: [usize; 3],
    },
    /// The patch does not fit inside the image on some axis.
    #[error("patch extent {extent:?} exceeds the image shape {shape:?}")]
    PatchExceedsImage {
        /// The patch extent.
        extent: [usize; 3],
        /// The image shape.
        shape: [usize; 3],
    },
    /// A volume's length disagrees with the image shape.
    #[error("volume {volume} holds {len} samples; the shape {shape:?} needs {expected}")]
    VolumeLength {
        /// Index of the offending volume.
        volume: usize,
        /// Its length.
        len: usize,
        /// The length the shape implies.
        expected: usize,
        /// The image shape.
        shape: [usize; 3],
    },
    /// A sample is NaN or infinite.
    #[error("volume {volume} sample {sample} is not finite")]
    NonFinite {
        /// Index of the offending volume.
        volume: usize,
        /// Index of the offending sample within it.
        sample: usize,
    },
    /// The eigendecomposition of a window's Gram matrix failed.
    #[error("eigendecomposition of the window Gram matrix failed")]
    Eigen(#[source] leto::LetoError),
}
