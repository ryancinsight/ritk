//! Mutual Information variant and normalization method type definitions.

/// Normalization method for Normalized Mutual Information (NMI).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormalizationMethod {
    /// Normalize by joint entropy: (H(X) + H(Y)) / H(X,Y)
    JointEntropy,
    /// Normalize by average of marginal entropies: 2 * MI / (H(X) + H(Y))
    AverageEntropy,
    /// Normalize by minimum: MI / min(H(X), H(Y))
    MinEntropy,
    /// Normalize by maximum: MI / max(H(X), H(Y))
    MaxEntropy }

/// Variant of Mutual Information to compute.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MutualInformationVariant {
    /// Standard Mutual Information (Viola-Wells).
    Standard,
    /// Mattes Mutual Information (Cubic B-Spline approximation via parameterization).
    Mattes,
    /// Normalized Mutual Information.
    Normalized(NormalizationMethod) }
