//! Validated descriptor geometry and deterministic center selection.

use super::MindSscError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct DescriptorGeometry {
    patch_radius: [usize; 3],
    neighbour_dilation: [usize; 3],
}

impl DescriptorGeometry {
    pub(super) const fn patch_radius(self) -> [usize; 3] {
        self.patch_radius
    }

    pub(super) const fn neighbour_dilation(self) -> [usize; 3] {
        self.neighbour_dilation
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum SamplingSelection {
    Dense,
    Stratified { max_samples: usize },
    Indices(Box<[usize]>),
}

/// Deterministic fixed-domain sampling policy for MIND-SSC.
///
/// Stratified sampling recursively bisects the longest physical grid cells,
/// assigns a population-proportional quota to each resulting spatial stratum,
/// and selects the lowest fixed-seed hash ranks without replacement. It uses
/// every eligible center when the domain fits the budget. Caller-provided
/// indices are linear fixed-image C-order indices and remain in caller order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MindSscSampling {
    selection: SamplingSelection,
}

impl MindSscSampling {
    /// Select every masked complete-support center.
    #[must_use]
    pub const fn dense() -> Self {
        Self {
            selection: SamplingSelection::Dense,
        }
    }

    /// Select at most `max_samples` deterministically ranked samples.
    ///
    /// # Errors
    ///
    /// Returns [`MindSscError::EmptySampleBudget`] when `max_samples` is zero.
    pub fn try_stratified(max_samples: usize) -> Result<Self, MindSscError> {
        if max_samples == 0 {
            return Err(MindSscError::EmptySampleBudget);
        }
        Ok(Self {
            selection: SamplingSelection::Stratified { max_samples },
        })
    }

    /// Select explicit linear fixed-image C-order indices.
    ///
    /// # Errors
    ///
    /// Returns [`MindSscError::EmptySampleIndices`] for an empty set and
    /// [`MindSscError::DuplicateSampleIndex`] for a duplicate. Image bounds,
    /// complete support, and mask membership are validated during preparation.
    pub fn try_indices(indices: impl IntoIterator<Item = usize>) -> Result<Self, MindSscError> {
        let indices = indices.into_iter().collect::<Vec<_>>();
        if indices.is_empty() {
            return Err(MindSscError::EmptySampleIndices);
        }
        let mut sorted = indices.clone();
        sorted.sort_unstable();
        if let Some(index) = sorted
            .windows(2)
            .find_map(|pair| (pair[0] == pair[1]).then_some(pair[0]))
        {
            return Err(MindSscError::DuplicateSampleIndex { index });
        }
        Ok(Self {
            selection: SamplingSelection::Indices(indices.into_boxed_slice()),
        })
    }

    pub(super) fn selection(&self) -> &SamplingSelection {
        &self.selection
    }
}

/// Validated MIND-SSC descriptor and sampling configuration.
///
/// The default is the MICCAI 2013 SSC geometry: a `3×3×3` patch and neighbour
/// distance two voxels. Per-center normalized Hamming loss lies in `[0, 1]`.
/// Hoeffding's inequality requires 6,623 independent uniform samples for a
/// two-sided error at most 0.02 with probability at least 0.99; 8,192 is the
/// next power of two. The deterministic design reproduces one fixed sample set;
/// the probability statement characterizes random uniform sampling and does
/// not make fixed-seed population or clinical-validation claims.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MindSscConfig {
    patch_radius: [usize; 3],
    neighbour_dilation: [usize; 3],
    sampling: MindSscSampling,
}

impl MindSscConfig {
    /// Default deterministic sample cap from a 99% Hoeffding bound at ±0.02.
    pub const DEFAULT_MAX_SAMPLES: usize = 8_192;

    /// Validate descriptor geometry and a fixed-domain sampling policy.
    ///
    /// # Errors
    ///
    /// Returns a typed error when a radius or dilation is zero, their sum
    /// overflows, or the support cannot be represented by signed offsets.
    pub fn try_new(
        patch_radius: [usize; 3],
        neighbour_dilation: [usize; 3],
        sampling: MindSscSampling,
    ) -> Result<Self, MindSscError> {
        for axis in 0..3 {
            if patch_radius[axis] == 0 {
                return Err(MindSscError::InvalidPatchRadius {
                    axis,
                    value: patch_radius[axis],
                });
            }
            if neighbour_dilation[axis] == 0 {
                return Err(MindSscError::InvalidNeighbourDilation {
                    axis,
                    value: neighbour_dilation[axis],
                });
            }
            let halo = patch_radius[axis]
                .checked_add(neighbour_dilation[axis])
                .ok_or(MindSscError::SupportOverflow {
                    axis,
                    patch_radius: patch_radius[axis],
                    neighbour_dilation: neighbour_dilation[axis],
                })?;
            if isize::try_from(halo).is_err() {
                return Err(MindSscError::SupportOverflow {
                    axis,
                    patch_radius: patch_radius[axis],
                    neighbour_dilation: neighbour_dilation[axis],
                });
            }
        }
        Ok(Self {
            patch_radius,
            neighbour_dilation,
            sampling,
        })
    }

    /// Patch half-width in fixed-grid voxels, ordered `[z, y, x]`.
    #[must_use]
    pub const fn patch_radius(&self) -> [usize; 3] {
        self.patch_radius
    }

    /// Six-neighbour displacement in fixed-grid voxels, ordered `[z, y, x]`.
    #[must_use]
    pub const fn neighbour_dilation(&self) -> [usize; 3] {
        self.neighbour_dilation
    }

    /// Fixed-domain center-selection policy.
    #[must_use]
    pub const fn sampling(&self) -> &MindSscSampling {
        &self.sampling
    }

    pub(super) fn halo(&self) -> [usize; 3] {
        std::array::from_fn(|axis| self.patch_radius[axis] + self.neighbour_dilation[axis])
    }

    pub(super) const fn geometry(&self) -> DescriptorGeometry {
        DescriptorGeometry {
            patch_radius: self.patch_radius,
            neighbour_dilation: self.neighbour_dilation,
        }
    }
}

impl Default for MindSscConfig {
    fn default() -> Self {
        Self {
            patch_radius: [1; 3],
            neighbour_dilation: [2; 3],
            sampling: MindSscSampling {
                selection: SamplingSelection::Stratified {
                    max_samples: Self::DEFAULT_MAX_SAMPLES,
                },
            },
        }
    }
}
