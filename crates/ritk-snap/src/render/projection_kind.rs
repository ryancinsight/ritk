//! Content selector for the projection pane.
//!
//! The projection pane renders either a scalar slab reduction or a
//! front-to-back volume rendering of the loaded volume. The selector names
//! which renderer owns the pane; slab reductions stay in the slab module
//! while volume rendering stays in the `mip_vr` seam.

use super::slab::ProjectionStatistic;

/// Content rendered into the projection pane.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProjectionKind {
    /// A scalar slab reduction over the loaded volume.
    Slab(ProjectionStatistic),
    /// Front-to-back volume rendering over the loaded volume.
    VolumeRender,
}

impl ProjectionKind {
    /// Return the stable presentation label for this projection content.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Slab(statistic) => statistic.label(),
            Self::VolumeRender => "VR",
        }
    }
}
