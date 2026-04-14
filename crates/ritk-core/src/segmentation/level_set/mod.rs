//! Level set segmentation methods for 3-D medical images.
//!
//! This module provides implicit-surface evolution algorithms that segment
//! images by evolving a level set function φ according to a PDE derived
//! from an energy functional or geometric flow.
//!
//! # Submodules
//!
//! - [`chan_vese`]: Chan–Vese "Active Contours Without Edges" (Chan & Vese 2001).
//! - [`geodesic_active_contour`]: Geodesic Active Contour (Caselles et al. 1997).
//!
//! # Mathematical Background
//!
//! A level set function φ: ℝ³ → ℝ represents a surface Γ as its zero level set
//! Γ = { x : φ(x) = 0 }. The interior is { x : φ(x) < 0 } and the exterior
//! is { x : φ(x) > 0 }. Evolution of Γ is achieved by solving a PDE for φ on
//! a fixed Eulerian grid, avoiding explicit surface tracking.

pub mod chan_vese;
pub mod geodesic_active_contour;

pub use chan_vese::ChanVeseSegmentation;
pub use geodesic_active_contour::GeodesicActiveContourSegmentation;
