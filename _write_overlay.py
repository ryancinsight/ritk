//! Overlay composition state for annotation rendering.
//
//! # Mathematical Specification
//
//! An overlay state O = (image_overlays, contour_overlays, mask_overlays).
//! Each component type represents a separate rendering layer:
//
//! - ImageOverlay:   secondary image I: Z^3 -> R, rendered with colormap and opacity.
//! - ContourOverlay: closed 3-D polygons C = { c_i : |c_i| >= 2 }.
//! - MaskOverlay:    dense label volume M: Z^3 -> N (0 = transparent).
//
//! # Invariants
//! - ImageOverlay: data.len() == dims[0] * dims[1] * dims[2].
//! - ContourOverlay: each contour >= 2 points (enforced via add_contour).
//! - MaskOverlay: data.len() == dims[0] * dims[1] * dims[2].

use std::collections::HashSet;
use serde::{Deserialize, Serialize};
