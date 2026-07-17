//! WGPU dispatch-limit constants.
//!
//! WGPU limits individual dispatch dimensions to 65 535 invocations.
//! All point-batch and tensor-row operations that may exceed this limit
//! should respect these chunk ceilings.
//!
//! The chunked-row-apply helper that previously lived here has moved to
//! `ritk_image::burn_compat_row_chunks` (Burn-compat feature) so this crate
//! no longer needs a direct `burn` dependency.

/// Maximum rows dispatched per shader invocation for 2-D/3-D operations.
///
/// Half the WGPU hard-limit (65 535) provides a conservative safe margin.
pub const WGPU_CHUNK_SIZE: usize = 32_768;

/// Maximum rows dispatched per shader invocation for the 4-D B-spline path.
///
/// The 4-D kernel issues more sub-dispatches per point than the 3-D path,
/// so a smaller ceiling is required to stay inside the WGPU limit.
pub const WGPU_CHUNK_SIZE_4D: usize = 16_384;
