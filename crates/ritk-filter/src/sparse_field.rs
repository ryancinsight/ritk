//! ITK's SparseField narrow band, owned once: layer lists, raster topology, and
//! the evolution engine both level-set filters run.
//!
//! `canny_segmentation_level_set` and `anti_alias_binary` are the same algorithm
//! with different physics. ITK's `SparseFieldLevelSetImageFilter` keeps
//! `2·NL+1` doubly linked lists of voxel indices, builds the band from the zero
//! crossing of the input, propagates values outward through the layers, evolves
//! the active layer until the RMS change falls below the tolerance, and writes a
//! `±(NL+1)` far field. That all lives here; a filter supplies only its per-voxel
//! update through [`SparseFieldStep`].
//!
//! - `topology` — [`GridTopology`]: the raster, the bounds-checked neighbours,
//!   the Neumann clamp, and the one face-offset table, in ITK's order.
//! - `layers` — [`SparseFieldLayers`]: an index-based doubly linked list per
//!   layer over a node slab with a free list, so relinking a voxel (`move_to`,
//!   the inner loop's most frequent mutation) is `O(1)`.
//! - `band` — `PropagateLayerValues` and the `ProcessStatusList` cascade.
//! - `engine` — [`evolve`], generic over the scalar and the step policy, so
//!   each filter's physics is monomorphised into the loop rather than dispatched
//!   at run time.
//!
//! Two invariants decide the shapes here.
//!
//! **Order is load-bearing.** The active layer's iteration order feeds the update
//! sweep, which writes into neighbouring layers' `phi`, and this crate's oracles
//! pin the float-exact ITK result. An array-backed layer store could free in
//! `O(1)` only by `swap_remove`, which reorders the layer, so it is not a valid
//! substitute for the list. **Membership is the handle, not `status`.** The
//! filters deliberately leave a voxel linked for the rest of the sweep after
//! marking it `ST_CUP`/`ST_CDN`, dropping it in the layer rebuild, so deriving
//! membership from `status` would unlink a voxel the algorithm keeps linked.
//!
//! Memory is band-proportional: one `u32` handle per voxel plus a 16-byte node per
//! band member, recycled through the free list.

mod band;
mod engine;
mod layers;
mod scalar;
mod topology;

pub(crate) use engine::{evolve, SparseFieldConfig, SparseFieldStep};
pub(crate) use topology::GridTopology;
