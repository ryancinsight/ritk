//! Batch tensor coordinate transforms for [`crate::Image`].
//!
//! The single-point `transform_physical_point_to_continuous_index` and
//! `transform_continuous_index_to_physical_point` methods now live directly
//! on `Image` in [`crate::types`]. The batch tensor transforms
//! (`world_to_index_tensor` / `index_to_world_tensor`) also live on `Image`
//! in [`crate::types`] via coeus-native implementations.

// All transform methods are now impl'd directly on `Image<T, B, D>` in
// `types.rs`. This module is preserved as an organizational placeholder for
// future transform-specific logic that does not fit on the base type.

#[cfg(test)]
#[path = "tests_transform.rs"]
mod tests;
