//! Coordinate transforms between an [`Image`]'s index space and physical space.
//!
//! Split by granularity rather than direction, because granularity is what
//! changes the cost model: [`point`] converts one coordinate at a time and
//! returns a [`ritk_spatial::Point`], [`batch`] converts a whole tensor under a
//! single backend dispatch, and [`grid`] hands out a reusable pair with the
//! direction inverse hoisted for a per-voxel sweep. [`point`] and [`batch`]
//! honour the image's [`CoordinateMap`](ritk_spatial::CoordinateMap); [`grid`]
//! rejects a non-Cartesian one rather than answering wrongly.
//!
//! [`Image`]: crate::Image

pub mod batch;
pub mod grid;
pub mod point;

#[cfg(test)]
#[path = "../tests_transform.rs"]
mod tests;

#[cfg(test)]
#[path = "../tests_transform_beam.rs"]
mod tests_beam;
