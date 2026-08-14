//! Coordinate transforms between an [`Image`]'s index space and physical space.
//!
//! Split by granularity rather than direction, because granularity is what
//! changes the cost model: [`point`] converts one coordinate at a time and
//! returns a [`ritk_spatial::Point`], while [`batch`] converts a whole tensor
//! under a single backend dispatch. Both honour the image's
//! [`CoordinateMap`](ritk_spatial::CoordinateMap).
//!
//! [`Image`]: crate::Image

pub mod batch;
pub mod point;

#[cfg(test)]
#[path = "../tests_transform.rs"]
mod tests;

#[cfg(test)]
#[path = "../tests_transform_beam.rs"]
mod tests_beam;
