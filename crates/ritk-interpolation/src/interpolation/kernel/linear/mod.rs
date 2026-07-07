//! Linear interpolation implementation.
//!
//! This module provides linear interpolation for 1D, 2D, 3D, and 4D data.

pub mod dim1;
pub mod dim2;
pub mod dim3;
pub mod dim4;

use super::BoundsPolicy;
use crate::interpolation::dispatch::dispatch_linear;
use ritk_core::interpolation::Interpolator;
use ritk_image::burn::module::{
    AutodiffModule, Content, Module, ModuleDisplay, ModuleDisplayDefault, ModuleMapper,
    ModuleVisitor,
};
use ritk_image::burn::record::{PrecisionSettings, Record};
use ritk_image::tensor::backend::{AutodiffBackend, Backend};
use ritk_image::tensor::Tensor;
use serde::{Deserialize, Serialize};

#[inline]
pub(super) fn slice_batch<B: Backend>(
    values: Tensor<B, 1>,
    start: usize,
    end: usize,
) -> Tensor<B, 1> {
    values.slice_dim(0, start..end)
}

/// Linear Interpolator.
///
/// Performs linear interpolation natively.
/// When [`BoundsPolicy::Extend`] (the default), out-of-bounds coordinates are clamped to the
/// nearest edge voxel. When [`BoundsPolicy::ZeroPad`], out-of-bounds samples return `0.0`,
/// which prevents spurious correlation peaks in MI-based registration metrics.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct LinearInterpolator {
    /// Boundary handling policy. Default: `Extend`.
    pub bounds_policy: BoundsPolicy,
}

impl<B: Backend> Record<B> for LinearInterpolator {
    type Item<S: PrecisionSettings> = Self;

    fn into_item<S: PrecisionSettings>(self) -> Self::Item<S> {
        self
    }

    fn from_item<S: PrecisionSettings>(item: Self::Item<S>, _device: &B::Device) -> Self {
        item
    }
}

impl<B: Backend> Module<B> for LinearInterpolator {
    type Record = Self;

    fn visit<V: ModuleVisitor<B>>(&self, _visitor: &mut V) {
        // No tensors to visit
    }

    fn map<M: ModuleMapper<B>>(self, _mapper: &mut M) -> Self {
        self
    }

    fn into_record(self) -> Self::Record {
        self
    }

    fn load_record(self, record: Self::Record) -> Self {
        record
    }

    fn collect_devices(&self, devices: Vec<B::Device>) -> Vec<B::Device> {
        devices
    }

    fn to_device(self, _device: &B::Device) -> Self {
        self
    }

    fn fork(self, _device: &B::Device) -> Self {
        self
    }
}

impl<B: AutodiffBackend> AutodiffModule<B> for LinearInterpolator {
    type InnerModule = LinearInterpolator;

    fn valid(&self) -> Self::InnerModule {
        *self
    }
}

impl ModuleDisplayDefault for LinearInterpolator {
    fn content(&self, content: Content) -> Option<Content> {
        Some(content.set_top_level_type("LinearInterpolator"))
    }
}

impl ModuleDisplay for LinearInterpolator {}

impl LinearInterpolator {
    /// Create a new linear interpolator with edge-clamping (default behaviour).
    pub fn new() -> Self {
        Self {
            bounds_policy: BoundsPolicy::Extend,
        }
    }

    /// Create a linear interpolator that returns `0.0` for out-of-bounds samples.
    pub fn new_zero_pad() -> Self {
        Self {
            bounds_policy: BoundsPolicy::ZeroPad,
        }
    }

    /// Builder-style setter for the bounds policy.
    pub fn with_bounds_policy(mut self, policy: BoundsPolicy) -> Self {
        self.bounds_policy = policy;
        self
    }
}

impl Default for LinearInterpolator {
    fn default() -> Self {
        Self::new()
    }
}

impl<B: Backend> Interpolator<B> for LinearInterpolator {
    fn interpolate<const D: usize>(
        &self,
        data: &Tensor<B, D>,
        indices: Tensor<B, 2>,
    ) -> Tensor<B, 1> {
        dispatch_linear(data, indices, self.bounds_policy.as_out_of_bounds_mode())
    }
}

#[cfg(test)]
#[path = "tests_linear.rs"]
mod tests_linear;
