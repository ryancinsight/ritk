//! Histogram computation utilities for Mutual Information metrics.
//!
//! This module provides shared implementations for differentiable soft histogramming
//! using Parzen windowing.

mod cache;
mod masked;
mod parzen;

pub use parzen::ParzenJointHistogram;
