//! Histogram computation utilities for Mutual Information metrics.
//!
//! This module provides shared implementations for differentiable soft histogramming
//! using Parzen windowing.

mod cache;
mod masked;
mod parzen;

pub use parzen::direct::{
    build_sparse_w_fixed_transposed, compaction_sizes, compute_joint_histogram_direct,
    compute_joint_histogram_from_cache_sparse, CompactionSizes, SparseWFixedEntry, SparseWFixedT,
};
pub use parzen::ParzenJointHistogram;
