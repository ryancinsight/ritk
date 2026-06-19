//! Clustering-based segmentation algorithms.
//!
//! Provides unsupervised partitioning of image intensities into K clusters
//! using iterative refinement methods.
//!
//! # Module Structure
//! - [`kmeans`]: K-Means clustering via Lloyd's algorithm with k-means++ initialization.
//! - [`slic`]: SLIC super-pixel segmentation (Achanta et al. 2012).

pub mod kmeans;
pub use kmeans::{kmeans_segment, KMeansSegmentation};

pub mod slic;
pub use slic::{slic_itk_segment, SlicConfig, SlicSuperpixelFilter};
