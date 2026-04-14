//! Clustering-based segmentation algorithms.
//!
//! Provides unsupervised partitioning of image intensities into K clusters
//! using iterative refinement methods.
//!
//! # Module Structure
//! - [`kmeans`]: K-Means clustering via Lloyd's algorithm with k-means++ initialization.

pub mod kmeans;

pub use kmeans::{kmeans_segment, KMeansSegmentation};
