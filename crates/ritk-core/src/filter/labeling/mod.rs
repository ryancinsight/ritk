//! Connected-component labeling filter (ITK `ConnectedComponentImageFilter` parity).
//!
//! Re-exports `ConnectedComponentsFilter`, `connected_components`, and
//! `LabelStatistics` from `crate::segmentation::labeling`, making them
//! available under the `filter::` path alongside all other ritk-core filters.
//!
//! # ITK parity
//! Matches `itk::ConnectedComponentImageFilter` semantics:
//! - Any pixel not equal to `background_value` (default 0.0) is foreground.
//! - 6-connectivity (face adjacency) is the default, matching ITK 3-D default.
//! - 26-connectivity (face + edge + corner) is available via
//!   `ConnectedComponentsFilter::with_connectivity(26)`.
//!
//! # Usage
//! ```ignore
//! use ritk_core::filter::ConnectedComponentsFilter;
//!
//! let filter = ConnectedComponentsFilter::with_connectivity(6)
//!     .with_background(0.0);
//! let (label_image, stats) = filter.apply(&binary_mask);
//! ```

pub use crate::segmentation::labeling::{
    connected_components, ConnectedComponentsFilter, LabelStatistics,
    RelabelComponentFilter, RelabelStatistics,
};
