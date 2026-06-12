//! Shared pixel-buffer I/O helpers — re-exported from [`ritk_tensor_ops`].
//!
//! The canonical [`extract_vec`], [`extract_vec_infallible`], [`rebuild`],
//! [`rebuild_with_origin`], and [`rebuild_with_metadata`] functions live in
//! `ritk_tensor_ops`.  This module is a thin compatibility shim so existing
//! `ritk_tensor_ops::*` import paths continue to resolve.

pub use ritk_tensor_ops::*;
