//! Annotation primitives — re-exported from the [`ritk_annotation`] crate.
//!
//! The concrete types ([`AnnotationState`], [`LabelMap`], [`LabelTable`],
//! [`RgbaBytes`], [`RgbaLinear`], [`LabelId`], [`UndoRedoStack`],
//! [`OverlayState`], [`Visibility`], [`Opacity`], [`Colormap`]) live in
//! `ritk_annotation`.  This module is a thin compatibility shim so existing
//! `ritk_core::annotation::*` import paths continue to resolve.

pub use ritk_annotation::*;
