//! FreeSurfer surface-family formats.
//!
//! | Module | Format | Files |
//! |--------|--------|-------|
//! | [`surface`] | Binary triangle surface | `lh.white`, `lh.pial`, `lh.inflated` |
//! | [`morphometry`] | New-format per-vertex scalars | `lh.curv`, `lh.thickness`, `lh.sulc` |
//! | [`annotation`] | Per-vertex parcellation with embedded colour table | `lh.aparc.annot` |
//! | [`label`] | ASCII vertex set | `lh.cortex.label` |
//! | [`lut`] | Text colour lookup table, read into a [`ritk_annotation::LabelTable`] | `FreeSurferColorLUT.txt` |
//! | [`ribbon`] | Rasterising an annotation into a volume | — |
//!
//! Every reader treats its input as hostile: counts are bounded, storage grows
//! only as input backs it, and every failure is a typed [`FreeSurferError`]
//! rather than a panic. Each module cites the reference its layout follows.
//!
//! These formats live here rather than in a format crate of their own because
//! they are parcellation vocabulary — an annotation is a surface parcellation,
//! and a lookup table is the `region_names` of any volumetric one — and they
//! need nothing beyond `std`. FreeSurfer *volumes* (`.mgz`) are `ritk-mgh`.
//!
//! # Conversion to volume
//!
//! A [`SurfaceAnnotation`] labels *vertices of a mesh*, not voxels. Turning one
//! into a [`crate::Parcellation`] needs the geometry those vertices belong to —
//! [`Surface`] reads it — and a rasterisation of the cortical ribbon between the
//! white and pial surfaces, which [`rasterise_ribbon`] performs.
//!
//! Mind the frame: FreeSurfer stores surfaces in surface RAS, which differs
//! from the scanner frame a volume carries by that volume's `c_ras`
//! translation. [`Surface::translated`] applies it; [`surface`] sets out why it
//! cannot be applied automatically.

mod big_endian;
mod error;

pub mod annotation;
pub mod label;
pub mod lut;
pub mod morphometry;
pub mod ribbon;
pub mod surface;

pub use annotation::SurfaceAnnotation;
pub use error::{FreeSurferError, FreeSurferFormat};
pub use label::{LabelVertex, SurfaceLabel};
pub use morphometry::Morphometry;
pub use ribbon::{RibbonError, RibbonReport, rasterise_ribbon};
pub use surface::Surface;

#[cfg(test)]
mod tests;
