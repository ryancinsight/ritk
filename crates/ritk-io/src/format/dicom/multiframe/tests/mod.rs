#[cfg(test)]
mod per_frame;
#[cfg(test)]
mod reader;
#[cfg(test)]
mod roundtrip;
#[cfg(test)]
mod writer;

pub(super) use super::per_frame::extract_functional_groups;
pub(super) use super::types::{PerFrameInfo, MF_GRAYSCALE_WORD_SC_UID};
pub(super) use super::*;
pub(super) use coeus_core::MoiraiBackend;
pub(super) use dicom::core::smallvec::SmallVec;
pub(super) use dicom::core::{DataElement, PrimitiveValue, Tag, VR};
pub(super) use dicom::object::meta::FileMetaTableBuilder;
pub(super) use dicom::object::InMemDicomObject;
pub(super) use ritk_image::Image as NativeImage;
pub(super) use ritk_spatial::{Direction, Point, Spacing};

/// Build a native `Image<f32, MoiraiBackend, 3>` test carrier from a flat
/// `[frames, rows, cols]` buffer with the given spatial metadata.
pub(super) fn native_image(
    data: Vec<f32>,
    dims: [usize; 3],
    origin: [f64; 3],
    spacing: [f64; 3],
) -> NativeImage<f32, MoiraiBackend, 3> {
    NativeImage::<f32, MoiraiBackend, 3>::from_flat(
        data,
        dims,
        Point::new(origin),
        Spacing::new(spacing),
        Direction::identity(),
    )
    .expect("native multiframe image construction")
}
