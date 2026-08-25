//! Native representation of a three-dimensional displacement field.

use coeus_core::ComputeBackend;
use ritk_image::Image;

/// Three physical-axis components of a Coeus-native displacement field.
///
/// Each component has the same `[z, y, x]` shape and spatial frame. `x`,
/// `y`, and `z` contain the physical displacements along their correspondingly
/// named axes.
pub struct NativeDisplacementField<B: ComputeBackend> {
    /// Physical x-axis displacement component.
    pub x: Image<f32, B, 3>,
    /// Physical y-axis displacement component.
    pub y: Image<f32, B, 3>,
    /// Physical z-axis displacement component.
    pub z: Image<f32, B, 3>,
}

/// The same three components as a positional `(x, y, z)` triple.
///
/// The shape the pre-native `apply` entry points return. This is a spelling
/// convenience for that one signature, not a distinct domain type — the named
/// form is [`NativeDisplacementField`].
pub type DisplacementComponents<B> = (Image<f32, B, 3>, Image<f32, B, 3>, Image<f32, B, 3>);
