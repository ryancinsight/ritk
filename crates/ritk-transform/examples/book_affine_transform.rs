//! Book example: host-backed affine point transformation.
//!
//! Demonstrates the Atlas-side transform seam over Coeus backend types without
//! introducing training-only module contracts.

use coeus_core::MoiraiBackend;
use ritk_image::Image;
use ritk_spatial::{Direction, Point, Spacing};
use ritk_transform::transform::affine::AtlasAffineTransform;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let matrix = [
        1.0f32, 0.0, 0.0, //
        0.0, 1.0, 0.0, //
        0.0, 0.0, 1.0,
    ];
    let translation = [2.0f32, -1.0, 0.5];
    let center = [0.0f32, 0.0, 0.0];

    let transform =
        AtlasAffineTransform::<MoiraiBackend, 3>::try_new(&matrix, &translation, &center)?;

    let points = Image::<f32, MoiraiBackend, 2>::from_flat(
        vec![1.0, 2.0, 3.0, -2.0, 4.0, 0.5],
        [2usize, 3usize],
        Point::origin(),
        Spacing::uniform(1.0),
        Direction::identity(),
    )?;

    let transformed = transform.transform_points::<MoiraiBackend>(&points)?;
    let _host_values = transformed.data_slice()?;

    Ok(())
}
