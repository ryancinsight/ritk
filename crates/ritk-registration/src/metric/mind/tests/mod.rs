use anyhow::Result;
use coeus_core::SequentialBackend;
use ritk_image::Image;
use ritk_spatial::{Direction, Point, Spacing};

use crate::types::AffineTransform;

pub(super) type TestImage = Image<f32, SequentialBackend, 3>;

pub(super) fn image(
    values: Vec<f32>,
    shape: [usize; 3],
    origin: Point<3>,
    spacing: Spacing<3>,
    direction: Direction<3>,
) -> Result<TestImage> {
    Image::from_flat_on(
        values,
        shape,
        origin,
        spacing,
        direction,
        &SequentialBackend,
    )
}

pub(super) fn synthetic_values(shape: [usize; 3]) -> Vec<f32> {
    let mut values = Vec::with_capacity(shape.into_iter().product());
    for z in 0..shape[0] {
        for y in 0..shape[1] {
            for x in 0..shape[2] {
                let z = u16::try_from(z).expect("test extent fits u16");
                let y = u16::try_from(y).expect("test extent fits u16");
                let x = u16::try_from(x).expect("test extent fits u16");
                let value = (z * z + 3 * y * y + 5 * x * x + 7 * z * y + 11 * x) % 251;
                values.push(f32::from(value));
            }
        }
    }
    values
}

pub(super) fn identity_image(values: Vec<f32>, shape: [usize; 3]) -> Result<TestImage> {
    image(
        values,
        shape,
        Point::origin(),
        Spacing::uniform(1.0),
        Direction::identity(),
    )
}

pub(super) fn translation(z: f64, y: f64, x: f64) -> AffineTransform {
    AffineTransform::new([
        1.0, 0.0, 0.0, z, 0.0, 1.0, 0.0, y, 0.0, 0.0, 1.0, x, 0.0, 0.0, 0.0, 1.0,
    ])
}

mod descriptor;
mod geometry;
mod integration;
mod sampling;
mod validation;
