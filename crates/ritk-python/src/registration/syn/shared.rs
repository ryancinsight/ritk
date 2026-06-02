use crate::errors::{RitkPyError, RitkResult};
use crate::image::{image_to_vec, into_py_image, vec_to_image, PyImage};
use ritk_core::spatial::{Direction, Point, Spacing};

#[derive(Clone)]
pub(crate) struct MatchingImageInputs {
    pub(crate) fixed_vals: Vec<f32>,
    pub(crate) moving_vals: Vec<f32>,
    pub(crate) fixed_shape: [usize; 3],
    pub(crate) fixed_origin: Point<3>,
    pub(crate) fixed_spacing: Spacing<3>,
    pub(crate) fixed_direction: Direction<3>,
    pub(crate) moving_origin: Point<3>,
    pub(crate) moving_spacing: Spacing<3>,
    pub(crate) moving_direction: Direction<3>,
}

pub(crate) fn load_matching_inputs(
    fixed: &PyImage,
    moving: &PyImage,
) -> RitkResult<MatchingImageInputs> {
    let (fixed_vals, fixed_shape) = image_to_vec(fixed.inner.as_ref());
    let (moving_vals, moving_shape) = image_to_vec(moving.inner.as_ref());

    if fixed_shape != moving_shape {
        return Err(RitkPyError::runtime(format!(
            "fixed shape {:?} != moving shape {:?}",
            fixed_shape, moving_shape
        )));
    }

    Ok(MatchingImageInputs {
        fixed_vals,
        moving_vals,
        fixed_shape,
        fixed_origin: *fixed.inner.origin(),
        fixed_spacing: *fixed.inner.spacing(),
        fixed_direction: *fixed.inner.direction(),
        moving_origin: *moving.inner.origin(),
        moving_spacing: *moving.inner.spacing(),
        moving_direction: *moving.inner.direction(),
    })
}

pub(crate) fn to_py_pair(
    warped_fixed: Vec<f32>,
    warped_moving: Vec<f32>,
    inputs: &MatchingImageInputs,
) -> (PyImage, PyImage) {
    let warped_fixed_img = vec_to_image(
        warped_fixed,
        inputs.fixed_shape,
        inputs.fixed_origin,
        inputs.fixed_spacing,
        inputs.fixed_direction,
    );
    let warped_moving_img = vec_to_image(
        warped_moving,
        inputs.fixed_shape,
        inputs.moving_origin,
        inputs.moving_spacing,
        inputs.moving_direction,
    );
    (
        into_py_image(warped_fixed_img),
        into_py_image(warped_moving_img),
    )
}

pub(crate) fn to_py_moving(warped_moving: Vec<f32>, inputs: &MatchingImageInputs) -> PyImage {
    into_py_image(vec_to_image(
        warped_moving,
        inputs.fixed_shape,
        inputs.fixed_origin,
        inputs.fixed_spacing,
        inputs.fixed_direction,
    ))
}

pub(crate) fn to_py_warped_and_displacement(
    warped_moving: Vec<f32>,
    displacement_field: (Vec<f32>, Vec<f32>, Vec<f32>),
    inputs: &MatchingImageInputs,
) -> (PyImage, PyImage) {
    let warped_image = vec_to_image(
        warped_moving,
        inputs.fixed_shape,
        inputs.fixed_origin,
        inputs.fixed_spacing,
        inputs.fixed_direction,
    );

    let [nz, ny, nx] = inputs.fixed_shape;
    let n = nz * ny * nx;
    let mut disp_packed = Vec::with_capacity(3 * n);
    disp_packed.extend_from_slice(&displacement_field.0);
    disp_packed.extend_from_slice(&displacement_field.1);
    disp_packed.extend_from_slice(&displacement_field.2);
    let disp_image = vec_to_image(
        disp_packed,
        [3 * nz, ny, nx],
        Point::new([0.0, 0.0, 0.0]),
        Spacing::new([1.0, 1.0, 1.0]),
        Direction::identity(),
    );

    (into_py_image(warped_image), into_py_image(disp_image))
}
