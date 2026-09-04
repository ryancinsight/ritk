use anyhow::Result;
use eunomia::CastFrom;
use ritk_spatial::{Direction, Point, Spacing};

use super::{identity_image, image, synthetic_values};
use crate::metric::mind::geometry::{sample_for_test, trilinear_background};
use crate::types::AffineTransform;

#[test]
fn native_sampling_matches_independent_classical_physical_oracle() -> Result<()> {
    let shape = [9, 10, 11];
    let values = synthetic_values(shape);
    let fixed_origin = [12.0, -7.0, 30.0];
    let fixed_spacing = [1.25, 2.0, 3.5];
    let fixed_direction = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
    let moving_origin = [-5.0, 9.0, 4.0];
    let moving_spacing = [0.75, 1.5, 2.25];
    let moving_direction = [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]];
    let fixed = image(
        vec![0.0; shape.into_iter().product()],
        shape,
        Point::new(fixed_origin),
        Spacing::new(fixed_spacing),
        Direction::from_rows(fixed_direction),
    )?;
    let moving = image(
        values.clone(),
        shape,
        Point::new(moving_origin),
        Spacing::new(moving_spacing),
        Direction::from_rows(moving_direction),
    )?;
    let transform = AffineTransform::new([
        0.0, -1.0, 0.0, 3.0, 1.0, 0.0, 0.0, -2.0, 0.0, 0.0, 1.0, 5.0, 0.0, 0.0, 0.0, 1.0,
    ]);
    let fixed_index = [4, 5, 6];
    let actual = sample_for_test(&moving, &fixed, &transform, fixed_index)?;
    let expected = classical_sample_oracle(
        &values,
        shape,
        fixed_index,
        fixed_origin,
        fixed_spacing,
        fixed_direction,
        moving_origin,
        moving_spacing,
        moving_direction,
        transform.as_array(),
    );
    let bound = 128.0 * f32::EPSILON * expected.abs().max(1.0);
    assert!(
        (actual - expected).abs() <= bound,
        "actual={actual}, expected={expected}, bound={bound}"
    );
    Ok(())
}

#[test]
fn scalar_sampler_matches_native_half_voxel_boundaries() -> Result<()> {
    let shape = [2; 3];
    let values = (1_u8..=8).map(f32::from).collect::<Vec<_>>();
    let moving = identity_image(values.clone(), shape)?;
    for (coordinate, expected) in [
        (-0.500_1_f32, 0.0),
        (-0.5, 1.0),
        (-0.25, 1.0),
        (1.25, 8.0),
        (1.499_9, 8.0),
        (1.5, 0.0),
    ] {
        let point = [coordinate; 3];
        let scalar = trilinear_background(&values, shape, point)?;
        let native = ritk_filter::resample::native::sample_moving_at_world(&moving, &point)?;
        assert_eq!(scalar, expected, "scalar sample at {coordinate}");
        assert_eq!(native, [expected], "native sample at {coordinate}");
    }
    Ok(())
}

#[expect(
    clippy::too_many_arguments,
    reason = "independent test oracle exposes both physical frames"
)]
fn classical_sample_oracle(
    values: &[f32],
    shape: [usize; 3],
    fixed_index: [usize; 3],
    fixed_origin: [f64; 3],
    fixed_spacing: [f64; 3],
    fixed_direction: [[f64; 3]; 3],
    moving_origin: [f64; 3],
    moving_spacing: [f64; 3],
    moving_direction: [[f64; 3]; 3],
    transform: &[f64; 16],
) -> f32 {
    let fixed_metadata_index = [fixed_index[2], fixed_index[1], fixed_index[0]]
        .map(|v| f64::from(u32::try_from(v).expect("test index fits u32")));
    let fixed_world = mat_vec_add(
        fixed_direction,
        std::array::from_fn(|axis| fixed_metadata_index[axis] * fixed_spacing[axis]),
        fixed_origin,
    );
    let classical = [fixed_world[2], fixed_world[1], fixed_world[0]];
    let moved_classical: [f64; 3] = std::array::from_fn(|row| {
        transform[row * 4] * classical[0]
            + transform[row * 4 + 1] * classical[1]
            + transform[row * 4 + 2] * classical[2]
            + transform[row * 4 + 3]
    });
    let moved_world = [moved_classical[2], moved_classical[1], moved_classical[0]];
    let relative = std::array::from_fn(|axis| moved_world[axis] - moving_origin[axis]);
    let inverse = transpose(moving_direction);
    let metadata_index = mat_vec(inverse, relative);
    let data_index = [
        metadata_index[2] / moving_spacing[2],
        metadata_index[1] / moving_spacing[1],
        metadata_index[0] / moving_spacing[0],
    ];
    trilinear_oracle(values, shape, data_index)
}

fn mat_vec(matrix: [[f64; 3]; 3], vector: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|row| {
        matrix[row][0] * vector[0] + matrix[row][1] * vector[1] + matrix[row][2] * vector[2]
    })
}

fn mat_vec_add(matrix: [[f64; 3]; 3], vector: [f64; 3], add: [f64; 3]) -> [f64; 3] {
    let product = mat_vec(matrix, vector);
    std::array::from_fn(|axis| product[axis] + add[axis])
}

fn transpose(matrix: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    std::array::from_fn(|row| std::array::from_fn(|column| matrix[column][row]))
}

fn trilinear_oracle(values: &[f32], shape: [usize; 3], point: [f64; 3]) -> f32 {
    if (0..3).any(|axis| {
        let extent = f64::from(u32::try_from(shape[axis]).expect("test extent fits u32"));
        point[axis] < -0.5 || point[axis] >= extent - 0.5
    }) {
        return 0.0;
    }
    let point: [f64; 3] = std::array::from_fn(|axis| {
        let extent = f64::from(u32::try_from(shape[axis]).expect("test extent fits u32"));
        point[axis].clamp(0.0, extent - 1.0)
    });
    let lower = point.map(|coordinate| usize::cast_from(coordinate.floor()));
    let upper: [usize; 3] = std::array::from_fn(|axis| (lower[axis] + 1).min(shape[axis] - 1));
    let fraction: [f64; 3] = std::array::from_fn(|axis| {
        point[axis] - f64::from(u32::try_from(lower[axis]).expect("test index fits u32"))
    });
    let mut result = 0.0_f64;
    for high_z in [false, true] {
        for high_y in [false, true] {
            for high_x in [false, true] {
                let high = [high_z, high_y, high_x];
                let index: [usize; 3] =
                    std::array::from_fn(|axis| if high[axis] { upper[axis] } else { lower[axis] });
                let weight = (0..3)
                    .map(|axis| {
                        if high[axis] {
                            fraction[axis]
                        } else {
                            1.0 - fraction[axis]
                        }
                    })
                    .product::<f64>();
                let linear = index[0] * shape[1] * shape[2] + index[1] * shape[2] + index[2];
                result += f64::from(values[linear]) * weight;
            }
        }
    }
    f32::cast_from(result)
}
