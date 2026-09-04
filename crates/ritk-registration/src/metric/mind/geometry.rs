//! Cartesian image geometry and scalar trilinear sampling.

use coeus_core::ComputeBackend;
#[cfg(test)]
use coeus_core::CpuAddressableStorage;
use eunomia::CastFrom;
use ritk_image::Image;
use ritk_transform::transform::affine::AtlasAffineTransform;

#[cfg(test)]
use crate::classical::rigid_physical_affine_to_native;
#[cfg(test)]
use crate::types::AffineTransform;

use super::sampling::linear_index;
use super::MindSscError;

#[derive(Debug, Clone, Copy)]
pub(super) struct CartesianGeometry {
    origin: [f32; 3],
    data_index_to_world: [[f32; 3]; 3],
    world_to_data_index: [[f32; 3]; 3],
}

impl CartesianGeometry {
    pub(super) fn try_from_image<B>(
        image: &Image<f32, B, 3>,
        role: &'static str,
    ) -> Result<Self, MindSscError>
    where
        B: ComputeBackend,
    {
        if !image.coordinate_map().is_cartesian() {
            return Err(MindSscError::NonCartesianImage {
                image: role,
                coordinate_map: format!("{:?}", image.coordinate_map()),
            });
        }
        let inverse = image
            .direction()
            .try_inverse()
            .ok_or(MindSscError::SingularDirection { image: role })?;
        let mut origin = [0.0_f32; 3];
        let mut data_index_to_world = [[0.0_f32; 3]; 3];
        let mut world_to_data_index = [[0.0_f32; 3]; 3];
        for metadata_axis in 0..3 {
            let origin_value = image.origin()[metadata_axis];
            validate_geometry_value(role, "origin", metadata_axis, origin_value, false)?;
            origin[metadata_axis] = f32::cast_from(origin_value);
            let spacing = image.spacing()[metadata_axis];
            validate_geometry_value(role, "spacing", metadata_axis, spacing, true)?;
            let data_axis = 2 - metadata_axis;
            for world_axis in 0..3 {
                let direction = image.direction()[(world_axis, metadata_axis)];
                validate_geometry_value(
                    role,
                    "direction",
                    world_axis * 3 + metadata_axis,
                    direction,
                    false,
                )?;
                data_index_to_world[world_axis][data_axis] = f32::cast_from(direction * spacing);
                world_to_data_index[data_axis][world_axis] =
                    f32::cast_from(inverse[(metadata_axis, world_axis)] / spacing);
            }
        }
        Ok(Self {
            origin,
            data_index_to_world,
            world_to_data_index,
        })
    }

    pub(super) fn index_to_world(self, index: [usize; 3]) -> Result<[f32; 3], MindSscError> {
        let mut world = self.origin;
        for world_axis in 0..3 {
            for (data_axis, coordinate) in index.iter().copied().enumerate() {
                let coordinate =
                    u32::try_from(coordinate).map_err(|_| MindSscError::IndexOverflow)?;
                world[world_axis] = self.data_index_to_world[world_axis][data_axis]
                    .mul_add(f32::cast_from(coordinate), world[world_axis]);
            }
        }
        Ok(world)
    }

    pub(super) fn world_to_index(self, world: [f32; 3]) -> Result<[f32; 3], MindSscError> {
        let mut index = [0.0_f32; 3];
        for data_axis in 0..3 {
            for world_axis in 0..3 {
                index[data_axis] = self.world_to_data_index[data_axis][world_axis].mul_add(
                    world[world_axis] - self.origin[world_axis],
                    index[data_axis],
                );
            }
            if !index[data_axis].is_finite() {
                return Err(MindSscError::NonFiniteMovingCoordinate {
                    axis: data_axis,
                    value: index[data_axis],
                });
            }
        }
        Ok(index)
    }
}

pub(super) fn validate_transform<B>(
    transform: &AtlasAffineTransform<B, 3>,
) -> Result<(), MindSscError>
where
    B: ComputeBackend,
{
    for (field, values) in [
        ("matrix", transform.matrix()),
        ("translation", transform.translation()),
        ("center", transform.center()),
    ] {
        if let Some((index, value)) = values
            .iter()
            .copied()
            .enumerate()
            .find(|(_, value)| !value.is_finite())
        {
            return Err(MindSscError::NonFiniteTransform {
                field,
                index,
                value,
            });
        }
    }
    Ok(())
}

pub(super) fn apply_native_affine<B>(
    transform: &AtlasAffineTransform<B, 3>,
    point: [f32; 3],
) -> [f32; 3]
where
    B: ComputeBackend,
{
    std::array::from_fn(|row| {
        let mut value = transform.center()[row] + transform.translation()[row];
        for (column, coordinate) in point.iter().copied().enumerate() {
            value = transform.matrix()[row * 3 + column]
                .mul_add(coordinate - transform.center()[column], value);
        }
        value
    })
}

pub(super) fn trilinear_background(
    values: &[f32],
    shape: [usize; 3],
    point: [f32; 3],
) -> Result<f32, MindSscError> {
    if shape.contains(&0) {
        return Err(MindSscError::EmptyMovingImage { shape });
    }
    let mut clamped = [0.0_f32; 3];
    for axis in 0..3 {
        if !point[axis].is_finite() {
            return Err(MindSscError::NonFiniteMovingCoordinate {
                axis,
                value: point[axis],
            });
        }
        let extent = u32::try_from(shape[axis]).map_err(|_| MindSscError::IndexOverflow)?;
        let extent = f32::cast_from(extent);
        if point[axis] < -0.5 || point[axis] >= extent - 0.5 {
            return Ok(0.0);
        }
        clamped[axis] = point[axis].clamp(0.0, extent - 1.0);
    }
    let lower = clamped.map(|coordinate| usize::cast_from(coordinate.floor()));
    let upper: [usize; 3] = std::array::from_fn(|axis| (lower[axis] + 1).min(shape[axis] - 1));
    let fraction: [f32; 3] = std::array::from_fn(|axis| {
        let lower = u32::try_from(lower[axis])
            .expect("invariant: lower coordinate is less than validated image extent");
        clamped[axis] - f32::cast_from(lower)
    });
    let mut result = 0.0_f32;
    for z_upper in [false, true] {
        for y_upper in [false, true] {
            for x_upper in [false, true] {
                let choose_upper = [z_upper, y_upper, x_upper];
                let index = std::array::from_fn(|axis| {
                    if choose_upper[axis] {
                        upper[axis]
                    } else {
                        lower[axis]
                    }
                });
                let linear = linear_index(index, shape)?;
                let value = *values.get(linear).ok_or(MindSscError::IndexOverflow)?;
                if !value.is_finite() {
                    return Err(MindSscError::NonFiniteImageSample {
                        image: "moving",
                        index: linear,
                        value,
                    });
                }
                let weight = (0..3).fold(1.0_f32, |weight, axis| {
                    weight
                        * if choose_upper[axis] {
                            fraction[axis]
                        } else {
                            1.0 - fraction[axis]
                        }
                });
                result = value.mul_add(weight, result);
            }
        }
    }
    Ok(result)
}

fn validate_geometry_value(
    image: &'static str,
    field: &'static str,
    index: usize,
    value: f64,
    positive: bool,
) -> Result<(), MindSscError> {
    if !value.is_finite() || (positive && value <= 0.0) || value.abs() > f64::from(f32::MAX) {
        Err(MindSscError::InvalidGeometry {
            image,
            field,
            index,
            value,
        })
    } else {
        Ok(())
    }
}

#[cfg(test)]
pub(super) fn sample_for_test<B>(
    moving: &Image<f32, B, 3>,
    fixed: &Image<f32, B, 3>,
    transform: &AffineTransform,
    fixed_index: [usize; 3],
) -> Result<f32, MindSscError>
where
    B: ComputeBackend,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
{
    let fixed_geometry = CartesianGeometry::try_from_image(fixed, "fixed")?;
    let moving_geometry = CartesianGeometry::try_from_image(moving, "moving")?;
    let native = rigid_physical_affine_to_native::<B>(transform)?;
    let point = apply_native_affine(&native, fixed_geometry.index_to_world(fixed_index)?);
    trilinear_background(
        moving
            .data_slice()
            .map_err(|error| MindSscError::NonContiguousImage {
                image: "moving",
                reason: error.to_string(),
            })?,
        moving.shape(),
        moving_geometry.world_to_index(point)?,
    )
}
