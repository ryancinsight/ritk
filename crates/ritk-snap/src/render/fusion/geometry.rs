//! Voxel and pixel index arithmetic shared by the fused render path.

use crate::LoadedVolume;

pub(super) fn slice_center_voxel(axis: usize, slice: usize, shape: [usize; 3]) -> [f64; 3] {
    let mut voxel = [0.0; 3];
    for (index, extent) in shape.into_iter().enumerate() {
        voxel[index] = (extent.saturating_sub(1) as f64) * 0.5;
    }
    voxel[axis] = slice as f64;
    voxel
}

pub(super) fn voxel_for_slice(axis: usize, slice: usize, row: usize, col: usize) -> [f64; 3] {
    match axis {
        0 => [slice as f64, row as f64, col as f64],
        1 => [row as f64, slice as f64, col as f64],
        _ => [row as f64, col as f64, slice as f64],
    }
}

pub(super) fn nearest_secondary_pixel(
    volume: &LoadedVolume,
    continuous_voxel: [f64; 3],
    axis: usize,
    slice: usize,
) -> Option<f32> {
    let mut indices = [0_usize; 3];
    indices[axis] = slice;
    for (index, extent) in volume.shape.into_iter().enumerate() {
        if index == axis {
            continue;
        }
        let coordinate = continuous_voxel[index];
        if !coordinate.is_finite()
            || coordinate < -0.5
            || coordinate > extent.saturating_sub(1) as f64 + 0.5
        {
            return None;
        }
        let nearest = coordinate.round();
        if nearest < 0.0 || nearest >= extent as f64 {
            return None;
        }
        indices[index] = nearest as usize;
    }
    Some(volume.pixel_at(indices[0], indices[1], indices[2]))
}
