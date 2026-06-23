//! Volume warping via 4×4 homogeneous transforms (nearest-neighbour interpolation).

use leto::Array3;

use super::transform::transform_point;
use crate::types::AffineTransform;

/// Apply a 4×4 homogeneous transformation to a 3D volume.
///
/// Each output voxel (z, y, x) is filled from the nearest source voxel at
/// `transform_point([x, y, z], transform)`, with zero padding outside bounds.
pub fn apply_transform(volume: &Array3<f64>, transform: &AffineTransform) -> Array3<f64> {
    let shape = volume.shape();
    let depth = shape[0];
    let height = shape[1];
    let width = shape[2];
    let mut result = Array3::zeros([depth, height, width]);

    for z in 0..depth {
        for y in 0..height {
            for x in 0..width {
                let src = transform_point(&[x as f64, y as f64, z as f64], transform);
                let sx = src[0].round() as isize;
                let sy = src[1].round() as isize;
                let sz = src[2].round() as isize;

                if sx >= 0
                    && sx < width as isize
                    && sy >= 0
                    && sy < height as isize
                    && sz >= 0
                    && sz < depth as isize
                {
                    let val = *volume.get([sz as usize, sy as usize, sx as usize]).unwrap();
                    *result.get_mut([z, y, x]).unwrap() = val;
                }
            }
        }
    }
    result
}
