//! Pyramid helpers: average-pool downsampling and trilinear upsampling.

use crate::deformable_field_ops::{flat, trilinear_interpolate};

/// Downsample a 3-D image by factor `f` via average pooling with stride `f`.
///
/// Output dimension per axis: `new_d = max(1, d / f)`.
pub(crate) fn downsample(image: &[f32], dims: [usize; 3], factor: usize) -> Vec<f32> {
    let [nz, ny, nx] = dims;
    let od = [
        (nz / factor).max(1),
        (ny / factor).max(1),
        (nx / factor).max(1),
    ];
    let mut out = vec![0.0_f32; od[0] * od[1] * od[2]];
    for oz in 0..od[0] {
        for oy in 0..od[1] {
            for ox in 0..od[2] {
                let (mut sum, mut cnt) = (0.0_f64, 0u32);
                for dz in 0..factor {
                    let iz = oz * factor + dz;
                    if iz >= nz {
                        break;
                    }
                    for dy in 0..factor {
                        let iy = oy * factor + dy;
                        if iy >= ny {
                            break;
                        }
                        for dx in 0..factor {
                            let ix = ox * factor + dx;
                            if ix >= nx {
                                break;
                            }
                            sum += image[flat(iz, iy, ix, ny, nx)] as f64;
                            cnt += 1;
                        }
                    }
                }
                out[flat(oz, oy, ox, od[1], od[2])] = (sum / cnt as f64) as f32;
            }
        }
    }
    out
}

/// Upsample a single displacement-field component via trilinear interpolation.
///
/// `component` ∈ {0,1,2} selects the axis whose ratio scales displacement
/// values to preserve physical displacement magnitude across voxel-size changes.
pub(crate) fn upsample_field(
    field: &[f32],
    old: [usize; 3],
    new: [usize; 3],
    component: usize,
) -> Vec<f32> {
    let nn = new[0] * new[1] * new[2];
    let scale = if old[component] > 0 {
        new[component] as f32 / old[component] as f32
    } else {
        1.0
    };
    let mut out = vec![0.0_f32; nn];
    let map = |n_new: usize, n_old: usize, idx: usize| -> f32 {
        if n_new > 1 && n_old > 1 {
            idx as f32 * (n_old - 1) as f32 / (n_new - 1) as f32
        } else {
            0.0
        }
    };
    for iz in 0..new[0] {
        let oz = map(new[0], old[0], iz);
        for iy in 0..new[1] {
            let oy = map(new[1], old[1], iy);
            for ix in 0..new[2] {
                let ox = map(new[2], old[2], ix);
                out[flat(iz, iy, ix, new[1], new[2])] =
                    trilinear_interpolate(field, old.into(), oz, oy, ox) * scale;
            }
        }
    }
    out
}
