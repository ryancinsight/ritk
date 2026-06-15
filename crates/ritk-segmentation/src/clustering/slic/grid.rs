//! Grid initialization and center perturbation for SLIC.

use super::coords::encode_coords;

/// Cluster center: intensity + spatial coordinates.
pub struct Center {
    pub intensity: f64,
    pub pos: Vec<f64>,
}

/// Initialize K cluster centers on a regular grid, perturbed to the
/// lowest-gradient voxel in a 3^ndim neighbourhood.
///
/// If the initial grid produces fewer than k points, step sizes are
/// adaptively reduced along the most compressible axis until at least
/// k centers are generated (or no further reduction is possible).
///
/// # Panics
/// Panics if `ndim` is not in {2, 3}.
pub fn init_centers(
    data: &[f64],
    shape: &[usize],
    ndim: usize,
    steps: &[usize],
    gradient: &[f64],
    k: usize,
) -> Vec<Center> {
    match ndim {
        2 => init_centers_impl::<2>(data, shape, steps, gradient, k),
        3 => init_centers_impl::<3>(data, shape, steps, gradient, k),
        _ => panic!("init_centers: unsupported dimensionality {}", ndim),
    }
}

/// Const-generic implementation of [`init_centers`].
fn init_centers_impl<const D: usize>(
    data: &[f64],
    shape: &[usize],
    steps: &[usize],
    gradient: &[f64],
    k: usize,
) -> Vec<Center> {
    let shape_arr: [usize; D] = {
        let mut arr = [0usize; D];
        arr.copy_from_slice(shape);
        arr
    };
    let steps_arr: [usize; D] = {
        let mut arr = [0usize; D];
        arr.copy_from_slice(steps);
        arr
    };

    let mut grid_points: Vec<[usize; D]> = Vec::with_capacity(k);
    let mut current = [0usize; D];
    generate_grid_points(shape_arr, steps_arr, &mut grid_points, &mut current, 0);

    // If the grid produced fewer than k points, try reducing step sizes
    // to generate more positions.
    let mut adjusted_steps = steps_arr;
    while grid_points.len() < k {
        // Find the axis with the largest step that still has room for more points.
        let best_axis = (0..D)
            .filter(|&d| adjusted_steps[d] > 1 && shape_arr[d] > adjusted_steps[d])
            .max_by_key(|&d| adjusted_steps[d]);
        match best_axis {
            Some(d) => {
                adjusted_steps[d] = (adjusted_steps[d] - 1).max(1);
                grid_points.clear();
                generate_grid_points(
                    shape_arr,
                    adjusted_steps,
                    &mut grid_points,
                    &mut current,
                    0,
                );
            }
            None => break,
        }
    }

    let mut centers = Vec::with_capacity(k.min(grid_points.len()));
    for coords in grid_points.iter().take(k) {
        // Perturb to lowest-gradient voxel in 3^D neighbourhood.
        let best = perturb_center(coords, shape_arr, gradient);
        let best_flat = encode_coords(&best, shape_arr);

        centers.push(Center {
            intensity: data[best_flat],
            pos: best.iter().map(|&c| c as f64).collect(),
        });
    }

    centers
}

/// Recursively generate grid center positions.
///
/// For each axis, centers are placed at `step/2 + n*step` for n = 0,1,2,...
/// Fallback: if an axis is too small for any grid point, one is placed at
/// its midpoint.
fn generate_grid_points<const D: usize>(
    shape: [usize; D],
    steps: [usize; D],
    points: &mut Vec<[usize; D]>,
    current: &mut [usize; D],
    depth: usize,
) {
    if depth == D {
        points.push(*current);
        return;
    }
    let step = steps[depth].max(1);
    let half = step / 2;
    let mut pos = half;
    let mut any = false;
    while pos < shape[depth] {
        current[depth] = pos;
        generate_grid_points(shape, steps, points, current, depth + 1);
        pos += step;
        any = true;
    }
    // Fallback: if no point was placed on this axis, put one at midpoint.
    if !any && shape[depth] > 0 {
        current[depth] = shape[depth] / 2;
        generate_grid_points(shape, steps, points, current, depth + 1);
    }
}

/// Perturb a cluster center to the lowest-gradient voxel in a 3^D neighbourhood.
fn perturb_center<const D: usize>(
    center: &[usize; D],
    shape: [usize; D],
    gradient: &[f64],
) -> [usize; D] {
    let mut best_coords = *center;
    let mut best_grad = f64::MAX;
    let mut offset = [0isize; D];
    find_min_gradient(
        &mut offset,
        0,
        center,
        shape,
        gradient,
        &mut best_coords,
        &mut best_grad,
    );
    best_coords
}

/// Recursively enumerate a 3^D neighbourhood, tracking the minimum gradient.
#[allow(clippy::too_many_arguments)]
fn find_min_gradient<const D: usize>(
    offset: &mut [isize; D],
    depth: usize,
    center: &[usize; D],
    shape: [usize; D],
    gradient: &[f64],
    best_coords: &mut [usize; D],
    best_grad: &mut f64,
) {
    if depth == D {
        let mut coords = [0usize; D];
        for d in 0..D {
            let c = center[d] as isize + offset[d];
            if c < 0 || c >= shape[d] as isize {
                return;
            }
            coords[d] = c as usize;
        }
        let flat = encode_coords(&coords, shape);
        let g = gradient[flat];
        if g < *best_grad {
            *best_grad = g;
            *best_coords = coords;
        }
        return;
    }

    for delta in -1isize..=1 {
        offset[depth] = delta;
        find_min_gradient(
            offset,
            depth + 1,
            center,
            shape,
            gradient,
            best_coords,
            best_grad,
        );
    }
}
