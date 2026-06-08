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
pub fn init_centers(
    data: &[f64],
    shape: &[usize],
    ndim: usize,
    steps: &[usize],
    gradient: &[f64],
    k: usize,
) -> Vec<Center> {
    let mut grid_points: Vec<Vec<usize>> = Vec::with_capacity(k);
    let mut current = vec![0usize; ndim];
    generate_grid_points(shape, steps, &mut grid_points, &mut current, 0, ndim);

    // If the grid produced fewer than k points, try reducing step sizes
    // to generate more positions.
    let mut adjusted_steps = steps.to_vec();
    while grid_points.len() < k {
        // Find the axis with the largest step that still has room for more points.
        let best_axis = (0..ndim)
            .filter(|&d| adjusted_steps[d] > 1 && shape[d] > adjusted_steps[d])
            .max_by_key(|&d| adjusted_steps[d]);
        match best_axis {
            Some(d) => {
                adjusted_steps[d] = (adjusted_steps[d] - 1).max(1);
                grid_points.clear();
                generate_grid_points(
                    shape,
                    &adjusted_steps,
                    &mut grid_points,
                    &mut current,
                    0,
                    ndim,
                );
            }
            None => break,
        }
    }

    let mut centers = Vec::with_capacity(k.min(grid_points.len()));
    for coords in grid_points.iter().take(k) {
        // Perturb to lowest-gradient voxel in 3^ndim neighbourhood.
        let best = perturb_center(coords, shape, gradient, ndim);
        let best_flat = encode_coords(&best, shape);

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
fn generate_grid_points(
    shape: &[usize],
    steps: &[usize],
    points: &mut Vec<Vec<usize>>,
    current: &mut [usize],
    depth: usize,
    ndim: usize,
) {
    if depth == ndim {
        points.push(current.to_vec());
        return;
    }
    let step = steps[depth].max(1);
    let half = step / 2;
    let mut pos = half;
    let mut any = false;
    while pos < shape[depth] {
        current[depth] = pos;
        generate_grid_points(shape, steps, points, current, depth + 1, ndim);
        pos += step;
        any = true;
    }
    // Fallback: if no point was placed on this axis, put one at midpoint.
    if !any && shape[depth] > 0 {
        current[depth] = shape[depth] / 2;
        generate_grid_points(shape, steps, points, current, depth + 1, ndim);
    }
}

/// Perturb a cluster center to the lowest-gradient voxel in a 3^ndim neighbourhood.
fn perturb_center(center: &[usize], shape: &[usize], gradient: &[f64], ndim: usize) -> Vec<usize> {
    let mut best_coords = center.to_vec();
    let mut best_grad = f64::MAX;
    let mut offset = vec![0isize; ndim];
    find_min_gradient(
        &mut offset,
        0,
        center,
        shape,
        gradient,
        &mut best_coords,
        &mut best_grad,
        ndim,
    );
    best_coords
}

/// Recursively enumerate a 3^ndim neighbourhood, tracking the minimum gradient.
#[allow(clippy::too_many_arguments)]
fn find_min_gradient(
    offset: &mut [isize],
    depth: usize,
    center: &[usize],
    shape: &[usize],
    gradient: &[f64],
    best_coords: &mut [usize],
    best_grad: &mut f64,
    ndim: usize,
) {
    if depth == ndim {
        let mut coords = vec![0usize; ndim];
        for d in 0..ndim {
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
            best_coords.copy_from_slice(&coords);
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
            ndim,
        );
    }
}
