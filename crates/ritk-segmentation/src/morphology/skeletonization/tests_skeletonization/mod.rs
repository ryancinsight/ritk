//! Tests for skeletonization
//! Extracted from the main module to keep the 500-line structural limit.
#![allow(clippy::needless_range_loop)]

use super::*;
use burn::tensor::{Shape, Tensor, TensorData};
use burn_ndarray::NdArray;
use ritk_core::image::Image;
use ritk_core::spatial::{Direction, Point, Spacing};

type TestBackend = NdArray<f32>;

// ── 1-D helpers ───────────────────────────────────────────────────────

pub(super) fn make_mask_1d(values: Vec<f32>, nx: usize) -> Image<TestBackend, 1> {
    let device: <TestBackend as burn::tensor::backend::Backend>::Device = Default::default();
    let td = TensorData::new(values, Shape::new([nx]));
    let tensor = Tensor::<TestBackend, 1>::from_data(td, &device);
    Image::new(
        tensor,
        Point::new([0.0]),
        Spacing::new([1.0]),
        Direction::identity(),
    )
}

pub(super) fn values_1d(image: &Image<TestBackend, 1>) -> Vec<f32> {
    image
        .data()
        .clone()
        .into_data()
        .as_slice::<f32>()
        .unwrap()
        .to_vec()
}

pub(super) fn count_fg_1d(image: &Image<TestBackend, 1>) -> usize {
    values_1d(image).iter().filter(|&&v| v > 0.5).count()
}

// ── 2-D helpers ───────────────────────────────────────────────────────

pub(super) fn make_mask_2d(values: Vec<f32>, shape: [usize; 2]) -> Image<TestBackend, 2> {
    let device: <TestBackend as burn::tensor::backend::Backend>::Device = Default::default();
    let td = TensorData::new(values, Shape::new(shape));
    let tensor = Tensor::<TestBackend, 2>::from_data(td, &device);
    Image::new(
        tensor,
        Point::new([0.0; 2]),
        Spacing::new([1.0; 2]),
        Direction::identity(),
    )
}

pub(super) fn values_2d(image: &Image<TestBackend, 2>) -> Vec<f32> {
    image
        .data()
        .clone()
        .into_data()
        .as_slice::<f32>()
        .unwrap()
        .to_vec()
}

pub(super) fn count_fg_2d(image: &Image<TestBackend, 2>) -> usize {
    values_2d(image).iter().filter(|&&v| v > 0.5).count()
}

// ── 3-D helpers ───────────────────────────────────────────────────────

pub(super) fn make_mask_3d(values: Vec<f32>, shape: [usize; 3]) -> Image<TestBackend, 3> {
    let device: <TestBackend as burn::tensor::backend::Backend>::Device = Default::default();
    let td = TensorData::new(values, Shape::new(shape));
    let tensor = Tensor::<TestBackend, 3>::from_data(td, &device);
    Image::new(
        tensor,
        Point::new([0.0; 3]),
        Spacing::new([1.0; 3]),
        Direction::identity(),
    )
}

pub(super) fn values_3d(image: &Image<TestBackend, 3>) -> Vec<f32> {
    image
        .data()
        .clone()
        .into_data()
        .as_slice::<f32>()
        .unwrap()
        .to_vec()
}

pub(super) fn count_fg_3d(image: &Image<TestBackend, 3>) -> usize {
    values_3d(image).iter().filter(|&&v| v > 0.5).count()
}

// ── Connected component counting (for topology checks) ───────────────

/// Count 8-connected foreground components in a 2-D flat mask.
pub(super) fn count_components_2d(flat: &[f32], ny: usize, nx: usize) -> usize {
    let n = ny * nx;
    let mask: Vec<bool> = flat.iter().map(|&v| v > 0.5).collect();
    let mut visited = vec![false; n];
    let mut components = 0usize;
    for start in 0..n {
        if !mask[start] || visited[start] {
            continue;
        }
        components += 1;
        let mut stack = vec![start];
        visited[start] = true;
        while let Some(idx) = stack.pop() {
            let iy = idx / nx;
            let ix = idx % nx;
            for dy in -1isize..=1 {
                for dx in -1isize..=1 {
                    if dy == 0 && dx == 0 {
                        continue;
                    }
                    let ny_i = iy as isize + dy;
                    let nx_i = ix as isize + dx;
                    if ny_i < 0 || ny_i >= ny as isize || nx_i < 0 || nx_i >= nx as isize {
                        continue;
                    }
                    let ni = ny_i as usize * nx + nx_i as usize;
                    if mask[ni] && !visited[ni] {
                        visited[ni] = true;
                        stack.push(ni);
                    }
                }
            }
        }
    }
    components
}

/// Count 26-connected foreground components in a 3-D flat mask.
pub(super) fn count_components_3d(flat: &[f32], nz: usize, ny: usize, nx: usize) -> usize {
    let n = nz * ny * nx;
    let mask: Vec<bool> = flat.iter().map(|&v| v > 0.5).collect();
    let mut visited = vec![false; n];
    let mut components = 0usize;
    for start in 0..n {
        if !mask[start] || visited[start] {
            continue;
        }
        components += 1;
        let mut stack = vec![start];
        visited[start] = true;
        while let Some(idx) = stack.pop() {
            let iz = idx / (ny * nx);
            let rem = idx % (ny * nx);
            let iy = rem / nx;
            let ix = rem % nx;
            for dz in -1isize..=1 {
                for dy in -1isize..=1 {
                    for dx in -1isize..=1 {
                        if dz == 0 && dy == 0 && dx == 0 {
                            continue;
                        }
                        let gz = iz as isize + dz;
                        let gy = iy as isize + dy;
                        let gx = ix as isize + dx;
                        if gz < 0
                            || gz >= nz as isize
                            || gy < 0
                            || gy >= ny as isize
                            || gx < 0
                            || gx >= nx as isize
                        {
                            continue;
                        }
                        let ni = gz as usize * ny * nx + gy as usize * nx + gx as usize;
                        if mask[ni] && !visited[ni] {
                            visited[ni] = true;
                            stack.push(ni);
                        }
                    }
                }
            }
        }
    }
    components
}

/// Check that no 2×2 foreground block exists (thinness in 2-D).
pub(super) fn has_2x2_block(flat: &[f32], ny: usize, nx: usize) -> bool {
    for iy in 0..ny.saturating_sub(1) {
        for ix in 0..nx.saturating_sub(1) {
            if flat[iy * nx + ix] > 0.5
                && flat[iy * nx + ix + 1] > 0.5
                && flat[(iy + 1) * nx + ix] > 0.5
                && flat[(iy + 1) * nx + ix + 1] > 0.5
            {
                return true;
            }
        }
    }
    false
}

mod thin_2d;
mod thin_3d;

// ── D = 1 tests ──────────────────────────────────────────────────────

#[test]
fn test_1d_empty_stays_empty() {
    let image = make_mask_1d(vec![0.0; 5], 5);
    let result = Skeletonization::new().apply(&image);
    assert_eq!(count_fg_1d(&result), 0);
}

#[test]
fn test_1d_single_voxel_preserved() {
    let mut vals = vec![0.0_f32; 7];
    vals[3] = 1.0;
    let image = make_mask_1d(vals, 7);
    let result = Skeletonization::new().apply(&image);
    let v = values_1d(&result);
    assert_eq!(v[3], 1.0);
    assert_eq!(count_fg_1d(&result), 1);
}

#[test]
fn test_1d_run_produces_midpoint() {
    // Run of 5: indices 1..=5 → midpoint = (1+5)/2 = 3.
    let mut vals = vec![0.0_f32; 8];
    for i in 1..=5 {
        vals[i] = 1.0;
    }
    let image = make_mask_1d(vals, 8);
    let result = Skeletonization::new().apply(&image);
    let v = values_1d(&result);
    assert_eq!(count_fg_1d(&result), 1, "single run → single midpoint");
    assert_eq!(v[3], 1.0, "midpoint at index 3");
}

#[test]
fn test_1d_two_runs_two_midpoints() {
    // Two runs: [0,1,1,0,0,1,1,1,0]
    let vals = vec![0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0];
    let image = make_mask_1d(vals, 9);
    let result = Skeletonization::new().apply(&image);
    let v = values_1d(&result);
    assert_eq!(count_fg_1d(&result), 2, "two runs → two midpoints");
    // Run 1: [1,2] → midpoint 1. Run 2: [5,6,7] → midpoint 6.
    assert_eq!(v[1], 1.0);
    assert_eq!(v[6], 1.0);
}

#[test]
fn test_1d_all_foreground() {
    // Entire image foreground: run [0, nx-1] → midpoint at nx/2.
    let nx = 9;
    let image = make_mask_1d(vec![1.0; nx], nx);
    let result = Skeletonization::new().apply(&image);
    assert_eq!(count_fg_1d(&result), 1);
    let v = values_1d(&result);
    assert_eq!(v[4], 1.0, "midpoint of [0,8] is 4");
}
