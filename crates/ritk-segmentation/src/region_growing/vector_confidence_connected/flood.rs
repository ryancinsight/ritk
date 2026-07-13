//! Allocation-reusing face-connected Mahalanobis flood.

use std::collections::VecDeque;

use ritk_core::spatial::VoxelIndex;

use super::statistics::mahalanobis_squared;

pub(super) struct FloodWorkspace {
    mask: Vec<bool>,
    queue: VecDeque<usize>,
    visit_order: Vec<usize>,
    delta: Vec<f64>,
}

impl FloodWorkspace {
    pub(super) fn new(voxel_count: usize, channel_count: usize) -> Self {
        Self {
            mask: vec![false; voxel_count],
            queue: VecDeque::new(),
            visit_order: Vec::new(),
            delta: vec![0.0; channel_count],
        }
    }

    pub(super) fn mask(&self) -> &[bool] {
        &self.mask
    }

    pub(super) fn visit_order(&self) -> &[usize] {
        &self.visit_order
    }

    pub(super) fn fill<S: AsRef<[f32]>>(
        &mut self,
        channels: &[S],
        dimensions: [usize; 3],
        seeds: &[VoxelIndex],
        mean: &[f64],
        inverse: &[f64],
        threshold: f64,
    ) {
        self.mask.fill(false);
        self.queue.clear();
        self.visit_order.clear();
        let threshold_squared = threshold * threshold;
        for &seed in seeds {
            let index = flatten(seed, dimensions);
            if !self.mask[index]
                && mahalanobis_squared(channels, index, mean, inverse, &mut self.delta)
                    <= threshold_squared
            {
                self.mask[index] = true;
                self.queue.push_back(index);
                self.visit_order.push(index);
            }
        }

        let [depth, height, width] = dimensions;
        let plane = height * width;
        while let Some(index) = self.queue.pop_front() {
            let z = index / plane;
            let remainder = index % plane;
            let y = remainder / width;
            let x = remainder % width;
            for neighbor in [
                z.checked_sub(1).map(|_| index - plane),
                (z + 1 < depth).then_some(index + plane),
                y.checked_sub(1).map(|_| index - width),
                (y + 1 < height).then_some(index + width),
                x.checked_sub(1).map(|_| index - 1),
                (x + 1 < width).then_some(index + 1),
            ]
            .into_iter()
            .flatten()
            {
                if !self.mask[neighbor]
                    && mahalanobis_squared(channels, neighbor, mean, inverse, &mut self.delta)
                        <= threshold_squared
                {
                    self.mask[neighbor] = true;
                    self.queue.push_back(neighbor);
                    self.visit_order.push(neighbor);
                }
            }
        }
    }
}

fn flatten(seed: VoxelIndex, dimensions: [usize; 3]) -> usize {
    (seed[0] * dimensions[1] + seed[1]) * dimensions[2] + seed[2]
}
