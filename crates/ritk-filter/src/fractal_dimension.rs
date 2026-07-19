//! Stochastic fractal dimension filter for 3-D images.
//!
//! # Mathematical Specification
//!
//! This implements `itk::StochasticFractalDimensionImageFilter`. For every
//! voxel, it estimates the local fractal dimension from the scaling of
//! intensity differences with physical distance inside a `(2R + 1)^3`
//! neighborhood, using the default radius `R = 2` on each axis.
//!
//! For each ordered in-bounds pair of distinct neighborhood members `(p, q)`, it
//! greedily bins the squared physical distance: a pair joins the first stored
//! distance whose difference is below `0.5 * min(spacing)`. Each bin accumulates
//! the mean absolute intensity difference. A least-squares fit of
//! `(ln(sqrt(d^2)), ln(mean))` gives `D = 3 - slope`, where
//! `slope = (N * sum(xy) - sum(x) * sum(y)) / (N * sum(x^2) - sum(x)^2)`.
//! Distances are formed from absolute physical coordinates before subtraction,
//! preserving ITK's bin-boundary rounding. Edge neighborhoods omit
//! out-of-bounds members; constant neighborhoods intentionally produce
//! non-finite values, matching ITK's degenerate logarithm.

use coeus_core::{ComputeBackend, CpuAddressableStorage};
use ritk_image::Image;

/// Stochastic fractal dimension filter (`itk::StochasticFractalDimensionImageFilter`).
#[derive(Debug, Clone, Copy)]
pub struct StochasticFractalDimensionFilter {
    /// Neighborhood radius per axis in `[z, y, x]` order.
    pub radius: [usize; 3],
}

impl Default for StochasticFractalDimensionFilter {
    fn default() -> Self {
        Self { radius: [2; 3] }
    }
}

impl StochasticFractalDimensionFilter {
    /// Construct with a per-axis neighborhood radius (`[z, y, x]`).
    #[must_use]
    pub fn new(radius: [usize; 3]) -> Self {
        Self { radius }
    }

    /// Estimate the per-voxel stochastic fractal dimension of a native 3-D image.
    ///
    /// # Errors
    ///
    /// Returns an error when the image storage cannot be read or rebuilt on the
    /// selected compute backend.
    pub fn apply<B>(
        &self,
        image: &Image<f32, B, 3>,
        backend: &B,
    ) -> anyhow::Result<Image<f32, B, 3>>
    where
        B: ComputeBackend + Default,
        B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
    {
        let (values, [nz, ny, nx]) = ritk_tensor_ops::native::extract_image_vec(image)?;
        let spacing = [image.spacing()[0], image.spacing()[1], image.spacing()[2]];
        let origin = [image.origin()[0], image.origin()[1], image.origin()[2]];
        let mut direction = [[0.0f64; 3]; 3];
        for (column, output) in direction.iter_mut().enumerate() {
            for (axis, value) in output.iter_mut().enumerate() {
                *value = image.direction()[(column, axis)];
            }
        }
        let tolerance = 0.5 * spacing.iter().copied().fold(f64::INFINITY, f64::min);
        let [radius_z, radius_y, radius_x] = self.radius;

        let physical_point = |z: f64, y: f64, x: f64| {
            let scaled = [z * spacing[0], y * spacing[1], x * spacing[2]];
            let mut point = origin;
            for (column, value) in point.iter_mut().enumerate() {
                *value += direction[column][0] * scaled[0]
                    + direction[column][1] * scaled[1]
                    + direction[column][2] * scaled[2];
            }
            point
        };

        let mut offsets =
            Vec::with_capacity((2 * radius_z + 1) * (2 * radius_y + 1) * (2 * radius_x + 1));
        for z in -(radius_z as isize)..=(radius_z as isize) {
            for y in -(radius_y as isize)..=(radius_y as isize) {
                for x in -(radius_x as isize)..=(radius_x as isize) {
                    offsets.push([z, y, x]);
                }
            }
        }

        let output =
            moirai::map_collect_index_with::<moirai::Adaptive, _, _>(values.len(), |flat| {
                let center_z = flat / (ny * nx);
                let remainder = flat % (ny * nx);
                let center_y = remainder / nx;
                let center_x = remainder % nx;

                let mut members = Vec::with_capacity(offsets.len());
                for offset in &offsets {
                    let z = center_z as isize + offset[0];
                    let y = center_y as isize + offset[1];
                    let x = center_x as isize + offset[2];
                    if z < 0
                        || y < 0
                        || x < 0
                        || z >= nz as isize
                        || y >= ny as isize
                        || x >= nx as isize
                    {
                        continue;
                    }
                    let value = values[(z as usize) * ny * nx + (y as usize) * nx + x as usize];
                    members.push((physical_point(z as f64, y as f64, x as f64), value));
                }

                let mut distances = Vec::<f64>::new();
                let mut frequencies = Vec::<f64>::new();
                let mut differences = Vec::<f64>::new();
                for (left_index, &(left_point, left_value)) in members.iter().enumerate() {
                    for (right_index, &(right_point, right_value)) in members.iter().enumerate() {
                        if left_index == right_index {
                            continue;
                        }
                        let delta_x = left_point[2] - right_point[2];
                        let delta_y = left_point[1] - right_point[1];
                        let delta_z = left_point[0] - right_point[0];
                        let squared_distance =
                            delta_x * delta_x + delta_y * delta_y + delta_z * delta_z;
                        let difference = (left_value - right_value).abs() as f64;
                        if let Some(index) = distances
                            .iter()
                            .position(|&distance| (distance - squared_distance).abs() < tolerance)
                        {
                            frequencies[index] += 1.0;
                            differences[index] += difference;
                        } else {
                            distances.push(squared_distance);
                            frequencies.push(1.0);
                            differences.push(difference);
                        }
                    }
                }

                let (mut sum_x, mut sum_y, mut sum_xx, mut sum_xy) = (0.0, 0.0, 0.0, 0.0);
                for index in 0..distances.len() {
                    let mean = differences[index] / frequencies[index];
                    let x = distances[index].sqrt().ln();
                    let y = mean.ln();
                    sum_x += x;
                    sum_y += y;
                    sum_xx += x * x;
                    sum_xy += x * y;
                }
                let count = distances.len() as f64;
                let slope = (count * sum_xy - sum_x * sum_y) / (count * sum_xx - sum_x * sum_x);
                (3.0 - slope) as f32
            });

        crate::native_support::rebuild_image(output, [nz, ny, nx], image, backend)
    }
}
