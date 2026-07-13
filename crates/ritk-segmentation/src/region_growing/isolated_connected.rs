//! Isolated-connected region growing.
//!
//! Given seeds `s1` and `s2`, this filter binary-searches the inclusive
//! connected-threshold band that still contains `s1` but excludes `s2`.
//! Search arithmetic uses `f64`, matching ITK's accumulator contract, while
//! image samples and threshold predicates remain `f32`.

use std::collections::VecDeque;

use ritk_core::spatial::VoxelIndex;
use ritk_image::tensor::Backend;
use ritk_image::Image;
use ritk_tensor_ops::{extract_vec, rebuild};

use super::connected_threshold::flood_fill;
use super::intensity::within_finite_bounds;

/// Threshold endpoint varied by isolated-connected search.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum IsolationThreshold {
    /// Search the upper endpoint while holding the lower endpoint fixed.
    #[default]
    Upper,
    /// Search the lower endpoint while holding the upper endpoint fixed.
    Lower,
}

/// Validated isolated-connected search configuration.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IsolatedConnectedConfig {
    lower: f32,
    upper: f32,
    replace_value: f32,
    tolerance: f64,
    threshold: IsolationThreshold,
}

impl IsolatedConnectedConfig {
    /// Construct a validated search configuration.
    ///
    /// # Errors
    ///
    /// Returns an error unless bounds and replacement are finite and tolerance
    /// is finite and positive.
    pub fn new(
        lower: f32,
        upper: f32,
        replace_value: f32,
        tolerance: f64,
        threshold: IsolationThreshold,
    ) -> anyhow::Result<Self> {
        anyhow::ensure!(
            lower.is_finite() && upper.is_finite() && lower <= upper,
            "isolated connected bounds must be finite and ordered, got [{lower}, {upper}]"
        );
        anyhow::ensure!(
            replace_value.is_finite(),
            "isolated connected replacement must be finite, got {replace_value}"
        );
        anyhow::ensure!(
            tolerance.is_finite() && tolerance > 0.0,
            "isolated connected tolerance must be finite and positive, got {tolerance}"
        );
        Ok(Self {
            lower,
            upper,
            replace_value,
            tolerance,
            threshold,
        })
    }

    /// Return the lower intensity bound.
    pub fn lower(self) -> f32 {
        self.lower
    }

    /// Return the upper intensity bound.
    pub fn upper(self) -> f32 {
        self.upper
    }

    /// Return the nonzero output label.
    pub fn replace_value(self) -> f32 {
        self.replace_value
    }

    /// Return the binary-search tolerance.
    pub fn tolerance(self) -> f64 {
        self.tolerance
    }

    /// Return the searched threshold endpoint.
    pub fn threshold(self) -> IsolationThreshold {
        self.threshold
    }
}

impl Default for IsolatedConnectedConfig {
    fn default() -> Self {
        Self {
            lower: 0.0,
            upper: 1.0,
            replace_value: 1.0,
            tolerance: 1.0,
            threshold: IsolationThreshold::Upper,
        }
    }
}

/// Isolated-connected region-growing filter.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IsolatedConnectedFilter {
    seed1: VoxelIndex,
    seed2: VoxelIndex,
    config: IsolatedConnectedConfig,
}

/// Result of isolated-connected region growing.
#[derive(Debug)]
pub struct IsolatedConnectedOutput<I> {
    image: I,
    thresholding_failed: bool,
}

impl<I> IsolatedConnectedOutput<I> {
    /// Return whether the final ITK-compatible band failed to separate the seeds.
    pub fn thresholding_failed(&self) -> bool {
        self.thresholding_failed
    }

    /// Borrow the produced image.
    pub fn image(&self) -> &I {
        &self.image
    }

    /// Consume this result and return the produced image.
    pub fn into_image(self) -> I {
        self.image
    }
}

impl IsolatedConnectedFilter {
    /// Construct an isolated-connected filter.
    pub fn new(
        seed1: impl Into<VoxelIndex>,
        seed2: impl Into<VoxelIndex>,
        config: IsolatedConnectedConfig,
    ) -> Self {
        Self {
            seed1: seed1.into(),
            seed2: seed2.into(),
            config,
        }
    }

    /// Apply isolated-connected growth to a legacy image.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid image storage, seed coordinates, non-finite
    /// samples, or output reconstruction failure.
    pub fn apply<B: Backend>(
        &self,
        image: &Image<B, 3>,
    ) -> anyhow::Result<IsolatedConnectedOutput<Image<B, 3>>> {
        let (values, dimensions) = extract_vec(image)?;
        let output =
            isolated_connected_values(&values, dimensions, self.seed1, self.seed2, self.config)?;
        Ok(IsolatedConnectedOutput {
            image: rebuild(output.values, dimensions, image),
            thresholding_failed: output.thresholding_failed,
        })
    }

    /// Apply isolated-connected growth directly to a Coeus-native image.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid image storage, seed coordinates, non-finite
    /// samples, or output construction failure. An unseparable final band is
    /// returned with `thresholding_failed` set, matching ITK.
    pub fn apply_native<B>(
        &self,
        image: &ritk_image::native::Image<f32, B, 3>,
        backend: &B,
    ) -> anyhow::Result<IsolatedConnectedOutput<ritk_image::native::Image<f32, B, 3>>>
    where
        B: coeus_core::ComputeBackend,
        B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    {
        let output = isolated_connected_values(
            image.data_slice()?,
            image.shape(),
            self.seed1,
            self.seed2,
            self.config,
        )?;
        Ok(IsolatedConnectedOutput {
            image: crate::native_output::from_values(image, output.values, backend)?,
            thresholding_failed: output.thresholding_failed,
        })
    }
}

fn isolated_connected_values(
    values: &[f32],
    dimensions: [usize; 3],
    seed1: VoxelIndex,
    seed2: VoxelIndex,
    config: IsolatedConnectedConfig,
) -> anyhow::Result<FlatOutput> {
    validate_input(values, dimensions, seed1, seed2)?;
    let seed2_flat = flatten(seed2, dimensions);
    let mut workspace = FloodWorkspace::new(values.len());
    let isolated = match config.threshold {
        IsolationThreshold::Upper => {
            let mut lower = f64::from(config.lower);
            let mut upper = f64::from(config.upper);
            let mut guess = upper;
            while lower + config.tolerance < guess {
                if workspace.reaches(
                    values,
                    dimensions,
                    seed1,
                    seed2_flat,
                    config.lower,
                    guess as f32,
                ) {
                    upper = guess;
                } else {
                    lower = guess;
                }
                guess = (upper + lower) * 0.5;
            }
            lower as f32
        }
        IsolationThreshold::Lower => {
            let mut lower = f64::from(config.lower);
            let mut upper = f64::from(config.upper);
            let mut guess = lower;
            while guess < upper - config.tolerance {
                if workspace.reaches(
                    values,
                    dimensions,
                    seed1,
                    seed2_flat,
                    guess as f32,
                    config.upper,
                ) {
                    lower = guess;
                } else {
                    upper = guess;
                }
                guess = (upper + lower) * 0.5;
            }
            upper as f32
        }
    };
    let (band_lower, band_upper) = match config.threshold {
        IsolationThreshold::Upper => (config.lower, isolated),
        IsolationThreshold::Lower => (isolated, config.upper),
    };
    let mask = flood_fill(values, dimensions, seed1, band_lower, band_upper);
    let seed1_flat = flatten(seed1, dimensions);
    let thresholding_failed = mask[seed1_flat] == 0.0 || mask[seed2_flat] != 0.0;
    Ok(FlatOutput {
        values: mask
            .into_iter()
            .map(|value| {
                if value != 0.0 {
                    config.replace_value
                } else {
                    0.0
                }
            })
            .collect(),
        thresholding_failed,
    })
}

struct FlatOutput {
    values: Vec<f32>,
    thresholding_failed: bool,
}

fn validate_input(
    values: &[f32],
    dimensions: [usize; 3],
    seed1: VoxelIndex,
    seed2: VoxelIndex,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        dimensions.iter().all(|&extent| extent > 0),
        "isolated connected requires nonzero dimensions, got {dimensions:?}"
    );
    let expected = dimensions
        .iter()
        .try_fold(1usize, |count, &extent| count.checked_mul(extent))
        .ok_or_else(|| {
            anyhow::anyhow!("isolated connected shape product overflows usize: {dimensions:?}")
        })?;
    anyhow::ensure!(
        values.len() == expected,
        "isolated connected shape {dimensions:?} requires {expected} samples, got {}",
        values.len()
    );
    for (name, seed) in [("seed1", seed1), ("seed2", seed2)] {
        anyhow::ensure!(
            seed[0] < dimensions[0] && seed[1] < dimensions[1] && seed[2] < dimensions[2],
            "isolated connected {name} {:?} is outside shape {dimensions:?}",
            seed.as_array()
        );
    }
    if let Some((index, value)) = values
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        anyhow::bail!(
            "isolated connected sample at flat index {index} must be finite, got {value}"
        );
    }
    Ok(())
}

struct FloodWorkspace {
    visited: Vec<u32>,
    generation: u32,
    queue: VecDeque<usize>,
}

impl FloodWorkspace {
    fn new(sample_count: usize) -> Self {
        Self {
            visited: vec![0; sample_count],
            generation: 0,
            queue: VecDeque::with_capacity(sample_count.min(1024)),
        }
    }

    fn reaches(
        &mut self,
        values: &[f32],
        dimensions: [usize; 3],
        seed: VoxelIndex,
        target: usize,
        lower: f32,
        upper: f32,
    ) -> bool {
        self.generation = self.generation.wrapping_add(1);
        if self.generation == 0 {
            self.visited.fill(0);
            self.generation = 1;
        }
        self.queue.clear();
        let seed_flat = flatten(seed, dimensions);
        if !within_finite_bounds(values[seed_flat], lower, upper) {
            return false;
        }
        self.visited[seed_flat] = self.generation;
        self.queue.push_back(seed_flat);
        let [depth, height, width] = dimensions;
        while let Some(index) = self.queue.pop_front() {
            if index == target {
                return true;
            }
            let z = index / (height * width);
            let remainder = index % (height * width);
            let y = remainder / width;
            let x = remainder % width;
            for (nz, ny, nx) in [
                (z.checked_sub(1), Some(y), Some(x)),
                ((z + 1 < depth).then_some(z + 1), Some(y), Some(x)),
                (Some(z), y.checked_sub(1), Some(x)),
                (Some(z), (y + 1 < height).then_some(y + 1), Some(x)),
                (Some(z), Some(y), x.checked_sub(1)),
                (Some(z), Some(y), (x + 1 < width).then_some(x + 1)),
            ] {
                let (Some(nz), Some(ny), Some(nx)) = (nz, ny, nx) else {
                    continue;
                };
                let neighbor = (nz * height + ny) * width + nx;
                if self.visited[neighbor] != self.generation
                    && within_finite_bounds(values[neighbor], lower, upper)
                {
                    self.visited[neighbor] = self.generation;
                    self.queue.push_back(neighbor);
                }
            }
        }
        false
    }
}

fn flatten(seed: VoxelIndex, dimensions: [usize; 3]) -> usize {
    (seed[0] * dimensions[1] + seed[1]) * dimensions[2] + seed[2]
}

#[cfg(test)]
#[path = "tests_isolated_connected.rs"]
mod tests_isolated_connected;

#[cfg(test)]
mod workspace_tests {
    use super::{FloodWorkspace, VoxelIndex};

    #[test]
    fn search_workspace_reuses_visited_and_queue_storage() {
        let values = vec![1.0; 64];
        let dimensions = [1, 8, 8];
        let mut workspace = FloodWorkspace::new(values.len());
        assert!(workspace.reaches(&values, dimensions, VoxelIndex::new(0, 0, 0), 63, 0.0, 2.0));
        let visited = workspace.visited.as_ptr();
        let capacity = workspace.queue.capacity();
        assert!(workspace.reaches(&values, dimensions, VoxelIndex::new(0, 0, 0), 63, 0.0, 2.0));
        assert_eq!(workspace.visited.as_ptr(), visited);
        assert_eq!(workspace.queue.capacity(), capacity);
    }
}
