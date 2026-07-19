//! Isolated watershed segmentation.
//!
//! # Mathematical Specification
//!
//! Given two seeds s1, s2 in a scalar image, constructs the watershed merge
//! hierarchy of the gradient magnitude `g` (ITK `GradientMagnitudeImageFilter`,
//! unit spacing) and binary-searches the last flood level separating the seeds.
//!
//! ## Algorithm
//!
//! The initial plateau-aware basins drain to local minima. Adjacent basins form
//! edges whose height is the maximum incident sample. Directed merges are
//! ordered by saliency (`edge height - source minimum`) and replayed at each
//! candidate flood level during the isolation search.
//!
//! Almost-equal flat zones are identified once, each zone records its lowest
//! boundary destination, and the resulting strictly descending component graph
//! is resolved with path compression.
//!
//! ## Output Label Convention
//!
//! - Label 1 (`f32` 1.0): voxels whose gradient-descent basin contains s1.
//! - Label 2 (`f32` 2.0): voxels whose gradient-descent basin contains s2
//!   (when s1 and s2 are in different basins).
//! - Label 0 (`f32` 0.0): remaining voxels.
//!
//! ## Edge Cases
//!
//! - Identical seeds: all voxels assigned label 1.
//! - Seeds in the same basin: the shared basin is returned as label 1; no
//!   label-2 region is produced.
//!
//! # Complexity
//!
//! O(n + e log e), where n is the voxel count and e is the number of distinct
//! inter-basin edges sorted into the merge hierarchy.
//!
//! # References
//!
//! - ITK `itk::IsolatedWatershedImageFilter`

use ritk_image::tensor::{Backend, Tensor};
use ritk_image::Image;
use ritk_tensor_ops::extract_vec;

use super::hierarchy::WatershedHierarchy;

// 6-connected face-neighbour offsets (dz, dy, dx) for a [nz, ny, nx] grid.
const NEIGHBOUR_OFFSETS: [(i64, i64, i64); 6] = [
    (-1, 0, 0),
    (0, -1, 0),
    (0, 0, -1),
    (0, 0, 1),
    (0, 1, 0),
    (1, 0, 0),
];

/// In-bounds 6-connected neighbours of flat index `idx` as flat indices.
pub(super) fn neighbours(idx: usize, dims: [usize; 3]) -> impl Iterator<Item = usize> {
    let [nz, ny, nx] = dims;
    let z = idx / (ny * nx);
    let rem = idx % (ny * nx);
    let y = rem / nx;
    let x = rem % nx;
    NEIGHBOUR_OFFSETS.iter().filter_map(move |&(dz, dy, dx)| {
        let zi = z as i64 + dz;
        let yi = y as i64 + dy;
        let xi = x as i64 + dx;
        if zi < 0 || zi >= nz as i64 || yi < 0 || yi >= ny as i64 || xi < 0 || xi >= nx as i64 {
            None
        } else {
            Some(zi as usize * ny * nx + yi as usize * nx + xi as usize)
        }
    })
}

fn almost_equal(left: f32, right: f32) -> bool {
    let absolute_difference = (left - right).abs();
    if absolute_difference <= f32::EPSILON * 0.1 {
        return true;
    }
    if left.is_sign_negative() != right.is_sign_negative() {
        return false;
    }
    left.to_bits().abs_diff(right.to_bits()) <= 4
}

/// ITK `GradientMagnitudeImageFilter`: per-axis central difference
/// `(f[+1] âˆ’ f[âˆ’1]) / 2` with ZeroFluxNeumann (edge-clamp) boundaries, magnitude
/// `sqrt(Î£ dáµ¢Â²)`. Unit spacing is the isolated-watershed internal convention. A
/// `z == 1` volume yields `dz == 0` via the clamp, reducing to the 2-D gradient.
fn gradient_magnitude(vals: &[f32], dims: [usize; 3]) -> Vec<f32> {
    let [nz, ny, nx] = dims;
    let mut out = vec![0.0_f32; nz * ny * nx];
    moirai::enumerate_mut_with::<moirai::Adaptive, _, _>(&mut out, |i, val| {
        let z = i / (ny * nx);
        let rem = i % (ny * nx);
        let y = rem / nx;
        let x = rem % nx;

        let zm = z.saturating_sub(1);
        let zp = (z + 1).min(nz - 1);
        let ym = y.saturating_sub(1);
        let yp = (y + 1).min(ny - 1);
        let xm = x.saturating_sub(1);
        let xp = (x + 1).min(nx - 1);

        let dz = vals[zp * ny * nx + y * nx + x] * 0.5 - vals[zm * ny * nx + y * nx + x] * 0.5;
        let dy = vals[z * ny * nx + yp * nx + x] * 0.5 - vals[z * ny * nx + ym * nx + x] * 0.5;
        let dx = vals[z * ny * nx + y * nx + xp] * 0.5 - vals[z * ny * nx + y * nx + xm] * 0.5;
        *val = dz.hypot(dy).hypot(dx);
    });
    out
}

/// Assign each voxel the representative of the plateau minimum reached by its
/// component's lowest boundary.
///
/// # Algorithm
///
/// 1. Build face-connected flat zones under ITK's four-ULP equality contract.
/// 2. Record the first strictly lowest boundary neighbor for every zone.
/// 3. Resolve the strictly descending component graph to its minimum, stamping
///    every traversed component with the same representative.
///
/// # Complexity
///
/// O(n) expected for bounded face connectivity; every component enters a
/// descending path at most once before path compression labels it.
pub(super) fn watershed_basins_gd(g: &[f32], dims: [usize; 3]) -> Vec<usize> {
    let n: usize = dims.iter().product();
    let mut component_of = vec![usize::MAX; n];
    let mut components = Vec::<Vec<usize>>::new();
    let mut queue = std::collections::VecDeque::new();
    for start in 0..n {
        if component_of[start] != usize::MAX {
            continue;
        }
        let component = components.len();
        component_of[start] = component;
        queue.push_back(start);
        let mut members = Vec::new();
        while let Some(index) = queue.pop_front() {
            members.push(index);
            for neighbor in neighbours(index, dims) {
                if component_of[neighbor] == usize::MAX && almost_equal(g[neighbor], g[index]) {
                    component_of[neighbor] = component;
                    queue.push_back(neighbor);
                }
            }
        }
        components.push(members);
    }

    let mut drains_to = vec![None; components.len()];
    for (component, members) in components.iter().enumerate() {
        let value = g[members[0]];
        let mut lowest = None;
        for &index in members {
            for neighbor in neighbours(index, dims) {
                if g[neighbor] < value
                    && !almost_equal(g[neighbor], value)
                    && lowest.is_none_or(|current| g[neighbor] < g[current])
                {
                    lowest = Some(neighbor);
                }
            }
        }
        drains_to[component] = lowest.map(|index| component_of[index]);
    }

    let mut representative = vec![usize::MAX; components.len()];
    let mut path = Vec::new();
    for start in 0..components.len() {
        if representative[start] != usize::MAX {
            continue;
        }
        path.clear();
        let mut component = start;
        while representative[component] == usize::MAX {
            path.push(component);
            let Some(next) = drains_to[component] else {
                representative[component] = components[component][0];
                break;
            };
            component = next;
        }
        let basin = representative[component];
        for component in path.drain(..) {
            representative[component] = basin;
        }
    }
    component_of
        .into_iter()
        .map(|component| representative[component])
        .collect()
}

// â”€â”€ Configuration â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Parameters for isolated watershed segmentation.
///
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IsolatedWatershedConfig {
    threshold: f32,
    isolated_value_tolerance: f64,
    upper_value_limit: f64,
}

impl IsolatedWatershedConfig {
    /// Construct validated hierarchy-search parameters.
    ///
    /// # Errors
    ///
    /// Returns an error unless `0 <= threshold <= upper_value_limit <= 1` and
    /// `isolated_value_tolerance` is finite and strictly positive.
    pub fn new(
        threshold: f32,
        isolated_value_tolerance: f64,
        upper_value_limit: f64,
    ) -> anyhow::Result<Self> {
        anyhow::ensure!(
            threshold.is_finite() && (0.0..=1.0).contains(&threshold),
            "isolated watershed threshold must be finite and in [0, 1], got {threshold}"
        );
        anyhow::ensure!(
            upper_value_limit.is_finite()
                && (f64::from(threshold)..=1.0).contains(&upper_value_limit),
            "isolated watershed upper value limit must be finite and in [{threshold}, 1], got {upper_value_limit}"
        );
        anyhow::ensure!(
            isolated_value_tolerance.is_finite() && isolated_value_tolerance > 0.0,
            "isolated watershed tolerance must be finite and positive, got {isolated_value_tolerance}"
        );
        Ok(Self {
            threshold,
            isolated_value_tolerance,
            upper_value_limit,
        })
    }

    /// Return the watershed threshold fraction.
    pub fn threshold(self) -> f32 {
        self.threshold
    }

    /// Return the binary-search tolerance.
    pub fn isolated_value_tolerance(self) -> f64 {
        self.isolated_value_tolerance
    }

    /// Return the maximum hierarchy level searched.
    pub fn upper_value_limit(self) -> f64 {
        self.upper_value_limit
    }
}

impl Default for IsolatedWatershedConfig {
    fn default() -> Self {
        Self {
            threshold: 0.0,
            isolated_value_tolerance: 0.001,
            upper_value_limit: 1.0,
        }
    }
}

// â”€â”€ Core algorithm â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Gradient-descent watershed segmentation matching ITK's
/// `IsolatedWatershedImageFilter`.
///
/// Each voxel is assigned the label of the seed whose gradient-descent basin
/// it belongs to.  `seed1`/`seed2` are flat linear indices
/// (`flat = zÂ·nyÂ·nx + yÂ·nx + x`).
///
fn isolated_watershed_values(
    vals: &[f32],
    dims: [usize; 3],
    seed1: usize,
    seed2: usize,
    config: &IsolatedWatershedConfig,
) -> Vec<f32> {
    let g = gradient_magnitude(vals, dims);
    let hierarchy = WatershedHierarchy::build(&g, dims, config.threshold, config.upper_value_limit);
    let mut lower = f64::from(config.threshold);
    let mut upper = config.upper_value_limit;
    let mut guess = upper;
    let mut labels = hierarchy.labels_at(lower);
    while lower + config.isolated_value_tolerance < guess {
        labels = hierarchy.labels_at(guess);
        if labels[seed1] == labels[seed2] {
            upper = guess;
        } else {
            lower = guess;
        }
        guess = (upper + lower) * 0.5;
    }
    if labels[seed1] == labels[seed2] {
        labels = hierarchy.labels_at(lower);
    }
    let seed1_label = labels[seed1];
    let seed2_label = labels[seed2];
    labels
        .into_iter()
        .map(|label| {
            if label == seed1_label {
                1.0
            } else if label == seed2_label {
                2.0
            } else {
                0.0
            }
        })
        .collect()
}

// â”€â”€ Public filter struct â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Isolated watershed segmentation filter.
///
/// Builds the watershed merge hierarchy of the gradient magnitude, isolates
/// the last flood level separating the seeds, then labels their two regions:
///
/// | Label | Meaning |
/// |-------|---------|
/// | 1.0   | Gradient-descent basin of `seed1` |
/// | 2.0   | Gradient-descent basin of `seed2` (when separate from `seed1`) |
/// | 0.0   | Remaining voxels |
///
/// Corresponds to ITK `itk::IsolatedWatershedImageFilter`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IsolatedWatershed {
    seed1: [usize; 3],
    seed2: [usize; 3],
    config: IsolatedWatershedConfig,
}

impl IsolatedWatershed {
    /// Construct an isolated watershed filter.
    pub fn new(seed1: [usize; 3], seed2: [usize; 3], config: IsolatedWatershedConfig) -> Self {
        Self {
            seed1,
            seed2,
            config,
        }
    }

    /// Apply the isolated watershed filter to a 3-D scalar image.
    ///
    /// Returns a label image with the same shape and spatial metadata as `image`.
    /// Labels are encoded as `f32`: 1.0 (seed1 region), 2.0 (seed2 region), 0.0 (rest).
    ///
    /// # Errors
    ///
    /// Returns an error if tensor extraction fails or shape, seed, or sample
    /// validation fails.
    pub fn apply<B: Backend>(&self, image: &Image<f32, B, 3>) -> anyhow::Result<Image<f32, B, 3>> {
        let (vals, dims) = extract_vec(image)?;
        validate_image_and_seeds(&vals, dims, self.seed1, self.seed2)?;
        let [_, ny, nx] = dims;

        let seed1_flat = self.seed1[0] * ny * nx + self.seed1[1] * nx + self.seed1[2];
        let seed2_flat = self.seed2[0] * ny * nx + self.seed2[1] * nx + self.seed2[2];

        let labels = isolated_watershed_values(&vals, dims, seed1_flat, seed2_flat, &self.config);

        let device = B::default();
        let tensor = Tensor::<f32, B>::from_slice_on(dims, &labels, &device);

        Image::new(
            tensor,
            *image.origin(),
            *image.spacing(),
            *image.direction(),
        )
    }

    /// Apply isolated watershed to a Coeus-native image.
    pub fn apply_native<B>(
        &self,
        image: &ritk_image::Image<f32, B, 3>,
        backend: &B,
    ) -> anyhow::Result<ritk_image::Image<f32, B, 3>>
    where
        B: coeus_core::ComputeBackend,
        B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    {
        let values = image.data_slice()?;
        let dimensions = image.shape();
        validate_image_and_seeds(values, dimensions, self.seed1, self.seed2)?;
        let [_, height, width] = dimensions;
        let seed1 = self.seed1[0] * height * width + self.seed1[1] * width + self.seed1[2];
        let seed2 = self.seed2[0] * height * width + self.seed2[1] * width + self.seed2[2];
        crate::native_output::from_values(
            image,
            isolated_watershed_values(values, dimensions, seed1, seed2, &self.config),
            backend,
        )
    }
}

fn validate_image_and_seeds(
    values: &[f32],
    dimensions: [usize; 3],
    seed1: [usize; 3],
    seed2: [usize; 3],
) -> anyhow::Result<()> {
    anyhow::ensure!(
        dimensions.iter().all(|&extent| extent > 0),
        "isolated watershed requires nonzero dimensions, got {dimensions:?}"
    );
    let expected = dimensions
        .iter()
        .try_fold(1usize, |count, &extent| count.checked_mul(extent))
        .ok_or_else(|| {
            anyhow::anyhow!("isolated watershed shape product overflows usize: {dimensions:?}")
        })?;
    anyhow::ensure!(
        values.len() == expected,
        "isolated watershed shape {dimensions:?} requires {expected} samples, got {}",
        values.len()
    );
    for (name, seed) in [("seed1", seed1), ("seed2", seed2)] {
        anyhow::ensure!(
            seed.iter()
                .zip(dimensions)
                .all(|(&index, extent)| index < extent),
            "isolated watershed {name} {seed:?} is outside shape {dimensions:?}"
        );
    }
    if let Some((index, value)) = values
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        anyhow::bail!(
            "isolated watershed sample at flat index {index} must be finite, got {value}"
        );
    }
    Ok(())
}

// â”€â”€ Tests â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

#[cfg(test)]
#[path = "tests_isolated.rs"]
mod tests_isolated;
