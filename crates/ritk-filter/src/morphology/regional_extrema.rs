//! Regional-extrema grayscale morphological filters (regional maxima / minima,
//! valued and binary) for 3-D images.
//!
//! # Mathematical Specification
//!
//! A **regional maximum** is a connected flat zone (a maximal connected set of
//! voxels of identical value) from which it is impossible to reach a voxel of
//! strictly greater value without first descending. Equivalently: a flat zone
//! all of whose external neighbours are strictly lower. **Regional minima** are
//! the dual (all external neighbours strictly higher).
//!
//! This is computed exactly by a flat-zone flood: BFS each connected
//! equal-valued component, then mark it extremal iff no out-of-zone neighbour is
//! strictly more extreme. Unlike the `f − R^δ_f(f − 1)` reconstruction shortcut,
//! this is correct for arbitrary float values (the shortcut merges extrema whose
//! contrast is below the unit shift).
//!
//! # ITK / SimpleITK Parity
//!
//! | Filter                       | ITK class                          | SimpleITK              |
//! |------------------------------|------------------------------------|------------------------|
//! | `RegionalMaximaFilter`       | `RegionalMaximaImageFilter`        | `RegionalMaxima`       |
//! | `RegionalMinimaFilter`       | `RegionalMinimaImageFilter`        | `RegionalMinima`       |
//! | `ValuedRegionalMaximaFilter` | `ValuedRegionalMaximaImageFilter`  | `ValuedRegionalMaxima` |
//! | `ValuedRegionalMinimaFilter` | `ValuedRegionalMinimaImageFilter`  | `ValuedRegionalMinima` |
//!
//! Binary filters emit `foreground` on extrema and `background` elsewhere
//! (ITK defaults 1 / 0). Valued filters keep the input value on extrema and set
//! non-extrema to the type's non-extremal sentinel — `f32::MIN` (`−FLT_MAX`) for
//! maxima, `f32::MAX` (`+FLT_MAX`) for minima — matching ITK's `NumericTraits`.
//!
//! # References
//! - Soille, P. (2003). *Morphological Image Analysis*, 2nd ed. Springer, §6.3.

use crate::morphology::Connectivity;
use ritk_image::tensor::Backend;
use ritk_image::Image;
use ritk_tensor_ops::{extract_vec, rebuild};

/// Which extremum to detect.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExtremaKind {
    Maxima,
    Minima,
}

/// Collect the 6/26-connected in-bounds neighbours of a flat index.
fn neighbours(flat: usize, dims: [usize; 3], conn: Connectivity, out: &mut Vec<usize>) {
    let [nz, ny, nx] = dims;
    let iz = (flat / (ny * nx)) as i32;
    let iy = ((flat % (ny * nx)) / nx) as i32;
    let ix = (flat % nx) as i32;
    out.clear();
    for dz in -1i32..=1 {
        for dy in -1i32..=1 {
            for dx in -1i32..=1 {
                if (dz, dy, dx) == (0, 0, 0) || !conn.includes(dz, dy, dx) {
                    continue;
                }
                let zz = iz + dz;
                let yy = iy + dy;
                let xx = ix + dx;
                if zz < 0
                    || zz >= nz as i32
                    || yy < 0
                    || yy >= ny as i32
                    || xx < 0
                    || xx >= nx as i32
                {
                    continue;
                }
                out.push(zz as usize * ny * nx + yy as usize * nx + xx as usize);
            }
        }
    }
}

/// Mark every voxel belonging to a regional extremum flat zone.
fn regional_extrema_mask(
    data: &[f32],
    dims: [usize; 3],
    conn: Connectivity,
    kind: ExtremaKind,
) -> Vec<bool> {
    let n = data.len();
    let mut visited = vec![false; n];
    let mut is_ext = vec![false; n];
    let mut stack: Vec<usize> = Vec::new();
    let mut members: Vec<usize> = Vec::new();
    let mut nbrs: Vec<usize> = Vec::new();

    for seed in 0..n {
        if visited[seed] {
            continue;
        }
        let val = data[seed];
        visited[seed] = true;
        stack.clear();
        members.clear();
        stack.push(seed);
        let mut is_extremum = true;

        while let Some(p) = stack.pop() {
            members.push(p);
            neighbours(p, dims, conn, &mut nbrs);
            for &nb in &nbrs {
                let nv = data[nb];
                if nv == val {
                    if !visited[nb] {
                        visited[nb] = true;
                        stack.push(nb);
                    }
                } else {
                    // A strictly more-extreme out-of-zone neighbour disqualifies
                    // the whole flat zone. Keep flooding so all members are
                    // visited and marked consistently.
                    let more_extreme = match kind {
                        ExtremaKind::Maxima => nv > val,
                        ExtremaKind::Minima => nv < val,
                    };
                    if more_extreme {
                        is_extremum = false;
                    }
                }
            }
        }

        if is_extremum {
            for &m in &members {
                is_ext[m] = true;
            }
        }
    }
    is_ext
}

fn validate_samples(data: &[f32]) -> anyhow::Result<()> {
    if let Some((index, value)) = data
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        anyhow::bail!("regional-extrema sample at flat index {index} must be finite, got {value}");
    }
    Ok(())
}

/// Shared driver: extract, detect, map each voxel through `value`, rebuild.
fn run<B: Backend>(
    image: &Image<B, 3>,
    conn: Connectivity,
    kind: ExtremaKind,
    value: impl Fn(bool, f32) -> f32,
) -> anyhow::Result<Image<B, 3>> {
    let (vals, dims) = extract_vec(image)?;
    validate_samples(&vals)?;
    let mask = regional_extrema_mask(&vals, dims, conn, kind);
    let out: Vec<f32> = mask
        .iter()
        .zip(vals.iter())
        .map(|(&m, &v)| value(m, v))
        .collect();
    Ok(rebuild(out, dims, image))
}

fn run_native<B>(
    image: &ritk_image::native::Image<f32, B, 3>,
    conn: Connectivity,
    kind: ExtremaKind,
    value: impl Fn(bool, f32) -> f32,
    backend: &B,
) -> anyhow::Result<ritk_image::native::Image<f32, B, 3>>
where
    B: coeus_core::ComputeBackend,
    B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
{
    let values = image.data_slice()?;
    validate_samples(values)?;
    let mask = regional_extrema_mask(values, image.shape(), conn, kind);
    ritk_image::native::Image::from_flat_on(
        mask.iter()
            .zip(values)
            .map(|(&is_extremum, &sample)| value(is_extremum, sample))
            .collect(),
        image.shape(),
        *image.origin(),
        *image.spacing(),
        *image.direction(),
        backend,
    )
}

// ── Binary regional maxima / minima ───────────────────────────────────────────

/// Binary regional-maxima filter: `foreground` on regional maxima, `background`
/// elsewhere. ITK `RegionalMaximaImageFilter` (defaults foreground 1.0 / 0.0).
#[derive(Debug, Clone)]
pub struct RegionalMaximaFilter {
    foreground: f32,
    background: f32,
    connectivity: Connectivity,
}

impl Default for RegionalMaximaFilter {
    fn default() -> Self {
        Self {
            foreground: 1.0,
            background: 0.0,
            connectivity: Connectivity::Face6,
        }
    }
}

impl RegionalMaximaFilter {
    /// Create a regional-maxima filter (foreground 1.0, background 0.0).
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the foreground/background output values.
    pub fn with_values(mut self, foreground: f32, background: f32) -> Self {
        self.foreground = foreground;
        self.background = background;
        self
    }

    /// Set the structuring-element adjacency (defaults to [`Connectivity::Face6`]).
    pub fn with_connectivity(mut self, connectivity: Connectivity) -> Self {
        self.connectivity = connectivity;
        self
    }

    /// Apply the binary regional-maxima transform.
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> anyhow::Result<Image<B, 3>> {
        let (fg, bg) = (self.foreground, self.background);
        run(
            image,
            self.connectivity,
            ExtremaKind::Maxima,
            move |m, _| {
                if m {
                    fg
                } else {
                    bg
                }
            },
        )
    }
}

/// Binary regional-minima filter. ITK `RegionalMinimaImageFilter`.
#[derive(Debug, Clone)]
pub struct RegionalMinimaFilter {
    foreground: f32,
    background: f32,
    connectivity: Connectivity,
}

impl Default for RegionalMinimaFilter {
    fn default() -> Self {
        Self {
            foreground: 1.0,
            background: 0.0,
            connectivity: Connectivity::Face6,
        }
    }
}

impl RegionalMinimaFilter {
    /// Create a regional-minima filter (foreground 1.0, background 0.0).
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the foreground/background output values.
    pub fn with_values(mut self, foreground: f32, background: f32) -> Self {
        self.foreground = foreground;
        self.background = background;
        self
    }

    /// Set the structuring-element adjacency (defaults to [`Connectivity::Face6`]).
    pub fn with_connectivity(mut self, connectivity: Connectivity) -> Self {
        self.connectivity = connectivity;
        self
    }

    /// Apply the binary regional-minima transform.
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> anyhow::Result<Image<B, 3>> {
        let (fg, bg) = (self.foreground, self.background);
        run(
            image,
            self.connectivity,
            ExtremaKind::Minima,
            move |m, _| {
                if m {
                    fg
                } else {
                    bg
                }
            },
        )
    }

    /// Apply the binary regional-minima transform to a Coeus-native image.
    ///
    /// # Errors
    ///
    /// Returns an error for non-finite input, inaccessible backend storage, or
    /// native output construction failure.
    pub fn apply_native<B>(
        &self,
        image: &ritk_image::native::Image<f32, B, 3>,
        backend: &B,
    ) -> anyhow::Result<ritk_image::native::Image<f32, B, 3>>
    where
        B: coeus_core::ComputeBackend,
        B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    {
        let (foreground, background) = (self.foreground, self.background);
        run_native(
            image,
            self.connectivity,
            ExtremaKind::Minima,
            move |is_minimum, _| {
                if is_minimum {
                    foreground
                } else {
                    background
                }
            },
            backend,
        )
    }
}

// ── Valued regional maxima / minima ───────────────────────────────────────────

/// Valued regional-maxima filter: keep the input value on regional maxima, set
/// non-maxima to `f32::MIN` (`−FLT_MAX`). ITK `ValuedRegionalMaximaImageFilter`.
#[derive(Debug, Clone)]
pub struct ValuedRegionalMaximaFilter {
    connectivity: Connectivity,
}

impl Default for ValuedRegionalMaximaFilter {
    fn default() -> Self {
        Self {
            connectivity: Connectivity::Face6,
        }
    }
}

impl ValuedRegionalMaximaFilter {
    /// Create a valued regional-maxima filter.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the structuring-element adjacency (defaults to [`Connectivity::Face6`]).
    pub fn with_connectivity(mut self, connectivity: Connectivity) -> Self {
        self.connectivity = connectivity;
        self
    }

    /// Apply the valued regional-maxima transform.
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> anyhow::Result<Image<B, 3>> {
        run(image, self.connectivity, ExtremaKind::Maxima, |m, v| {
            if m {
                v
            } else {
                f32::MIN
            }
        })
    }
}

/// Valued regional-minima filter: keep the input value on regional minima, set
/// non-minima to `f32::MAX` (`+FLT_MAX`). ITK `ValuedRegionalMinimaImageFilter`.
#[derive(Debug, Clone)]
pub struct ValuedRegionalMinimaFilter {
    connectivity: Connectivity,
}

impl Default for ValuedRegionalMinimaFilter {
    fn default() -> Self {
        Self {
            connectivity: Connectivity::Face6,
        }
    }
}

impl ValuedRegionalMinimaFilter {
    /// Create a valued regional-minima filter.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the structuring-element adjacency (defaults to [`Connectivity::Face6`]).
    pub fn with_connectivity(mut self, connectivity: Connectivity) -> Self {
        self.connectivity = connectivity;
        self
    }

    /// Apply the valued regional-minima transform.
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> anyhow::Result<Image<B, 3>> {
        run(image, self.connectivity, ExtremaKind::Minima, |m, v| {
            if m {
                v
            } else {
                f32::MAX
            }
        })
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "tests_regional_extrema.rs"]
mod tests_regional_extrema;
