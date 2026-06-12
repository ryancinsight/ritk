//! Label dilation, erosion, opening, and closing for 3-D label volumes.
//!
//! # Mathematical Specification
//!
//! Given a label volume L: [nz x ny x nx] -> N (0=background, k>0 label k),
//! label dilation with radius r expands each labeled region outward:
//!
//! For each background voxel x:
//!   N_r(x) = { y : max|y_i - x_i| <= r }  (cubic neighbourhood)
//!   labels(x) = { L(y) : y in N_r(x), L(y) > 0 }
//!   L_out(x) = min(labels(x))  if labels(x) non-empty, else 0
//!
//! Existing labelled voxels are preserved: L_out(x) = L(x) for L(x) > 0.
//! Conflict resolution: where two label regions meet, the minimum label ID wins.
//!
//! # References
//! - Serra, J. (1982). Image Analysis and Mathematical Morphology. Academic Press.

use burn::tensor::backend::Backend;
use burn::tensor::{Shape, Tensor, TensorData};
use ritk_image::Image;
use ritk_tensor_ops::extract_vec;

// ════════════════════════════════════════════════════════════════════════════
// LabelDilation
// ════════════════════════════════════════════════════════════════════════════

/// Label dilation for 3-D label volumes.
///
/// Expands each labeled region by radius voxels. For conflicting dilations
/// (two label regions growing into the same background voxel), the label
/// with the minimum ID takes priority.
#[derive(Debug, Clone)]
pub struct LabelDilation {
    pub radius: usize,
}

impl LabelDilation {
    pub fn new(radius: usize) -> Self {
        Self { radius }
    }
    pub fn with_radius(mut self, r: usize) -> Self {
        self.radius = r;
        self
    }

    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> anyhow::Result<Image<B, 3>> {
        let (vals, dims) = extract_vec(image)?;
        let result = dilate_labels(&vals, dims, self.radius);
        let device = image.data().device();
        let t = Tensor::<B, 3>::from_data(TensorData::new(result, Shape::new(dims)), &device);
        Ok(Image::new(
            t,
            *image.origin(),
            *image.spacing(),
            *image.direction(),
        ))
    }
}

fn dilate_labels(data: &[f32], dims: [usize; 3], radius: usize) -> Vec<f32> {
    let [nz, ny, nx] = dims;
    let r = radius as isize;
    let _n = nz * ny * nx;
    let mut out = data.to_vec();
    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let idx = iz * ny * nx + iy * nx + ix;
                // Only expand into background voxels
                if data[idx] > 0.5 {
                    continue;
                }
                let mut min_label: f32 = 0.0;
                for dz in -r..=r {
                    for dy in -r..=r {
                        for dx in -r..=r {
                            let (zz, yy, xx) =
                                (iz as isize + dz, iy as isize + dy, ix as isize + dx);
                            if zz < 0
                                || zz >= nz as isize
                                || yy < 0
                                || yy >= ny as isize
                                || xx < 0
                                || xx >= nx as isize
                            {
                                continue;
                            }
                            let v = data[zz as usize * ny * nx + yy as usize * nx + xx as usize];
                            if v > 0.5 && (min_label < 0.5 || v < min_label) {
                                min_label = v;
                            }
                        }
                    }
                }
                out[idx] = min_label;
            }
        }
    }
    out
}

// ════════════════════════════════════════════════════════════════════════════
// LabelErosion
// ════════════════════════════════════════════════════════════════════════════

/// Label erosion for 3-D label volumes.
///
/// # Mathematical Specification
///
/// For each voxel x in a label volume L: [nz x ny x nx] -> N (0=background):
/// - If L(x) = 0: L_out(x) = 0  (background remains background)
/// - If L(x) > 0: examine N_r(x) = { y : max_i|y_i - x_i| <= r }
///   - If any y in N_r(x) has L(y) = 0 -> L_out(x) = 0  (border voxel eroded)
///   - Else L_out(x) = L(x)  (interior voxel preserved)
///
/// # Properties
/// - Anti-extensivity: L_out(x) <= L(x) for all x.
/// - r=0 identity: N_0(x) = {x}, so only checks itself; no change.
/// - Dual to LabelDilation under label-complement.
///
/// # References
/// - Serra, J. (1982). *Image Analysis and Mathematical Morphology*. Academic Press.
#[derive(Debug, Clone)]
pub struct LabelErosion {
    pub radius: usize,
}

impl LabelErosion {
    pub fn new(radius: usize) -> Self {
        Self { radius }
    }
    pub fn with_radius(mut self, r: usize) -> Self {
        self.radius = r;
        self
    }

    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> anyhow::Result<Image<B, 3>> {
        let (vals, dims) = extract_vec(image)?;
        let result = erode_labels(&vals, dims, self.radius);
        let device = image.data().device();
        let t = Tensor::<B, 3>::from_data(TensorData::new(result, Shape::new(dims)), &device);
        Ok(Image::new(
            t,
            *image.origin(),
            *image.spacing(),
            *image.direction(),
        ))
    }
}

fn erode_labels(data: &[f32], dims: [usize; 3], radius: usize) -> Vec<f32> {
    let [nz, ny, nx] = dims;
    let r = radius as isize;
    let mut out = data.to_vec();
    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let idx = iz * ny * nx + iy * nx + ix;
                if data[idx] < 0.5 {
                    continue; // background stays background
                }
                let mut eroded = false;
                'outer: for dz in -r..=r {
                    for dy in -r..=r {
                        for dx in -r..=r {
                            let (zz, yy, xx) =
                                (iz as isize + dz, iy as isize + dy, ix as isize + dx);
                            if zz < 0
                                || zz >= nz as isize
                                || yy < 0
                                || yy >= ny as isize
                                || xx < 0
                                || xx >= nx as isize
                            {
                                eroded = true;
                                break 'outer;
                            }
                            let v = data[zz as usize * ny * nx + yy as usize * nx + xx as usize];
                            if v < 0.5 {
                                eroded = true;
                                break 'outer;
                            }
                        }
                    }
                }
                if eroded {
                    out[idx] = 0.0;
                }
            }
        }
    }
    out
}

// ════════════════════════════════════════════════════════════════════════════
// LabelOpening
// ════════════════════════════════════════════════════════════════════════════

/// Morphological opening on label volumes: LabelDilation composed with LabelErosion.
///
/// Removes labeled regions smaller than radius r, smooths region boundaries.
/// Opening = Dilation(Erosion(L)).
#[derive(Debug, Clone)]
pub struct LabelOpening {
    pub radius: usize,
}

impl LabelOpening {
    pub fn new(radius: usize) -> Self {
        Self { radius }
    }

    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> anyhow::Result<Image<B, 3>> {
        let eroded = LabelErosion::new(self.radius).apply(image)?;
        LabelDilation::new(self.radius).apply(&eroded)
    }
}

// ════════════════════════════════════════════════════════════════════════════
// LabelClosing
// ════════════════════════════════════════════════════════════════════════════

/// Morphological closing on label volumes: LabelErosion composed with LabelDilation.
///
/// Fills background holes smaller than radius r within labeled regions.
/// Closing = Erosion(Dilation(L)).
#[derive(Debug, Clone)]
pub struct LabelClosing {
    pub radius: usize,
}

impl LabelClosing {
    pub fn new(radius: usize) -> Self {
        Self { radius }
    }

    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> anyhow::Result<Image<B, 3>> {
        let dilated = LabelDilation::new(self.radius).apply(image)?;
        LabelErosion::new(self.radius).apply(&dilated)
    }
}
