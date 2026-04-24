//! Label dilation for 3-D label volumes.
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

use crate::image::Image;
use burn::tensor::backend::Backend;
use burn::tensor::{Shape, Tensor, TensorData};

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
    pub fn new(radius: usize) -> Self { Self { radius } }
    pub fn with_radius(mut self, r: usize) -> Self { self.radius = r; self }

    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> anyhow::Result<Image<B, 3>> {
        let dims = image.shape();
        let td = image.data().clone().into_data();
        let vals: &[f32] = td.as_slice::<f32>()
            .map_err(|e| anyhow::anyhow!("LabelDilation requires f32 data: {:?}", e))?;
        let result = dilate_labels(vals, dims, self.radius);
        let device = image.data().device();
        let t = Tensor::<B, 3>::from_data(TensorData::new(result, Shape::new(dims)), &device);
        Ok(Image::new(t, *image.origin(), *image.spacing(), *image.direction()))
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
                if data[idx] > 0.5 { continue; }
                let mut min_label: f32 = 0.0;
                for dz in -r..=r {
                    for dy in -r..=r {
                        for dx in -r..=r {
                            let (zz,yy,xx)=(iz as isize+dz,iy as isize+dy,ix as isize+dx);
                            if zz<0||zz>=nz as isize||yy<0||yy>=ny as isize||xx<0||xx>=nx as isize { continue; }
                            let v = data[zz as usize*ny*nx+yy as usize*nx+xx as usize];
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
    pub fn new(radius: usize) -> Self { Self { radius } }
    pub fn with_radius(mut self, r: usize) -> Self { self.radius = r; self }

    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> anyhow::Result<Image<B, 3>> {
        let dims = image.shape();
        let td = image.data().clone().into_data();
        let vals: &[f32] = td.as_slice::<f32>()
            .map_err(|e| anyhow::anyhow!("LabelErosion requires f32 data: {:?}", e))?;
        let result = erode_labels(vals, dims, self.radius);
        let device = image.data().device();
        let t = Tensor::<B, 3>::from_data(TensorData::new(result, Shape::new(dims)), &device);
        Ok(Image::new(t, *image.origin(), *image.spacing(), *image.direction()))
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
                if data[idx] <= 0.5 {
                    out[idx] = 0.0;
                    continue;
                }
                // Labeled voxel: erode if any neighbour is background
                let mut erode = false;
                'outer: for dz in -r..=r {
                    for dy in -r..=r {
                        for dx in -r..=r {
                            let zz = iz as isize + dz;
                            let yy = iy as isize + dy;
                            let xx = ix as isize + dx;
                            if zz < 0 || zz >= nz as isize
                                || yy < 0 || yy >= ny as isize
                                || xx < 0 || xx >= nx as isize
                            {
                                // Out-of-bounds treated as background
                                erode = true;
                                break 'outer;
                            }
                            let nidx = zz as usize * ny * nx
                                + yy as usize * nx
                                + xx as usize;
                            if data[nidx] <= 0.5 {
                                erode = true;
                                break 'outer;
                            }
                        }
                    }
                }
                if erode {
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
    pub fn new(radius: usize) -> Self { Self { radius } }

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
    pub fn new(radius: usize) -> Self { Self { radius } }

    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> anyhow::Result<Image<B, 3>> {
        let dilated = LabelDilation::new(self.radius).apply(image)?;
        LabelErosion::new(self.radius).apply(&dilated)
    }
}

// ════════════════════════════════════════════════════════════════════════════
// MorphologicalReconstruction
// ════════════════════════════════════════════════════════════════════════════

/// Reconstruction mode for geodesic morphological reconstruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReconstructionMode {
    /// Geodesic dilation: marker expands upward constrained by mask (M <= I).
    Dilation,
    /// Geodesic erosion: marker contracts downward constrained by mask (M >= I).
    Erosion,
}

/// Geodesic morphological reconstruction for grayscale f32 images.
///
/// # Mathematical Specification
///
/// **Dilation reconstruction** (Vincent 1993):
///   Given marker M and mask I with M <= I:
///   M* = lim_{k->inf} min(dilate_1(M_k), I)
///   where dilate_1 is one-step dilation with the unit-radius cubic B_1.
///
/// **Erosion reconstruction**:
///   Given marker M and mask I with M >= I:
///   M* = lim_{k->inf} max(erode_1(M_k), I)
///
/// Convergence criterion: max_x |M_{k+1}(x) - M_k(x)| < 1e-5, or max_iter reached.
///
/// # References
/// - Vincent, L. (1993). Morphological grayscale reconstruction in image analysis.
///   *IEEE Trans. Image Process.* 2(2):176-201.
#[derive(Debug, Clone)]
pub struct MorphologicalReconstruction {
    pub mode: ReconstructionMode,
    pub max_iter: usize,
}

impl MorphologicalReconstruction {
    pub fn new(mode: ReconstructionMode) -> Self {
        Self { mode, max_iter: 200 }
    }

    pub fn with_max_iter(mut self, n: usize) -> Self {
        self.max_iter = n;
        self
    }

    /// Apply geodesic reconstruction.
    ///
    /// # Arguments
    /// - `marker`: initial marker image; must have same shape as `mask`
    /// - `mask`: constraint mask image
    ///
    /// # Errors
    /// Returns `Err` when marker and mask shapes differ.
    pub fn apply<B: Backend>(
        &self,
        marker: &Image<B, 3>,
        mask: &Image<B, 3>,
    ) -> anyhow::Result<Image<B, 3>> {
        let marker_dims = marker.shape();
        let mask_dims = mask.shape();
        if marker_dims != mask_dims {
            anyhow::bail!(
                "MorphologicalReconstruction: marker shape {:?} != mask shape {:?}",
                marker_dims,
                mask_dims
            );
        }
        let dims = marker_dims;

        let md = marker.data().clone().into_data();
        let id_ = mask.data().clone().into_data();
        let marker_vals = md
            .as_slice::<f32>()
            .map_err(|e| anyhow::anyhow!("marker: {:?}", e))?;
        let mask_vals = id_
            .as_slice::<f32>()
            .map_err(|e| anyhow::anyhow!("mask: {:?}", e))?;

        // Clamp marker to enforce M <= I (dilation) or M >= I (erosion)
        let mut current: Vec<f32> = match self.mode {
            ReconstructionMode::Dilation => marker_vals
                .iter()
                .zip(mask_vals.iter())
                .map(|(&m, &i)| m.min(i))
                .collect(),
            ReconstructionMode::Erosion => marker_vals
                .iter()
                .zip(mask_vals.iter())
                .map(|(&m, &i)| m.max(i))
                .collect(),
        };
        let mask_vec: Vec<f32> = mask_vals.to_vec();

        for _ in 0..self.max_iter {
            let next = match self.mode {
                ReconstructionMode::Dilation => {
                    let dilated = dilate1_scalar(&current, dims);
                    dilated
                        .iter()
                        .zip(mask_vec.iter())
                        .map(|(&d, &m)| d.min(m))
                        .collect::<Vec<f32>>()
                }
                ReconstructionMode::Erosion => {
                    let eroded = erode1_scalar(&current, dims);
                    eroded
                        .iter()
                        .zip(mask_vec.iter())
                        .map(|(&e, &m)| e.max(m))
                        .collect::<Vec<f32>>()
                }
            };

            // Convergence check
            let max_delta = current
                .iter()
                .zip(next.iter())
                .map(|(&a, &b)| (a - b).abs())
                .fold(0.0f32, f32::max);

            current = next;

            if max_delta < 1e-5 {
                break;
            }
        }

        let device = marker.data().device();
        let t = Tensor::<B, 3>::from_data(
            TensorData::new(current, Shape::new(dims)),
            &device,
        );
        Ok(Image::new(t, *marker.origin(), *marker.spacing(), *marker.direction()))
    }
}

/// One-step grayscale dilation (max in 3x3x3 neighbourhood, clamp padding).
fn dilate1_scalar(data: &[f32], dims: [usize; 3]) -> Vec<f32> {
    let [nz, ny, nx] = dims;
    let mut out = Vec::with_capacity(data.len());
    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let mut mx = f32::NEG_INFINITY;
                for dz in -1i32..=1 {
                    for dy in -1i32..=1 {
                        for dx in -1i32..=1 {
                            let zz = (iz as i32 + dz).clamp(0, nz as i32 - 1) as usize;
                            let yy = (iy as i32 + dy).clamp(0, ny as i32 - 1) as usize;
                            let xx = (ix as i32 + dx).clamp(0, nx as i32 - 1) as usize;
                            let v = data[zz * ny * nx + yy * nx + xx];
                            if v > mx { mx = v; }
                        }
                    }
                }
                out.push(mx);
            }
        }
    }
    out
}

/// One-step grayscale erosion (min in 3x3x3 neighbourhood).
///
/// Out-of-bounds positions contribute f32::NEG_INFINITY so that boundary
/// voxels erode toward the mask value during geodesic erosion reconstruction.
/// This is mathematically required: on a finite-support domain the exterior
/// acts as a strict lower bound, enabling convergence from marker to mask.
fn erode1_scalar(data: &[f32], dims: [usize; 3]) -> Vec<f32> {
    let [nz, ny, nx] = dims;
    let mut out = Vec::with_capacity(data.len());
    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let mut mn = f32::INFINITY;
                'outer_e: for dz in -1i32..=1 {
                    for dy in -1i32..=1 {
                        for dx in -1i32..=1 {
                            let zz = iz as i32 + dz;
                            let yy = iy as i32 + dy;
                            let xx = ix as i32 + dx;
                            if zz < 0 || zz >= nz as i32
                                || yy < 0 || yy >= ny as i32
                                || xx < 0 || xx >= nx as i32
                            {
                                mn = f32::NEG_INFINITY;
                                break 'outer_e;
                            }
                            let v = data[zz as usize * ny * nx
                                + yy as usize * nx
                                + xx as usize];
                            if v < mn { mn = v; }
                        }
                    }
                }
                out.push(mn);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::Image;
    use crate::spatial::{Direction, Point, Spacing};
    use burn::tensor::{Shape, Tensor, TensorData};
    use burn_ndarray::NdArray;
    type B = NdArray<f32>;
    fn img(v: Vec<f32>, d: [usize;3]) -> Image<B,3> {
        let t=Tensor::<B,3>::from_data(TensorData::new(v,Shape::new(d)),&Default::default());
        Image::new(t,Point::new([0.0,0.0,0.0]),Spacing::new([1.0,1.0,1.0]),Direction::identity())
    }
    fn vv(i: &Image<B,3>) -> Vec<f32> { i.data().clone().into_data().into_vec::<f32>().unwrap() }

    #[test] fn test_all_background_unchanged() {
        let d=[6,6,6]; let n=d[0]*d[1]*d[2];
        let v=vec![0.0_f32;n];
        let out=vv(&LabelDilation::new(1).apply(&img(v.clone(),d)).unwrap());
        for &x in &out { assert!(x.abs()<1e-6,"all-bg: {x}"); }
    }
    #[test] fn test_label_expands_into_background() {
        let d=[7,7,7]; let [nz,ny,nx]=d; let n=nz*ny*nx;
        let mut v=vec![0.0_f32;n]; let c=3*ny*nx+3*nx+3; v[c]=1.0;
        let out=vv(&LabelDilation::new(1).apply(&img(v,d)).unwrap());
        assert!((out[c]-1.0).abs()<1e-6,"centre preserved");
        let neighbour=3*ny*nx+3*nx+4;
        assert!((out[neighbour]-1.0).abs()<1e-6,"neighbour expanded to label 1");
    }
    #[test] fn test_conflict_min_label_wins() {
        let d=[5,5,5]; let [_,ny,nx]=d; let n=d[0]*d[1]*d[2];
        let mut v=vec![0.0_f32;n];
        let a=2*ny*nx+1*nx+1; v[a]=1.0;
        let b=2*ny*nx+1*nx+3; v[b]=2.0;
        let out=vv(&LabelDilation::new(1).apply(&img(v,d)).unwrap());
        let middle=2*ny*nx+1*nx+2;
        assert!((out[middle]-1.0).abs()<1e-6,"conflict: min label wins (got {})",out[middle]);
    }
    #[test] fn test_radius_zero_identity() {
        let d=[6,6,6]; let [_,ny,nx]=d; let n=d[0]*d[1]*d[2];
        let mut v=vec![0.0_f32;n]; v[2*ny*nx+2*nx+2]=1.0; v[3*ny*nx+3*nx+3]=2.0;
        let out=vv(&LabelDilation::new(0).apply(&img(v.clone(),d)).unwrap());
        for (i,(&e,&a)) in v.iter().zip(out.iter()).enumerate() {
            assert!((a-e).abs()<1e-6,"r=0 identity voxel {i}: {a} != {e}");
        }
    }
    #[test] fn test_metadata_preserved() {
        let d=[5,5,5]; let n=d[0]*d[1]*d[2];
        let t=Tensor::<B,3>::from_data(TensorData::new(vec![0.0_f32;n],Shape::new(d)),&Default::default());
        let o=Point::new([1.0,2.0,3.0]); let s=Spacing::new([0.5,0.5,0.5]);
        let r=LabelDilation::new(1).apply(&Image::new(t,o,s,Direction::identity())).unwrap();
        assert_eq!(*r.origin(),o); assert_eq!(*r.spacing(),s);
    }

    // ── LabelErosion tests ─────────────────────────────────────────────────

    #[test]
    fn test_label_erosion_all_background_unchanged() {
        let d = [5, 5, 5];
        let n = d[0] * d[1] * d[2];
        let out = vv(&LabelErosion::new(1).apply(&img(vec![0.0_f32; n], d)).unwrap());
        for &x in &out { assert!(x.abs() < 1e-6, "all-bg erosion: {x}"); }
    }

    #[test]
    fn test_label_erosion_single_voxel_erodes_to_zero() {
        let d = [5, 5, 5];
        let [_, ny, nx] = d;
        let n = d[0] * d[1] * d[2];
        let mut v = vec![0.0_f32; n];
        let c = 2 * ny * nx + 2 * nx + 2;
        v[c] = 1.0;
        let out = vv(&LabelErosion::new(1).apply(&img(v, d)).unwrap());
        assert!(out[c].abs() < 1e-6, "single voxel should erode to 0, got {}", out[c]);
    }

    #[test]
    fn test_label_erosion_radius_zero_is_identity() {
        let d = [7, 7, 7];
        let [_, ny, nx] = d;
        let n = d[0] * d[1] * d[2];
        let mut v = vec![0.0_f32; n];
        v[3 * ny * nx + 3 * nx + 3] = 1.0;
        let out = vv(&LabelErosion::new(0).apply(&img(v.clone(), d)).unwrap());
        for (i, (&e, &a)) in v.iter().zip(out.iter()).enumerate() {
            assert!((a - e).abs() < 1e-6, "r=0 identity voxel {i}: {a} != {e}");
        }
    }

    #[test]
    fn test_label_erosion_interior_preserved() {
        let d = [9, 9, 9];
        let [_, ny, nx] = d;
        let n = d[0] * d[1] * d[2];
        let mut v = vec![0.0_f32; n];
        for iz in 1..8 {
            for iy in 1..8 {
                for ix in 1..8 {
                    v[iz * ny * nx + iy * nx + ix] = 1.0;
                }
            }
        }
        let out = vv(&LabelErosion::new(1).apply(&img(v, d)).unwrap());
        let centre = 4 * ny * nx + 4 * nx + 4;
        assert!((out[centre] - 1.0).abs() < 1e-6, "interior preserved, got {}", out[centre]);
    }

    #[test]
    fn test_label_erosion_metadata_preserved() {
        let d = [5, 5, 5];
        let n = d[0] * d[1] * d[2];
        let t = Tensor::<B, 3>::from_data(TensorData::new(vec![0.0_f32; n], Shape::new(d)), &Default::default());
        let o = Point::new([1.0, 2.0, 3.0]);
        let s = Spacing::new([0.5, 0.5, 0.5]);
        let r = LabelErosion::new(1).apply(&Image::new(t, o, s, Direction::identity())).unwrap();
        assert_eq!(*r.origin(), o);
        assert_eq!(*r.spacing(), s);
    }

    // ── LabelOpening tests ─────────────────────────────────────────────────

    #[test]
    fn test_label_opening_removes_isolated_voxel() {
        let d = [7, 7, 7];
        let [_, ny, nx] = d;
        let n = d[0] * d[1] * d[2];
        let mut v = vec![0.0_f32; n];
        for iz in 2..6 { for iy in 2..6 { for ix in 2..6 {
            v[iz * ny * nx + iy * nx + ix] = 1.0;
        }}}
        v[0] = 1.0;
        let out = vv(&LabelOpening::new(1).apply(&img(v, d)).unwrap());
        assert!(out[0].abs() < 1e-6, "isolated voxel removed by opening, got {}", out[0]);
    }

    #[test]
    fn test_label_opening_empty_is_identity() {
        let d = [5, 5, 5];
        let n = d[0] * d[1] * d[2];
        let out = vv(&LabelOpening::new(1).apply(&img(vec![0.0_f32; n], d)).unwrap());
        for &x in &out { assert!(x.abs() < 1e-6, "empty opening: {x}"); }
    }

    #[test]
    fn test_label_opening_metadata_preserved() {
        let d = [5, 5, 5];
        let n = d[0] * d[1] * d[2];
        let t = Tensor::<B, 3>::from_data(TensorData::new(vec![0.0_f32; n], Shape::new(d)), &Default::default());
        let o = Point::new([1.0, 0.0, 0.0]);
        let s = Spacing::new([2.0, 2.0, 2.0]);
        let r = LabelOpening::new(1).apply(&Image::new(t, o, s, Direction::identity())).unwrap();
        assert_eq!(*r.origin(), o);
        assert_eq!(*r.spacing(), s);
    }

    // ── LabelClosing tests ─────────────────────────────────────────────────

    #[test]
    fn test_label_closing_fills_background_hole() {
        let d = [7, 7, 7];
        let [_, ny, nx] = d;
        let n = d[0] * d[1] * d[2];
        let mut v = vec![1.0_f32; n];
        let hole = 3 * ny * nx + 3 * nx + 3;
        v[hole] = 0.0;
        let out = vv(&LabelClosing::new(1).apply(&img(v, d)).unwrap());
        assert!((out[hole] - 1.0).abs() < 1e-6, "hole should be filled, got {}", out[hole]);
    }

    #[test]
    fn test_label_closing_empty_is_identity() {
        let d = [5, 5, 5];
        let n = d[0] * d[1] * d[2];
        let out = vv(&LabelClosing::new(1).apply(&img(vec![0.0_f32; n], d)).unwrap());
        for &x in &out { assert!(x.abs() < 1e-6, "empty closing: {x}"); }
    }

    #[test]
    fn test_label_closing_metadata_preserved() {
        let d = [5, 5, 5];
        let n = d[0] * d[1] * d[2];
        let t = Tensor::<B, 3>::from_data(TensorData::new(vec![0.0_f32; n], Shape::new(d)), &Default::default());
        let o = Point::new([0.0, 1.0, 2.0]);
        let s = Spacing::new([1.5, 1.5, 1.5]);
        let r = LabelClosing::new(1).apply(&Image::new(t, o, s, Direction::identity())).unwrap();
        assert_eq!(*r.origin(), o);
        assert_eq!(*r.spacing(), s);
    }

    // ── MorphologicalReconstruction tests ─────────────────────────────────

    #[test]
    fn test_recon_dilation_expands_to_mask() {
        let d = [5, 5, 5];
        let [_, ny, nx] = d;
        let n = d[0] * d[1] * d[2];
        let mask = img(vec![1.0_f32; n], d);
        let mut mv = vec![0.0_f32; n];
        mv[2 * ny * nx + 2 * nx + 2] = 1.0;
        let marker = img(mv, d);
        let out = vv(
            &MorphologicalReconstruction::new(ReconstructionMode::Dilation)
                .with_max_iter(500)
                .apply(&marker, &mask)
                .unwrap(),
        );
        let mean: f32 = out.iter().sum::<f32>() / n as f32;
        assert!((mean - 1.0).abs() < 1e-3, "dilation should fill mask, mean={mean}");
    }

    #[test]
    fn test_recon_erosion_contracts_to_mask() {
        let d = [5, 5, 5];
        let n = d[0] * d[1] * d[2];
        let mask = img(vec![0.0_f32; n], d);
        let marker = img(vec![1.0_f32; n], d);
        let out = vv(
            &MorphologicalReconstruction::new(ReconstructionMode::Erosion)
                .apply(&marker, &mask)
                .unwrap(),
        );
        let max_val = out.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        assert!(max_val < 1e-4, "erosion should contract to 0, max={max_val}");
    }

    #[test]
    fn test_recon_marker_equals_mask_converges_immediately() {
        let d = [4, 4, 4];
        let n = d[0] * d[1] * d[2];
        let vals: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
        let marker = img(vals.clone(), d);
        let mask = img(vals.clone(), d);
        let out = vv(
            &MorphologicalReconstruction::new(ReconstructionMode::Dilation)
                .apply(&marker, &mask)
                .unwrap(),
        );
        for (i, (&e, &a)) in vals.iter().zip(out.iter()).enumerate() {
            assert!((a - e).abs() < 1e-4, "identity voxel {i}: {a} != {e}");
        }
    }

    #[test]
    fn test_recon_shape_mismatch_returns_error() {
        let marker = img(vec![0.0_f32; 27], [3, 3, 3]);
        let mask = img(vec![1.0_f32; 64], [4, 4, 4]);
        let result = MorphologicalReconstruction::new(ReconstructionMode::Dilation)
            .apply(&marker, &mask);
        assert!(result.is_err(), "shape mismatch must return Err");
    }

    #[test]
    fn test_recon_metadata_preserved() {
        let d = [3, 3, 3];
        let n = d[0] * d[1] * d[2];
        let o = Point::new([1.0, 2.0, 3.0]);
        let s = Spacing::new([0.5, 0.5, 0.5]);
        let t = Tensor::<B, 3>::from_data(TensorData::new(vec![0.5_f32; n], Shape::new(d)), &Default::default());
        let marker = Image::new(t.clone(), o, s, Direction::identity());
        let mask = Image::new(t, o, s, Direction::identity());
        let out = MorphologicalReconstruction::new(ReconstructionMode::Dilation)
            .apply(&marker, &mask)
            .unwrap();
        assert_eq!(*out.origin(), o);
        assert_eq!(*out.spacing(), s);
    }

}
