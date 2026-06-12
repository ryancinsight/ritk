//! Label contour filter — boundaries between labelled regions.
//!
//! # Mathematical Specification
//!
//! Given a label image `L : ℤ³ → ℕ₀`, a voxel `p` is a **label contour voxel** if:
//! - `L(p) ≠ background_label`, AND
//! - at least one neighbour `q ∈ N(p)` satisfies `L(q) ≠ L(p)`.
//!
//! The connectivity topology `N(p)` is determined by [`Connectivity`]:
//! - [`Connectivity::Face6`] (default): 6-connected face neighbours (ITK default).
//! - [`Connectivity::Vertex26`]: 26-connected neighbours.
//!
//! # ITK Parity
//!
//! Corresponds to `itk::LabelContourImageFilter<TInputImage, TOutputImage>`.
//! ITK defaults: `FullyConnected = false`, `BackgroundValue = 0`.
//!
//! Output: a contour voxel retains its original label value;
//! non-contour and background voxels are set to `background_value` (default 0).
//!
//! # Reference
//!
//! - Malandain, G. & Bertrand, G. (1992). Fast characterization of 3D simple points.

use super::Connectivity;
use burn::tensor::backend::Backend;
use burn::tensor::{Shape, Tensor, TensorData};
use ritk_image::Image;
use ritk_tensor_ops::extract_vec;

/// Label contour filter.
///
/// Marks voxels that lie on the boundary between distinct labelled regions.
#[derive(Debug, Clone)]
pub struct LabelContourImageFilter {
    /// Neighbourhood connectivity topology (ITK default: `Face6`).
    pub connectivity: Connectivity,
    /// Label value used for background. Default 0.
    pub background_value: f32,
}

impl LabelContourImageFilter {
    /// Construct with explicit parameters.
    pub fn new(connectivity: Connectivity, background_value: f32) -> Self {
        Self {
            connectivity,
            background_value,
        }
    }

    /// Set connectivity.
    pub fn with_connectivity(mut self, connectivity: Connectivity) -> Self {
        self.connectivity = connectivity;
        self
    }

    /// Backward-compatible accessor for the former `fully_connected: bool` field.
    pub fn fully_connected(&self) -> bool {
        self.connectivity.fully_connected()
    }
}

impl Default for LabelContourImageFilter {
    fn default() -> Self {
        Self::new(Connectivity::Face6, 0.0)
    }
}

/// 6-connected face neighbours.
const N6: [(i32, i32, i32); 6] = [
    (-1, 0, 0),
    (1, 0, 0),
    (0, -1, 0),
    (0, 1, 0),
    (0, 0, -1),
    (0, 0, 1),
];

fn n26() -> Vec<(i32, i32, i32)> {
    let mut v = Vec::with_capacity(26);
    for dz in -1i32..=1 {
        for dy in -1i32..=1 {
            for dx in -1i32..=1 {
                if dz != 0 || dy != 0 || dx != 0 {
                    v.push((dz, dy, dx));
                }
            }
        }
    }
    v
}

impl LabelContourImageFilter {
    /// Apply the label contour filter to a 3-D image.
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> anyhow::Result<Image<B, 3>> {
        let (vals, dims) = extract_vec(image)?;
        let [nz, ny, nx] = dims;
        let device = image.data().device();

        let bg = self.background_value;
        let n26 = n26();

        let mut out = vec![bg; nz * ny * nx];

        for iz in 0..nz {
            for iy in 0..ny {
                for ix in 0..nx {
                    let label = vals[iz * ny * nx + iy * nx + ix];
                    if (label - bg).abs() < 1e-5 {
                        continue; // background stays background
                    }
                    let is_contour = match self.connectivity {
                        Connectivity::Vertex26 => n26.iter().any(|&(dz, dy, dx)| {
                            let qz = iz as i32 + dz;
                            let qy = iy as i32 + dy;
                            let qx = ix as i32 + dx;
                            if qz < 0
                                || qy < 0
                                || qx < 0
                                || qz >= nz as i32
                                || qy >= ny as i32
                                || qx >= nx as i32
                            {
                                return true;
                            }
                            let nl = vals[qz as usize * ny * nx + qy as usize * nx + qx as usize];
                            (nl - label).abs() > 1e-5
                        }),
                        Connectivity::Face6 => N6.iter().any(|&(dz, dy, dx)| {
                            let qz = iz as i32 + dz;
                            let qy = iy as i32 + dy;
                            let qx = ix as i32 + dx;
                            if qz < 0
                                || qy < 0
                                || qx < 0
                                || qz >= nz as i32
                                || qy >= ny as i32
                                || qx >= nx as i32
                            {
                                return true;
                            }
                            let nl = vals[qz as usize * ny * nx + qy as usize * nx + qx as usize];
                            (nl - label).abs() > 1e-5
                        }),
                    };
                    if is_contour {
                        out[iz * ny * nx + iy * nx + ix] = label;
                    }
                }
            }
        }

        let shape = Shape::new([nz, ny, nx]);
        let data = TensorData::new(out, shape);
        let tensor = Tensor::<B, 3>::from_data(data, &device);
        Ok(Image::new(
            tensor,
            *image.origin(),
            *image.spacing(),
            *image.direction(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::TensorData;
    use burn_ndarray::NdArray;
    use ritk_spatial::{Direction, Point, Spacing};

    type B = NdArray<f32>;

    fn make_image(data: Vec<f32>, shape: [usize; 3]) -> Image<B, 3> {
        let device = Default::default();
        let td = TensorData::new(data, Shape::new(shape));
        let tensor = Tensor::<B, 3>::from_data(td, &device);
        Image::new(
            tensor,
            Point::new([0.0_f64, 0.0, 0.0]),
            Spacing::new([1.0_f64, 1.0, 1.0]),
            Direction::identity(),
        )
    }

    fn voxels(img: &Image<B, 3>) -> Vec<f32> {
        img.data_slice().into_owned()
    }

    /// All-background image → all background in output.
    #[test]
    fn all_background_zero() {
        let img = make_image(vec![0.0f32; 27], [3, 3, 3]);
        let out = LabelContourImageFilter::default().apply(&img).unwrap();
        assert!(voxels(&out).iter().all(|&v| v == 0.0));
    }

    /// Single label region filling entire image: outer shell is border; center is interior.
    #[test]
    fn single_label_all_border() {
        let img = make_image(vec![2.0f32; 27], [3, 3, 3]);
        let out = LabelContourImageFilter::default().apply(&img).unwrap();
        let v = voxels(&out);
        // Center (1,1,1) = index 13: all 6 neighbors in-bounds and same label → interior → bg.
        assert_eq!(v[13], 0.0, "center of 3×3×3 single-label is interior");
        // All other 26 voxels border out-of-bounds (treated as different label) → contour.
        assert!(
            v.iter()
                .enumerate()
                .filter(|&(i, _)| i != 13)
                .all(|(_, &x)| (x - 2.0).abs() < 1e-5),
            "outer shell of 3×3×3 single-label must be contour"
        );
    }

    /// Two labels side-by-side (left half = label 1, right half = label 2).
    /// Interface voxels should be marked; interior voxels (no different-label neighbour) → 0.
    #[test]
    fn two_labels_interface_marked() {
        // 1×1×6: [1,1,1,2,2,2]
        let img = make_image(vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0], [1, 1, 6]);
        let out = LabelContourImageFilter::default().apply(&img).unwrap();
        let v = voxels(&out);
        // voxel 2 (label 1, right edge of label-1): neighbour voxel 3 = label 2 → contour
        assert!(
            (v[2] - 1.0).abs() < 1e-5,
            "v[2] should be 1 (contour), got {}",
            v[2]
        );
        // voxel 3 (label 2, left edge of label-2): neighbour voxel 2 = label 1 → contour
        assert!(
            (v[3] - 2.0).abs() < 1e-5,
            "v[3] should be 2 (contour), got {}",
            v[3]
        );
        // voxels 0 and 1 are at left image boundary → also border
        // voxels 4 and 5 are at right image boundary → also border
        // voxel 1 borders voxel 0 (same label) and voxel 2 (same label) but image left boundary treated as bg → border
    }

    /// Labels are preserved in contour voxels (label value, not a binary mask).
    #[test]
    fn contour_voxels_preserve_label_value() {
        // 1×1×4: [0, 5, 5, 0]
        let img = make_image(vec![0.0, 5.0, 5.0, 0.0], [1, 1, 4]);
        let out = LabelContourImageFilter::default().apply(&img).unwrap();
        let v = voxels(&out);
        // voxel 1 (label 5): neighbour 0 is bg → contour → value = 5
        assert!((v[1] - 5.0).abs() < 1e-5, "v[1]={}", v[1]);
        // voxel 2 (label 5): neighbour 3 is bg → contour → value = 5
        assert!((v[2] - 5.0).abs() < 1e-5, "v[2]={}", v[2]);
    }

    /// Spatial metadata preserved.
    #[test]
    fn preserves_metadata() {
        let img = make_image(vec![0.0f32; 8], [2, 2, 2]);
        let out = LabelContourImageFilter::default().apply(&img).unwrap();
        assert_eq!(out.shape(), [2, 2, 2]);
    }
}
