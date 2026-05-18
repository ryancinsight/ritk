//! Binary contour filter — foreground border voxels of binary objects.
//!
//! # Mathematical Specification
//!
//! Given a binary image `I ∈ {0, fg}`, a voxel `p` is a **border voxel** if:
//! - `I(p) = fg` (is foreground), AND
//! - at least one neighbour `q ∈ N(p)` satisfies `I(q) ≠ fg`.
//!
//! The connectivity topology `N(p)` is determined by `fully_connected`:
//! - `false` (default): 6-connected — 6 face-neighbours in ℤ³ (ITK default).
//! - `true`: 26-connected — all 26 neighbours within the unit cube.
//!
//! # ITK Parity
//!
//! Corresponds to `itk::BinaryContourImageFilter<TInputImage, TOutputImage>`.
//! ITK defaults: `FullyConnected = false`, `ForegroundValue = 1`, `BackgroundValue = 0`.
//!
//! Output: foreground border voxels set to `foreground_value`; all others 0.
//!
//! # Reference
//!
//! - Malandain, G. & Bertrand, G. (1992). Fast characterization of 3D simple points.
//!   *ICPR 1992*.

use crate::filter::ops::extract_vec;
use crate::image::Image;
use burn::tensor::backend::Backend;
use burn::tensor::{Shape, Tensor, TensorData};

/// Binary contour filter.
///
/// Marks only the border (surface) voxels of foreground objects.
/// Interior foreground voxels (fully surrounded by foreground) are set to 0.
#[derive(Debug, Clone)]
pub struct BinaryContourImageFilter {
    /// Whether to use 26-connectivity (`true`) or 6-connectivity (`false`, ITK default).
    pub fully_connected: bool,
    /// Foreground intensity value. Default 1.0.
    pub foreground_value: f32,
}

impl BinaryContourImageFilter {
    /// Construct with explicit parameters.
    pub fn new(fully_connected: bool, foreground_value: f32) -> Self {
        Self {
            fully_connected,
            foreground_value,
        }
    }
}

impl Default for BinaryContourImageFilter {
    fn default() -> Self {
        Self::new(false, 1.0)
    }
}

/// 6-connected face neighbours (±z, ±y, ±x).
const N6: [(i32, i32, i32); 6] = [
    (-1, 0, 0),
    (1, 0, 0),
    (0, -1, 0),
    (0, 1, 0),
    (0, 0, -1),
    (0, 0, 1),
];

/// 26-connected neighbours (all offsets in {-1,0,1}³ except (0,0,0)).
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

impl BinaryContourImageFilter {
    /// Apply the binary contour filter to a 3-D image.
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> anyhow::Result<Image<B, 3>> {
        let (vals, dims) = extract_vec(image)?;
        let [nz, ny, nx] = dims;
        let device = image.data().device();

        let fg = self.foreground_value;
        let fully = self.fully_connected;
        let n26 = n26();

        let mut out = vec![0.0f32; nz * ny * nx];

        for iz in 0..nz {
            for iy in 0..ny {
                for ix in 0..nx {
                    let v = vals[iz * ny * nx + iy * nx + ix];
                    if (v - fg).abs() > 1e-5 {
                        continue; // background
                    }
                    let is_border = if fully {
                        n26.iter().any(|&(dz, dy, dx)| {
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
                            let nv = vals[qz as usize * ny * nx + qy as usize * nx + qx as usize];
                            (nv - fg).abs() > 1e-5
                        })
                    } else {
                        N6.iter().any(|&(dz, dy, dx)| {
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
                            let nv = vals[qz as usize * ny * nx + qy as usize * nx + qx as usize];
                            (nv - fg).abs() > 1e-5
                        })
                    };
                    if is_border {
                        out[iz * ny * nx + iy * nx + ix] = fg;
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
    use crate::spatial::{Direction, Point, Spacing};
    use burn::tensor::TensorData;
    use burn_ndarray::NdArray;

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
        img.data_vec()
    }

    /// All-background → all-zero output.
    #[test]
    fn all_background_zero() {
        let img = make_image(vec![0.0f32; 27], [3, 3, 3]);
        let out = BinaryContourImageFilter::default().apply(&img).unwrap();
        assert!(voxels(&out).iter().all(|&v| v == 0.0));
    }

    /// All-foreground solid block: only the outer shell is a border.
    /// 3×3×3 all-fg: corner/edge/face voxels are borders; center (1,1,1) is interior (6-conn).
    #[test]
    fn solid_block_all_border() {
        let img = make_image(vec![1.0f32; 27], [3, 3, 3]);
        let out = BinaryContourImageFilter::default().apply(&img).unwrap();
        let v = voxels(&out);
        // Center voxel (1,1,1) = index 1*9+1*3+1 = 13: has all 6 face-neighbors in-bounds and fg → interior.
        assert_eq!(v[13], 0.0, "center of 3×3×3 is interior in 6-conn");
        // All other 26 voxels are on the outer shell and neighbor out-of-bounds → borders.
        assert!(
            v.iter()
                .enumerate()
                .filter(|&(i, _)| i != 13)
                .all(|(_, &x)| (x - 1.0).abs() < 1e-5),
            "all non-center voxels of 3×3×3 must be borders"
        );
    }

    /// 5×5×5 block: all-fg. Center voxel (2,2,2) is interior → 0 (6-connected).
    #[test]
    fn five_cube_center_is_interior_6conn() {
        let img = make_image(vec![1.0f32; 125], [5, 5, 5]);
        let out = BinaryContourImageFilter::new(false, 1.0)
            .apply(&img)
            .unwrap();
        let v = voxels(&out);
        // Center voxel index = 2*25+2*5+2 = 62
        assert_eq!(v[62], 0.0, "center of 5×5×5 should be interior (6-conn)");
    }

    /// Single foreground voxel in a background image is a border voxel.
    #[test]
    fn single_fg_voxel_is_border() {
        let mut data = vec![0.0f32; 27];
        data[13] = 1.0; // center of 3×3×3
        let img = make_image(data, [3, 3, 3]);
        let out = BinaryContourImageFilter::default().apply(&img).unwrap();
        let v = voxels(&out);
        assert!((v[13] - 1.0).abs() < 1e-5, "single fg voxel must be border");
        let others: f32 = v
            .iter()
            .enumerate()
            .filter(|&(i, _)| i != 13)
            .map(|(_, &x)| x)
            .sum();
        assert_eq!(others, 0.0);
    }

    /// Spatial metadata preserved.
    #[test]
    fn preserves_metadata() {
        let img = make_image(vec![0.0f32; 8], [2, 2, 2]);
        let out = BinaryContourImageFilter::default().apply(&img).unwrap();
        assert_eq!(out.shape(), [2, 2, 2]);
        assert_eq!(*out.origin(), *img.origin());
        assert_eq!(*out.spacing(), *img.spacing());
    }

    /// 26-connected: center of 5×5×5 all-fg block is also interior.
    #[test]
    fn five_cube_center_interior_26conn() {
        let img = make_image(vec![1.0f32; 125], [5, 5, 5]);
        let out = BinaryContourImageFilter::new(true, 1.0)
            .apply(&img)
            .unwrap();
        let v = voxels(&out);
        assert_eq!(v[62], 0.0, "center of 5×5×5 should be interior (26-conn)");
    }
}
