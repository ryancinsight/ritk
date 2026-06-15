//! Atlas / label-map transfer — the ANTs-style "apply a transform to a label map".
//!
//! ritk has registration (rigid [`crate::register_rigid_ngf`], affine/MI
//! [`crate::GlobalMiRegistration`], deformable [`crate::MultiResSyNRegistration`])
//! and intensity resampling, but no single primitive to carry an integer LABEL /
//! atlas map through a recovered transform. [`warp_label_map`] fills that gap: it
//! resamples a moving-space label image onto a reference grid with NEAREST-NEIGHBOUR
//! interpolation, so integer region IDs are preserved exactly (no label blending).
//!
//! Atlas-to-patient pipeline (e.g. MNI152 → patient T1):
//! 1. Register the patient (fixed) to the atlas (moving) with any ritk registration,
//!    obtaining a [`Transform`] that maps patient → atlas world coordinates.
//! 2. `let patient_labels = warp_label_map(&atlas_labels, &transform, &patient_t1);`
//!    — the atlas region labels now live on the patient grid; read region centroids
//!    for targeting.
//!
//! For a deformable result the displacement/velocity field implements [`Transform`]
//! too, so the same call warps labels through a SyN/Demons/B-spline field.

use burn::tensor::backend::Backend;
use ritk_image::{grid, Image};
use ritk_interpolation::{Interpolator, NearestNeighborInterpolator};
use ritk_transform::Transform;
use std::collections::BTreeMap;

/// Warp a moving-space label/atlas image onto `reference`'s grid through
/// `transform` (mapping reference → moving world coordinates), using
/// nearest-neighbour interpolation so integer labels are preserved exactly.
/// Reference-grid voxels whose mapped point falls outside the label image are set
/// to `0` (background).
///
/// The result shares `reference`'s shape and spatial metadata, so it overlays the
/// reference image voxel-for-voxel.
#[must_use]
pub fn warp_label_map<B: Backend>(
    labels: &Image<B, 3>,
    transform: &impl Transform<B, 3>,
    reference: &Image<B, 3>,
) -> Image<B, 3> {
    let device = reference.data().device();
    let shape = reference.shape();

    // reference voxel → world → (transform) → label-image world → label index.
    let indices = grid::generate_grid(shape, &device);
    let ref_world = reference.index_to_world_tensor(indices);
    let label_world = transform.transform_points(ref_world);
    let label_idx = labels.world_to_index_tensor(label_world);

    // Nearest-neighbour keeps integer labels intact; zero-pad → background outside FOV.
    let warped = NearestNeighborInterpolator::new_zero_pad()
        .interpolate(labels.data(), label_idx)
        .reshape(shape);

    Image::new(
        warped,
        *reference.origin(),
        *reference.spacing(),
        *reference.direction(),
    )
}

/// World-space centroid of every non-zero label in `labels`, returned as
/// `(label_id, [x, y, z])` pairs sorted by id. Turns a warped atlas label map
/// into target points: register the patient to an atlas, [`warp_label_map`] the
/// atlas regions into patient space, then read each region's centroid as an
/// array-center / focus target. Centroids are in the image's physical (LPS) world
/// frame, consistent with [`Image::index_to_world_tensor`].
pub fn label_centroids<B: Backend>(labels: &Image<B, 3>) -> anyhow::Result<Vec<(u32, [f64; 3])>> {
    let [d0, d1, d2] = labels.shape();
    let data = labels.try_data_slice()?;
    let origin = labels.origin();
    let spacing = labels.spacing();
    let direction = labels.direction();

    // Accumulate per-label sum of voxel multi-index [d0, d1, d2] + count.
    // 3D loop iteration pattern avoids expensive division and modulo operations per voxel.
    let mut acc: BTreeMap<u32, ([f64; 3], f64)> = BTreeMap::new();
    let mut flat = 0;
    for iz in 0..d0 {
        let z_f = iz as f64;
        for iy in 0..d1 {
            let y_f = iy as f64;
            for ix in 0..d2 {
                let v = data[flat];
                flat += 1;
                if v <= 0.5 {
                    continue; // background / unlabelled
                }
                let x_f = ix as f64;
                let lab = (v + 0.5) as u32; // nearest integer label id
                let e = acc.entry(lab).or_insert(([0.0; 3], 0.0));
                e.0[0] += z_f;
                e.0[1] += y_f;
                e.0[2] += x_f;
                e.1 += 1.0;
            }
        }
    }

    let centroids = acc
        .into_iter()
        .map(|(lab, (sum, cnt))| {
            let mean = [sum[0] / cnt, sum[1] / cnt, sum[2] / cnt];
            // world[c] = origin[c] + Σ_axis mean[axis]·spacing[axis]·direction[(c, axis)]
            let mut world = [0.0_f64; 3];
            for (c, wc) in world.iter_mut().enumerate() {
                let mut a = origin[c];
                for (axis, &mi) in mean.iter().enumerate() {
                    a += mi * spacing[axis] * direction[(c, axis)];
                }
                *wc = a;
            }
            (lab, world)
        })
        .collect();
    Ok(centroids)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Shape, Tensor, TensorData};
    use burn_ndarray::NdArray;
    use ritk_spatial::{Direction, Point, Spacing};
    use ritk_transform::TranslationTransform;

    type B = NdArray<f32>;

    fn label_image(data: Vec<f32>, shape: [usize; 3]) -> Image<B, 3> {
        let device = Default::default();
        let t = Tensor::<B, 3>::from_data(TensorData::new(data, Shape::new(shape)), &device);
        Image::new(
            t,
            Point::new([0.0; 3]),
            Spacing::new([1.0; 3]),
            Direction::identity(),
        )
    }

    fn host(img: &Image<B, 3>) -> Vec<f32> {
        let n: usize = img.shape().iter().product();
        img.data()
            .clone()
            .reshape([n])
            .into_data()
            .to_vec()
            .unwrap()
    }

    /// Identity warp reproduces the label map exactly (validates the full chain:
    /// grid → index↔world → nearest interpolate → reshape).
    #[test]
    fn identity_warp_preserves_labels_exactly() {
        let (d, h, w) = (6usize, 6, 6);
        let mut v = vec![0.0f32; d * h * w];
        for z in 2..4 {
            for y in 2..4 {
                for x in 2..4 {
                    v[(z * h + y) * w + x] = 7.0; // a labelled block
                }
            }
        }
        let img = label_image(v.clone(), [d, h, w]);
        let device = Default::default();
        let ident = TranslationTransform::<B, 3>::new(Tensor::from_data(
            TensorData::new(vec![0.0f32, 0.0, 0.0], [3]),
            &device,
        ));
        let out = warp_label_map(&img, &ident, &img);
        assert_eq!(host(&out), v, "identity warp must preserve labels exactly");
    }

    /// A translation moves the labels and keeps them INTEGER (nearest-neighbour, no
    /// blending), with the same voxel count when the block stays in the FOV.
    #[test]
    fn translation_warp_keeps_integer_labels() {
        let (d, h, w) = (8usize, 8, 8);
        let mut v = vec![0.0f32; d * h * w];
        for z in 0..d {
            for y in 0..h {
                for x in 3..5 {
                    v[(z * h + y) * w + x] = 5.0;
                }
            }
        }
        let img = label_image(v.clone(), [d, h, w]);
        let device = Default::default();
        // Small in-bounds shift along the x/d2 world axis (component 2, identity dir).
        let t = TranslationTransform::<B, 3>::new(Tensor::from_data(
            TensorData::new(vec![0.0f32, 0.0, -2.0], [3]),
            &device,
        ));
        let out = host(&warp_label_map(&img, &t, &img));
        // No blending: only the original label values appear.
        for &val in &out {
            assert!(
                val == 0.0 || val == 5.0,
                "nearest-neighbour must keep integer labels, got {val}"
            );
        }
        let count = |s: &[f32]| s.iter().filter(|&&x| x == 5.0).count();
        assert_eq!(
            count(&out),
            count(&v),
            "in-bounds shift preserves label volume"
        );
        assert_ne!(out, v, "a non-zero translation must move the labels");
    }

    /// Region centroids land at the geometric centres (identity direction + unit
    /// spacing + zero origin ⇒ world centroid == mean voxel multi-index [d0,d1,d2]).
    #[test]
    fn label_centroids_finds_region_centres() {
        let (d, h, w) = (10usize, 10, 10);
        let mut v = vec![0.0f32; d * h * w];
        // Label 1: single voxel at index (d0,d1,d2) = (2, 3, 4).
        v[(2 * h + 3) * w + 4] = 1.0;
        // Label 2: 2×2×2 block over [6,8) on each axis → centroid 6.5 each.
        for z in 6..8 {
            for y in 6..8 {
                for x in 6..8 {
                    v[(z * h + y) * w + x] = 2.0;
                }
            }
        }
        let c = label_centroids(&label_image(v, [d, h, w])).unwrap();
        assert_eq!(c.len(), 2);
        let close = |a: f64, b: f64| (a - b).abs() < 1e-6;
        assert_eq!(c[0].0, 1);
        assert!(
            close(c[0].1[0], 2.0) && close(c[0].1[1], 3.0) && close(c[0].1[2], 4.0),
            "label 1 centroid {:?}",
            c[0].1
        );
        assert_eq!(c[1].0, 2);
        assert!(
            close(c[1].1[0], 6.5) && close(c[1].1[1], 6.5) && close(c[1].1[2], 6.5),
            "label 2 centroid {:?}",
            c[1].1
        );
    }
}
