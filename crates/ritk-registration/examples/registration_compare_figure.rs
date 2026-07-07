//! Visual comparison of CT↔MR rigid registration: identity (before) vs ritk
//! multi-resolution center-weighted NGF vs the SimpleElastix reference.
//!
//! Renders a mid-axial slice as an RGB overlay (R = CT, G = MR). Aligned anatomy
//! appears yellow/grey; misalignment shows as red/green fringes — most visible at
//! the ventricles and cortical rim. Three panels: identity | ritk | elastix.
//!
//! Usage: `cargo run --release -p ritk-registration --example registration_compare_figure`
//! (paths default to the RIRE-109 pair + elastix result under leoneuro/).

use burn_ndarray::NdArray;
use image::{Rgb, RgbImage};
use ritk_image::tensor::{Tensor, TensorData};
use ritk_image::{grid, Image};
use ritk_interpolation::{Interpolator, LinearInterpolator};
use ritk_io::read_nifti;
use ritk_registration::{
    ct_brain_mask, default_ngf_pyramid, register_rigid_ngf_multires,
    translation_from_centers_of_mass, CtBrainMaskConfig,
};
use ritk_transform::{RigidTransform, Transform};

type B = NdArray<f32>;

/// Resample `moving` onto `fixed`'s grid through `transform`; row-major host vec.
fn resample(
    fixed: &Image<B, 3>,
    moving: &Image<B, 3>,
    transform: &RigidTransform<B, 3>,
) -> Vec<f32> {
    let device = fixed.data().device();
    let idx = grid::generate_grid(fixed.shape(), &device);
    let world = fixed.index_to_world_tensor(idx);
    let mworld = transform.transform_points(world);
    let midx = moving.world_to_index_tensor(mworld);
    LinearInterpolator::new()
        .interpolate(moving.data(), midx)
        .into_data()
        .to_vec()
        .expect("resampled host vec")
}

fn host(img: &Image<B, 3>) -> Vec<f32> {
    let n: usize = img.shape().iter().product();
    img.data()
        .clone()
        .reshape([n])
        .into_data()
        .to_vec()
        .expect("host vec")
}

/// Normalise an axial slice `z0` of a `[nz,ny,nx]` row-major volume to `[0,1]`
/// using a robust [p2,p98] window.
fn slice_norm(vol: &[f32], shape: [usize; 3], z0: usize) -> Vec<f32> {
    let [_nz, ny, nx] = shape;
    let base = z0 * ny * nx;
    let s = &vol[base..base + ny * nx];
    let mut sorted: Vec<f32> = s.iter().copied().filter(|v| v.is_finite()).collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p = |q: f32| sorted[((sorted.len() as f32 * q) as usize).min(sorted.len() - 1)];
    let (lo, hi) = (p(0.02), p(0.98));
    let d = (hi - lo).max(1e-6);
    s.iter().map(|&v| ((v - lo) / d).clamp(0.0, 1.0)).collect()
}

fn main() -> anyhow::Result<()> {
    let ct_path = "D:/kwavers/leoneuro/data/brain_ct.nii.gz";
    let mr_path = "D:/kwavers/leoneuro/data/brain_mri_t1.nii.gz";
    let elastix_path = "D:/kwavers/leoneuro/scripts/elastix_result_mr_on_ct.nii.gz";
    let out = "D:/kwavers/leoneuro/scripts/registration_compare.png";

    let device = Default::default();
    let ct = read_nifti::<B, _>(ct_path, &device)?;
    let mri = read_nifti::<B, _>(mr_path, &device)?;
    let shape = ct.shape();
    let [nz, ny, nx] = shape;

    // ritk registration (multi-res, sampled, center-weighted).
    let mask = ct_brain_mask(&ct, &CtBrainMaskConfig::default());
    let coarse = ritk_filter::BinShrinkImageFilter::new(vec![1, 4, 4]);
    let com_t = translation_from_centers_of_mass(&coarse.apply(&ct), &coarse.apply(&mri));
    let (ritk_t, res) = register_rigid_ngf_multires::<B>(
        &ct,
        &mri,
        [0.0; 3],
        com_t,
        Some(&mask),
        &default_ngf_pyramid(None), // uniform NGF (skull-driven) on correct geometry
    );
    let tr = res.matrix[0] + res.matrix[5] + res.matrix[10];
    let ang = (((tr - 1.0) / 2.0).clamp(-1.0, 1.0)).acos().to_degrees();
    println!("ritk recovered composite rotation {ang:.2}°");

    // The three moving volumes on the CT grid.
    let zero = || Tensor::<B, 1>::from_data(TensorData::from([0.0_f32, 0.0, 0.0]), &device);
    let identity = RigidTransform::<B, 3>::new(zero(), zero(), zero());
    let mr_identity = resample(&ct, &mri, &identity);
    // Write the ritk identity-resampled MR for a pixelwise check against sitk's
    // own identity resample (isolates resample correctness from the optimiser).
    {
        let n: usize = ct.shape().iter().product();
        let t =
            Tensor::<B, 3>::from_data(TensorData::new(mr_identity.clone(), ct.shape()), &device);
        let img = Image::new(t, *ct.origin(), *ct.spacing(), *ct.direction());
        ritk_io::write_nifti(
            "D:/kwavers/leoneuro/scripts/ritk_identity_mr_on_ct.nii.gz",
            &img,
        )?;
        let _ = n;
    }
    let mr_ritk = resample(&ct, &mri, &ritk_t);
    let mr_elastix = host(&read_nifti::<B, _>(elastix_path, &device)?);

    let ct_host = host(&ct);

    // Quantify alignment: masked NCC (Pearson) of CT vs each resampled MR over
    // the brain mask — a modality-agnostic-ish overlap score for ranking poses.
    let mask_host = host(&mask);
    let ncc = |a: &[f32], b: &[f32]| -> f64 {
        let idx: Vec<usize> = (0..a.len()).filter(|&i| mask_host[i] > 0.5).collect();
        let n = idx.len() as f64;
        let (ma, mb) = (
            idx.iter().map(|&i| a[i] as f64).sum::<f64>() / n,
            idx.iter().map(|&i| b[i] as f64).sum::<f64>() / n,
        );
        let (mut num, mut da, mut db) = (0.0, 0.0, 0.0);
        for &i in &idx {
            let (x, y) = (a[i] as f64 - ma, b[i] as f64 - mb);
            num += x * y;
            da += x * x;
            db += y * y;
        }
        num / (da.sqrt() * db.sqrt()).max(1e-12)
    };
    println!(
        "masked NCC(CT, MR):  identity {:.4}   ritk {:.4}   elastix {:.4}",
        ncc(&ct_host, &mr_identity),
        ncc(&ct_host, &mr_ritk),
        ncc(&ct_host, &mr_elastix),
    );

    let z0 = nz / 2;
    let ct_s = slice_norm(&ct_host, shape, z0);
    let panels = [
        ("identity", slice_norm(&mr_identity, shape, z0)),
        ("ritk", slice_norm(&mr_ritk, shape, z0)),
        ("elastix", slice_norm(&mr_elastix, shape, z0)),
    ];

    // Compose: 3 panels side by side, R = CT, G = MR.
    let gap = 8u32;
    let (w, h) = (nx as u32, ny as u32);
    let mut img = RgbImage::from_pixel(w * 3 + gap * 2, h, Rgb([16, 16, 16]));
    for (pi, (_name, mr_s)) in panels.iter().enumerate() {
        let xoff = pi as u32 * (w + gap);
        for y in 0..ny {
            for x in 0..nx {
                let r = (ct_s[y * nx + x] * 255.0) as u8;
                let g = (mr_s[y * nx + x] * 255.0) as u8;
                img.put_pixel(xoff + x as u32, y as u32, Rgb([r, g, 0]));
            }
        }
    }
    img.save(out)?;
    println!("wrote {out}  (panels: identity | ritk | elastix; R=CT, G=MR; axial z={z0})");
    Ok(())
}
