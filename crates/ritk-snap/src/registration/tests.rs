use super::*;
use crate::app::tests::test_volume;

#[test]
fn moving_image_is_resampled_on_the_fixed_patient_grid() {
    let mut fixed = test_volume([10, 12, 14]);
    fixed.spacing = [2.0, 1.5, 0.75];
    fixed.origin = [-48.0, 11.0, 17.0];
    let mut moving = test_volume([10, 14, 12]);
    moving.spacing = [2.0, 0.75, 1.5];
    moving.direction = [1., 0., 0., 0., 0., -1., 0., 1., 0.];
    moving.origin = [50., 137., -63.];
    moving.modality = Some(arrayvec::ArrayString::from("MR").expect("DICOM modality"));
    let mut values = Vec::new();
    let mut mr = vec![0.; moving.data.len()];
    for z in 0..10 {
        for y in 0..12 {
            for x in 0..14 {
                let v = ((z * 11 + y * 3 + x * x + z * x) % 31) as f32;
                values.push(v);
                mr[(z * 14 + x) * 12 + (11 - y)] = 100. - 2. * v;
            }
        }
    }
    fixed.data = Arc::new(values);
    moving.data = Arc::new(mr);
    let result = run(&fixed, &moving, false, &AtomicBool::new(false)).expect("rigid registration");
    assert_eq!(
        (
            result.shape,
            result.origin,
            result.direction,
            result.spacing
        ),
        (fixed.shape, fixed.origin, fixed.direction, fixed.spacing)
    );
    assert_eq!(result.modality, moving.modality);
    assert!(result.source.is_none() && result.metadata.is_none());
    for (&a, &b) in fixed.data.iter().zip(result.data.iter()) {
        assert!((b - (100. - 2. * a)).abs() < 0.01);
    }
}

#[test]
fn flat_images_and_non_volumetric_inputs_are_rejected() {
    let flat = test_volume([8; 3]);
    assert!(run(&flat, &flat, false, &AtomicBool::new(false)).is_err());
    assert!(scalar_input(&test_volume([1, 8, 8])).is_err());
}

#[test]
fn optimizer_recovers_an_internal_rigid_displacement_in_a_shared_frame() {
    let mut fixed = test_volume([24; 3]);
    let mut moving = fixed.clone();
    fn anatomy(z: f64, y: f64, x: f64) -> f32 {
        let lobes = [
            (8., 12., 10., 90., 10.),
            (16., 7., 12., 65., 7.),
            (13., 15., 16., 48., 4.),
            (9., 14., 6., 55., 6.),
        ];
        lobes
            .into_iter()
            .map(|(cz, cy, cx, a, w)| {
                a * (-((z - cz).powi(2) + (y - cy).powi(2) + (x - cx).powi(2)) / w).exp()
            })
            .sum::<f64>() as f32
    }
    let indices = || {
        (0..24).flat_map(|z| {
            (0..24)
                .flat_map(move |y| (0..24).map(move |x| (f64::from(z), f64::from(y), f64::from(x))))
        })
    };
    let (s, c) = 8_f64.to_radians().sin_cos();
    fixed.data = Arc::new(indices().map(|(z, y, x)| anatomy(z, y, x)).collect());
    moving.data = Arc::new(
        indices()
            .map(|(z, y, x)| {
                let dy = y - 12. + 2.;
                let dx = x - 12. - 1.;
                200. - 2. * anatomy(z - 2., 12. + c * dy + s * dx, 12. - s * dy + c * dx)
            })
            .collect(),
    );
    let aligned = run(&fixed, &moving, true, &AtomicBool::new(false)).expect("rigid fit");
    let error = |volume: &LoadedVolume| -> f64 {
        fixed
            .data
            .iter()
            .zip(volume.data.iter())
            .filter(|(a, _)| **a > 10.)
            .map(|(&a, &b)| f64::from((200. - 2. * a - b).abs()))
            .sum()
    };
    assert!(
        error(&aligned) < 0.3 * error(&moving),
        "before {}, after {}",
        error(&moving),
        error(&aligned)
    );
}

#[test]
fn register_public_rire_ct_mr_against_the_independent_fiducial_transform() {
    use crate::dicom::loader::load_volume_from_path;
    let root =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test_data/registration/rire");
    let mut ct = load_volume_from_path(root.join("training_001_ct.mha")).expect("RIRE CT");
    let mr = load_volume_from_path(root.join("training_001_mr_T1.mha")).expect("RIRE MR");
    ct.modality = Some(arrayvec::ArrayString::from("CT").expect("modality"));
    let pose = estimate(
        &ct,
        &mr,
        validate_volume(&ct).expect("CT geometry"),
        validate_volume(&mr).expect("MR geometry"),
        false,
        &AtomicBool::new(false),
    )
    .expect("estimate RIRE rigid pose");
    let reference =
        std::fs::read_to_string(root.join("ct_T1.standard")).expect("RIRE independent reference");
    let mut errors = Vec::new();
    for line in reference.lines() {
        let row: Vec<f64> = line
            .split_whitespace()
            .filter_map(|s| s.parse().ok())
            .collect();
        if let [_, x, y, z, u, v, w] = row.as_slice() {
            let actual = transform_point(&[*x, *y, *z], &pose);
            errors.push(
                ((actual[0] - u).powi(2) + (actual[1] - v).powi(2) + (actual[2] - w).powi(2))
                    .sqrt(),
            );
        }
    }
    assert_eq!(errors.len(), 8);
    // Each supplied corner must land within one 4-mm CT slice of its reference.
    assert!(
        errors.iter().all(|&error| error < 4.),
        "RIRE corner errors in mm: {errors:?}, pose {pose:?}"
    );
}
