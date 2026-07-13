use coeus_core::MoiraiBackend;
use coeus_tensor::Tensor;
use ritk_core::spatial::{Direction, Point, Spacing};
use ritk_image::native::Image;
use ritk_model::ssmmorph::{IntegrationMode, SSMMorphConfig};
use ritk_registration::registration::dl_ssm_registration::DiffeomorphicSSMMorph;

#[test]
fn test_diffeomorphic_ssmmorph_integration() {
    let mut config = SSMMorphConfig::for_3d_registration();
    config.encoder.num_stages = 2;
    config.encoder.base_channels = 4;
    config.encoder.blocks_per_stage = 1;
    config.encoder.in_channels = 2;
    config.integration = IntegrationMode::Diffeomorphic;
    config.integration_steps = 5;
    let model = DiffeomorphicSSMMorph::<MoiraiBackend>::new(&config);

    let extent = 16;
    let shape = [extent; 3];
    let make_sphere = |center: [f32; 3], radius: f32| -> Vec<f32> {
        let mut values = Vec::with_capacity(extent * extent * extent);
        for z in 0..extent {
            for y in 0..extent {
                for x in 0..extent {
                    let delta = [
                        x as f32 - center[0],
                        y as f32 - center[1],
                        z as f32 - center[2],
                    ];
                    let squared = delta.iter().map(|value| value * value).sum::<f32>();
                    values.push((-squared / (2.0 * radius * radius)).exp());
                }
            }
        }
        values
    };
    let backend = MoiraiBackend;
    let geometry = (
        Point::origin(),
        Spacing::new([1.0; 3]),
        Direction::identity(),
    );
    let fixed = Image::from_flat_on(
        make_sphere([8.0; 3], 4.0),
        shape,
        geometry.0,
        geometry.1,
        geometry.2,
        &backend,
    )
    .expect("valid fixed image");
    let moving = Image::from_flat_on(
        make_sphere([9.0, 9.0, 8.0], 4.0),
        shape,
        geometry.0,
        geometry.1,
        geometry.2,
        &backend,
    )
    .expect("valid moving image");

    let transform = model
        .register_diffeomorphic(&fixed, &moving)
        .expect("native images satisfy SSMMorph contracts");
    assert_eq!(transform.field().components().len(), 3);
    for component in transform.field().components() {
        assert_eq!(component.shape(), &[16, 16, 16]);
        assert!(
            component.as_slice().iter().all(|value| *value == 0.0),
            "zero-initialized projection must encode identity displacement"
        );
    }
}

#[test]
fn ssmmorph_rejects_noncontiguous_native_input_without_copying() {
    let config = SSMMorphConfig::lightweight();
    let model = DiffeomorphicSSMMorph::<MoiraiBackend>::new(&config);
    let backend = MoiraiBackend;
    let tensor = Tensor::from_slice_on([2, 2, 2], &[0.0; 8], &backend).permute(&[1, 0, 2]);
    let image = Image::new(
        tensor,
        Point::origin(),
        Spacing::new([1.0; 3]),
        Direction::identity(),
    )
    .expect("rank remains three");
    let error = model
        .register_diffeomorphic(&image, &image)
        .err()
        .expect("noncontiguous input must be rejected");
    assert_eq!(
        error.to_string(),
        "SSMMorph native image tensors must be contiguous for zero-copy reshape"
    );
}
