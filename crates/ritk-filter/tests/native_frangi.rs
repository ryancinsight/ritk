//! Native-image contracts for Frangi vesselness.

use coeus_core::SequentialBackend;
use ritk_filter::{FrangiConfig, FrangiVesselnessFilter, VesselPolarity};
use ritk_image::native::Image;
use ritk_spatial::{Direction, Point, Spacing};

type B = SequentialBackend;

fn image(values: Vec<f32>, shape: [usize; 3]) -> Image<f32, B, 3> {
    Image::from_flat_on(
        values,
        shape,
        Point::origin(),
        Spacing::uniform(1.0),
        Direction::identity(),
        &B::default(),
    )
    .expect("invariant: valid native test image")
}

fn values(image: &Image<f32, B, 3>) -> &[f32] {
    image
        .data_slice()
        .expect("invariant: contiguous native test image")
}

fn bright_tube(size: usize, radius: f64) -> Vec<f32> {
    let centre = (size as f64 - 1.0) / 2.0;
    (0..size * size * size)
        .map(|flat| {
            let y = (flat / size) % size;
            let x = flat % size;
            let distance = ((y as f64 - centre).powi(2) + (x as f64 - centre).powi(2)).sqrt();
            if distance < radius {
                100.0
            } else {
                0.0
            }
        })
        .collect()
}

fn bright_sphere(size: usize, radius: f64) -> Vec<f32> {
    let centre = (size as f64 - 1.0) / 2.0;
    (0..size * size * size)
        .map(|flat| {
            let z = flat / (size * size);
            let y = (flat / size) % size;
            let x = flat % size;
            let distance = ((z as f64 - centre).powi(2)
                + (y as f64 - centre).powi(2)
                + (x as f64 - centre).powi(2))
            .sqrt();
            if distance < radius {
                100.0
            } else {
                0.0
            }
        })
        .collect()
}

fn bright_config() -> FrangiConfig {
    FrangiConfig {
        scales: vec![1.0, 2.0],
        alpha: 0.5,
        beta: 0.5,
        gamma: 15.0,
        polarity: VesselPolarity::Bright,
    }
}

#[test]
fn bright_cylindrical_tube_has_high_central_vesselness() {
    const SIZE: usize = 20;
    let output = FrangiVesselnessFilter::new(bright_config())
        .apply_native(
            &image(bright_tube(SIZE, 3.0), [SIZE, SIZE, SIZE]),
            &B::default(),
        )
        .expect("native Frangi succeeds");
    for z in [9usize, 10] {
        for y in [9usize, 10] {
            for x in [9usize, 10] {
                let value = values(&output)[z * SIZE * SIZE + y * SIZE + x];
                assert!(value > 0.05, "tube center ({z}, {y}, {x}): {value}");
            }
        }
    }
    assert!(values(&output)[0] < 0.02);
    assert!(values(&output)[SIZE * SIZE * SIZE - 1] < 0.02);
}

#[test]
fn uniform_field_has_zero_vesselness() {
    const SIZE: usize = 10;
    let output = FrangiVesselnessFilter::new(FrangiConfig::default())
        .apply_native(
            &image(vec![42.0; SIZE * SIZE * SIZE], [SIZE, SIZE, SIZE]),
            &B::default(),
        )
        .expect("native Frangi succeeds");
    assert!(values(&output).iter().all(|&value| value < 1e-6));
}

#[test]
fn bright_sphere_has_low_central_vesselness() {
    const SIZE: usize = 30;
    let output = FrangiVesselnessFilter::new(bright_config())
        .apply_native(
            &image(bright_sphere(SIZE, 5.0), [SIZE, SIZE, SIZE]),
            &B::default(),
        )
        .expect("native Frangi succeeds");
    let centre = 14 * SIZE * SIZE + 14 * SIZE + 14;
    assert!(
        values(&output)[centre] < 0.4,
        "sphere response: {}",
        values(&output)[centre]
    );
}

#[test]
fn tube_response_exceeds_sphere_response() {
    const SIZE: usize = 20;
    let filter = FrangiVesselnessFilter::new(bright_config());
    let tube = filter
        .apply_native(
            &image(bright_tube(SIZE, 3.0), [SIZE, SIZE, SIZE]),
            &B::default(),
        )
        .expect("native tube Frangi succeeds");
    let sphere = filter
        .apply_native(
            &image(bright_sphere(SIZE, 3.0), [SIZE, SIZE, SIZE]),
            &B::default(),
        )
        .expect("native sphere Frangi succeeds");
    let centre = 9 * SIZE * SIZE + 9 * SIZE + 9;
    assert!(
        values(&tube)[centre] > values(&sphere)[centre],
        "tube={} sphere={}",
        values(&tube)[centre],
        values(&sphere)[centre]
    );
}

#[test]
fn dark_polarity_rejects_a_bright_tube() {
    const SIZE: usize = 12;
    let filter = FrangiVesselnessFilter::new(FrangiConfig {
        scales: vec![1.5],
        polarity: VesselPolarity::Dark,
        ..Default::default()
    });
    let output = filter
        .apply_native(
            &image(bright_tube(SIZE, 2.5), [SIZE, SIZE, SIZE]),
            &B::default(),
        )
        .expect("native Frangi succeeds");
    let centre = SIZE / 2 * SIZE * SIZE + SIZE / 2 * SIZE + SIZE / 2;
    assert!(
        values(&output)[centre] < 1e-6,
        "dark-polarity response: {}",
        values(&output)[centre]
    );
}
