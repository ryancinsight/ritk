//! Repeated-pose runtime and retained-memory benchmark for packed MIND-SSC.
//!
//! The 32-cubed image has 17,576 complete-support centers, so the default
//! deterministic policy exercises its 8,192-center cap. The timed region is
//! one pose evaluation; fixed descriptor preparation and image allocation are
//! outside it.

use std::hint::black_box;
use std::mem::size_of;
use std::time::Duration;

use coeus_core::SequentialBackend;
use criterion::{criterion_group, criterion_main, Criterion, SamplingMode};
use ritk_image::Image;
use ritk_registration::metric::mind::{MindSscConfig, MindSscFixedPrep};
use ritk_registration::types::AffineTransform;
use ritk_spatial::{Direction, Point, Spacing};

const SHAPE: [usize; 3] = [32; 3];

fn synthetic_image() -> Image<f32, SequentialBackend, 3> {
    let values = (0..SHAPE[0])
        .flat_map(|z| {
            (0..SHAPE[1]).flat_map(move |y| {
                (0..SHAPE[2]).map(move |x| {
                    let z = u16::try_from(z).expect("benchmark extent fits u16");
                    let y = u16::try_from(y).expect("benchmark extent fits u16");
                    let x = u16::try_from(x).expect("benchmark extent fits u16");
                    f32::from((z * z + 3 * y * y + 5 * x * x + 7 * z * y + 11 * x) % 251)
                })
            })
        })
        .collect();
    Image::from_flat_on(
        values,
        SHAPE,
        Point::origin(),
        Spacing::uniform(1.0),
        Direction::identity(),
        &SequentialBackend,
    )
    .expect("benchmark image geometry and storage are valid")
}

fn benchmark_mind_ssc(criterion: &mut Criterion) {
    let image = synthetic_image();
    let prepared = MindSscFixedPrep::try_new(&image, MindSscConfig::default(), None, None)
        .expect("benchmark image contains complete MIND-SSC support");
    let memory = prepared.memory_usage();
    assert_eq!(memory.selected_centers, MindSscConfig::DEFAULT_MAX_SAMPLES);
    assert_eq!(
        memory.index_bytes,
        MindSscConfig::DEFAULT_MAX_SAMPLES * size_of::<usize>()
    );
    assert_eq!(
        memory.descriptor_bytes,
        MindSscConfig::DEFAULT_MAX_SAMPLES * size_of::<u64>()
    );
    assert_eq!(memory.weight_bytes, 0);
    assert_eq!(
        memory.heap_payload_bytes,
        memory.index_bytes + memory.descriptor_bytes
    );
    assert_eq!(memory.per_center_scratch_bytes, 72);

    let pose = AffineTransform::new([
        1.0, 0.0, 0.0, 0.35, 0.0, 1.0, 0.0, -0.2, 0.0, 0.0, 1.0, 0.1, 0.0, 0.0, 0.0, 1.0,
    ]);
    let mut group = criterion.benchmark_group("mind_ssc");
    group
        .sample_size(10)
        .sampling_mode(SamplingMode::Flat)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(3));
    group.bench_function("8192_centers/pose", |bencher| {
        bencher.iter(|| {
            black_box(
                prepared
                    .eval(black_box(&image), black_box(&pose))
                    .expect("benchmark pose and moving image are valid"),
            )
        });
    });
    group.finish();
}

criterion_group!(benches, benchmark_mind_ssc);
criterion_main!(benches);
