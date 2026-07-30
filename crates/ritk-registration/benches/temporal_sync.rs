use std::hint::black_box;
use std::time::Duration;

use criterion::{criterion_group, criterion_main, Criterion, SamplingMode};
use leto::Array1;
use ritk_registration::classical::temporal::TemporalSyncConfig;
use ritk_registration::TemporalSync;

const SAMPLE_COUNT: usize = 4_096;
const SEARCH_RANGE: usize = 64;
const DELAY_FRAMES: f64 = 7.25;

fn waveform(sample: f64) -> f64 {
    let normalized = sample / SAMPLE_COUNT as f64;
    (normalized * 37.0).sin()
        + 0.45 * (normalized * 113.0 + 0.3).cos()
        + 0.2 * (normalized * 241.0 - 0.7).sin()
}

fn temporal_signals() -> (Array1<f64>, Array1<f64>) {
    let reference = (0..SAMPLE_COUNT)
        .map(|index| waveform(index as f64))
        .collect::<Vec<_>>();
    let moving = (0..SAMPLE_COUNT)
        .map(|index| waveform(index as f64 - DELAY_FRAMES))
        .collect::<Vec<_>>();

    (
        Array1::from_vec([SAMPLE_COUNT], reference).expect("benchmark shape is valid"),
        Array1::from_vec([SAMPLE_COUNT], moving).expect("benchmark shape is valid"),
    )
}

fn bench_temporal_sync(criterion: &mut Criterion) {
    let (reference, moving) = temporal_signals();
    let synchronizer = TemporalSync::with_config(TemporalSyncConfig {
        frame_spacing: 0.02,
        search_range: SEARCH_RANGE,
        min_correlation: 0.8,
    });

    let mut group = criterion.benchmark_group("temporal_sync");
    group
        .sample_size(20)
        .sampling_mode(SamplingMode::Flat)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(3));
    group.bench_function("4096_samples/64_frame_search", |bencher| {
        bencher.iter(|| {
            black_box(
                synchronizer
                    .synchronize(black_box(&reference), black_box(&moving))
                    .expect("benchmark signals are finite and identifiable"),
            )
        });
    });
    group.finish();
}

criterion_group!(benches, bench_temporal_sync);
criterion_main!(benches);
