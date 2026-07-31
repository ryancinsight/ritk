//! MGH/MGZ read-throughput baselines.
//!
//! The benchmark prepares deterministic files outside the timed region and
//! measures the public reader, including file open, optional gzip
//! decompression, big-endian voxel conversion, and image construction.
//!
//! The fixed 128 × 128 × 64 volume contains 1,048,576 `f32` voxels (4 MiB).
//! This is large enough to expose complete-payload allocation and conversion
//! costs while keeping the two-workload binary within the 300-second budget.
//! Criterion records the host and statistical distribution in its baseline.

use coeus_core::SequentialBackend;
use criterion::{criterion_group, criterion_main, Criterion};
use ritk_image::Image;
use ritk_mgh::{read_mgh, write_mgh};
use ritk_spatial::{Direction, Point, Spacing};
use std::hint::black_box;
use tempfile::TempDir;

const SHAPE: [usize; 3] = [64, 128, 128];

fn benchmark_image() -> Image<f32, SequentialBackend, 3> {
    let [nz, ny, nx] = SHAPE;
    let voxel_count = nz
        .checked_mul(ny)
        .and_then(|plane| plane.checked_mul(nx))
        .expect("invariant: benchmark shape fits usize");
    let values = (0..voxel_count)
        .map(|index| {
            let x = index % nx;
            let y = index / nx % ny;
            let z = index / (nx * ny);
            ((x * 17 + y * 29 + z * 43) % 4096) as f32
        })
        .collect();
    Image::from_flat_on(
        values,
        SHAPE,
        Point::new([10.0, 20.0, 30.0]),
        Spacing::new([0.8, 0.8, 1.2]),
        Direction::identity(),
        &SequentialBackend,
    )
    .expect("invariant: benchmark voxel count matches shape")
}

fn prepare_file(extension: &str) -> (TempDir, std::path::PathBuf) {
    let directory = tempfile::tempdir().expect("benchmark temporary directory");
    let path = directory.path().join(format!("volume.{extension}"));
    let image = benchmark_image();
    write_mgh(&image, &path, &SequentialBackend).expect("benchmark fixture must serialize");
    (directory, path)
}

fn bench_reader(criterion: &mut Criterion) {
    let (_mgh_directory, mgh_path) = prepare_file("mgh");
    let (_mgz_directory, mgz_path) = prepare_file("mgz");
    let mut group = criterion.benchmark_group("mgh_reader_128x128x64");
    group.warm_up_time(std::time::Duration::from_secs(1));
    group.measurement_time(std::time::Duration::from_secs(3));
    group.sample_size(20);

    group.bench_function("uncompressed", |bencher| {
        bencher.iter(|| {
            black_box(
                read_mgh(black_box(&mgh_path), &SequentialBackend)
                    .expect("benchmark MGH fixture must decode"),
            )
        });
    });
    group.bench_function("gzip", |bencher| {
        bencher.iter(|| {
            black_box(
                read_mgh(black_box(&mgz_path), &SequentialBackend)
                    .expect("benchmark MGZ fixture must decode"),
            )
        });
    });
    group.finish();
}

criterion_group!(benches, bench_reader);
criterion_main!(benches);
