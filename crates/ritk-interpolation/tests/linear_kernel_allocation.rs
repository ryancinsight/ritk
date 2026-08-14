//! Measures that the linear kernel's allocation is independent of volume size.
//!
//! The claim under test is that the borrowed-view seam removed the
//! whole-volume copy from `LinearInterpolator::interpolate`. Value tests cannot
//! show that — the copying implementation produced the same numbers. A
//! counting allocator can, and the decisive form is a *scaling* assertion
//! rather than a threshold: sample the same points from a 32³ volume and from a
//! 64³ volume (8× the voxels) and require the bytes allocated to be **equal**.
//! A whole-volume copy cannot satisfy that; nothing else in the kernel depends
//! on volume size. The threshold assertion that follows it is the weaker
//! backstop.
//!
//! Its own test binary with a single test, because the counter is process-wide:
//! a second test running concurrently in the same binary would race it.
//! `SequentialBackend` for the same reason — no worker threads allocating
//! underneath the measurement.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use coeus_core::SequentialBackend;
use coeus_tensor::Tensor;
use ritk_core::interpolation::Interpolator;
use ritk_interpolation::LinearInterpolator;

static BYTES: AtomicUsize = AtomicUsize::new(0);
static ARMED: AtomicBool = AtomicBool::new(false);

struct CountingAllocator;

// SAFETY: every method forwards to `System`, which upholds the `GlobalAlloc`
// contract; the counters are `Relaxed` atomics that observe allocation sizes
// without affecting the pointers returned or the layouts passed through.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if ARMED.load(Ordering::Relaxed) {
            BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        }
        // SAFETY: `layout` is forwarded unchanged from the caller, which the
        // `GlobalAlloc` contract already requires to be valid and non-zero-sized.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` was returned by `Self::alloc` for this exact `layout`,
        // which forwarded it to `System`, so `System` owns the allocation.
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

/// Bytes allocated while sampling `points` from a **permuted** `edge`³ volume.
///
/// Permuted, not contiguous, because that is where the cost actually was.
/// `Tensor::to_contiguous` returns an `Arc` clone when the tensor is contiguous
/// *and* zero-offset, so the removed call was already free on a plain volume —
/// measuring one would show nothing and prove nothing. The copy the seam
/// removes is the strided and offset path, so that is the path measured.
fn bytes_to_sample(edge: usize, points: &[f32]) -> usize {
    let voxels = edge * edge * edge;
    let values = (0..voxels).map(|value| value as f32).collect::<Vec<_>>();
    let volume = Tensor::<f32, SequentialBackend>::from_slice([edge, edge, edge], &values)
        .permute(&[2, 1, 0]);
    assert!(!volume.is_contiguous(), "the measured case must be strided");
    let indices = Tensor::<f32, SequentialBackend>::from_slice([points.len() / 3, 3], points);
    let interpolator = LinearInterpolator::new();

    // Warm up first: lazily initialized backend state must not land inside the
    // measured window.
    let _ = interpolator.interpolate(&volume, indices.clone());

    BYTES.store(0, Ordering::Relaxed);
    ARMED.store(true, Ordering::Relaxed);
    let sampled = interpolator.interpolate(&volume, indices);
    ARMED.store(false, Ordering::Relaxed);
    let measured = BYTES.load(Ordering::Relaxed);

    assert_eq!(
        sampled.shape(),
        [points.len() / 3],
        "kernel must still sample"
    );
    measured
}

#[test]
fn linear_kernel_allocation_is_independent_of_volume_size() {
    let points: Vec<f32> = (0..16)
        .flat_map(|p| [p as f32, p as f32, p as f32])
        .collect();

    let small = bytes_to_sample(32, &points);
    let large = bytes_to_sample(64, &points);

    assert_eq!(
        small,
        large,
        "allocation must not scale with volume: 32³ allocated {small} bytes, \
         64³ (8x the voxels, {} more bytes of data) allocated {large}; a \
         whole-volume copy would show the difference",
        (64usize.pow(3) - 32usize.pow(3)) * size_of::<f32>()
    );

    // Backstop: the absolute figure must also sit far below one volume. The
    // output tensor is 16 samples — the only allocation the kernel still makes.
    let volume_bytes = 64usize.pow(3) * size_of::<f32>();
    assert!(
        large < volume_bytes / 64,
        "sampling 16 points allocated {large} bytes against a {volume_bytes}-byte volume"
    );

    // And no per-voxel or per-sample heap traffic: 16 samples of f32 output
    // plus its tensor metadata, nothing proportional to 2^rank corner reads.
    assert!(
        large < 4096,
        "kernel allocated {large} bytes for a 16-sample output; \
         expected one output buffer and its layout metadata"
    );
}
