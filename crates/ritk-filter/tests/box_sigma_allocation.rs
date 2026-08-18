//! Allocation measurement for the region-based box sigma kernel.
//!
//! The claim under test is that `BoxSigmaImageFilter` no longer copies its
//! input volume. Asserting that from the source is not evidence, so this binary
//! installs a counting global allocator and measures it.
//!
//! # Method
//!
//! [`Counting`] wraps the system allocator and accumulates allocation events
//! and requested bytes. The test brackets three operations with a counter reset:
//!
//! 1. `try_data_vec()` — the whole-volume host copy the filter used to perform
//!    on entry. This calibrates the measurement: it establishes, in this
//!    process and on this allocator, what one copy of this volume costs.
//! 2. The bare parallel collect over the same element count with a trivial
//!    closure — the scheduler's steady-state cost for producing one output
//!    volume, reported so the filter's number can be read against it.
//! 3. `apply_native()` — the filter itself.
//!
//! The oracle is absolute, not a difference: the filter must allocate its output
//! volume and essentially nothing else. A kernel that copied its input would
//! allocate a second volume; a kernel that allocated per voxel or per window
//! would make allocations on the order of the voxel count. Both are excluded by
//! asserting bytes against the output size and count against a small constant.
//!
//! Ordering matters and is controlled: the parallel runtime allocates its worker
//! buffers on first substantial use — measured at 841 allocations / 7.4 MB — and
//! reuses them afterwards. That one-time cost lands on whichever call runs
//! first, so a full-size warm-up runs before any measurement. Without it the
//! numbers describe pool construction rather than either kernel.
//!
//! # Limits
//!
//! The counter is process-wide and records *requested* bytes at the allocator,
//! not peak resident set: an allocation freed immediately still counts, and
//! reuse of a freed block counts again. It therefore bounds copying from above,
//! which is the direction this test needs, but it is not a memory-footprint
//! measurement. The parallel runtime's thread pool and per-task bookkeeping also
//! allocate inside the measured window; the warm-up below pays the one-time pool
//! initialisation outside the window, and the volume is sized so that scheduler
//! traffic cannot approach the whole-volume threshold. `dhat` was not used
//! because it is not a dependency anywhere in this workspace and would add one
//! for a measurement an exact counter already provides; the tradeoff is that
//! this test reports totals rather than dhat's per-call-stack attribution.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use coeus_core::SequentialBackend;
use ritk_filter::BoxSigmaImageFilter;
use ritk_image::test_support::make_image;
use ritk_image::Image;

struct Counting;

static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
static BYTES: AtomicUsize = AtomicUsize::new(0);

// SAFETY: every method forwards to `System`, which is a valid allocator, with
// the same layout it was given; the counters are plain atomics and perform no
// allocation, so no re-entrancy into the allocator is possible.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        BYTES.fetch_add(new_size.saturating_sub(layout.size()), Ordering::Relaxed);
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

/// Allocation events and requested bytes accumulated while running `body`.
fn measure<R>(body: impl FnOnce() -> R) -> (usize, usize, R) {
    ALLOCATIONS.store(0, Ordering::SeqCst);
    BYTES.store(0, Ordering::SeqCst);
    let value = body();
    (
        ALLOCATIONS.load(Ordering::SeqCst),
        BYTES.load(Ordering::SeqCst),
        value,
    )
}

#[test]
fn box_sigma_does_not_copy_its_input_volume() {
    type B = SequentialBackend;

    const EDGE: usize = 64;
    let voxels = EDGE * EDGE * EDGE;
    let volume_bytes = voxels * std::mem::size_of::<f32>();

    let data: Vec<f32> = (0..voxels).map(|i| (i % 251) as f32).collect();
    let image: Image<f32, B, 3> = make_image(data, [EDGE, EDGE, EDGE]);
    let filter = BoxSigmaImageFilter::new([1, 1, 1]);
    let backend = B::default();

    // Warm-up at full size: the parallel runtime builds its worker buffers on
    // first substantial use, and that one-time cost would otherwise be charged
    // to whichever measurement happened to run first.
    let warm: Image<f32, B, 3> = make_image(vec![1.0; voxels], [EDGE, EDGE, EDGE]);
    let _ = filter.apply_native(&warm, &backend).expect("warm-up runs");
    drop(warm);

    // Calibration: what one whole-volume host copy costs, measured here.
    let (copy_events, copy_bytes, copied) = measure(|| image.try_data_vec().expect("host copy"));
    assert_eq!(copied.len(), voxels);
    assert!(
        copy_bytes >= volume_bytes,
        "calibration must observe at least one volume: {copy_bytes} < {volume_bytes}"
    );
    drop(copied);

    // Scheduler steady state: the same parallel collect producing the same
    // output volume with a closure that reads nothing — the floor any kernel
    // built on this collect must pay.
    let (base_events, base_bytes, baseline) =
        measure(|| moirai::map_collect_index_with::<moirai::Adaptive, _, _>(voxels, |_| 0.0f32));
    assert_eq!(baseline.len(), voxels);
    drop(baseline);

    // The filter.
    let (filter_events, filter_bytes, out) = measure(|| {
        filter
            .apply_native(&image, &backend)
            .expect("filter succeeds")
    });
    assert_eq!(out.shape(), [EDGE, EDGE, EDGE]);

    println!(
        "volume = {volume_bytes} B ({voxels} voxels)\n\
         one host copy      : {copy_events:>6} allocations, {copy_bytes:>10} B\n\
         scheduler floor    : {base_events:>6} allocations, {base_bytes:>10} B\n\
         box sigma          : {filter_events:>6} allocations, {filter_bytes:>10} B"
    );

    // The claim: the filter allocates its output and nothing of volume scale.
    // An implementation that extracted its input into a `Vec` first — which is
    // what this filter did before the region rewrite — would land at or above
    // two volumes.
    assert!(
        filter_bytes < volume_bytes * 5 / 4,
        "box sigma allocated {filter_bytes} B against a {volume_bytes} B output; \
         a volume-scale temporary such as an input copy is present"
    );

    // The kernel builds one clipped-window region and one row walker per output
    // voxel. If either allocated, this count would be on the order of {voxels}.
    assert!(
        filter_events < 64,
        "box sigma made {filter_events} allocations across {voxels} voxels; \
         per-voxel or per-window allocation is present"
    );
}
