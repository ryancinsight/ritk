//! Peak-allocation measurement for tests that assert what a read costs.
//!
//! The companion to [`crate::io_bounds`]: that module bounds speculative
//! allocation, this one lets a test prove the bound holds.
//!
//! Reader defects of this class are invisible to an ordinary assertion. A
//! length field driving `Vec::with_capacity` does not necessarily abort — a
//! multi-gigabyte request succeeds wherever the memory exists, so the read
//! still returns the same truncation error it would have returned anyway, and
//! only the memory tells you the difference. Elapsed time is not a substitute
//! either: a wall-clock assertion is flaky by construction.
//!
//! `#[global_allocator]` is per binary, so each test binary installs its own
//! static. The type and the measurement live here so only that one-line
//! declaration repeats:
//!
//! ```ignore
//! use ritk_core::alloc_probe::{peak_bytes_during, PeakTrackingAllocator};
//!
//! #[global_allocator]
//! static ALLOCATOR: PeakTrackingAllocator = PeakTrackingAllocator;
//!
//! let (result, peak) = peak_bytes_during(|| read_thing(&hostile_input));
//! assert!(peak < LIMIT);
//! ```
//!
//! The mark is process-wide, so a concurrently running sibling test inflates
//! it. That is workable because the gap this measures is large: a correct read
//! of a kilobyte fixture touches kilobytes, while the defect asks for
//! gigabytes. Choose a threshold orders of magnitude above the honest cost and
//! orders below the defect's, and state the derivation at the assertion.

use mnemosyne::Mnemosyne;
use std::alloc::{GlobalAlloc, Layout};
use std::sync::atomic::{AtomicUsize, Ordering};

/// Live heap bytes across the test binary.
static LIVE_BYTES: AtomicUsize = AtomicUsize::new(0);

/// High-water mark of [`LIVE_BYTES`], rebased by [`peak_bytes_during`].
static PEAK_BYTES: AtomicUsize = AtomicUsize::new(0);

/// Allocator that records the high-water mark of live bytes.
///
/// Forwards every request to the system allocator unchanged; the counters are
/// the only added behaviour. Orderings are `Relaxed` because no test depends on
/// ordering between threads, only on the magnitude of the peak.
#[derive(Debug, Default, Clone, Copy)]
pub struct PeakTrackingAllocator;

// SAFETY: every method forwards to `Mnemosyne` with the caller's layout unchanged,
// so the allocator contract is whatever `Mnemosyne` already guarantees. The
// counters are side effects on atomics and affect no returned pointer.
unsafe impl GlobalAlloc for PeakTrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: `layout` is forwarded exactly as received.
        let ptr = unsafe { Mnemosyne.alloc(layout) };
        if !ptr.is_null() {
            let live = LIVE_BYTES.fetch_add(layout.size(), Ordering::Relaxed) + layout.size();
            PEAK_BYTES.fetch_max(live, Ordering::Relaxed);
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        LIVE_BYTES.fetch_sub(layout.size(), Ordering::Relaxed);
        // SAFETY: `ptr` and `layout` are forwarded exactly as received.
        unsafe { Mnemosyne.dealloc(ptr, layout) }
    }
}

/// Run `body` and return its value alongside the peak live-byte mark reached.
///
/// The mark is rebased to the current live total first, so the result measures
/// `body` rather than everything the binary has allocated so far.
///
/// Requires [`PeakTrackingAllocator`] to be installed as the test binary's
/// `#[global_allocator]`; without it the returned peak never moves.
pub fn peak_bytes_during<T>(body: impl FnOnce() -> T) -> (T, usize) {
    PEAK_BYTES.store(LIVE_BYTES.load(Ordering::Relaxed), Ordering::Relaxed);
    let value = body();
    (value, PEAK_BYTES.load(Ordering::Relaxed))
}
