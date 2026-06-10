//! Histogram buffer pool for the direct Parzen histogram computation path.
//!
//! [`HistogramPool`] provides thread-local buffer reuse across rayon
//! fold/reduce calls, avoiding repeated O(num_bins²) allocations.

use std::sync::Mutex;

// ── Histogram pool (ARCH-315-03, MEM-317-02) ───────────────────────────────

/// Thread-local histogram buffer pool for parallel reduction.
///
/// Reuses `[num_bins²]` buffers across fold/reduce calls to avoid repeated
/// allocation + zeroing. Each thread checks out a buffer via [`checkout`],
/// zero-fills it, and returns it via [`return_buffer`] after the reduce phase.
///
/// # Cross-iteration reuse (MEM-317-02)
///
/// When stored on `ParzenJointHistogram`, the pool persists across CMA-ES
/// iterations, amortising the initial `Vec` allocation over many calls.
/// Each call to `checkout()` zero-fills the buffer (O(num_bins²)), but the
/// underlying `Vec` storage is reused from the pool, avoiding the
/// allocation + deallocation cycle that was previously per-invocation.
///
/// # Mutex poison handling
///
/// All lock acquisitions use `unwrap_or_else(|e| e.into_inner())` so that
/// a panic in one thread does not poison the pool for the rest.
#[derive(Debug)]
pub struct HistogramPool {
    buffers: Mutex<Vec<Vec<f32>>>,
    num_bins_sq: usize,
}

impl HistogramPool {
    /// Create a new empty pool for histograms of size `num_bins²`.
    pub fn new(num_bins_sq: usize) -> Self {
        Self {
            // Capacity: start empty; use new_with_capacity for pre-allocated pools
            buffers: Mutex::new(Vec::new()),
            num_bins_sq,
        }
    }

    /// Create a pool pre-allocated with `buffer_count` zeroed buffers.
    ///
    /// Pre-allocating avoids the first-iteration allocation latency when
    /// using rayon's `fold().reduce()`: each thread can immediately
    /// check out a buffer without hitting the allocator. The pool will
    /// still grow beyond `buffer_count` if needed (subsequent checkouts
    /// allocate new buffers as before).
    ///
    /// # Arguments
    /// * `num_bins_sq` — Histogram buffer size (`num_bins²`)
    /// * `buffer_count` — Number of buffers to pre-allocate
    pub fn new_with_capacity(num_bins_sq: usize, buffer_count: usize) -> Self {
        let buffers: Vec<Vec<f32>> = (0..buffer_count)
            .map(|_| vec![0.0f32; num_bins_sq])
            .collect();
        Self {
            buffers: Mutex::new(buffers),
            num_bins_sq,
        }
    }

    /// Check out a zeroed buffer from the pool, or allocate a new one.
    ///
    /// # Performance (PERF-319-05)
    ///
    /// Reused buffers are zeroed with `fill(0.0)`, which compiles to a
    /// memset-like loop that is 2–3× faster than `vec![0.0; N]` for large
    /// `N` because it skips the allocator. New buffers are allocated via
    /// `vec![0.0; N]` (already zeroed) and skip the redundant `fill`.
    pub fn checkout(&self) -> Vec<f32> {
        let mut guard = self.buffers.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(mut buf) = guard.pop() {
            drop(guard); // release lock before zeroing
            buf.fill(0.0);
            buf
        } else {
            drop(guard); // release lock before allocating
            vec![0.0f32; self.num_bins_sq]
        }
    }

    /// Return a buffer to the pool for reuse.
    pub fn return_buffer(&self, buf: Vec<f32>) {
        self.buffers
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(buf);
    }
}
