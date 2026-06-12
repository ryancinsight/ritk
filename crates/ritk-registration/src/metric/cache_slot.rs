//! Single-slot lazy cache used by metric implementations.
//!
//! Encapsulates the `Arc<Mutex<Option<T>>>` pattern that appears in
//! `MutualInformation`, `ParzenJointHistogram`, and `LnccMetric` caches.
//! Provides `get_or_init`, `invalidate`, and `is_populated` so callers
//! express intent rather than locking mechanics.
use std::sync::{Arc, Mutex};

/// A single-slot lazy cache: `Arc<Mutex<Option<T>>>`.
///
/// - `get_or_init` populates the slot on first call and returns a clone on
///   subsequent calls — suitable for read-heavy metric caches.
/// - `invalidate` clears the slot so the next `get_or_init` recomputes.
/// - `is_populated` is a non-mutating probe.
///
/// `Clone` performs a shallow clone (shared underlying `Arc`), so two
/// `CacheSlot<T>` clones refer to the same slot — consistent with the
/// existing `Arc`-sharing semantics in metric structs.
#[derive(Clone, Debug)]
pub(crate) struct CacheSlot<T: Clone>(Arc<Mutex<Option<T>>>);

impl<T: Clone> CacheSlot<T> {
    /// Creates an empty (unpopulated) slot.
    pub(crate) fn empty() -> Self {
        Self(Arc::new(Mutex::new(None)))
    }

    /// Returns the cached value if present; otherwise calls `init`, stores
    /// the result, and returns a clone of it.
    ///
    /// # Panics
    /// Panics if the mutex has been poisoned (invariant: never poison the lock).
    pub(crate) fn get_or_init<F: FnOnce() -> T>(&self, init: F) -> T {
        let mut guard = self
            .0
            .lock()
            .expect("invariant: CacheSlot mutex is not poisoned");
        if guard.is_none() {
            *guard = Some(init());
        }
        // SAFETY: guard is Some after the block above.
        guard.clone().unwrap()
    }

    /// Returns the cached value if present **and** valid; otherwise (re-)initialises.
    ///
    /// If the slot is populated but `valid` returns `false`, the stale value is
    /// discarded. Then `init` is called (outside the lock) and the result is
    /// stored. Concurrent reinitializations are possible if two threads both
    /// observe a stale value; last-writer-wins, which is correct for deterministic
    /// caches where every thread computes the same result.
    ///
    /// # Panics
    /// Panics if the mutex has been poisoned.
    pub(crate) fn get_or_reinit_if<F, P>(&self, valid: P, init: F) -> T
    where
        F: FnOnce() -> T,
        P: FnOnce(&T) -> bool,
    {
        {
            let guard = self
                .0
                .lock()
                .expect("invariant: CacheSlot mutex is not poisoned");
            if let Some(ref v) = *guard {
                if valid(v) {
                    return v.clone();
                }
            }
        }
        // Either empty or the validity predicate rejected the cached entry.
        let val = init();
        *self
            .0
            .lock()
            .expect("invariant: CacheSlot mutex is not poisoned") = Some(val.clone());
        val
    }

    /// Clears the cached value; the next `get_or_init` will recompute.
    ///
    /// # Panics
    /// Panics if the mutex has been poisoned.
    pub(crate) fn invalidate(&self) {
        *self
            .0
            .lock()
            .expect("invariant: CacheSlot mutex is not poisoned") = None;
    }

    /// Returns `true` if a value has been cached (without mutating the slot).
    pub(crate) fn is_populated(&self) -> bool {
        self.0
            .lock()
            .expect("invariant: CacheSlot mutex is not poisoned")
            .is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_slot_is_unpopulated() {
        let slot: CacheSlot<u32> = CacheSlot::empty();
        assert!(!slot.is_populated());
    }

    #[test]
    fn get_or_init_populates_on_first_call() {
        let slot: CacheSlot<u32> = CacheSlot::empty();
        let v = slot.get_or_init(|| 42);
        assert_eq!(v, 42);
        assert!(slot.is_populated());
    }

    #[test]
    fn get_or_init_does_not_reinit_on_subsequent_calls() {
        let slot: CacheSlot<u32> = CacheSlot::empty();
        slot.get_or_init(|| 1);
        let v = slot.get_or_init(|| 999); // init not called again
        assert_eq!(v, 1, "cached value must be returned, not re-initialized");
    }

    #[test]
    fn invalidate_clears_slot() {
        let slot: CacheSlot<u32> = CacheSlot::empty();
        slot.get_or_init(|| 7);
        slot.invalidate();
        assert!(!slot.is_populated());
        let v = slot.get_or_init(|| 99);
        assert_eq!(v, 99);
    }

    #[test]
    fn clone_shares_underlying_slot() {
        let slot: CacheSlot<u32> = CacheSlot::empty();
        let clone = slot.clone();
        slot.get_or_init(|| 5);
        assert!(clone.is_populated(), "clone must see the populated slot");
    }
}
