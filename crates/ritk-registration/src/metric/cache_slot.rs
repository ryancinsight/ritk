//! Single-slot lazy cache used by metric implementations.
//!
//! Encapsulates the `Arc<Mutex<Option<T>>>` pattern that appears in
//! `MutualInformation`, `ParzenJointHistogram`, and `LnccMetric` caches.
//! Provides `get_or_init`, `invalidate`, and `is_populated` so callers
//! express intent rather than locking mechanics.
use std::sync::{Arc, RwLock};

/// A single-slot lazy cache: `Arc<RwLock<Option<T>>>`.
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
pub(crate) struct CacheSlot<T: Clone>(Arc<RwLock<Option<T>>>);

impl<T: Clone> CacheSlot<T> {
    /// Creates an empty (unpopulated) slot.
    pub(crate) fn empty() -> Self {
        Self(Arc::new(RwLock::new(None)))
    }

    /// Returns the cached value if present; otherwise calls `init`, stores
    /// the result, and returns a clone of it.
    ///
    /// # Panics
    /// Panics if the RwLock has been poisoned (invariant: never poison the lock).
    #[allow(dead_code)]
    pub(crate) fn get_or_init<F: FnOnce() -> T>(&self, init: F) -> T {
        {
            let guard = self
                .0
                .read()
                .expect("invariant: CacheSlot rwlock is not poisoned");
            if let Some(ref v) = *guard {
                return v.clone();
            }
        }
        let mut guard = self
            .0
            .write()
            .expect("invariant: CacheSlot rwlock is not poisoned");
        if guard.is_none() {
            *guard = Some(init());
        }
        // SAFETY: guard is Some after the check/initialization above.
        guard
            .clone()
            .expect("invariant: cached value is always Some after get_or_init populates it")
    }

    /// Returns the cached value if present **and** valid; otherwise (re-)initialises.
    ///
    /// If the slot is empty or the cached value is stale (i.e. `valid` returns `false`),
    /// `init` is evaluated while holding the write lock. This enforces serialisation
    /// and prevents concurrent threads from executing the expensive initialization code
    /// redundantly on a cache miss.
    ///
    /// # Panics
    /// Panics if the RwLock has been poisoned.
    pub(crate) fn get_or_reinit_if<F, P>(&self, mut valid: P, init: F) -> T
    where
        F: FnOnce() -> T,
        P: FnMut(&T) -> bool,
    {
        {
            let guard = self
                .0
                .read()
                .expect("invariant: CacheSlot rwlock is not poisoned");
            if let Some(ref v) = *guard {
                if valid(v) {
                    return v.clone();
                }
            }
        }
        // Either empty or the validity predicate rejected the cached entry.
        let mut guard = self
            .0
            .write()
            .expect("invariant: CacheSlot rwlock is not poisoned");
        if let Some(ref v) = *guard {
            if valid(v) {
                return v.clone();
            }
        }
        let val = init();
        *guard = Some(val.clone());
        val
    }

    /// Clears the cached value; the next `get_or_init` will recompute.
    ///
    /// # Panics
    /// Panics if the RwLock has been poisoned.
    pub(crate) fn invalidate(&self) {
        *self
            .0
            .write()
            .expect("invariant: CacheSlot rwlock is not poisoned") = None;
    }

    /// Apply `f` to a shared reference to the cached `Option<T>`, returning `f`'s result.
    ///
    /// Holds the read lock only for the duration of `f`.
    ///
    /// # Panics
    /// Panics if the RwLock has been poisoned.
    pub(crate) fn with_ref<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&Option<T>) -> R,
    {
        f(&self
            .0
            .read()
            .expect("invariant: CacheSlot rwlock is not poisoned"))
    }

    /// Apply `f` to an exclusive reference to the cached `Option<T>`, returning `f`'s result.
    ///
    /// Holds the write lock only for the duration of `f`.
    ///
    /// # Panics
    /// Panics if the RwLock has been poisoned.
    pub(crate) fn with_mut<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&mut Option<T>) -> R,
    {
        f(&mut self
            .0
            .write()
            .expect("invariant: CacheSlot rwlock is not poisoned"))
    }

    /// Returns `true` if a value has been cached (without mutating the slot).
    #[allow(dead_code)]
    pub(crate) fn is_populated(&self) -> bool {
        self.0
            .read()
            .expect("invariant: CacheSlot rwlock is not poisoned")
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
