//! VTK-style modification time (MTime) for lazy pipeline re-execution.
//!
//! # Mathematical Specification
//!
//! Let T: ℕ be a global monotonically increasing counter.
//! `ModifiedTime::tick()` atomically increments T and returns the new value.
//!
//! Re-execution invariant: given pipeline stage P with output mtime M_out and
//! input mtime M_in, P must re-execute iff M_in > M_out.  When M_in ≤ M_out,
//! the cached output is valid.
//!
//! Formally: `needs_update(dep_mtime) ⟺ dep_mtime > self.get_mtime()`.

use std::sync::atomic::{AtomicU64, Ordering};

static GLOBAL_MTIME: AtomicU64 = AtomicU64::new(0);

/// Monotonically increasing modification timestamp.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub struct ModifiedTime(u64);

impl ModifiedTime {
    /// The zero timestamp — before any modification.
    pub const ZERO: ModifiedTime = ModifiedTime(0);

    /// Returns the raw counter value.
    #[inline]
    pub fn value(self) -> u64 {
        self.0
    }

    /// Reconstruct a `ModifiedTime` from a raw counter value.
    ///
    /// This is intended for test infrastructure that needs to store and
    /// reload mtime values through atomic types. Production code should
    /// use `tick()` and `modified()` exclusively.
    #[inline]
    pub fn from_raw(raw: u64) -> Self {
        Self(raw)
    }

    /// Atomically increments the global counter and returns the new `ModifiedTime`.
    ///
    /// The returned value is strictly greater than all previously returned values.
    pub fn tick() -> Self {
        Self(GLOBAL_MTIME.fetch_add(1, Ordering::SeqCst) + 1)
    }
}

/// Trait for objects that track modification time.
///
/// Implementors store a `ModifiedTime` field and delegate `get_mtime`/`modified`
/// to it.  The default `needs_update` method encodes the re-execution invariant.
pub trait Modifiable {
    /// Returns the current modification time of this object.
    fn get_mtime(&self) -> ModifiedTime;

    /// Updates the object's modification time to a fresh tick.
    fn modified(&mut self);

    /// Returns `true` if `dependency_mtime` is strictly newer than this object's
    /// stored mtime, indicating the output is stale and must be recomputed.
    #[inline]
    fn needs_update(&self, dependency_mtime: ModifiedTime) -> bool {
        dependency_mtime > self.get_mtime()
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    struct Obj {
        mtime: ModifiedTime,
    }

    impl Obj {
        fn new() -> Self {
            Self {
                mtime: ModifiedTime::tick(),
            }
        }
    }

    impl Modifiable for Obj {
        fn get_mtime(&self) -> ModifiedTime {
            self.mtime
        }
        fn modified(&mut self) {
            self.mtime = ModifiedTime::tick();
        }
    }

    #[test]
    fn mtime_tick_strictly_increases() {
        let t0 = ModifiedTime::tick();
        let t1 = ModifiedTime::tick();
        let t2 = ModifiedTime::tick();
        assert!(t0 < t1, "t0({}) must be < t1({})", t0.value(), t1.value());
        assert!(t1 < t2, "t1({}) must be < t2({})", t1.value(), t2.value());
    }

    #[test]
    fn mtime_zero_is_smallest() {
        let any = ModifiedTime::tick();
        assert!(
            ModifiedTime::ZERO < any,
            "ZERO({}) must be < any tick({})",
            ModifiedTime::ZERO.value(),
            any.value()
        );
    }

    #[test]
    fn mtime_two_objects_have_distinct_mtimes() {
        let a = Obj::new();
        let b = Obj::new();
        assert_ne!(
            a.get_mtime(),
            b.get_mtime(),
            "distinct objects created sequentially must have distinct mtimes"
        );
    }

    #[test]
    fn mtime_modified_strictly_increments() {
        let mut obj = Obj::new();
        let before = obj.get_mtime();
        obj.modified();
        let after = obj.get_mtime();
        assert!(
            after > before,
            "mtime after modified() must exceed mtime before: before={}, after={}",
            before.value(),
            after.value()
        );
    }

    #[test]
    fn mtime_needs_update_true_when_dependency_is_newer() {
        // input created after output → input.mtime > output.mtime
        let output = Obj::new();
        let input = Obj::new();
        assert!(
            output.needs_update(input.get_mtime()),
            "output must need update when input was created (and thus stamped) after output"
        );
    }

    #[test]
    fn mtime_needs_update_false_after_re_execution() {
        let input = Obj::new();
        let mut output = Obj::new(); // created after input — but we simulate re-execution
        output.modified(); // output stamps itself after input
        assert!(
            !output.needs_update(input.get_mtime()),
            "output does not need update after modified() makes it newer than input"
        );
    }

    #[test]
    fn mtime_needs_update_false_for_equal_zero_dependency() {
        struct Frozen {
            mtime: ModifiedTime,
        }
        impl Modifiable for Frozen {
            fn get_mtime(&self) -> ModifiedTime {
                self.mtime
            }
            fn modified(&mut self) {
                self.mtime = ModifiedTime::tick();
            }
        }
        // Both at ZERO — not strictly greater, so no update needed.
        let frozen = Frozen {
            mtime: ModifiedTime::ZERO,
        };
        assert!(
            !frozen.needs_update(ModifiedTime::ZERO),
            "equal mtime must not trigger update"
        );
    }
}
