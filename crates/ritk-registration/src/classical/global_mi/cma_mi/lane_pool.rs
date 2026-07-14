use std::ops::Deref;
use std::sync::{Condvar, Mutex, MutexGuard};

/// Bounded pool of independent stateful evaluation lanes.
///
/// A lease removes one lane from the pool, so no two evaluations can access
/// the same state concurrently. Dropping the lease returns the lane even while
/// unwinding, and the condition variable bounds waiting evaluators without
/// spinning.
pub(super) struct LanePool<T> {
    available: Mutex<Vec<T>>,
    ready: Condvar,
}

impl<T> LanePool<T> {
    pub(super) fn new(lanes: impl IntoIterator<Item = T>) -> Self {
        let available: Vec<T> = lanes.into_iter().collect();
        assert!(
            !available.is_empty(),
            "a lane pool requires at least one lane"
        );
        Self {
            available: Mutex::new(available),
            ready: Condvar::new(),
        }
    }

    pub(super) fn lease(&self) -> LaneLease<'_, T> {
        let mut available = lock_unpoisoned(&self.available);
        loop {
            if let Some(lane) = available.pop() {
                return LaneLease {
                    lane: Some(lane),
                    pool: self,
                };
            }
            available = self
                .ready
                .wait(available)
                .unwrap_or_else(|poisoned| poisoned.into_inner());
        }
    }
}

pub(super) struct LaneLease<'pool, T> {
    lane: Option<T>,
    pool: &'pool LanePool<T>,
}

impl<T> Deref for LaneLease<'_, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.lane
            .as_ref()
            .expect("invariant: a live lane lease owns its lane")
    }
}

impl<T> Drop for LaneLease<'_, T> {
    fn drop(&mut self) {
        let lane = self
            .lane
            .take()
            .expect("invariant: a lane lease returns its lane exactly once");
        lock_unpoisoned(&self.pool.available).push(lane);
        self.pool.ready.notify_one();
    }
}

fn lock_unpoisoned<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Barrier;

    use super::LanePool;

    #[test]
    fn leases_each_lane_to_at_most_one_concurrent_evaluation() {
        const LANE_COUNT: usize = 3;
        const EVALUATION_COUNT: usize = 12;
        let active = std::array::from_fn::<_, LANE_COUNT, _>(|_| AtomicUsize::new(0));
        let visits = std::array::from_fn::<_, LANE_COUNT, _>(|_| AtomicUsize::new(0));
        let pool = LanePool::new(0..LANE_COUNT);
        let start = Barrier::new(EVALUATION_COUNT);
        let leased = Barrier::new(LANE_COUNT);

        std::thread::scope(|scope| {
            for _ in 0..EVALUATION_COUNT {
                scope.spawn(|| {
                    start.wait();
                    let lane = pool.lease();
                    let lane_index = *lane;
                    assert_eq!(active[lane_index].fetch_add(1, Ordering::AcqRel), 0);
                    visits[lane_index].fetch_add(1, Ordering::Relaxed);
                    leased.wait();
                    assert_eq!(active[lane_index].fetch_sub(1, Ordering::AcqRel), 1);
                });
            }
        });

        assert_eq!(
            visits
                .iter()
                .map(|count| count.load(Ordering::Relaxed))
                .sum::<usize>(),
            EVALUATION_COUNT
        );
        assert_eq!(super::lock_unpoisoned(&pool.available).len(), LANE_COUNT);
    }

    #[test]
    fn unwinding_returns_the_leased_lane() {
        let pool = LanePool::new([7]);
        let unwind = std::panic::catch_unwind(|| {
            let lane = pool.lease();
            assert_eq!(*lane, 7);
            panic!("exercise lane lease unwinding");
        });

        assert!(unwind.is_err());
        assert_eq!(*pool.lease(), 7);
    }
}
