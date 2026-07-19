//! VTK-style observer/event notification system.
//!
//! # Mathematical Specification
//!
//! Let O = { (tag, event, callback) } be the observer registry.
//! `add_observer(event, cb)` appends a new entry with a fresh unique tag τ
//! and returns τ.  Tag uniqueness invariant: τ ≠ τ' for all distinct
//! `add_observer` calls on the same `EventHandlers` instance.
//! `remove_observer(tag)` removes the unique entry with the given tag (O(n)).
//! `invoke_event(event)` calls cb(event) for every entry where event' == event,
//! in registration order.
//!
//! Thread-safety: `EventHandlers` is `Send + Sync` because callbacks are
//! `Arc<dyn Fn(EventId) + Send + Sync>`.

use std::sync::Arc;

/// Discriminant for VTK pipeline events.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EventId {
    /// An object's internal state changed (most common event).
    Modified,
    /// A pipeline algorithm is about to begin execution.
    StartEvent,
    /// A pipeline algorithm finished execution.
    EndEvent,
    /// Progress update during a long-running execution.
    ProgressEvent,
    /// A recoverable error occurred inside a pipeline stage.
    ErrorEvent,
    /// A non-fatal warning was issued by a pipeline stage.
    WarningEvent,
    /// An actor was picked (selected interactively) in the scene.
    PickEvent,
    /// The renderer completed a frame.
    RenderEvent,
}

/// Opaque registration handle returned by `add_observer`.
///
/// Use it with `remove_observer` to deregister a specific callback.
pub type ObserverTag = u64;

/// Type-erased, thread-safe callback that receives the fired `EventId`.
pub type ObserverCallback = Arc<dyn Fn(EventId) + Send + Sync>;

/// Registry of event handlers attached to a single observable object.
///
/// `EventHandlers` is `Default` (empty registry) and cheaply constructible.
#[derive(Default)]
pub struct EventHandlers {
    next_tag: ObserverTag,
    handlers: Vec<(ObserverTag, EventId, ObserverCallback)>,
}

impl EventHandlers {
    /// Construct an empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register `cb` to be called when `event` fires.
    ///
    /// Returns an `ObserverTag` that uniquely identifies this registration and
    /// can be passed to `remove_observer` to deregister it.
    pub fn add_observer(&mut self, event: EventId, cb: ObserverCallback) -> ObserverTag {
        self.next_tag += 1;
        let tag = self.next_tag;
        self.handlers.push((tag, event, cb));
        tag
    }

    /// Deregister the observer with the given `tag`.
    ///
    /// Returns `true` if the tag was found and removed; `false` if it was not
    /// present (already removed or never registered).
    pub fn remove_observer(&mut self, tag: ObserverTag) -> bool {
        if let Some(pos) = self.handlers.iter().position(|(t, _, _)| *t == tag) {
            self.handlers.remove(pos);
            true
        } else {
            false
        }
    }

    /// Call all callbacks registered for `event`, in registration order.
    ///
    /// Callbacks registered for other events are not called.
    pub fn invoke_event(&self, event: EventId) {
        for (_, registered_event, cb) in &self.handlers {
            if *registered_event == event {
                cb(event);
            }
        }
    }

    /// Total number of registered observers across all event types.
    pub fn observer_count(&self) -> usize {
        self.handlers.len()
    }
}

/// Trait implemented by objects whose state changes can be observed.
///
/// Provides default implementations of `add_observer`, `remove_observer`, and
/// `invoke_event` by delegating to the object's `EventHandlers`.
pub trait Observable {
    fn event_handlers(&self) -> &EventHandlers;
    fn event_handlers_mut(&mut self) -> &mut EventHandlers;

    /// Register an observer for `event`.  Returns a registration tag.
    fn add_observer(&mut self, event: EventId, cb: ObserverCallback) -> ObserverTag {
        self.event_handlers_mut().add_observer(event, cb)
    }

    /// Remove an observer by tag.  Returns `true` if found and removed.
    fn remove_observer(&mut self, tag: ObserverTag) -> bool {
        self.event_handlers_mut().remove_observer(tag)
    }

    /// Fire all callbacks registered for `event`.
    fn invoke_event(&self, event: EventId) {
        self.event_handlers().invoke_event(event);
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn counter_cb(counter: Arc<AtomicUsize>) -> ObserverCallback {
        Arc::new(move |_event| {
            counter.fetch_add(1, Ordering::SeqCst);
        })
    }

    #[test]
    fn add_observer_returns_unique_tags() {
        let mut h = EventHandlers::new();
        let t1 = h.add_observer(EventId::Modified, counter_cb(Arc::new(AtomicUsize::new(0))));
        let t2 = h.add_observer(EventId::Modified, counter_cb(Arc::new(AtomicUsize::new(0))));
        assert_ne!(t1, t2, "consecutive tags must be distinct");
    }

    #[test]
    fn invoke_event_fires_all_matching_callbacks() {
        let counter = Arc::new(AtomicUsize::new(0));
        let mut h = EventHandlers::new();
        h.add_observer(EventId::Modified, counter_cb(Arc::clone(&counter)));
        h.add_observer(EventId::Modified, counter_cb(Arc::clone(&counter)));
        h.invoke_event(EventId::Modified);
        assert_eq!(
            counter.load(Ordering::SeqCst),
            2,
            "both observers must fire"
        );
    }

    #[test]
    fn invoke_event_does_not_fire_mismatched_event() {
        let counter = Arc::new(AtomicUsize::new(0));
        let mut h = EventHandlers::new();
        h.add_observer(EventId::StartEvent, counter_cb(Arc::clone(&counter)));
        h.invoke_event(EventId::Modified); // different event
        assert_eq!(
            counter.load(Ordering::SeqCst),
            0,
            "mismatched event must not trigger callback"
        );
    }

    #[test]
    fn remove_observer_stops_callback_from_firing() {
        let counter = Arc::new(AtomicUsize::new(0));
        let mut h = EventHandlers::new();
        let tag = h.add_observer(EventId::EndEvent, counter_cb(Arc::clone(&counter)));
        h.invoke_event(EventId::EndEvent);
        assert_eq!(counter.load(Ordering::SeqCst), 1, "fires before removal");
        assert!(h.remove_observer(tag), "tag must be found");
        h.invoke_event(EventId::EndEvent);
        assert_eq!(
            counter.load(Ordering::SeqCst),
            1,
            "must not fire after removal"
        );
    }

    #[test]
    fn remove_observer_unknown_tag_returns_false() {
        let mut h = EventHandlers::new();
        assert!(!h.remove_observer(9999), "unknown tag must return false");
    }

    #[test]
    fn observer_count_reflects_additions_and_removals() {
        let mut h = EventHandlers::new();
        assert_eq!(h.observer_count(), 0);
        let t1 = h.add_observer(EventId::Modified, counter_cb(Arc::new(AtomicUsize::new(0))));
        h.add_observer(
            EventId::RenderEvent,
            counter_cb(Arc::new(AtomicUsize::new(0))),
        );
        assert_eq!(h.observer_count(), 2);
        h.remove_observer(t1);
        assert_eq!(h.observer_count(), 1);
    }

    #[test]
    fn invoke_event_passes_correct_event_id_to_callback() {
        use std::sync::Mutex;
        let last: Arc<Mutex<Option<EventId>>> = Arc::new(Mutex::new(None));
        let last_clone = Arc::clone(&last);
        let cb: ObserverCallback = Arc::new(move |ev| {
            *last_clone.lock().unwrap() = Some(ev);
        });
        let mut h = EventHandlers::new();
        h.add_observer(EventId::ProgressEvent, cb);
        h.invoke_event(EventId::ProgressEvent);
        assert_eq!(
            *last.lock().unwrap(),
            Some(EventId::ProgressEvent),
            "callback must receive the fired EventId"
        );
    }

    #[test]
    fn invoke_event_fires_in_registration_order() {
        use std::sync::Mutex;
        let order: Arc<Mutex<Vec<u8>>> = Arc::new(Mutex::new(Vec::new()));
        let mut h = EventHandlers::new();
        for id in 0u8..4 {
            let order_clone = Arc::clone(&order);
            h.add_observer(
                EventId::Modified,
                Arc::new(move |_| {
                    order_clone.lock().unwrap().push(id);
                }),
            );
        }
        h.invoke_event(EventId::Modified);
        assert_eq!(
            *order.lock().unwrap(),
            vec![0u8, 1, 2, 3],
            "callbacks must fire in registration order"
        );
    }
}
