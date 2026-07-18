//! VTK-style data-flow pipeline abstractions.
//!
//! # Architecture
//!
//! The pipeline follows the VTK data-flow model:
//! Source -> [Filter1 -> Filter2 -> ...] -> optional Sink
//!
//! Each stage is a trait object enabling runtime composition without generic proliferation.
//!
//! # Mathematical Specification
//!
//! Let D0 = Source::produce(). For k filters, Dn = Filterk::execute(Dn-1), n = 1..k.
//! The final Dk is returned by `VtkPipeline::execute()` and forwarded to the sink if present.
//! A filter must preserve or document which `VtkDataObject` variant it accepts and emits.
//!
//! # Modification-Time Re-execution Invariant
//!
//! Given pipeline P with output mtime M_out and the maximum mtime M_dep of
//! its source and filter stages, P must re-execute iff M_dep > M_out.
//! When M_dep â‰¤ M_out, the cached output is valid and `execute_if_needed`
//! returns `Ok(None)`.
//!
//! # Event Notification Invariant
//!
//! On execution, P fires `StartEvent` before the first stage and `EndEvent` after
//! successful completion. On failure, `ErrorEvent` is fired instead of `EndEvent`.

use super::mtime::{Modifiable, ModifiedTime};
use super::observer::{EventHandlers, EventId, Observable};
use super::vtk_data_object::VtkDataObject;
use anyhow::Result;
use std::any::Any;

/// Produces a `VtkDataObject` from an external source (file, procedural generator).
pub trait VtkSource: Send + Sync {
    /// Generate the initial data object.
    fn produce(&self) -> Result<VtkDataObject>;

    /// Returns the modification time of this source.
    ///
    /// The default implementation returns `ModifiedTime::ZERO`, meaning the
    /// source never forces re-execution through its own mtime. Sources that
    /// can change after construction must override this.
    fn mtime(&self) -> ModifiedTime {
        ModifiedTime::ZERO
    }
}

/// Transforms one `VtkDataObject` into another.
///
/// Implementations must document which variants they accept and which they produce.
pub trait VtkFilter: Send + Sync {
    /// Apply the filter transformation.
    fn execute(&self, input: VtkDataObject) -> Result<VtkDataObject>;

    /// Returns the modification time of this filter, if it tracks one.
    ///
    /// The default implementation returns `ModifiedTime::ZERO`, meaning the
    /// filter never forces re-execution through its own mtime.
    fn mtime(&self) -> ModifiedTime {
        ModifiedTime::ZERO
    }

    /// Returns `Some(&mut dyn Any)` for downcasting to the concrete filter type
    /// for runtime parameter mutation through the trait object boundary.
    ///
    /// Stateless filters (e.g., `ComputeNormalsFilter`) return `None`.
    /// Stateful filters with mutable parameters override this to return `Some(self)`.
    ///
    /// # Usage
    ///
    /// ```ignore
    /// if let Some(sf) = filter.as_any_mut().and_then(|a| a.downcast_mut::<SmoothFilter>()) {
    ///     sf.set_relaxation_factor(0.8);
    /// }
    /// ```
    fn as_any_mut(&mut self) -> Option<&mut dyn Any> {
        None
    }
}

/// Consumes a `VtkDataObject` (writes to disk, renders, accumulates statistics, etc.).
pub trait VtkSink: Send + Sync {
    /// Process the final data object.
    fn consume(&self, data: &VtkDataObject) -> Result<()>;
}

/// An ordered source -> filters -> sink data-flow pipeline.
///
/// Implements `Modifiable` for lazy re-execution and `Observable` for event
/// notification (StartEvent, EndEvent, ErrorEvent).
pub struct VtkPipeline {
    source: Box<dyn VtkSource>,
    filters: Vec<Box<dyn VtkFilter>>,
    sink: Option<Box<dyn VtkSink>>,
    mtime: ModifiedTime,
    event_handlers: EventHandlers,
    cached_output: Option<VtkDataObject> }

impl VtkPipeline {
    /// Construct a pipeline with the given source and no filters or sink.
    pub fn new(source: Box<dyn VtkSource>) -> Self {
        Self {
            source,
            filters: Vec::new(),
            sink: None,
            mtime: ModifiedTime::tick(),
            event_handlers: EventHandlers::new(),
            cached_output: None }
    }

    /// Append a filter stage. Returns `&mut Self` for chaining.
    ///
    /// Bumps `self.mtime` so that a subsequent `execute_if_needed` call
    /// re-executes the pipeline.
    pub fn add_filter(&mut self, filter: Box<dyn VtkFilter>) -> &mut Self {
        self.filters.push(filter);
        self.modified();
        self
    }

    /// Set the terminal sink. Returns `&mut Self` for chaining.
    ///
    /// Bumps `self.mtime` for the same reason as `add_filter`.
    pub fn set_sink(&mut self, sink: Box<dyn VtkSink>) -> &mut Self {
        self.sink = Some(sink);
        self.modified();
        self
    }

    /// Returns a mutable reference to the boxed filter at `index`, if present.
    ///
    /// This exposes the stored boxed trait object so callers can downcast through
    /// `VtkFilter::as_any_mut()` and mutate filter parameters in place.
    pub fn filter_mut(&mut self, index: usize) -> Option<&mut Box<dyn VtkFilter>> {
        self.filters.get_mut(index)
    }

    /// Execute the full pipeline.
    ///
    /// Fires `StartEvent` before execution begins. On success, fires `EndEvent`,
    /// caches the output, and stamps `self.modified()`. On failure, fires
    /// `ErrorEvent` instead of `EndEvent`.
    pub fn execute(&mut self) -> Result<VtkDataObject> {
        self.invoke_event(EventId::StartEvent);
        match self.execute_inner() {
            Ok(data) => {
                self.cached_output = Some(data.clone());
                self.modified();
                self.invoke_event(EventId::EndEvent);
                Ok(data)
            }
            Err(e) => {
                self.invoke_event(EventId::ErrorEvent);
                Err(e)
            }
        }
    }

    /// Internal execution logic: produce -> filter chain -> sink.
    fn execute_inner(&self) -> Result<VtkDataObject> {
        let mut data = self.source.produce()?;
        for filter in &self.filters {
            data = filter.execute(data)?;
        }
        if let Some(sink) = &self.sink {
            sink.consume(&data)?;
        }
        Ok(data)
    }

    /// Conditionally execute the pipeline based on modification-time staleness.
    ///
    /// Computes the maximum mtime among the source and all filter stages.
    /// If `self.needs_update(max_dep)`, calls `execute()` internally.
    ///
    /// # Returns
    ///
    /// - `Ok(None)` â€” no re-execution needed; cached output is valid.
    /// - `Ok(Some(data))` â€” execution happened and produced new output.
    /// - `Err(e)` â€” execution failed.
    pub fn execute_if_needed(&mut self) -> Result<Option<VtkDataObject>> {
        let max_dep = self.source.mtime().max(
            self.filters
                .iter()
                .fold(ModifiedTime::ZERO, |acc, f| acc.max(f.mtime())),
        );
        if self.needs_update(max_dep) {
            self.execute().map(Some)
        } else {
            Ok(None)
        }
    }
}

impl Modifiable for VtkPipeline {
    fn get_mtime(&self) -> ModifiedTime {
        self.mtime
    }
    fn modified(&mut self) {
        self.mtime = ModifiedTime::tick();
    }
}

impl Observable for VtkPipeline {
    fn event_handlers(&self) -> &EventHandlers {
        &self.event_handlers
    }
    fn event_handlers_mut(&mut self) -> &mut EventHandlers {
        &mut self.event_handlers
    }
}

#[cfg(test)]
mod tests;
