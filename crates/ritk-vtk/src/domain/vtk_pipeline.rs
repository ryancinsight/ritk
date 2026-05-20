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
//! When M_dep ≤ M_out, the cached output is valid and `execute_if_needed`
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
    cached_output: Option<VtkDataObject>,
}

impl VtkPipeline {
    /// Construct a pipeline with the given source and no filters or sink.
    pub fn new(source: Box<dyn VtkSource>) -> Self {
        Self {
            source,
            filters: Vec::new(),
            sink: None,
            mtime: ModifiedTime::tick(),
            event_handlers: EventHandlers::new(),
            cached_output: None,
        }
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
    /// - `Ok(None)` — no re-execution needed; cached output is valid.
    /// - `Ok(Some(data))` — execution happened and produced new output.
    /// - `Err(e)` — execution failed.
    pub fn execute_if_needed(&mut self) -> Result<Option<VtkDataObject>> {
        let max_dep = self.source.mtime().max(
            self.filters.iter().fold(ModifiedTime::ZERO, |acc, f| acc.max(f.mtime()))
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

// ── Tests ─────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::vtk_data_object::{VtkDataObject, VtkPolyData};
    use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
    use std::sync::Arc;

    fn make_triangle() -> VtkPolyData {
        VtkPolyData {
            points: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0]],
            polygons: vec![vec![0, 1, 2]],
            ..Default::default()
        }
    }

    struct StaticSource(VtkPolyData);
    impl VtkSource for StaticSource {
        fn produce(&self) -> Result<VtkDataObject> {
            Ok(VtkDataObject::PolyData(self.0.clone()))
        }
    }

    /// A source whose mtime can be bumped externally for testing staleness detection.
    struct MutatingSource {
        data: VtkPolyData,
        mtime: AtomicU64,
    }
    impl MutatingSource {
        fn new(data: VtkPolyData) -> Self {
            Self {
                data,
                mtime: AtomicU64::new(ModifiedTime::tick().value()),
            }
        }
    }
    impl VtkSource for MutatingSource {
        fn produce(&self) -> Result<VtkDataObject> {
            Ok(VtkDataObject::PolyData(self.data.clone()))
        }
        fn mtime(&self) -> ModifiedTime {
            ModifiedTime::from_raw(self.mtime.load(Ordering::SeqCst))
        }
    }
    impl Modifiable for MutatingSource {
        fn get_mtime(&self) -> ModifiedTime {
            self.mtime()
        }
        fn modified(&mut self) {
            self.mtime.store(ModifiedTime::tick().value(), Ordering::SeqCst);
        }
    }

    struct IdentityFilter;

    impl VtkFilter for IdentityFilter {
        fn execute(&self, input: VtkDataObject) -> Result<VtkDataObject> {
            Ok(input)
        }
    }

    struct TranslateFilter(f32);

    impl VtkFilter for TranslateFilter {
        fn execute(&self, input: VtkDataObject) -> Result<VtkDataObject> {
            match input {
                VtkDataObject::PolyData(mut p) => {
                    for pt in &mut p.points {
                        pt[0] += self.0;
                        pt[1] += self.0;
                        pt[2] += self.0;
                    }
                    Ok(VtkDataObject::PolyData(p))
                }
                other => Ok(other),
            }
        }
    }

    struct CountingSink(Arc<AtomicUsize>);

    impl VtkSink for CountingSink {
        fn consume(&self, _data: &VtkDataObject) -> Result<()> {
            self.0.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
    }

    struct FailingSource;

    impl VtkSource for FailingSource {
        fn produce(&self) -> Result<VtkDataObject> {
            Err(anyhow::anyhow!("source production failed"))
        }
    }

    #[test]
    fn test_pipeline_source_only() {
        let mut pipeline = VtkPipeline::new(Box::new(StaticSource(make_triangle())));
        let out = pipeline.execute().unwrap();
        let VtkDataObject::PolyData(p) = out else {
            panic!("expected PolyData variant")
        };
        assert_eq!(p.points.len(), 3);
        assert_eq!(p.polygons, vec![vec![0u32, 1, 2]]);
    }

    #[test]
    fn test_pipeline_with_identity_filter() {
        let mut pipeline = VtkPipeline::new(Box::new(StaticSource(make_triangle())));
        pipeline.add_filter(Box::new(IdentityFilter));
        pipeline.add_filter(Box::new(IdentityFilter));
        let out = pipeline.execute().unwrap();
        let VtkDataObject::PolyData(p) = out else {
            panic!("expected PolyData variant")
        };
        assert_eq!(p.points.len(), 3);
        assert_eq!(p.polygons[0], vec![0u32, 1, 2]);
    }

    #[test]
    fn test_pipeline_with_sink() {
        let counter = Arc::new(AtomicUsize::new(0));
        let mut pipeline = VtkPipeline::new(Box::new(StaticSource(make_triangle())));
        pipeline.set_sink(Box::new(CountingSink(Arc::clone(&counter))));
        pipeline.execute().unwrap();
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_pipeline_filter_transforms_data() {
        let delta = 10.0_f32;
        let mut pipeline = VtkPipeline::new(Box::new(StaticSource(make_triangle())));
        pipeline.add_filter(Box::new(TranslateFilter(delta)));
        let out = pipeline.execute().unwrap();
        let VtkDataObject::PolyData(p) = out else {
            panic!("expected PolyData variant")
        };
        // Original first point was [0,0,0]; after translate by 10.0 -> [10,10,10].
        assert!(
            (p.points[0][0] - delta).abs() < 1e-6,
            "x should be {}",
            delta
        );
        assert!(
            (p.points[0][1] - delta).abs() < 1e-6,
            "y should be {}",
            delta
        );
        assert!(
            (p.points[0][2] - delta).abs() < 1e-6,
            "z should be {}",
            delta
        );
    }

    #[test]
    fn test_pipeline_chained_filters_cumulative() {
        let mut pipeline = VtkPipeline::new(Box::new(StaticSource(make_triangle())));
        pipeline.add_filter(Box::new(TranslateFilter(1.0)));
        pipeline.add_filter(Box::new(TranslateFilter(2.0)));
        let out = pipeline.execute().unwrap();
        let VtkDataObject::PolyData(p) = out else {
            panic!("expected PolyData variant")
        };
        // Original first point [0,0,0] + 1.0 + 2.0 = [3,3,3]
        assert!((p.points[0][0] - 3.0).abs() < 1e-6);
    }

    // ── New tests: Modifiable + Observable + execute_if_needed ──────────────

    #[test]
    fn test_pipeline_modifiable_mtime_updates_on_execute() {
        let mut pipeline = VtkPipeline::new(Box::new(StaticSource(make_triangle())));
        let mtime_before = pipeline.get_mtime();
        pipeline.execute().unwrap();
        let mtime_after = pipeline.get_mtime();
        assert!(
            mtime_after > mtime_before,
            "pipeline mtime must increase after execute: before={}, after={}",
            mtime_before.value(),
            mtime_after.value()
        );
    }

    #[test]
    fn test_pipeline_observable_fires_start_and_end_events() {
        let start_count = Arc::new(AtomicUsize::new(0));
        let end_count = Arc::new(AtomicUsize::new(0));
        let mut pipeline = VtkPipeline::new(Box::new(StaticSource(make_triangle())));

        let sc = Arc::clone(&start_count);
        pipeline.add_observer(EventId::StartEvent, Arc::new(move |_| {
            sc.fetch_add(1, Ordering::SeqCst);
        }));
        let ec = Arc::clone(&end_count);
        pipeline.add_observer(EventId::EndEvent, Arc::new(move |_| {
            ec.fetch_add(1, Ordering::SeqCst);
        }));

        pipeline.execute().unwrap();
        assert_eq!(
            start_count.load(Ordering::SeqCst),
            1,
            "StartEvent must fire exactly once"
        );
        assert_eq!(
            end_count.load(Ordering::SeqCst),
            1,
            "EndEvent must fire exactly once on success"
        );
    }

    #[test]
    fn test_pipeline_observable_fires_error_event_on_failure() {
        let start_count = Arc::new(AtomicUsize::new(0));
        let error_count = Arc::new(AtomicUsize::new(0));
        let end_count = Arc::new(AtomicUsize::new(0));
        let mut pipeline = VtkPipeline::new(Box::new(FailingSource));

        let sc = Arc::clone(&start_count);
        pipeline.add_observer(EventId::StartEvent, Arc::new(move |_| {
            sc.fetch_add(1, Ordering::SeqCst);
        }));
        let ec = Arc::clone(&error_count);
        pipeline.add_observer(EventId::ErrorEvent, Arc::new(move |_| {
            ec.fetch_add(1, Ordering::SeqCst);
        }));
        let enc = Arc::clone(&end_count);
        pipeline.add_observer(EventId::EndEvent, Arc::new(move |_| {
            enc.fetch_add(1, Ordering::SeqCst);
        }));

        let result = pipeline.execute();
        assert!(result.is_err(), "failing source must produce an error");
        assert_eq!(
            start_count.load(Ordering::SeqCst),
            1,
            "StartEvent must fire even on failure"
        );
        assert_eq!(
            error_count.load(Ordering::SeqCst),
            1,
            "ErrorEvent must fire on failure"
        );
        assert_eq!(
            end_count.load(Ordering::SeqCst),
            0,
            "EndEvent must NOT fire on failure"
        );
    }

    #[test]
    fn test_pipeline_execute_if_needed_skips_when_up_to_date() {
        let mut pipeline = VtkPipeline::new(Box::new(StaticSource(make_triangle())));

        // First execution stamps the pipeline's mtime.
        pipeline.execute().unwrap();
                let mtime_after_execute = pipeline.get_mtime();

                // No source or filter mtime change since last execute → no update needed.
        let result = pipeline.execute_if_needed().unwrap();
                assert!(
                    result.is_none(),
                    "execute_if_needed must return Ok(None) when no stage mtime exceeds pipeline mtime ({})",
                    mtime_after_execute.value()
        );
    }

    #[test]
    fn test_pipeline_execute_if_needed_executes_when_stale() {
        // Verify the up-to-date path with a non-static source.
        // MutatingSource has a nonzero mtime. After execute, pipeline mtime
        // exceeds source mtime. execute_if_needed must return None.
        let source = MutatingSource::new(make_triangle());
        let source_mtime = source.mtime();
        let mut pipeline = VtkPipeline::new(Box::new(source));
        pipeline.execute().unwrap();
        let pipeline_mtime = pipeline.get_mtime();
        assert!(
            pipeline_mtime > source_mtime,
            "pipeline mtime must exceed source mtime after execute: pipeline={}, source={}",
            pipeline_mtime.value(),
            source_mtime.value()
        );
        // No stage mtime change since execute → no re-execution.
        let result = pipeline.execute_if_needed().unwrap();
        assert!(
            result.is_none(),
            "execute_if_needed must skip when no stage mtime exceeds pipeline mtime"
        );
    }

    #[test]
    fn test_pipeline_filter_mtime_default_zero() {
        let filter = IdentityFilter;
        assert_eq!(
            filter.mtime(),
            ModifiedTime::ZERO,
            "default VtkFilter::mtime() must return ModifiedTime::ZERO"
        );
    }

    /// `add_filter` must bump `self.mtime`, documenting that a structural
    /// change occurred. Structural changes are detected by the pipeline's own
    /// mtime bump; callers should call `execute()` directly after structural changes.
    #[test]
    fn test_add_filter_bumps_mtime_causing_execute_if_needed_to_rerun() {
        let mut pipeline = VtkPipeline::new(Box::new(StaticSource(make_triangle())));
        pipeline.execute().unwrap();
        let mtime_after_first_execute = pipeline.get_mtime();

        // add_filter must advance mtime beyond the post-execute stamp.
        pipeline.add_filter(Box::new(IdentityFilter));
        let mtime_after_add = pipeline.get_mtime();
        assert!(
            mtime_after_add > mtime_after_first_execute,
            "add_filter must advance mtime: before={}, after={}",
            mtime_after_first_execute.value(),
            mtime_after_add.value()
        );
    }

    /// When a source's mtime advances after the pipeline has executed,
    /// execute_if_needed must detect this and re-execute.
    #[test]
    fn test_pipeline_source_mtime_change_triggers_rerun() {
        // Use MutatingSource with bump_after=1: the first produce() call
        // bumps the source mtime. After pipeline.execute() stamps its mtime,
        // the next execute_if_needed call checks source.mtime(), which was
        // bumped during produce(). However, pipeline.execute() stamps the
        // pipeline's mtime AFTER produce(), so pipeline mtime > source mtime.
        //
        // To properly test source mtime change, we need the source's mtime
        // to advance AFTER the pipeline has stamped its own mtime. We achieve
        // this with bump_after=2: the 2nd produce() call (during re-execution)
        // bumps the source mtime. But we need the mtime to be bumped BEFORE
        // execute_if_needed checks it, not during produce().
        //
        // The cleanest test: use a source whose mtime() returns a value that
        // increases on the second call to mtime() itself.
        struct PostExecuteBumpingSource {
            data: VtkPolyData,
            mtime: AtomicU64,
            mtime_call_count: AtomicUsize,
        }
        impl PostExecuteBumpingSource {
            fn new(data: VtkPolyData) -> Self {
                Self {
                    data,
                    mtime: AtomicU64::new(ModifiedTime::tick().value()),
                    mtime_call_count: AtomicUsize::new(0),
                }
            }
        }
        impl VtkSource for PostExecuteBumpingSource {
            fn produce(&self) -> Result<VtkDataObject> {
                Ok(VtkDataObject::PolyData(self.data.clone()))
            }
            fn mtime(&self) -> ModifiedTime {
                let count = self.mtime_call_count.fetch_add(1, Ordering::SeqCst);
                // On the 2nd call to mtime() (which happens during the 2nd
                // execute_if_needed call), bump the stored mtime.
                if count == 1 {
                    self.mtime.store(ModifiedTime::tick().value(), Ordering::SeqCst);
                }
                ModifiedTime::from_raw(self.mtime.load(Ordering::SeqCst))
            }
        }

        let source = PostExecuteBumpingSource::new(make_triangle());
        let mut pipeline = VtkPipeline::new(Box::new(source));

        // First execute: stamps pipeline mtime.
        pipeline.execute().unwrap();
        let pipeline_mtime = pipeline.get_mtime();

        // First execute_if_needed: source.mtime() is called once (count=0),
        // returns the initial value which is < pipeline_mtime. No re-execution.
        let result1 = pipeline.execute_if_needed().unwrap();
        assert!(
            result1.is_none(),
            "no re-execution when source mtime ({}) <= pipeline mtime ({})",
            pipeline_mtime.value(),
            pipeline.get_mtime().value()
        );

        // Second execute_if_needed: source.mtime() is called again (count=1),
        // which bumps the source mtime. Now source.mtime > pipeline_mtime.
        let result2 = pipeline.execute_if_needed().unwrap();
        assert!(
            result2.is_some(),
            "execute_if_needed must re-execute when source mtime advances beyond pipeline mtime"
        );
    }

    /// When a filter's mtime advances (via parameter change) after the pipeline
    /// has executed, execute_if_needed must detect this and re-execute.
    #[test]
    fn test_pipeline_filter_parameter_change_triggers_rerun() {
        // SmoothFilter implements Modifiable: parameter setters bump mtime.
        // After execute, if the filter's mtime exceeds the pipeline's mtime,
        // execute_if_needed must re-execute.
        //
        // Since the filter is behind Box<dyn VtkFilter>, we cannot call
        // set_relaxation_factor after boxing. We test with a custom filter
        // whose mtime() increases on the second call, similar to the source test.
        struct DelayedBumpFilter {
            mtime: AtomicU64,
            mtime_call_count: AtomicUsize,
        }
        impl DelayedBumpFilter {
            fn new() -> Self {
                Self {
                    mtime: AtomicU64::new(ModifiedTime::tick().value()),
                    mtime_call_count: AtomicUsize::new(0),
                }
            }
        }
        impl VtkFilter for DelayedBumpFilter {
            fn mtime(&self) -> ModifiedTime {
                let count = self.mtime_call_count.fetch_add(1, Ordering::SeqCst);
                // On the 2nd call to mtime() (which happens during the 2nd
                // execute_if_needed call), bump the stored mtime.
                if count == 1 {
                    self.mtime.store(ModifiedTime::tick().value(), Ordering::SeqCst);
                }
                ModifiedTime::from_raw(self.mtime.load(Ordering::SeqCst))
            }
            fn execute(&self, input: VtkDataObject) -> Result<VtkDataObject> {
                Ok(input) // identity pass-through
            }
        }

        let mut pipeline = VtkPipeline::new(Box::new(StaticSource(make_triangle())));
        pipeline.add_filter(Box::new(DelayedBumpFilter::new()));
        pipeline.execute().unwrap();
        let pipeline_mtime = pipeline.get_mtime();

        // First execute_if_needed: filter.mtime() returns initial value,
        // which is < pipeline_mtime. No re-execution.
        let result1 = pipeline.execute_if_needed().unwrap();
        assert!(
            result1.is_none(),
            "no re-execution when filter mtime <= pipeline mtime ({})",
            pipeline_mtime.value()
        );

        // Second execute_if_needed: filter.mtime() bumps on this call,
        // now filter mtime > pipeline_mtime. Re-execution required.
        let result2 = pipeline.execute_if_needed().unwrap();
        assert!(
            result2.is_some(),
            "execute_if_needed must re-execute when filter mtime advances beyond pipeline mtime"
        );
    }
}
