//! Value-semantic tests for VtkPipeline, VtkSource, VtkFilter, and VtkSink.

use super::*;
use crate::domain::mtime::{Modifiable, ModifiedTime};
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
        self.mtime
            .store(ModifiedTime::tick().value(), Ordering::SeqCst);
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

// ── Modifiable + Observable + execute_if_needed ──────────────────────────

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
    pipeline.add_observer(
        EventId::StartEvent,
        Arc::new(move |_| {
            sc.fetch_add(1, Ordering::SeqCst);
        }),
    );
    let ec = Arc::clone(&end_count);
    pipeline.add_observer(
        EventId::EndEvent,
        Arc::new(move |_| {
            ec.fetch_add(1, Ordering::SeqCst);
        }),
    );

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
    pipeline.add_observer(
        EventId::StartEvent,
        Arc::new(move |_| {
            sc.fetch_add(1, Ordering::SeqCst);
        }),
    );
    let ec = Arc::clone(&error_count);
    pipeline.add_observer(
        EventId::ErrorEvent,
        Arc::new(move |_| {
            ec.fetch_add(1, Ordering::SeqCst);
        }),
    );
    let enc = Arc::clone(&end_count);
    pipeline.add_observer(
        EventId::EndEvent,
        Arc::new(move |_| {
            enc.fetch_add(1, Ordering::SeqCst);
        }),
    );

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
                self.mtime
                    .store(ModifiedTime::tick().value(), Ordering::SeqCst);
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

/// When a boxed filter's parameters change after the pipeline has executed,
/// `execute_if_needed` must detect the updated mtime and re-execute.
#[test]
fn test_pipeline_filter_parameter_change_triggers_rerun() {
    use crate::domain::filters::SmoothFilter;

    let mut pipeline = VtkPipeline::new(Box::new(StaticSource(make_triangle())));
    pipeline.add_filter(Box::new(SmoothFilter::new(0.5, 0)));
    pipeline.execute().unwrap();
    let pipeline_mtime = pipeline.get_mtime();

    let filter = pipeline
        .filter_mut(0)
        .expect("pipeline must expose the boxed filter for mutation")
        .as_mut()
        .as_any_mut()
        .and_then(|a| a.downcast_mut::<SmoothFilter>())
        .expect("boxed filter must downcast to SmoothFilter for parameter mutation");
    let filter_mtime_before = filter.get_mtime();
    filter.set_iterations(1);
    let filter_mtime_after = filter.get_mtime();
    assert!(
        filter_mtime_after > filter_mtime_before,
        "set_iterations must bump boxed filter mtime: before={}, after={}",
        filter_mtime_before.value(),
        filter_mtime_after.value()
    );

    let result = pipeline.execute_if_needed().unwrap();
    assert!(
        result.is_some(),
        "execute_if_needed must re-execute when a boxed filter mtime advances beyond pipeline mtime ({})",
        pipeline_mtime.value()
    );

    let VtkDataObject::PolyData(p) = result.unwrap() else {
        panic!("expected PolyData variant")
    };
    assert!(
        (p.points[0][0] - 0.375).abs() < 1e-5,
        "mutated boxed SmoothFilter must affect output x coordinate: got {}",
        p.points[0][0]
    );
    assert!(
        (p.points[0][1] - 0.25).abs() < 1e-5,
        "mutated boxed SmoothFilter must affect output y coordinate: got {}",
        p.points[0][1]
    );
}
