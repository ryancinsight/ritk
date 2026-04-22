//! VTK-style data-flow pipeline abstractions.
//!
//! # Architecture
//!
//! The pipeline follows the VTK data-flow model:
//!   Source -> [Filter1 -> Filter2 -> ...] -> optional Sink
//!
//! Each stage is a trait object enabling runtime composition without generic proliferation.
//!
//! # Mathematical Specification
//!
//! Let D0 = Source::produce().  For k filters, Dn = Filterk::execute(Dn-1), n = 1..k.
//! The final Dk is returned by `VtkPipeline::execute()` and forwarded to the sink if present.
//! A filter must preserve or document which `VtkDataObject` variant it accepts and emits.

use super::vtk_data_object::VtkDataObject;
use anyhow::Result;

/// Produces a `VtkDataObject` from an external source (file, procedural generator).
pub trait VtkSource: Send + Sync {
    /// Generate the initial data object.
    fn produce(&self) -> Result<VtkDataObject>;
}

/// Transforms one `VtkDataObject` into another.
///
/// Implementations must document which variants they accept and which they produce.
pub trait VtkFilter: Send + Sync {
    /// Apply the filter transformation.
    fn execute(&self, input: VtkDataObject) -> Result<VtkDataObject>;
}

/// Consumes a `VtkDataObject` (writes to disk, renders, accumulates statistics, etc.).
pub trait VtkSink: Send + Sync {
    /// Process the final data object.
    fn consume(&self, data: &VtkDataObject) -> Result<()>;
}

/// An ordered source -> filters -> sink data-flow pipeline.
pub struct VtkPipeline {
    source: Box<dyn VtkSource>,
    filters: Vec<Box<dyn VtkFilter>>,
    sink: Option<Box<dyn VtkSink>>,
}

impl VtkPipeline {
    /// Construct a pipeline with the given source and no filters or sink.
    pub fn new(source: Box<dyn VtkSource>) -> Self {
        Self {
            source,
            filters: Vec::new(),
            sink: None,
        }
    }

    /// Append a filter stage. Returns `&mut Self` for chaining.
    pub fn add_filter(&mut self, filter: Box<dyn VtkFilter>) -> &mut Self {
        self.filters.push(filter);
        self
    }

    /// Set the terminal sink. Returns `&mut Self` for chaining.
    pub fn set_sink(&mut self, sink: Box<dyn VtkSink>) -> &mut Self {
        self.sink = Some(sink);
        self
    }

    /// Execute the full pipeline.
    ///
    /// Calls `produce`, chains all filters, calls the sink if present, and returns
    /// the final `VtkDataObject`.
    pub fn execute(&self) -> Result<VtkDataObject> {
        let mut data = self.source.produce()?;
        for filter in &self.filters {
            data = filter.execute(data)?;
        }
        if let Some(sink) = &self.sink {
            sink.consume(&data)?;
        }
        Ok(data)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::vtk_data_object::{VtkDataObject, VtkPolyData};
    use std::sync::atomic::{AtomicUsize, Ordering};
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

    #[test]
    fn test_pipeline_source_only() {
        let pipeline = VtkPipeline::new(Box::new(StaticSource(make_triangle())));
        let out = pipeline.execute().unwrap();
        let VtkDataObject::PolyData(p) = out else { panic!("expected PolyData variant") };
        assert_eq!(p.points.len(), 3);
        assert_eq!(p.polygons, vec![vec![0u32, 1, 2]]);
    }

    #[test]
    fn test_pipeline_with_identity_filter() {
        let mut pipeline = VtkPipeline::new(Box::new(StaticSource(make_triangle())));
        pipeline.add_filter(Box::new(IdentityFilter));
        pipeline.add_filter(Box::new(IdentityFilter));
        let out = pipeline.execute().unwrap();
        let VtkDataObject::PolyData(p) = out else { panic!("expected PolyData variant") };
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
        let VtkDataObject::PolyData(p) = out else { panic!("expected PolyData variant") };
        // Original first point was [0,0,0]; after translate by 10.0 -> [10,10,10].
        assert!((p.points[0][0] - delta).abs() < 1e-6, "x should be {}", delta);
        assert!((p.points[0][1] - delta).abs() < 1e-6, "y should be {}", delta);
        assert!((p.points[0][2] - delta).abs() < 1e-6, "z should be {}", delta);
    }

    #[test]
    fn test_pipeline_chained_filters_cumulative() {
        let mut pipeline = VtkPipeline::new(Box::new(StaticSource(make_triangle())));
        pipeline.add_filter(Box::new(TranslateFilter(1.0)));
        pipeline.add_filter(Box::new(TranslateFilter(2.0)));
        let out = pipeline.execute().unwrap();
        let VtkDataObject::PolyData(p) = out else { panic!("expected PolyData variant") };
        // Original first point [0,0,0] + 1.0 + 2.0 = [3,3,3]
        assert!((p.points[0][0] - 3.0).abs() < 1e-6);
    }
}
