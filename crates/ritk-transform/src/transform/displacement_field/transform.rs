use super::core::DisplacementField;
use super::module::DisplacementFieldRecord;
use crate::transform::{Resampleable, Transform};
use ritk_core::spatial::{Direction, Point, Spacing};
use ritk_image::burn::module::{
    AutodiffModule, Content, Module, ModuleDisplay, ModuleDisplayDefault, ModuleMapper,
    ModuleVisitor,
};
use ritk_image::burn::record::{PrecisionSettings, Record};
use ritk_image::tensor::backend::{AutodiffBackend, Backend};
use ritk_image::tensor::Tensor;
use ritk_interpolation::{Interpolator, LinearInterpolator};
use serde::{Deserialize, Serialize};

/// Displacement field transform.
///
/// Transforms points by adding a displacement vector interpolated strictly from a field.
#[derive(Debug, Clone)]
pub struct DisplacementFieldTransform<B: Backend, const D: usize> {
    field: DisplacementField<B, D>,
    interpolator: LinearInterpolator,
}

impl<B: Backend, const D: usize> DisplacementFieldTransform<B, D> {
    pub fn new(field: DisplacementField<B, D>, interpolator: LinearInterpolator) -> Self {
        Self {
            field,
            interpolator,
        }
    }

    pub fn field(&self) -> &DisplacementField<B, D> {
        &self.field
    }

    pub fn interpolator(&self) -> &LinearInterpolator {
        &self.interpolator
    }
}

// ── Record types ──────────────────────────────────────────────────────────────

/// Serialisable item produced by converting a `DisplacementFieldTransformRecord`
/// to a specific floating-point precision.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DisplacementFieldTransformRecordItem<FR> {
    field: FR,
}

/// In-memory record for `DisplacementFieldTransform`.
#[derive(Debug, Clone)]
pub struct DisplacementFieldTransformRecord<B: Backend, const D: usize> {
    field: DisplacementFieldRecord<B, D>,
}

impl<B: Backend, const D: usize> Record<B> for DisplacementFieldTransformRecord<B, D> {
    type Item<S: PrecisionSettings> = DisplacementFieldTransformRecordItem<
        <DisplacementFieldRecord<B, D> as Record<B>>::Item<S>,
    >;

    fn into_item<S: PrecisionSettings>(self) -> Self::Item<S> {
        DisplacementFieldTransformRecordItem {
            field: self.field.into_item::<S>(),
        }
    }

    fn from_item<S: PrecisionSettings>(item: Self::Item<S>, device: &B::Device) -> Self {
        DisplacementFieldTransformRecord {
            field: DisplacementFieldRecord::<B, D>::from_item::<S>(item.field, device),
        }
    }
}

// ── Module<B> ─────────────────────────────────────────────────────────────────

impl<B: Backend, const D: usize> Module<B> for DisplacementFieldTransform<B, D> {
    type Record = DisplacementFieldTransformRecord<B, D>;

    fn visit<V: ModuleVisitor<B>>(&self, visitor: &mut V) {
        self.field.visit(visitor);
    }

    fn map<M: ModuleMapper<B>>(self, mapper: &mut M) -> Self {
        Self {
            field: self.field.map(mapper),
            interpolator: self.interpolator,
        }
    }

    fn into_record(self) -> Self::Record {
        DisplacementFieldTransformRecord {
            field: self.field.into_record(),
        }
    }

    fn load_record(self, record: Self::Record) -> Self {
        Self {
            field: self.field.load_record(record.field),
            interpolator: self.interpolator,
        }
    }

    fn collect_devices(&self, devices: Vec<B::Device>) -> Vec<B::Device> {
        self.field.collect_devices(devices)
    }

    fn to_device(self, device: &B::Device) -> Self {
        Self {
            field: self.field.to_device(device),
            interpolator: self.interpolator,
        }
    }

    fn fork(self, device: &B::Device) -> Self {
        Self {
            field: self.field.fork(device),
            interpolator: self.interpolator,
        }
    }
}

// ── AutodiffModule<B> ────────────────────────────────────────────────────────

impl<B: AutodiffBackend, const D: usize> AutodiffModule<B> for DisplacementFieldTransform<B, D> {
    type InnerModule = DisplacementFieldTransform<B::InnerBackend, D>;

    fn valid(&self) -> Self::InnerModule {
        DisplacementFieldTransform {
            field: self.field.valid(),
            interpolator: self.interpolator,
        }
    }
}

// ── ModuleDisplay ─────────────────────────────────────────────────────────────

impl<B: Backend, const D: usize> ModuleDisplayDefault for DisplacementFieldTransform<B, D> {
    fn content(&self, content: Content) -> Option<Content> {
        content.add_single(&self.field).optional()
    }
}

impl<B: Backend, const D: usize> ModuleDisplay for DisplacementFieldTransform<B, D> {}

// ── Transform and Resampleable ───────────────────────────────────────────────

impl<B: Backend, const D: usize> Transform<B, D> for DisplacementFieldTransform<B, D> {
    fn transform_points(&self, points: Tensor<B, 2>) -> Tensor<B, 2> {
        let indices = self.field.world_to_index_tensor(points.clone());

        let mut displacement_components = Vec::with_capacity(D);
        for i in 0..D {
            let component = &self.field.components[i].val();
            let val = self.interpolator.interpolate(component, indices.clone());
            displacement_components.push(val);
        }

        let displacement = Tensor::stack(displacement_components, 1);
        points + displacement
    }
}

impl<B: Backend, const D: usize> Resampleable<B, D> for DisplacementFieldTransform<B, D> {
    fn resample(
        &self,
        shape: [usize; D],
        origin: Point<D>,
        spacing: Spacing<D>,
        direction: Direction<D>,
    ) -> Self {
        let new_field = self.field.resample(shape, origin, spacing, direction);
        Self::new(new_field, self.interpolator)
    }
}

// Dimension-specific convenience aliases removed.
// Use `DisplacementFieldTransform<B, 2>` or `DisplacementFieldTransform<B, 3>` directly.
