//! Burn `Module` / `AutodiffModule` implementations for `DisplacementField`.
//!
//! Keeps the trainable-parameter plumbing (visit, map, record serialisation,
//! device movement) separate from the geometric core in `core.rs`.

use super::core::DisplacementField;
use ritk_image::burn::module::{
    AutodiffModule, Content, Module, ModuleDisplayDefault, ModuleMapper, ModuleVisitor,
};
use ritk_image::burn::record::{PrecisionSettings, Record};
use ritk_image::tensor::backend::{AutodiffBackend, Backend};
use ritk_image::tensor::Tensor;
use ritk_image::burn::module::Param;
use serde::{Deserialize, Serialize};

// ── Clone ────────────────────────────────────────────────────────────────────

impl<B: Backend, const D: usize> Clone for DisplacementField<B, D> {
    fn clone(&self) -> Self {
        Self {
            components: self.components.clone(),
            origin: self.origin,
            spacing: self.spacing,
            direction: self.direction,
            world_to_index_matrix: self.world_to_index_matrix.clone(),
            origin_tensor: self.origin_tensor.clone(),
        }
    }
}

// ── Record types ─────────────────────────────────────────────────────────────

/// The serialisable *item* produced when a `DisplacementFieldRecord` is
/// converted to a specific floating-point precision.
///
/// `CR` is the concrete component-record item type, e.g.
/// `ParamRecord<TensorData>` at full-precision.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DisplacementFieldRecordItem<CR> {
    components: Vec<CR>,
    origin: Vec<f64>,
    spacing: Vec<f64>,
    direction: Vec<f64>, // D×D row-major
}

/// In-memory record produced by `Module::into_record()`.
#[derive(Debug, Clone)]
pub struct DisplacementFieldRecord<B: Backend, const D: usize> {
    pub(super) components: Vec<<Param<Tensor<B, D>> as Module<B>>::Record>,
    pub(super) origin: Vec<f64>,
    pub(super) spacing: Vec<f64>,
    pub(super) direction: Vec<f64>,
}

impl<B: Backend, const D: usize> Record<B> for DisplacementFieldRecord<B, D> {
    type Item<S: PrecisionSettings> = DisplacementFieldRecordItem<
        <<Param<Tensor<B, D>> as Module<B>>::Record as Record<B>>::Item<S>,
    >;

    fn into_item<S: PrecisionSettings>(self) -> Self::Item<S> {
        DisplacementFieldRecordItem {
            components: self
                .components
                .into_iter()
                .map(|r| r.into_item::<S>())
                .collect(),
            origin: self.origin,
            spacing: self.spacing,
            direction: self.direction,
        }
    }

    fn from_item<S: PrecisionSettings>(item: Self::Item<S>, device: &B::Device) -> Self {
        DisplacementFieldRecord {
            components: item
                .components
                .into_iter()
                .map(|r| {
                    <Param<Tensor<B, D>> as Module<B>>::Record::from_item::<S>(r, device)
                })
                .collect(),
            origin: item.origin,
            spacing: item.spacing,
            direction: item.direction,
        }
    }
}

// ── Module<B> ─────────────────────────────────────────────────────────────────

impl<B: Backend, const D: usize> Module<B> for DisplacementField<B, D> {
    type Record = DisplacementFieldRecord<B, D>;

    fn visit<V: ModuleVisitor<B>>(&self, visitor: &mut V) {
        for param in &self.components {
            param.visit(visitor);
        }
    }

    fn map<M: ModuleMapper<B>>(self, mapper: &mut M) -> Self {
        let components = self
            .components
            .into_iter()
            .map(|p| <Param<Tensor<B, D>> as Module<B>>::map(p, mapper))
            .collect();
        Self {
            components,
            origin: self.origin,
            spacing: self.spacing,
            direction: self.direction,
            world_to_index_matrix: self.world_to_index_matrix,
            origin_tensor: self.origin_tensor,
        }
    }

    fn into_record(self) -> Self::Record {
        DisplacementFieldRecord {
            components: self.components.into_iter().map(|p| p.into_record()).collect(),
            origin: (0..D).map(|i| self.origin[i]).collect(),
            spacing: (0..D).map(|i| self.spacing[i]).collect(),
            direction: (0..D)
                .flat_map(|r| (0..D).map(move |c| self.direction[(r, c)]))
                .collect(),
        }
    }

    fn load_record(self, record: Self::Record) -> Self {
        let components = self
            .components
            .into_iter()
            .zip(record.components)
            .map(|(p, r)| p.load_record(r))
            .collect();
        Self { components, ..self }
    }

    fn collect_devices(&self, mut devices: Vec<B::Device>) -> Vec<B::Device> {
        for p in &self.components {
            devices = p.collect_devices(devices);
        }
        devices
    }

    fn to_device(self, device: &B::Device) -> Self {
        let components = self.components.into_iter().map(|p| p.to_device(device)).collect();
        Self {
            components,
            world_to_index_matrix: self.world_to_index_matrix.to_device(device),
            origin_tensor: self.origin_tensor.to_device(device),
            origin: self.origin,
            spacing: self.spacing,
            direction: self.direction,
        }
    }

    fn fork(self, device: &B::Device) -> Self {
        let components = self.components.into_iter().map(|p| p.fork(device)).collect();
        Self {
            components,
            world_to_index_matrix: self.world_to_index_matrix.to_device(device),
            origin_tensor: self.origin_tensor.to_device(device),
            origin: self.origin,
            spacing: self.spacing,
            direction: self.direction,
        }
    }
}

// ── AutodiffModule<B> ────────────────────────────────────────────────────────

impl<B: AutodiffBackend, const D: usize> AutodiffModule<B> for DisplacementField<B, D> {
    type InnerModule = DisplacementField<B::InnerBackend, D>;

    fn valid(&self) -> Self::InnerModule {
        let components = self.components.iter().map(|p| p.valid()).collect();
        DisplacementField {
            components,
            origin: self.origin,
            spacing: self.spacing,
            direction: self.direction,
            world_to_index_matrix: self.world_to_index_matrix.clone().inner(),
            origin_tensor: self.origin_tensor.clone().inner(),
        }
    }
}

// ── ModuleDisplay ─────────────────────────────────────────────────────────────

impl<B: Backend, const D: usize> ModuleDisplayDefault for DisplacementField<B, D> {
    fn content(&self, content: Content) -> Option<Content> {
        content.optional()
    }
}

impl<B: Backend, const D: usize> ritk_image::burn::module::ModuleDisplay for DisplacementField<B, D> {}
