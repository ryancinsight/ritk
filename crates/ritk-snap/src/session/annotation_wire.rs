//! Private wire schema for nested completed annotations.

use crate::tools::interaction::{Annotation, AnnotationLocation, PatientLength};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

mod numbers;
#[cfg(test)]
mod tests;

#[derive(Serialize, Deserialize)]
#[serde(
    remote = "Annotation",
    rename = "Annotation",
    expecting = "enum Annotation"
)]
enum AnnotationWire {
    Length {
        #[serde(default)]
        plane: usize,
        #[serde(default)]
        location: Option<AnnotationLocation>,
        #[serde(with = "numbers")]
        p1: [f32; 2],
        #[serde(with = "numbers")]
        p2: [f32; 2],
        #[serde(with = "numbers")]
        length_mm: f32,
    },
    PatientLength {
        #[serde(default)]
        plane: usize,
        #[serde(default)]
        location: Option<AnnotationLocation>,
        length: PatientLength,
    },
    Angle {
        #[serde(default)]
        location: Option<AnnotationLocation>,
        #[serde(with = "numbers")]
        p1: [f32; 2],
        #[serde(with = "numbers")]
        p2: [f32; 2],
        #[serde(with = "numbers")]
        p3: [f32; 2],
        #[serde(with = "numbers")]
        angle_deg: f32,
    },
    RoiRect {
        #[serde(default)]
        location: Option<AnnotationLocation>,
        #[serde(with = "numbers")]
        top_left: [f32; 2],
        #[serde(with = "numbers")]
        bottom_right: [f32; 2],
        #[serde(with = "numbers")]
        mean: f32,
        #[serde(with = "numbers")]
        std_dev: f32,
        #[serde(with = "numbers")]
        min: f32,
        #[serde(with = "numbers")]
        max: f32,
        #[serde(with = "numbers")]
        area_mm2: f32,
    },
    RoiEllipse {
        #[serde(default)]
        location: Option<AnnotationLocation>,
        #[serde(with = "numbers")]
        center: [f32; 2],
        #[serde(with = "numbers")]
        radii: [f32; 2],
        #[serde(with = "numbers")]
        mean: f32,
        #[serde(with = "numbers")]
        std_dev: f32,
        #[serde(with = "numbers")]
        min: f32,
        #[serde(with = "numbers")]
        max: f32,
        #[serde(with = "numbers")]
        area_mm2: f32,
    },
    Deviation {
        #[serde(default)]
        location: Option<AnnotationLocation>,
        #[serde(with = "numbers")]
        start: [f32; 2],
        #[serde(with = "numbers")]
        end: [f32; 2],
        #[serde(with = "numbers")]
        reference: [f32; 2],
        #[serde(with = "numbers")]
        distance_mm: f32,
        #[serde(with = "numbers")]
        angle_deg: f32,
    },
    HuPoint {
        #[serde(default)]
        location: Option<AnnotationLocation>,
        #[serde(with = "numbers")]
        pos: [f32; 2],
        #[serde(with = "numbers")]
        value: f32,
    },
    ClosedPolygon {
        #[serde(default)]
        location: Option<AnnotationLocation>,
        #[serde(with = "numbers")]
        points: Vec<[f32; 2]>,
        #[serde(with = "numbers")]
        area_mm2: f32,
        #[serde(with = "numbers")]
        perimeter_mm: f32,
    },
    OpenPolygon {
        #[serde(default)]
        location: Option<AnnotationLocation>,
        #[serde(with = "numbers")]
        points: Vec<[f32; 2]>,
        #[serde(with = "numbers")]
        length_mm: f32,
    },
    Arrow {
        #[serde(default)]
        location: Option<AnnotationLocation>,
        #[serde(with = "numbers")]
        tail: [f32; 2],
        #[serde(with = "numbers")]
        head: [f32; 2],
    },
    Pencil {
        #[serde(default)]
        location: Option<AnnotationLocation>,
        #[serde(with = "numbers")]
        points: Vec<[f32; 2]>,
    },
    CobbAngle {
        #[serde(default)]
        location: Option<AnnotationLocation>,
        #[serde(with = "numbers")]
        points: [[f32; 2]; 4],
        #[serde(with = "numbers")]
        angle_deg: f32,
    },
}

pub(crate) fn serialize_annotation<S>(
    annotation: &Annotation,
    serializer: S,
) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    AnnotationWire::serialize(annotation, serializer)
}

pub(crate) fn deserialize_annotation<'de, D>(deserializer: D) -> Result<Annotation, D::Error>
where
    D: Deserializer<'de>,
{
    AnnotationWire::deserialize(deserializer)
}
