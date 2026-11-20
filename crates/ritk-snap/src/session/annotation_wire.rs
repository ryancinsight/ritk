//! Private wire schema for nested completed annotations.

use crate::tools::interaction::{Annotation, AnnotationLocation, PatientLength};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

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
        p1: [f32; 2],
        p2: [f32; 2],
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
        p1: [f32; 2],
        p2: [f32; 2],
        p3: [f32; 2],
        angle_deg: f32,
    },
    RoiRect {
        #[serde(default)]
        location: Option<AnnotationLocation>,
        top_left: [f32; 2],
        bottom_right: [f32; 2],
        mean: f32,
        std_dev: f32,
        min: f32,
        max: f32,
        area_mm2: f32,
    },
    RoiEllipse {
        #[serde(default)]
        location: Option<AnnotationLocation>,
        center: [f32; 2],
        radii: [f32; 2],
        mean: f32,
        std_dev: f32,
        min: f32,
        max: f32,
        area_mm2: f32,
    },
    Deviation {
        #[serde(default)]
        location: Option<AnnotationLocation>,
        start: [f32; 2],
        end: [f32; 2],
        reference: [f32; 2],
        distance_mm: f32,
        angle_deg: f32,
    },
    HuPoint {
        #[serde(default)]
        location: Option<AnnotationLocation>,
        pos: [f32; 2],
        value: f32,
    },
    ClosedPolygon {
        #[serde(default)]
        location: Option<AnnotationLocation>,
        points: Vec<[f32; 2]>,
        area_mm2: f32,
        perimeter_mm: f32,
    },
    OpenPolygon {
        #[serde(default)]
        location: Option<AnnotationLocation>,
        points: Vec<[f32; 2]>,
        length_mm: f32,
    },
    Arrow {
        #[serde(default)]
        location: Option<AnnotationLocation>,
        tail: [f32; 2],
        head: [f32; 2],
    },
    Pencil {
        #[serde(default)]
        location: Option<AnnotationLocation>,
        points: Vec<[f32; 2]>,
    },
    CobbAngle {
        #[serde(default)]
        location: Option<AnnotationLocation>,
        points: [[f32; 2]; 4],
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
