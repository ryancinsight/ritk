//! Annotation state for interactive segmentation workflows.
//!
//! # Mathematical Specification
//!
//! An annotation state A = (P, C, L) where:
//! - P is a finite ordered set of seed/landmark points in R^3
//! - C = { c1, c2, ... }: closed contours, each ci = [p0,...,pn] in R^(3n)
//! - L = { l1, l2, ... }: open polylines, each li = [q0,...,qm] in R^(3m)
//!
//! Points are stored as [x, y, z] physical-space coordinates in mm.
//! Contours and polylines require >= 2 points; single-point paths are rejected.

use ritk_spatial::Point;
use serde::{Deserialize, Serialize};

use super::error::AnnotationError;
use super::types::LabelId;

/// A single 3-D point annotation with an optional label association.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PointAnnotation {
    /// Physical-space coordinate in mm.
    pub position: Point<3>,
    /// Optional label ID this point belongs to (e.g., a seed for region growing).
    pub label_id: Option<LabelId> }

impl PointAnnotation {
    /// Construct a point annotation with no label association.
    pub fn new(position: impl Into<Point<3>>) -> Self {
        Self {
            position: position.into(),
            label_id: None }
    }

    /// Construct a point annotation bound to a label ID.
    pub fn with_label(position: impl Into<Point<3>>, label_id: impl Into<LabelId>) -> Self {
        Self {
            position: position.into(),
            label_id: Some(label_id.into()) }
    }
}

/// Annotation state accumulating seed points, contours, and polylines.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AnnotationState {
    /// Seed/landmark points.
    pub points: Vec<PointAnnotation>,
    /// Closed contours (each has >= 2 points).
    pub contours: Vec<Vec<Point<3>>>,
    /// Open polylines (each has >= 2 points).
    pub polylines: Vec<Vec<Point<3>>> }

impl AnnotationState {
    /// Construct an empty annotation state.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a seed point.
    pub fn add_point(&mut self, annotation: PointAnnotation) {
        self.points.push(annotation);
    }

    /// Add a closed contour. Returns `Err` if the contour has fewer than 2 points.
    pub fn add_contour(&mut self, points: Vec<Point<3>>) -> Result<(), AnnotationError> {
        if points.len() < 2 {
            return Err(AnnotationError::TooFewPoints {
                kind: "contour",
                count: points.len() });
        }
        self.contours.push(points);
        Ok(())
    }

    /// Add an open polyline. Returns `Err` if the polyline has fewer than 2 points.
    pub fn add_polyline(&mut self, points: Vec<Point<3>>) -> Result<(), AnnotationError> {
        if points.len() < 2 {
            return Err(AnnotationError::TooFewPoints {
                kind: "polyline",
                count: points.len() });
        }
        self.polylines.push(points);
        Ok(())
    }

    /// Remove all annotations.
    pub fn clear(&mut self) {
        self.points.clear();
        self.contours.clear();
        self.polylines.clear();
    }

    /// Remove the last added point. Returns the removed annotation if present.
    pub fn pop_point(&mut self) -> Option<PointAnnotation> {
        self.points.pop()
    }

    /// Remove the last added contour. Returns the removed contour if present.
    pub fn pop_contour(&mut self) -> Option<Vec<Point<3>>> {
        self.contours.pop()
    }

    /// Remove the last added polyline. Returns the removed polyline if present.
    pub fn pop_polyline(&mut self) -> Option<Vec<Point<3>>> {
        self.polylines.pop()
    }

    /// Total annotation count across all types.
    pub fn total_count(&self) -> usize {
        self.points.len() + self.contours.len() + self.polylines.len()
    }

    /// Collect all seed points for a given label ID.
    pub fn seeds_for_label(&self, label_id: impl Into<LabelId>) -> Vec<Point<3>> {
        let label_id = label_id.into();
        self.points
            .iter()
            .filter(|p| p.label_id == Some(label_id))
            .map(|p| p.position)
            .collect()
    }

    /// Serialize this state to JSON.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }

    /// Deserialize an annotation state from JSON.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }
}

#[cfg(test)]
#[path = "tests_annotation_state.rs"]
mod tests;
