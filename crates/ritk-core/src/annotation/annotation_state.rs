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

use serde::{Deserialize, Serialize};

/// A single 3-D point annotation with an optional label association.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PointAnnotation {
    /// Physical-space [x, y, z] coordinate in mm.
    pub position: [f64; 3],
    /// Optional label ID this point belongs to (e.g., a seed for region growing).
    pub label_id: Option<u32>,
}

impl PointAnnotation {
    /// Construct a point annotation with no label association.
    pub fn new(position: [f64; 3]) -> Self {
        Self { position, label_id: None }
    }

    /// Construct a point annotation bound to a label ID.
    pub fn with_label(position: [f64; 3], label_id: u32) -> Self {
        Self { position, label_id: Some(label_id) }
    }
}

/// Annotation state accumulating seed points, contours, and polylines.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AnnotationState {
    /// Seed/landmark points.
    pub points: Vec<PointAnnotation>,
    /// Closed contours (each has >= 2 points).
    pub contours: Vec<Vec<[f64; 3]>>,
    /// Open polylines (each has >= 2 points).
    pub polylines: Vec<Vec<[f64; 3]>>,
}

impl AnnotationState {
    /// Construct an empty annotation state.
    pub fn new() -> Self { Self::default() }

    /// Add a seed point.
    pub fn add_point(&mut self, annotation: PointAnnotation) {
        self.points.push(annotation);
    }

    /// Add a closed contour. Returns `Err` if the contour has fewer than 2 points.
    pub fn add_contour(&mut self, points: Vec<[f64; 3]>) -> Result<(), String> {
        if points.len() < 2 {
            return Err(format!("contour requires >= 2 points, got {}", points.len()));
        }
        self.contours.push(points);
        Ok(())
    }

    /// Add an open polyline. Returns `Err` if the polyline has fewer than 2 points.
    pub fn add_polyline(&mut self, points: Vec<[f64; 3]>) -> Result<(), String> {
        if points.len() < 2 {
            return Err(format!("polyline requires >= 2 points, got {}", points.len()));
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
    pub fn pop_point(&mut self) -> Option<PointAnnotation> { self.points.pop() }

    /// Remove the last added contour. Returns the removed contour if present.
    pub fn pop_contour(&mut self) -> Option<Vec<[f64; 3]>> { self.contours.pop() }

    /// Remove the last added polyline. Returns the removed polyline if present.
    pub fn pop_polyline(&mut self) -> Option<Vec<[f64; 3]>> { self.polylines.pop() }

    /// Total annotation count across all types.
    pub fn total_count(&self) -> usize {
        self.points.len() + self.contours.len() + self.polylines.len()
    }

    /// Collect all seed points for a given label ID.
    pub fn seeds_for_label(&self, label_id: u32) -> Vec<[f64; 3]> {
        self.points.iter()
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
mod tests {
    use super::*;

    #[test]
    fn test_annotation_state_empty() {
        let state = AnnotationState::new();
        assert_eq!(state.total_count(), 0);
        assert!(state.points.is_empty());
        assert!(state.contours.is_empty());
        assert!(state.polylines.is_empty());
    }

    #[test]
    fn test_add_point() {
        let mut state = AnnotationState::new();
        let ann = PointAnnotation::with_label([1.0, 2.0, 3.0], 5);
        state.add_point(ann);
        assert_eq!(state.points.len(), 1);
        assert_eq!(state.points[0].position, [1.0, 2.0, 3.0]);
        assert_eq!(state.points[0].label_id, Some(5));
    }

    #[test]
    fn test_add_contour_valid() {
        let mut state = AnnotationState::new();
        let pts = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0]];
        state.add_contour(pts.clone()).unwrap();
        assert_eq!(state.contours.len(), 1);
        assert_eq!(state.contours[0], pts);
    }

    #[test]
    fn test_add_contour_too_short() {
        let mut state = AnnotationState::new();
        let result = state.add_contour(vec![[0.0, 0.0, 0.0]]);
        assert!(result.is_err());
        let msg = result.unwrap_err();
        assert!(msg.contains("1"), "error must mention point count: {}", msg);
    }

    #[test]
    fn test_add_polyline_valid() {
        let mut state = AnnotationState::new();
        let pts = vec![[0.0,0.0,0.0],[1.0,0.0,0.0],[2.0,0.0,0.0],[3.0,0.0,0.0]];
        state.add_polyline(pts.clone()).unwrap();
        assert_eq!(state.polylines.len(), 1);
        assert_eq!(state.polylines[0], pts);
    }

    #[test]
    fn test_add_polyline_too_short() {
        let mut state = AnnotationState::new();
        let result = state.add_polyline(vec![[0.0, 0.0, 0.0]]);
        assert!(result.is_err());
        let msg = result.unwrap_err();
        assert!(msg.contains("1"), "error must mention point count: {}", msg);
    }

    #[test]
    fn test_clear() {
        let mut state = AnnotationState::new();
        state.add_point(PointAnnotation::new([0.0, 0.0, 0.0]));
        state.add_contour(vec![[0.0,0.0,0.0],[1.0,0.0,0.0]]).unwrap();
        state.add_polyline(vec![[0.0,0.0,0.0],[1.0,0.0,0.0]]).unwrap();
        state.clear();
        assert_eq!(state.total_count(), 0);
        assert!(state.points.is_empty());
        assert!(state.contours.is_empty());
        assert!(state.polylines.is_empty());
    }

    #[test]
    fn test_seeds_for_label() {
        let mut state = AnnotationState::new();
        state.add_point(PointAnnotation::with_label([1.0, 0.0, 0.0], 1));
        state.add_point(PointAnnotation::with_label([2.0, 0.0, 0.0], 1));
        state.add_point(PointAnnotation::with_label([3.0, 0.0, 0.0], 2));
        let seeds1 = state.seeds_for_label(1);
        assert_eq!(seeds1.len(), 2);
        assert_eq!(seeds1[0], [1.0, 0.0, 0.0]);
        assert_eq!(seeds1[1], [2.0, 0.0, 0.0]);
        let seeds2 = state.seeds_for_label(2);
        assert_eq!(seeds2.len(), 1);
        assert_eq!(seeds2[0], [3.0, 0.0, 0.0]);
    }

    #[test]
    fn test_json_roundtrip() {
        let mut state = AnnotationState::new();
        state.add_point(PointAnnotation::with_label([10.0, 20.0, 30.0], 3));
        state.add_contour(vec![[0.0,0.0,0.0],[1.0,0.0,0.0],[0.5,1.0,0.0]]).unwrap();
        state.add_polyline(vec![[5.0,5.0,5.0],[6.0,5.0,5.0]]).unwrap();
        let json = state.to_json().expect("serialization must succeed");
        let restored = AnnotationState::from_json(&json).expect("deserialization must succeed");
        assert_eq!(restored.points.len(), state.points.len());
        assert_eq!(restored.points[0].position, [10.0, 20.0, 30.0]);
        assert_eq!(restored.points[0].label_id, Some(3));
        assert_eq!(restored.contours.len(), 1);
        assert_eq!(restored.contours[0].len(), 3);
        assert_eq!(restored.polylines.len(), 1);
        assert_eq!(restored.polylines[0].len(), 2);
    }
}
