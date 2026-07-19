//! Laplacian surface smoothing filter for polygonal meshes.
//!
//! # Mathematical Specification
//!
//! Given a polygonal mesh M = (V, P), the Laplacian smoothing operator L is
//! defined for each vertex v_i as:
//!
//!   L(v_i) = (1 − λ) · v_i  +  λ · (1/|N(i)|) · Σ_{j ∈ N(i)} v_j
//!
//! where N(i) is the set of vertices sharing an edge with v_i, and λ ∈ (0, 1]
//! is the relaxation factor.
//!
//! This operator is applied `iterations` times.  For isolated vertices
//! (|N(i)| = 0) the position is left unchanged.
//!
//! Convergence behaviour: as iterations → ∞, the mesh shrinks toward its
//! barycentre.  For λ = 0, the mesh is unchanged.  The topology (connectivity)
//! is preserved; only vertex coordinates change.

use crate::domain::mtime::{Modifiable, ModifiedTime};
use crate::domain::vtk_data_object::VtkDataObject;
use crate::domain::vtk_pipeline::VtkFilter;
use anyhow::Result;
use std::any::Any;
use std::collections::HashSet;

/// Laplacian surface smoothing filter.
///
/// Smooths a `VtkPolyData` mesh by iteratively moving each vertex toward the
/// average position of its edge-neighbours.
#[derive(Debug, Clone)]
pub struct SmoothFilter {
    /// Relaxation factor λ ∈ (0, 1]. Default: 0.5.
    relaxation_factor: f32,
    /// Number of Laplacian smoothing iterations. Default: 20.
    iterations: usize,
    /// Modification timestamp; bumped on any parameter change.
    mtime: ModifiedTime,
}

impl SmoothFilter {
    /// Construct a new smoothing filter with the given parameters.
    pub fn new(relaxation_factor: f32, iterations: usize) -> Self {
        Self {
            relaxation_factor,
            iterations,
            mtime: ModifiedTime::tick(),
        }
    }

    /// Set the relaxation factor λ.
    ///
    /// Bumps the modification time so that downstream pipeline stages
    /// detect the parameter change.
    pub fn set_relaxation_factor(&mut self, lambda: f32) {
        self.relaxation_factor = lambda;
        self.modified();
    }

    /// Set the number of Laplacian smoothing iterations.
    ///
    /// Bumps the modification time so that downstream pipeline stages
    /// detect the parameter change.
    pub fn set_iterations(&mut self, n: usize) {
        self.iterations = n;
        self.modified();
    }

    /// Returns the relaxation factor λ.
    pub fn relaxation_factor(&self) -> f32 {
        self.relaxation_factor
    }

    /// Returns the number of smoothing iterations.
    pub fn iterations(&self) -> usize {
        self.iterations
    }
}

impl Default for SmoothFilter {
    fn default() -> Self {
        Self::new(0.5, 20)
    }
}

impl Modifiable for SmoothFilter {
    fn get_mtime(&self) -> ModifiedTime {
        self.mtime
    }

    fn modified(&mut self) {
        self.mtime = ModifiedTime::tick();
    }
}

impl VtkFilter for SmoothFilter {
    fn mtime(&self) -> ModifiedTime {
        self.get_mtime()
    }

    fn as_any_mut(&mut self) -> Option<&mut dyn Any> {
        Some(self)
    }

    fn execute(&self, input: VtkDataObject) -> Result<VtkDataObject> {
        match input {
            VtkDataObject::PolyData(mut poly) => {
                if self.iterations == 0 || self.relaxation_factor.abs() < f32::EPSILON {
                    return Ok(VtkDataObject::PolyData(poly));
                }
                let adj = build_adjacency(&poly);
                let mut pts = poly.points.clone();
                for _ in 0..self.iterations {
                    pts = laplacian_step(&pts, &adj, self.relaxation_factor);
                }
                poly.points = pts;
                Ok(VtkDataObject::PolyData(poly))
            }
            other => Err(anyhow::anyhow!(
                "SmoothFilter requires PolyData input; received {}",
                crate::domain::filters::normals::data_object_type_name(&other)
            )),
        }
    }
}

// ── Internal helpers ───────────────────────────────────────────────────────

/// Build an edge-based adjacency list from polygon connectivity.
///
/// For each polygon [v0, v1, …, vk], all consecutive pairs (vi, v_{i+1 mod k})
/// form edges; each edge contributes both directions to the adjacency.
fn build_adjacency(poly: &crate::domain::vtk_data_object::VtkPolyData) -> Vec<Vec<u32>> {
    let n = poly.points.len();
    let mut adj: Vec<HashSet<u32>> = vec![HashSet::new(); n];
    for polygon in &poly.polygons {
        let k = polygon.len();
        for i in 0..k {
            let a = polygon[i];
            let b = polygon[(i + 1) % k];
            adj[a as usize].insert(b);
            adj[b as usize].insert(a);
        }
    }
    for line in &poly.lines {
        for i in 0..line.len().saturating_sub(1) {
            adj[line[i] as usize].insert(line[i + 1]);
            adj[line[i + 1] as usize].insert(line[i]);
        }
    }
    adj.into_iter().map(|s| s.into_iter().collect()).collect()
}

/// Apply one Laplacian smoothing step.
fn laplacian_step(pts: &[[f32; 3]], adj: &[Vec<u32>], lambda: f32) -> Vec<[f32; 3]> {
    pts.iter()
        .enumerate()
        .map(|(i, &p)| {
            let neighbors = &adj[i];
            if neighbors.is_empty() {
                p
            } else {
                let k = neighbors.len() as f32;
                let sum = neighbors.iter().fold([0.0_f32; 3], |acc, &j| {
                    let q = pts[j as usize];
                    [acc[0] + q[0], acc[1] + q[1], acc[2] + q[2]]
                });
                let mean = [sum[0] / k, sum[1] / k, sum[2] / k];
                [
                    p[0] * (1.0 - lambda) + mean[0] * lambda,
                    p[1] * (1.0 - lambda) + mean[1] * lambda,
                    p[2] * (1.0 - lambda) + mean[2] * lambda,
                ]
            }
        })
        .collect()
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "tests_smooth.rs"]
mod tests;
