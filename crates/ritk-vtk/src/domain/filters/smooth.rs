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
//!
//! # Layout
//!
//! The edge adjacency is stored in CSR shape ([`Adjacency`]): one contiguous
//! neighbor-id buffer plus a per-vertex offset table — the same flat
//! buffer + offset layout as the VTK `VtkCellArray` cell connectivity. The
//! traversal-hot Laplacian loop therefore slices contiguous neighbor runs
//! instead of chasing a `Vec` allocation per vertex.

use crate::domain::mtime::{Modifiable, ModifiedTime};
use crate::domain::vtk_data_object::VtkDataObject;
use crate::domain::vtk_pipeline::VtkFilter;
use anyhow::Result;
use std::any::Any;

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
                let adj = Adjacency::build(&poly);
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

/// CSR-shaped edge adjacency for a polygonal mesh.
///
/// Neighbor ids live in one contiguous [`Adjacency::neighbors`] buffer; the
/// neighbors of vertex `v` occupy `neighbors[offsets[v]..offsets[v + 1]]`,
/// with `offsets.len() == vertex_count + 1` and the sentinel
/// `offsets[vertex_count] == neighbors.len()`. This is the same flat
/// buffer + offset-table shape as the VTK `VtkCellArray` cell-connectivity
/// layout.
///
/// Each vertex's neighbor run is deduplicated and sorted ascending, so the
/// produced layout is fully deterministic — the previous `HashSet`-based
/// jagged build left neighbor order implementation-defined.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Adjacency {
    /// `neighbors[offsets[v]..offsets[v + 1]]` are the edge-neighbors of
    /// vertex `v`. Non-decreasing, with the `vertex_count + 1` sentinel.
    pub offsets: Vec<usize>,
    /// Contiguous neighbor-id buffer: one sorted, deduplicated run per vertex.
    pub neighbors: Vec<u32>,
}

impl Adjacency {
    /// Build the edge adjacency of `poly` from its polygon and line
    /// connectivity.
    ///
    /// Each polygon `[v0, v1, …, vk]` contributes the undirected edges
    /// `(vi, v(i+1) mod k)`, and each polyline contributes every consecutive
    /// pair, so vertices sharing a polygon edge or a line segment become
    /// neighbors. Isolated vertices have an empty run.
    ///
    /// The build is jagged-free: degrees are counted once, prefix-summed into
    /// the offset table, neighbor ids are written directly into the reserved
    /// flat buffer, and each run is then sorted and deduplicated in place —
    /// no per-vertex heap allocation and no intermediate `Vec<Vec<_>>`.
    pub fn build(poly: &crate::domain::vtk_data_object::VtkPolyData) -> Self {
        let n = poly.points.len();

        // Pass 1: degree per vertex (each undirected edge counts twice).
        let mut degree = vec![0usize; n];
        for polygon in &poly.polygons {
            let k = polygon.len();
            for i in 0..k {
                degree[polygon[i] as usize] += 1;
                degree[polygon[(i + 1) % k] as usize] += 1;
            }
        }
        for line in &poly.lines {
            for i in 0..line.len().saturating_sub(1) {
                degree[line[i] as usize] += 1;
                degree[line[i + 1] as usize] += 1;
            }
        }

        // Pass 2: prefix-sum the degrees into the offset table.
        let mut offsets = Vec::with_capacity(n + 1);
        let mut count = 0usize;
        offsets.push(0);
        for &d in &degree {
            count += d;
            offsets.push(count);
        }

        // Pass 3: write neighbor ids into the reserved flat regions.
        let mut neighbors = vec![0u32; count];
        let mut cursor = offsets.clone();
        for polygon in &poly.polygons {
            let k = polygon.len();
            for i in 0..k {
                let a = polygon[i] as usize;
                let b = polygon[(i + 1) % k] as usize;
                neighbors[cursor[a]] = b as u32;
                cursor[a] += 1;
                neighbors[cursor[b]] = a as u32;
                cursor[b] += 1;
            }
        }
        for line in &poly.lines {
            for i in 0..line.len().saturating_sub(1) {
                let a = line[i] as usize;
                let b = line[i + 1] as usize;
                neighbors[cursor[a]] = b as u32;
                cursor[a] += 1;
                neighbors[cursor[b]] = a as u32;
                cursor[b] += 1;
            }
        }

        // Pass 4: sort and dedup each run in place, compacting leftward.
        let mut final_len = vec![0usize; n];
        let mut write = 0usize;
        for v in 0..n {
            let s = offsets[v];
            let e = offsets[v + 1];
            let run = &mut neighbors[s..e];
            run.sort_unstable();
            let mut w = 0usize;
            for r in 0..run.len() {
                if w == 0 || run[r] != run[w - 1] {
                    run[w] = run[r];
                    w += 1;
                }
            }
            neighbors.copy_within(s..s + w, write);
            write += w;
            final_len[v] = w;
        }
        neighbors.truncate(write);

        // Pass 5: rebuild offsets from the deduplicated run lengths.
        let mut final_offsets = Vec::with_capacity(n + 1);
        let mut final_count = 0usize;
        final_offsets.push(0);
        for &len in &final_len {
            final_count += len;
            final_offsets.push(final_count);
        }

        Self {
            offsets: final_offsets,
            neighbors,
        }
    }

    /// Neighbors of vertex `v` as one contiguous, sorted slice.
    #[inline]
    pub fn neighbors_of(&self, v: usize) -> &[u32] {
        &self.neighbors[self.offsets[v]..self.offsets[v + 1]]
    }
}

/// Apply one Laplacian smoothing step.
fn laplacian_step(pts: &[[f32; 3]], adj: &Adjacency, lambda: f32) -> Vec<[f32; 3]> {
    pts.iter()
        .enumerate()
        .map(|(i, &p)| {
            let neighbors = adj.neighbors_of(i);
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
