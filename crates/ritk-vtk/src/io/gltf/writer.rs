//! glTF 2.0 JSON writer ← VtkPolyData.
//!
//! Produces a single `.gltf` file (no sidecar `.bin`).  All geometry data is
//! embedded as a `data:application/octet-stream;base64,…` URI in the `buffers`
//! array.
//!
//! # Output structure
//! ```text
//! asset / scene / scenes / nodes / meshes
//!   └─ primitives[0]: POSITION accessor (VEC3 FLOAT) + INDICES accessor (SCALAR UINT)
//! accessors[0]: POSITION — VEC3, componentType=5126 (FLOAT), with min/max
//! accessors[1]: INDICES  — SCALAR, componentType=5125 (UNSIGNED_INT)
//! bufferViews[0]: vertex positions, target=34962 (ARRAY_BUFFER)
//! bufferViews[1]: face indices,     target=34963 (ELEMENT_ARRAY_BUFFER)
//! buffers[0]:  base64 data URI containing LE f32 positions + LE u32 indices
//! ```
//!
//! # Triangulation
//! Non-triangular polygons are fan-triangulated:
//! `[v0, v1, v2, v3, …] → (v0,v1,v2), (v0,v2,v3), …`
//!
//! # Buffer alignment
//! `byteOffset` of the index buffer view is always 4-byte aligned.
//! (Vertex positions use 12 bytes each — already 4-byte aligned.)

use crate::domain::vtk_data_object::VtkPolyData;
use anyhow::Result;
use serde_json::{json, Value};
use std::path::Path;

/// Write `poly` as a glTF 2.0 JSON file to `path`.
pub fn write_gltf(path: impl AsRef<Path>, poly: &VtkPolyData) -> Result<()> {
    let json = build_gltf_json(poly)?;
    let text = serde_json::to_string_pretty(&json)?;
    std::fs::write(path.as_ref(), text)?;
    Ok(())
}

// ── Core builder ──────────────────────────────────────────────────────────────

/// Exposed for in-memory testing.
pub(crate) fn build_gltf_json(poly: &VtkPolyData) -> Result<Value> {
    // ── Fan-triangulate all polygons ─────────────────────────────────────────
    let mut indices: Vec<u32> = Vec::new();
    for polygon in &poly.polygons {
        for [a, b, c] in fan_triangulate(polygon) {
            indices.push(a);
            indices.push(b);
            indices.push(c);
        }
    }

    let n_verts = poly.points.len();
    let n_indices = indices.len();

    // ── Compute POSITION AABB ────────────────────────────────────────────────
    let (min_pos, max_pos) = aabb(&poly.points);

    // ── Serialise vertex positions: 3 × f32 LE per vertex ───────────────────
    let mut vert_bytes: Vec<u8> = Vec::with_capacity(n_verts * 12);
    for [x, y, z] in &poly.points {
        vert_bytes.extend_from_slice(&x.to_le_bytes());
        vert_bytes.extend_from_slice(&y.to_le_bytes());
        vert_bytes.extend_from_slice(&z.to_le_bytes());
    }
    let vert_byte_len = vert_bytes.len(); // n_verts * 12 — always 4-byte aligned

    // ── Align index buffer offset to 4 bytes ────────────────────────────────
    // 12 % 4 == 0, so no padding needed, but enforce explicitly.
    let idx_byte_offset = (vert_byte_len + 3) & !3;
    let padding = idx_byte_offset - vert_byte_len;

    // ── Serialise face indices: u32 LE ───────────────────────────────────────
    let mut idx_bytes: Vec<u8> = Vec::with_capacity(n_indices * 4);
    for &idx in &indices {
        idx_bytes.extend_from_slice(&idx.to_le_bytes());
    }
    let idx_byte_len = idx_bytes.len();

    // ── Assemble and base64-encode the combined buffer ───────────────────────
    let total_bytes = idx_byte_offset + idx_byte_len;
    let mut buf: Vec<u8> = Vec::with_capacity(total_bytes);
    buf.extend_from_slice(&vert_bytes);
    buf.resize(buf.len() + padding, 0u8);
    buf.extend_from_slice(&idx_bytes);
    let encoded = base64_encode(&buf);
    let data_uri = format!("data:application/octet-stream;base64,{encoded}");

    // ── Build JSON value ─────────────────────────────────────────────────────
    Ok(json!({
        "asset": { "version": "2.0", "generator": "RITK" },
        "scene": 0,
        "scenes": [{ "nodes": [0] }],
        "nodes": [{ "mesh": 0 }],
        "meshes": [{
            "name": "mesh",
            "primitives": [{
                "attributes": { "POSITION": 0 },
                "indices": 1,
                "mode": 4
            }]
        }],
        "accessors": [
            {
                "bufferView": 0,
                "componentType": 5126,
                "count": n_verts,
                "type": "VEC3",
                "min": [min_pos[0] as f64, min_pos[1] as f64, min_pos[2] as f64],
                "max": [max_pos[0] as f64, max_pos[1] as f64, max_pos[2] as f64]
            },
            {
                "bufferView": 1,
                "componentType": 5125,
                "count": n_indices,
                "type": "SCALAR"
            }
        ],
        "bufferViews": [
            {
                "buffer": 0,
                "byteOffset": 0,
                "byteLength": vert_byte_len,
                "target": 34962
            },
            {
                "buffer": 0,
                "byteOffset": idx_byte_offset,
                "byteLength": idx_byte_len,
                "target": 34963
            }
        ],
        "buffers": [{
            "uri": data_uri,
            "byteLength": total_bytes
        }]
    }))
}

// ── Geometry helpers ──────────────────────────────────────────────────────────

/// Fan triangulation: polygon `[v0,v1,…,vN-1]` →
/// triangles `(v0,v1,v2), (v0,v2,v3), …, (v0,vN-2,vN-1)`.
fn fan_triangulate(polygon: &[u32]) -> Vec<[u32; 3]> {
    if polygon.len() < 3 {
        return Vec::new();
    }
    let v0 = polygon[0];
    (1..polygon.len() - 1)
        .map(|i| [v0, polygon[i], polygon[i + 1]])
        .collect()
}

/// Axis-aligned bounding box of a point set.
/// Returns `([0;3],[0;3])` for an empty set.
fn aabb(points: &[[f32; 3]]) -> ([f32; 3], [f32; 3]) {
    if points.is_empty() {
        return ([0.0; 3], [0.0; 3]);
    }
    let mut lo = points[0];
    let mut hi = points[0];
    for &[x, y, z] in points {
        lo[0] = lo[0].min(x);
        lo[1] = lo[1].min(y);
        lo[2] = lo[2].min(z);
        hi[0] = hi[0].max(x);
        hi[1] = hi[1].max(y);
        hi[2] = hi[2].max(z);
    }
    (lo, hi)
}

// ── Base64 encoder (no external crate) ───────────────────────────────────────

/// RFC 4648 §4 base64 encoding.
const B64: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

fn base64_encode(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = if chunk.len() > 1 { chunk[1] as u32 } else { 0 };
        let b2 = if chunk.len() > 2 { chunk[2] as u32 } else { 0 };
        let n = (b0 << 16) | (b1 << 8) | b2;
        out.push(B64[((n >> 18) & 0x3F) as usize] as char);
        out.push(B64[((n >> 12) & 0x3F) as usize] as char);
        if chunk.len() > 1 {
            out.push(B64[((n >> 6) & 0x3F) as usize] as char);
        } else {
            out.push('=');
        }
        if chunk.len() > 2 {
            out.push(B64[(n & 0x3F) as usize] as char);
        } else {
            out.push('=');
        }
    }
    out
}
