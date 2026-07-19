//! GPU uniform buffer structs and SSAO kernel for mesh rendering.
//!
//! # Layout invariants (std140 / WebGPU alignment)
//!
//! | Struct | Size (bytes) | Alignment |
//! |---------------------|-------------|-----------|
//! | `MeshVertex` | 32 | 16 |
//! | `SceneUniforms` | 144 | 16 |
//! | `LightUniform` | 48 | 16 |
//! | `LightBlock` | 96 | 16 |
//! | `MaterialUniforms` | 48 | 16 |
//! | `SsaoUniforms` | 48 | 16 |
//! | `CompositeUniforms` | 16 | 4 |
//!
//! All structs are `#[repr(C)]` + `bytemuck::{Pod, Zeroable}` for safe
//! byte-cast upload via `queue.write_buffer` / `create_buffer_init`.

use bytemuck::{Pod, Zeroable};
use std::f32::consts::TAU;

// â”€â”€ Vertex layout â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Interleaved vertex: position (xyz + pad) + normal (xyz + pad) = 32 bytes.
///
/// Stride 32 matches `wgpu::VertexFormat::Float32x4` Ã— 2 attributes.
/// Padding is required for 16-byte alignment of each attribute slot.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub(super) struct MeshVertex {
    /// World-space position, W = 1.0.
    pub position: [f32; 3],
    pub _pad0: f32,
    /// World-space normal (unit length).
    pub normal: [f32; 3],
    pub _pad1: f32,
}

impl MeshVertex {
    pub const STRIDE: u64 = 32;

    /// Vertex buffer layout for `wgpu::RenderPipeline`.
    ///
    /// Attribute 0 â†’ `@location(0)` position vec4
    /// Attribute 1 â†’ `@location(1)` normal vec4
    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        use wgpu::{VertexAttribute, VertexFormat, VertexStepMode};
        wgpu::VertexBufferLayout {
            array_stride: Self::STRIDE,
            step_mode: VertexStepMode::Vertex,
            attributes: &[
                VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: VertexFormat::Float32x4,
                },
                VertexAttribute {
                    offset: 16,
                    shader_location: 1,
                    format: VertexFormat::Float32x4,
                },
            ],
        }
    }
}

// â”€â”€ Scene / transform uniforms â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Model-view-projection and model-view matrices plus peel-pass index.
///
/// `mvp` (column-major): world â†’ clip
/// `mv` (column-major): world â†’ view, used for normal transform
/// `peel_pass`: 0 = base pass (no depth discard), i>0 = peel layer i
///
/// std140: 2 Ã— 64 + 4 + 12 = 144 bytes, align 16.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub(super) struct SceneUniforms {
    pub mvp: [f32; 16],
    pub mv: [f32; 16],
    pub peel_pass: u32,
    pub _pad: [u32; 3],
}

// â”€â”€ Light uniforms â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// One directional light in view space.
///
/// Each `vec3<f32>` field is padded to 16 bytes (WGSL std140 `vec3` alignment).
/// struct size: 48 bytes, align 16.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub(super) struct LightUniform {
    /// Unit direction toward the light in view space.
    pub direction_view: [f32; 3],
    pub _pad0: f32,
    /// RGB light color in [0, 1].
    pub color: [f32; 3],
    pub _pad1: f32,
    /// RGB ambient contribution in [0, 1].
    pub ambient: [f32; 3],
    pub _pad2: f32,
}

/// Block of 2 directional lights. 2 Ã— 48 = 96 bytes.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub(super) struct LightBlock {
    pub lights: [LightUniform; 2],
}

// â”€â”€ Material uniforms â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Phong material packed as three `vec4<f32>` (3 Ã— 16 = 48 bytes).
///
/// Layout:
/// - `diffuse` (16 bytes): RGBA diffuse color
/// - `specular_shine` (16 bytes): xyz = specular RGB, w = shininess exponent
/// - `opacity_pad` (16 bytes): x = opacity `[0, 1]`, yzw = padding
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub(super) struct MaterialUniforms {
    pub diffuse: [f32; 4],
    pub specular_shine: [f32; 4],
    pub opacity_pad: [f32; 4],
}

// â”€â”€ SSAO uniforms â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Parameters for the SSAO compute pass.
///
/// `focal_x`, `focal_y`: perspective focal lengths (`proj[0]`, `proj[5]`).
/// `n_samples`: active hemisphere samples in `[1, 16]`.
/// `strength`: AO blend factor; 0.0 disables occlusion effect.
///
/// std140: 12 scalars Ã— 4 = 48 bytes, align 4 (largest scalar).
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub(super) struct SsaoUniforms {
    pub near: f32,
    pub far: f32,
    pub focal_x: f32,
    pub focal_y: f32,
    pub radius: f32,
    pub bias: f32,
    pub n_samples: u32,
    pub strength: f32,
    pub viewport_w: u32,
    pub viewport_h: u32,
    pub _pad: [u32; 2],
}

// â”€â”€ Composite uniforms â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Viewport dimensions for the composite compute pass. 16 bytes.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub(super) struct CompositeUniforms {
    pub rows: u32,
    pub cols: u32,
    pub _pad: [u32; 2],
}

// â”€â”€ SSAO hemisphere kernel â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Build a 16-sample cosine-weighted hemisphere kernel in view space.
///
/// # Algorithm
///
/// For sample `i` (i = 1 .. 16):
/// - `h1 = halton(i, 2)` âˆˆ (0, 1) â€” controls polar angle
/// - `h2 = halton(i, 3)` âˆˆ (0, 1) â€” controls azimuth
/// - `z = h1` (always > 0 â†’ positive hemisphere, i.e., toward surface normal)
/// - `r_xy = âˆš(1 âˆ’ h1Â²)`
/// - `x = r_xy Â· cos(2Ï€Â·h2)`, `y = r_xy Â· sin(2Ï€Â·h2)`
/// - `scale = lerp(0.1, 1.0, (i/16)Â²)` â€” concentrate samples near origin
/// - Final entry: `(xÂ·scale, yÂ·scale, zÂ·scale, 1.0)`
///
/// # Invariant
///
/// `z > 0` for all 16 samples (Halton(i,2) âˆˆ (0,1) for i â‰¥ 1).
pub(super) fn build_ssao_kernel() -> [[f32; 4]; 16] {
    let mut k = [[0.0f32; 4]; 16];
    for (i, entry) in k.iter_mut().enumerate() {
        let h1 = halton((i + 1) as u32, 2);
        let h2 = halton((i + 1) as u32, 3);
        let phi = TAU * h2;
        let r_xy = (1.0 - h1 * h1).sqrt();
        let x = r_xy * phi.cos();
        let y = r_xy * phi.sin();
        let z = h1; // z > 0 âˆ€ i â‰¥ 1
        let t = (i + 1) as f32 / 16.0;
        let scale = 0.1 + 0.9 * t * t; // lerp(0.1, 1.0, tÂ²)
        *entry = [x * scale, y * scale, z * scale, 1.0];
    }
    k
}

/// Halton quasi-random low-discrepancy sequence in [0, 1).
///
/// `halton(i, b) = Î£_k d_k / b^k` where `d_k` is the k-th digit of `i` in base `b`.
fn halton(mut i: u32, base: u32) -> f32 {
    let mut f = 1.0f32;
    let mut r = 0.0f32;
    let b = base as f32;
    while i > 0 {
        f /= b;
        r += f * (i % base) as f32;
        i /= base;
    }
    r
}

#[cfg(test)]
mod tests {
    use super::*;

    /// SSAO kernel: all 16 samples must have z > 0 (positive hemisphere).
    #[test]
    fn ssao_kernel_all_positive_z() {
        let k = build_ssao_kernel();
        for (i, entry) in k.iter().enumerate() {
            assert!(entry[2] > 0.0, "sample {i}: z = {} must be > 0", entry[2]);
        }
    }

    /// SSAO kernel: scale factor must be in [0.1, 1.0] for all samples.
    #[test]
    fn ssao_kernel_scale_in_range() {
        let k = build_ssao_kernel();
        for (i, entry) in k.iter().enumerate() {
            let len = (entry[0].powi(2) + entry[1].powi(2) + entry[2].powi(2)).sqrt();
            // scale = lerp(0.1, 1.0, tÂ²) âˆˆ [0.1, 1.0]; the unscaled unit vector has norm 1.
            // Actual norm = scale Ã— ||(x,y,z)|| where ||(x,y,z)|| = 1 â†’ len = scale âˆˆ [0.1, 1.0].
            assert!(
                (0.09..=1.01).contains(&len),
                "sample {i}: len = {len} not in [0.1, 1.0]"
            );
        }
    }

    /// Halton base-2 sequence first four terms: 1/2, 1/4, 3/4, 1/8.
    #[test]
    fn halton_base2_first_four() {
        let expected = [0.5f32, 0.25, 0.75, 0.125];
        for (i, &exp) in expected.iter().enumerate() {
            let got = halton((i + 1) as u32, 2);
            assert!(
                (got - exp).abs() < 1e-6,
                "halton({}, 2) = {got}, expected {exp}",
                i + 1
            );
        }
    }
}
