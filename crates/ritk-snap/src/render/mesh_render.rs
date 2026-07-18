//! CPU Phong-shaded surface mesh renderer (GAP-262-VIZ-02).
//!
//! # Overview
//!
//! Converts a `VtkPolyData` mesh into an RGBA pixel buffer using a Z-buffer
//! rasterizer with per-face Phong lighting. The output buffer can be uploaded
//! as an [`egui::ColorImage`] for display in the viewer.
//!
//! # Algorithm
//!
//! 1. **Fan-triangulate** all polygons from `VtkPolyData::polygons`.
//! 2. **MVP transform**: world â†’ clip via view-projection matrix.
//! 3. **Clip & cull**: drop back-facing triangles and those outside the near/far planes.
//! 4. **Rasterize**: scan-line fill each triangle into an RGBA buffer with Z-buffer.
//! 5. **Phong shading**: per-face ambient + diffuse + specular lighting.
//!
//! # Limitations (documented)
//!
//! - Depth peeling OIT (order-independent transparency) and SSAO are deferred
//!   to a future GPU (wgpu) rendering pass.
//! - Back-face culling is performed in NDC space; winding order is assumed
//!   counter-clockwise when viewed from the front.
//! - Only `polygons` cells are rendered; `lines`, `vertices`, and
//!   `triangle_strips` are currently skipped.
//!
//! # Complexity
//!
//! O(T Â· A_avg) where T = number of triangles and A_avg = average triangle
//! screen area in pixels.

use ritk_io::VtkPolyData;
use std::f32::consts::PI;

// â”€â”€ Camera â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Pinhole perspective camera.
#[derive(Debug, Clone)]
pub struct MeshCamera {
    /// Eye position in world space.
    pub eye: [f32; 3],
    /// Look-at target in world space.
    pub target: [f32; 3],
    /// Up vector (normalized before use).
    pub up: [f32; 3],
    /// Vertical field of view in radians (default: Ï€/4).
    pub fov_y: f32,
    /// Viewport aspect ratio width/height.
    pub aspect: f32,
    /// Near clip plane distance (must be > 0).
    pub near: f32,
    /// Far clip plane distance (must be > near).
    pub far: f32 }

impl Default for MeshCamera {
    fn default() -> Self {
        Self {
            eye: [0.0, 0.0, 3.0],
            target: [0.0, 0.0, 0.0],
            up: [0.0, 1.0, 0.0],
            fov_y: PI / 4.0,
            aspect: 1.0,
            near: 0.1,
            far: 1000.0 }
    }
}

// â”€â”€ Material â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Phong material parameters.
#[derive(Debug, Clone)]
pub struct PhongMaterial {
    pub ambient: [f32; 3],
    pub diffuse: [f32; 3],
    pub specular: [f32; 3],
    pub shininess: f32,
    /// Overall opacity [0, 1].
    pub opacity: f32 }

impl Default for PhongMaterial {
    fn default() -> Self {
        Self {
            ambient: [0.1, 0.1, 0.1],
            diffuse: [0.8, 0.8, 0.8],
            specular: [0.5, 0.5, 0.5],
            shininess: 32.0,
            opacity: 1.0 }
    }
}

// â”€â”€ Light â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Directional (parallel) light source.
#[derive(Debug, Clone)]
pub struct DirectionalLight {
    /// Unit direction toward the light (world space, normalized).
    pub direction: [f32; 3],
    /// Light color RGB in [0, 1].
    pub color: [f32; 3] }

impl Default for DirectionalLight {
    fn default() -> Self {
        Self {
            direction: normalize([1.0, 1.0, 1.0]),
            color: [1.0, 1.0, 1.0] }
    }
}

// â”€â”€ Renderer â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// CPU Phong-shaded Z-buffer mesh renderer.
///
/// Renders a `VtkPolyData` into an RGBA byte buffer of shape
/// `[height * width * 4]` in row-major order.
pub struct MeshRenderer {
    pub width: usize,
    pub height: usize }

impl MeshRenderer {
    /// Create a renderer for a viewport of the given pixel dimensions.
    pub fn new(width: usize, height: usize) -> Self {
        Self { width, height }
    }

    /// Render `mesh` with the given camera, material, and lights.
    ///
    /// # Returns
    /// RGBA `Vec<u8>` of length `height * width * 4`, row-major (top row first).
    pub fn render(
        &self,
        mesh: &VtkPolyData,
        camera: &MeshCamera,
        material: &PhongMaterial,
        lights: &[DirectionalLight],
    ) -> Vec<u8> {
        let w = self.width;
        let h = self.height;
        let mut color_buf = vec![0u8; h * w * 4]; // RGBA
        let mut z_buf = vec![f32::INFINITY; h * w];

        // Build view and projection matrices.
        let view = look_at(camera.eye, camera.target, camera.up);
        let proj = perspective(camera.fov_y, camera.aspect, camera.near, camera.far);
        let mvp = mat4_mul(proj, view);

        let points = &mesh.points;
        if points.is_empty() {
            return color_buf;
        }

        // Fan-triangulate all polygons and render each triangle.
        for poly in &mesh.polygons {
            if poly.len() < 3 {
                continue;
            }
            let v0 = poly[0] as usize;
            for i in 1..(poly.len() - 1) {
                let v1 = poly[i] as usize;
                let v2 = poly[i + 1] as usize;
                if v0 >= points.len() || v1 >= points.len() || v2 >= points.len() {
                    continue;
                }
                rasterize_triangle(
                    points[v0],
                    points[v1],
                    points[v2],
                    &mvp,
                    camera,
                    material,
                    lights,
                    w,
                    h,
                    &mut color_buf,
                    &mut z_buf,
                );
            }
        }

        color_buf
    }
}

// â”€â”€ Triangle rasterization â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Project a world-space point through the MVP matrix and perform perspective
/// divide to obtain NDC coordinates.
///
/// Returns `(x_ndc, y_ndc, z_ndc, w_clip)`. If `w_clip <= 0`, the point is
/// behind the camera.
fn project(p: [f32; 3], mvp: &[f32; 16]) -> (f32, f32, f32, f32) {
    let x = mvp[0] * p[0] + mvp[4] * p[1] + mvp[8] * p[2] + mvp[12];
    let y = mvp[1] * p[0] + mvp[5] * p[1] + mvp[9] * p[2] + mvp[13];
    let z = mvp[2] * p[0] + mvp[6] * p[1] + mvp[10] * p[2] + mvp[14];
    let w = mvp[3] * p[0] + mvp[7] * p[1] + mvp[11] * p[2] + mvp[15];
    if w.abs() < 1e-10 {
        return (x, y, z, w);
    }
    (x / w, y / w, z / w, w)
}

/// Rasterize one triangle into the color/z buffers.
#[allow(clippy::too_many_arguments)]
fn rasterize_triangle(
    p0: [f32; 3],
    p1: [f32; 3],
    p2: [f32; 3],
    mvp: &[f32; 16],
    camera: &MeshCamera,
    material: &PhongMaterial,
    lights: &[DirectionalLight],
    w: usize,
    h: usize,
    color_buf: &mut [u8],
    z_buf: &mut [f32],
) {
    let (x0, y0, z0, w0) = project(p0, mvp);
    let (x1, y1, z1, w1) = project(p1, mvp);
    let (x2, y2, z2, w2) = project(p2, mvp);

    // Discard if any vertex is behind the camera
    if w0 <= 0.0 || w1 <= 0.0 || w2 <= 0.0 {
        return;
    }

    // Back-face culling in NDC (y-up): CCW triangles (cross_z > 0) are front-facing.
    // Cull when cross_z <= 0 (CW = back-facing).
    let cross_z = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0);
    if cross_z <= 0.0 {
        return;
    }

    // NDC â†’ screen coordinates (integer pixel centers)
    let to_screen = |xn: f32, yn: f32| -> (f32, f32) {
        ((xn * 0.5 + 0.5) * w as f32, (0.5 - yn * 0.5) * h as f32)
    };

    let (sx0, sy0) = to_screen(x0, y0);
    let (sx1, sy1) = to_screen(x1, y1);
    let (sx2, sy2) = to_screen(x2, y2);

    // Compute face normal in world space for Phong lighting
    let face_normal = compute_face_normal(p0, p1, p2);
    let phong_color = phong_shade(face_normal, camera.eye, p0, material, lights);

    let alpha = (material.opacity * super::U8_MAX_F32).clamp(0.0, super::U8_MAX_F32) as u8;

    // Bounding box scan-line rasterization with Z-buffer
    let min_x = (sx0.min(sx1).min(sx2).floor() as i32).max(0);
    let max_x = (sx0.max(sx1).max(sx2).ceil() as i32).min(w as i32 - 1);
    let min_y = (sy0.min(sy1).min(sy2).floor() as i32).max(0);
    let max_y = (sy0.max(sy1).max(sy2).ceil() as i32).min(h as i32 - 1);

    for py in min_y..=max_y {
        for px in min_x..=max_x {
            let fx = px as f32 + 0.5;
            let fy = py as f32 + 0.5;

            // Barycentric coordinates
            let denom = (sy1 - sy2) * (sx0 - sx2) + (sx2 - sx1) * (sy0 - sy2);
            if denom.abs() < 1e-10 {
                continue;
            }
            let b0 = ((sy1 - sy2) * (fx - sx2) + (sx2 - sx1) * (fy - sy2)) / denom;
            let b1 = ((sy2 - sy0) * (fx - sx2) + (sx0 - sx2) * (fy - sy2)) / denom;
            let b2 = 1.0 - b0 - b1;

            if b0 < 0.0 || b1 < 0.0 || b2 < 0.0 {
                continue;
            }

            // Interpolated z depth (NDC z)
            let z = b0 * z0 + b1 * z1 + b2 * z2;

            let idx = py as usize * w + px as usize;
            if z < z_buf[idx] {
                z_buf[idx] = z;
                let base = idx * 4;
                color_buf[base] =
                    (phong_color[0] * super::U8_MAX_F32).clamp(0.0, super::U8_MAX_F32) as u8;
                color_buf[base + 1] =
                    (phong_color[1] * super::U8_MAX_F32).clamp(0.0, super::U8_MAX_F32) as u8;
                color_buf[base + 2] =
                    (phong_color[2] * super::U8_MAX_F32).clamp(0.0, super::U8_MAX_F32) as u8;
                color_buf[base + 3] = alpha;
            }
        }
    }
}

// â”€â”€ Phong lighting â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Compute Phong shading for a face.
///
/// # Equation
///
/// ```text
/// I = k_a Â· I_a + Î£_i [ k_d Â· max(nÂ·l_i, 0) Â· I_i
///                       + k_s Â· max(r_iÂ·v, 0)^s Â· I_i ]
/// ```
///
/// where:
/// - `k_a, k_d, k_s` = material ambient/diffuse/specular
/// - `n` = face normal (normalized)
/// - `l_i` = unit vector toward light i
/// - `r_i = 2(nÂ·l_i)n âˆ’ l_i` = specular reflection vector
/// - `v` = unit vector toward camera (view direction)
/// - `s` = shininess exponent
pub fn phong_shade(
    normal: [f32; 3],
    eye: [f32; 3],
    surface_point: [f32; 3],
    material: &PhongMaterial,
    lights: &[DirectionalLight],
) -> [f32; 3] {
    let n = normalize(normal);
    let v = normalize(sub3(eye, surface_point));

    let mut color = material.ambient;

    for light in lights {
        let l = normalize(light.direction);
        let n_dot_l = dot(n, l).max(0.0);
        // Diffuse
        for ((c, diff), lc) in color.iter_mut().zip(material.diffuse).zip(light.color) {
            *c += diff * n_dot_l * lc;
        }
        // Specular (Phong reflection model)
        if n_dot_l > 0.0 {
            let r = sub3(scale3(n, 2.0 * dot(n, l)), l);
            let r_dot_v = dot(r, v).max(0.0);
            let spec = r_dot_v.powf(material.shininess);
            for ((c, spec_val), lc) in color.iter_mut().zip(material.specular).zip(light.color) {
                *c += spec_val * spec * lc;
            }
        }
    }

    [
        color[0].clamp(0.0, 1.0),
        color[1].clamp(0.0, 1.0),
        color[2].clamp(0.0, 1.0),
    ]
}

// â”€â”€ Matrix math â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Column-major 4Ã—4 matrix multiplication: C = A * B.
pub fn mat4_mul(a: [f32; 16], b: [f32; 16]) -> [f32; 16] {
    let mut c = [0.0_f32; 16];
    for row in 0..4 {
        for col in 0..4 {
            for k in 0..4 {
                c[col * 4 + row] += a[k * 4 + row] * b[col * 4 + k];
            }
        }
    }
    c
}

/// LookAt view matrix (column-major).
///
/// Constructs an orthonormal right-handed view basis from eye, target, up.
pub fn look_at(eye: [f32; 3], target: [f32; 3], up: [f32; 3]) -> [f32; 16] {
    let f = normalize(sub3(target, eye)); // forward
    let r = normalize(cross(f, normalize(up))); // right
    let u = cross(r, f); // true up

    // Column-major: column j is [r[j], u[j], -f[j], 0] for j<3, then translation
    [
        r[0],
        u[0],
        -f[0],
        0.0,
        r[1],
        u[1],
        -f[1],
        0.0,
        r[2],
        u[2],
        -f[2],
        0.0,
        -dot(r, eye),
        -dot(u, eye),
        dot(f, eye),
        1.0,
    ]
}

/// Symmetric perspective projection matrix (column-major, OpenGL convention).
///
/// Maps view frustum to NDC cube [-1,1]Â³ with depth range [-1,1].
pub fn perspective(fov_y: f32, aspect: f32, near: f32, far: f32) -> [f32; 16] {
    let f = 1.0 / (fov_y / 2.0).tan();
    let d = near - far;
    let mut m = [0.0_f32; 16];
    m[0] = f / aspect;
    m[5] = f;
    m[10] = (far + near) / d;
    m[11] = -1.0;
    m[14] = 2.0 * far * near / d;
    m
}

/// Compute the outward face normal of a triangle (p0, p1, p2).
///
/// The normal is normalized. If the triangle is degenerate (zero area),
/// returns `[0, 0, 1]`.
pub fn compute_face_normal(p0: [f32; 3], p1: [f32; 3], p2: [f32; 3]) -> [f32; 3] {
    let e0 = sub3(p1, p0);
    let e1 = sub3(p2, p0);
    let n = cross(e0, e1);
    let len = (dot(n, n)).sqrt();
    if len < 1e-12 {
        return [0.0, 0.0, 1.0];
    }
    [n[0] / len, n[1] / len, n[2] / len]
}

// â”€â”€ Vector helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

#[inline]
pub(crate) fn normalize(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len < 1e-12 {
        return [0.0, 0.0, 0.0];
    }
    [v[0] / len, v[1] / len, v[2] / len]
}

#[inline]
pub(crate) fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
pub(crate) fn sub3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline]
pub(crate) fn scale3(v: [f32; 3], s: f32) -> [f32; 3] {
    [v[0] * s, v[1] * s, v[2] * s]
}

#[inline]
pub(crate) fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

// â”€â”€ Tests â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

#[cfg(test)]
#[path = "tests_mesh_render.rs"]
mod tests_mesh_render;
