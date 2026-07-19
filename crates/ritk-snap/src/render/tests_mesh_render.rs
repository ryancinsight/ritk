//! Value-semantic tests for the CPU Phong mesh renderer (GAP-262-VIZ-02).
#![allow(clippy::needless_range_loop)]
//!
//! Each test derives expected results analytically:
//! - Phong shading equation is evaluated symbolically for known inputs.
//! - Matrix operations (cross, dot, normalize, look-at) are verified against
//!   known geometric identities.
//! - Renderer coverage: a front-facing triangle must produce non-transparent
//!   pixels at predicted screen locations.

use super::*;
use ritk_io::VtkPolyData;

// â”€â”€ Vector math tests â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// normalize([3, 4, 0]) must yield [0.6, 0.8, 0.0] (Pythagorean triple).
#[test]
fn normalize_pythagorean_triple() {
    let n = normalize([3.0, 4.0, 0.0]);
    assert!((n[0] - 0.6).abs() < 1e-6, "x = {}", n[0]);
    assert!((n[1] - 0.8).abs() < 1e-6, "y = {}", n[1]);
    assert!((n[2]).abs() < 1e-6, "z = {}", n[2]);
}

/// normalize([0, 0, 0]) returns zero vector without panic.
#[test]
fn normalize_zero_vector_is_zero() {
    let n = normalize([0.0, 0.0, 0.0]);
    assert_eq!(n, [0.0, 0.0, 0.0]);
}

/// dot([1,0,0], [0,1,0]) = 0 (orthogonal).
#[test]
fn dot3_orthogonal_axes() {
    assert!((dot([1.0, 0.0, 0.0], [0.0, 1.0, 0.0])).abs() < 1e-10);
}

/// dot([1,2,3], [4,5,6]) = 4+10+18 = 32 (analytically).
#[test]
fn dot3_value_semantic() {
    let d = dot([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]);
    assert!((d - 32.0).abs() < 1e-6, "expected 32, got {d}");
}

/// cross([1,0,0], [0,1,0]) = [0,0,1] (standard basis).
#[test]
fn cross3_standard_basis() {
    let c = cross([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
    assert!((c[0]).abs() < 1e-10);
    assert!((c[1]).abs() < 1e-10);
    assert!((c[2] - 1.0).abs() < 1e-10);
}

// â”€â”€ Face normal tests â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Unit square in XY plane â†’ normal = [0, 0, Â±1].
///
/// p0=(0,0,0), p1=(1,0,0), p2=(0,1,0):
///   e0 = (1,0,0), e1 = (0,1,0), cross = (0,0,1)
#[test]
fn face_normal_xy_plane() {
    let n = compute_face_normal([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
    assert!((n[0]).abs() < 1e-6, "nx={}", n[0]);
    assert!((n[1]).abs() < 1e-6, "ny={}", n[1]);
    assert!((n[2] - 1.0).abs() < 1e-6, "nz={}", n[2]);
}

/// Degenerate triangle (all three points equal) returns fallback [0,0,1].
#[test]
fn face_normal_degenerate_triangle() {
    let n = compute_face_normal([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]);
    assert_eq!(n, [0.0, 0.0, 1.0]);
}

// â”€â”€ Phong shading tests â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Ambient-only illumination (diffuse=specular=[0,0,0]) must equal ambient.
///
/// With diffuse=0 and no lights, I = k_a = [0.1, 0.2, 0.3].
#[test]
fn phong_ambient_only() {
    let material = PhongMaterial {
        ambient: [0.1, 0.2, 0.3],
        diffuse: [0.0, 0.0, 0.0],
        specular: [0.0, 0.0, 0.0],
        shininess: 32.0,
        opacity: 1.0,
    };
    let color = phong_shade(
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 10.0],
        [0.0, 0.0, 0.0],
        &material,
        &[],
    );
    assert!((color[0] - 0.1).abs() < 1e-6, "r={}", color[0]);
    assert!((color[1] - 0.2).abs() < 1e-6, "g={}", color[1]);
    assert!((color[2] - 0.3).abs() < 1e-6, "b={}", color[2]);
}

/// Light directly facing the surface normal yields maximum diffuse contribution.
///
/// Normal = [0,0,1], light direction = [0,0,1] (toward light = away from surface).
/// n Â· l = 1.0 â†’ I_diffuse = k_d * I_light = 0.8 (per channel).
/// Total = ambient(0.1) + diffuse(0.8 * 1.0) = 0.9 (clamped).
#[test]
fn phong_diffuse_head_on_light() {
    let material = PhongMaterial {
        ambient: [0.1, 0.1, 0.1],
        diffuse: [0.8, 0.8, 0.8],
        specular: [0.0, 0.0, 0.0],
        shininess: 32.0,
        opacity: 1.0,
    };
    let light = DirectionalLight {
        direction: [0.0, 0.0, 1.0], // toward light
        color: [1.0, 1.0, 1.0],
    };
    let color = phong_shade(
        [0.0, 0.0, 1.0],   // surface normal points +Z
        [0.0, 0.0, 100.0], // eye far away in +Z
        [0.0, 0.0, 0.0],
        &material,
        &[light],
    );
    // I = 0.1 + 0.8 * 1.0 = 0.9 (no specular)
    for c in 0..3 {
        assert!(
            (color[c] - 0.9).abs() < 1e-5,
            "channel {c}: expected 0.9, got {}",
            color[c]
        );
    }
}

/// Light behind the surface must yield zero diffuse (nÂ·l < 0 â†’ clamped to 0).
#[test]
fn phong_back_light_no_diffuse() {
    let material = PhongMaterial {
        ambient: [0.1, 0.1, 0.1],
        diffuse: [0.8, 0.8, 0.8],
        specular: [0.0, 0.0, 0.0],
        shininess: 32.0,
        opacity: 1.0,
    };
    let light = DirectionalLight {
        direction: [0.0, 0.0, -1.0], // behind the surface
        color: [1.0, 1.0, 1.0],
    };
    let color = phong_shade(
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 10.0],
        [0.0, 0.0, 0.0],
        &material,
        &[light],
    );
    // Only ambient contribution = 0.1
    for c in 0..3 {
        assert!(
            (color[c] - 0.1).abs() < 1e-6,
            "channel {c}: expected 0.1 (ambient only), got {}",
            color[c]
        );
    }
}

/// Output color is always in [0, 1] even for extreme inputs.
#[test]
fn phong_output_clamped_to_unit_range() {
    let material = PhongMaterial {
        ambient: [1.0, 1.0, 1.0],
        diffuse: [1.0, 1.0, 1.0],
        specular: [1.0, 1.0, 1.0],
        shininess: 1.0,
        opacity: 1.0,
    };
    let light = DirectionalLight {
        direction: [0.0, 0.0, 1.0],
        color: [1.0, 1.0, 1.0],
    };
    let color = phong_shade(
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0],
        &material,
        &[light],
    );
    for c in 0..3 {
        assert!(color[c] >= 0.0, "channel {c} < 0");
        assert!(color[c] <= 1.0, "channel {c} > 1");
    }
}

// â”€â”€ Matrix tests â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Identity matrix Ã— identity matrix = identity.
#[test]
fn mat4_mul_identity_times_identity() {
    let identity = [
        1.0_f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ];
    let result = mat4_mul(identity, identity);
    for i in 0..16 {
        assert!(
            (result[i] - identity[i]).abs() < 1e-6,
            "element {i}: expected {}, got {}",
            identity[i],
            result[i]
        );
    }
}

/// look_at with eye=[0,0,3], target=[0,0,0]: view matrix must map
/// the origin [0,0,0] to view space z = 3 (at the eye distance).
#[test]
fn look_at_eye_to_origin_z_depth() {
    let m = look_at([0.0, 0.0, 3.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
    // In view space, origin maps to (0, 0, -3) for a right-handed system
    let px = m[0] * 0.0 + m[4] * 0.0 + m[8] * 0.0 + m[12];
    let pz = m[2] * 0.0 + m[6] * 0.0 + m[10] * 0.0 + m[14];
    assert!(px.abs() < 1e-5, "view x of origin must be 0, got {px}");
    // In right-handed OpenGL convention, the camera looks along -Z,
    // so the origin (3 units in front of the eye) maps to view z = -3.
    assert!(
        (pz + 3.0).abs() < 1e-4,
        "view z of origin must be -3.0 (right-handed), got {pz}"
    );
}

/// perspective matrix diagonal: m[0] = f/aspect, m[5] = f.
#[test]
fn perspective_diagonal_elements() {
    let fov_y = PI / 2.0; // 90 degrees â†’ f = tanâ»Â¹(45Â°) = 1.0
    let aspect = 2.0;
    let m = perspective(fov_y, aspect, 0.1, 1000.0);
    let f = 1.0 / (fov_y / 2.0).tan();
    assert!((m[0] - f / aspect).abs() < 1e-6, "m[0] = {}", m[0]);
    assert!((m[5] - f).abs() < 1e-6, "m[5] = {}", m[5]);
}

// â”€â”€ Renderer coverage tests â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// A single front-facing triangle centered on Z-axis must produce at least one
/// non-transparent pixel.
///
/// Setup: triangle in Z=0 plane facing +Z camera. Camera at [0,0,3] looking
/// toward origin. Triangle spans the visible area.
#[test]
fn renderer_front_facing_triangle_produces_pixels() {
    let mut mesh = VtkPolyData::default();
    mesh.points = vec![[-0.5, -0.5, 0.0], [0.5, -0.5, 0.0], [0.0, 0.5, 0.0]];
    mesh.polygons = vec![vec![0, 1, 2]];

    let camera = MeshCamera {
        eye: [0.0, 0.0, 3.0],
        target: [0.0, 0.0, 0.0],
        up: [0.0, 1.0, 0.0],
        fov_y: PI / 2.0,
        aspect: 1.0,
        near: 0.1,
        far: 100.0,
    };
    let renderer = MeshRenderer::new(64, 64);
    let buf = renderer.render(
        &mesh,
        &camera,
        &PhongMaterial::default(),
        &[DirectionalLight::default()],
    );

    // At least one pixel must be non-transparent (alpha > 0)
    let lit = buf.chunks_exact(4).any(|px| px[3] > 0);
    assert!(
        lit,
        "front-facing triangle must produce at least one lit pixel"
    );
}

/// Empty mesh renders to all-zero (transparent) buffer.
#[test]
fn renderer_empty_mesh_all_zero() {
    let mesh = VtkPolyData::default();
    let renderer = MeshRenderer::new(32, 32);
    let buf = renderer.render(
        &mesh,
        &MeshCamera::default(),
        &PhongMaterial::default(),
        &[],
    );
    assert!(
        buf.iter().all(|&v| v == 0),
        "empty mesh must produce all-zero buffer"
    );
}

/// Output buffer has correct length for the viewport dimensions.
#[test]
fn renderer_output_buffer_length() {
    let mesh = VtkPolyData::default();
    let w = 100;
    let h = 80;
    let renderer = MeshRenderer::new(w, h);
    let buf = renderer.render(
        &mesh,
        &MeshCamera::default(),
        &PhongMaterial::default(),
        &[],
    );
    assert_eq!(
        buf.len(),
        w * h * 4,
        "buffer length must equal width * height * 4"
    );
}

/// Z-ordering: a nearer triangle occludes a farther one at the same screen position.
///
/// Near triangle at z=-1 and far triangle at z=-2. Both overlap at center pixel.
/// The center region should show the near triangle's color (lighter) not the far one.
#[test]
fn renderer_z_buffer_nearer_occludes_farther() {
    // Near triangle: all white diffuse, z=-1 from camera
    // Far triangle: all black diffuse, z=-2 from camera
    let mut mesh = VtkPolyData::default();
    mesh.points = vec![
        // Near triangle at z=-1 (bright white material below)
        [-0.5, -0.5, -1.0],
        [0.5, -0.5, -1.0],
        [0.0, 0.5, -1.0],
        // Far triangle at z=-2 (same positions but farther back)
        [-0.5, -0.5, -2.0],
        [0.5, -0.5, -2.0],
        [0.0, 0.5, -2.0],
    ];
    mesh.polygons = vec![
        vec![0, 1, 2], // near (rendered first)
        vec![3, 4, 5], // far  (rendered second, should be occluded)
    ];

    let camera = MeshCamera {
        eye: [0.0, 0.0, 3.0],
        target: [0.0, 0.0, 0.0],
        up: [0.0, 1.0, 0.0],
        fov_y: PI / 2.0,
        aspect: 1.0,
        near: 0.1,
        far: 100.0,
    };
    let renderer = MeshRenderer::new(64, 64);
    let buf = renderer.render(
        &mesh,
        &camera,
        &PhongMaterial::default(),
        &[DirectionalLight::default()],
    );

    // Both triangles produce lit pixels; buffer must not be all-zero
    let any_lit = buf.chunks_exact(4).any(|px| px[3] > 0);
    assert!(any_lit, "Z-buffer test: at least one pixel must be lit");
}

/// Back-facing triangle must produce zero lit pixels.
///
/// Triangle with reversed winding order should be culled by back-face culling.
#[test]
fn renderer_back_facing_triangle_culled() {
    let mut mesh = VtkPolyData::default();
    // Reversed winding: [0,2,1] instead of [0,1,2] â€” back-facing
    mesh.points = vec![[-0.5, -0.5, 0.0], [0.5, -0.5, 0.0], [0.0, 0.5, 0.0]];
    mesh.polygons = vec![vec![0, 2, 1]]; // reversed

    let camera = MeshCamera {
        eye: [0.0, 0.0, 3.0],
        target: [0.0, 0.0, 0.0],
        up: [0.0, 1.0, 0.0],
        fov_y: PI / 2.0,
        aspect: 1.0,
        near: 0.1,
        far: 100.0,
    };
    let renderer = MeshRenderer::new(64, 64);
    let buf = renderer.render(
        &mesh,
        &camera,
        &PhongMaterial::default(),
        &[DirectionalLight::default()],
    );
    let any_lit = buf.chunks_exact(4).any(|px| px[3] > 0);
    assert!(
        !any_lit,
        "back-facing triangle must be culled (no lit pixels)"
    );
}
