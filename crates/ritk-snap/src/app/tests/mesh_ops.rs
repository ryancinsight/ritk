use crate::app::state::SnapApp;
use ritk_io::VtkPolyData;

fn unit_triangle_poly() -> VtkPolyData {
    VtkPolyData {
        points: vec![[0.0_f32, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        polygons: vec![vec![0, 1, 2]],
        ..Default::default()
    }
}

/// AABB midpoint for a unit right-angle triangle is (0.5, 0.5, 0.0).
#[test]
fn auto_camera_centers_on_aabb() {
    let poly = unit_triangle_poly();
    let cam = SnapApp::auto_camera_for_poly(&poly, 512, 512);
    assert!(
        (cam.target[0] - 0.5).abs() < 1e-6,
        "target.x expected 0.5, got {}",
        cam.target[0]
    );
    assert!(
        (cam.target[1] - 0.5).abs() < 1e-6,
        "target.y expected 0.5, got {}",
        cam.target[1]
    );
    assert!(
        cam.target[2].abs() < 1e-6,
        "target.z expected 0.0, got {}",
        cam.target[2]
    );
}

/// Eye must be above (larger Z than) the target for Z-up placement.
#[test]
fn auto_camera_eye_above_target() {
    let poly = unit_triangle_poly();
    let cam = SnapApp::auto_camera_for_poly(&poly, 512, 512);
    assert!(
        cam.eye[2] > cam.target[2],
        "eye.z ({}) must be > target.z ({})",
        cam.eye[2],
        cam.target[2]
    );
}

/// aspect = w / h = 800 / 400 = 2.0 exactly.
#[test]
fn auto_camera_aspect_ratio() {
    let poly = unit_triangle_poly();
    let cam = SnapApp::auto_camera_for_poly(&poly, 800, 400);
    assert!(
        (cam.aspect - 2.0_f32).abs() < 1e-6,
        "aspect expected 2.0, got {}",
        cam.aspect
    );
}

/// near must be strictly positive; far must exceed near.
#[test]
fn auto_camera_near_far_positive() {
    let poly = unit_triangle_poly();
    let cam = SnapApp::auto_camera_for_poly(&poly, 512, 512);
    assert!(cam.near > 0.0, "near must be > 0, got {}", cam.near);
    assert!(
        cam.far > cam.near,
        "far ({}) must be > near ({})",
        cam.far,
        cam.near
    );
}

/// Empty poly must not panic and must return a camera with correct aspect.
#[test]
fn auto_camera_empty_poly_no_panic() {
    let poly = VtkPolyData::default();
    let cam = SnapApp::auto_camera_for_poly(&poly, 640, 480);
    let expected = 640.0_f32 / 480.0_f32;
    assert!(
        (cam.aspect - expected).abs() < 1e-6,
        "empty poly: aspect expected {}, got {}",
        expected,
        cam.aspect
    );
    assert!(cam.near > 0.0, "near must be > 0, got {}", cam.near);
    assert!(
        cam.far > cam.near,
        "far ({}) must be > near ({})",
        cam.far,
        cam.near
    );
}
