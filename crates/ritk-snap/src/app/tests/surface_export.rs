use super::*;

#[test]
fn binary_mask_reports_empty_for_all_background() {
    let labels = vec![0_u32; 8];
    assert!(binary_mask_from_labels(&labels).is_none());
}

#[test]
fn binary_mask_marks_foreground_labels() {
    let labels = vec![0_u32, 2_u32, 0_u32, 7_u32];
    let binary = binary_mask_from_labels(&labels).expect("foreground expected");
    assert_eq!(binary, vec![0.0, 1.0, 0.0, 1.0]);
}

#[test]
fn build_label_surface_mesh_emits_faces_for_center_cube() {
    let nz = 4usize;
    let ny = 4usize;
    let nx = 4usize;
    let mut binary = vec![0.0f32; nz * ny * nx];
    for iz in 1..3 {
        for iy in 1..3 {
            for ix in 1..3 {
                binary[iz * ny * nx + iy * nx + ix] = 1.0;
            }
        }
    }

    let mesh = build_label_surface_mesh(&binary, [nz, ny, nx], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]);
    assert_eq!(mesh.face_count(), 44);
}

#[test]
fn marching_cubes_physical_positions_match_spacing() {
    let mut data = vec![0.0f32; 8];
    data[0] = 1.0;
    let mesh = build_label_surface_mesh(&data, [2, 2, 2], [2.0, 3.0, 4.0], [0.0, 0.0, 0.0]);

    assert_eq!(mesh.face_count(), 1);
    let n = mesh.vertex_count();
    let mut xs: Vec<f64> = (0..n)
        .map(|i| {
            mesh.vertices
                .position(gaia::domain::core::index::VertexId::new(i as u32))
                .x
        })
        .collect();
    let mut ys: Vec<f64> = (0..n)
        .map(|i| {
            mesh.vertices
                .position(gaia::domain::core::index::VertexId::new(i as u32))
                .y
        })
        .collect();
    let mut zs: Vec<f64> = (0..n)
        .map(|i| {
            mesh.vertices
                .position(gaia::domain::core::index::VertexId::new(i as u32))
                .z
        })
        .collect();
    xs.sort_by(|a, b| a.partial_cmp(b).expect("x finite"));
    ys.sort_by(|a, b| a.partial_cmp(b).expect("y finite"));
    zs.sort_by(|a, b| a.partial_cmp(b).expect("z finite"));

    assert!(
        (xs[2] - 1.0_f64).abs() < 1e-4,
        "edge 0 midpoint x = 1.0, got {}",
        xs[2]
    );
    assert!(
        (ys[2] - 1.5_f64).abs() < 1e-4,
        "edge 3 midpoint y = 1.5, got {}",
        ys[2]
    );
    assert!(
        (zs[2] - 2.0_f64).abs() < 1e-4,
        "edge 8 midpoint z = 2.0, got {}",
        zs[2]
    );
}
