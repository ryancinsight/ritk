use super::*;
use crate::domain::vtk_data_object::{VtkDataObject, VtkPolyData};

/// Triangle: [0,0,0], [1,0,0], [0.5,1,0]
fn triangle() -> VtkPolyData {
    VtkPolyData {
        points: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0]],
        polygons: vec![vec![0, 1, 2]],
        ..Default::default()
    }
}

#[test]
fn zero_iterations_leaves_points_unchanged() {
    let f = SmoothFilter::new(0.5, 0);
    let original = triangle();
    let out = f
        .execute(VtkDataObject::PolyData(original.clone()))
        .expect("infallible: validated precondition");
    let VtkDataObject::PolyData(p) = out else {
        panic!()
    };
    for (orig, smoothed) in original.points.iter().zip(p.points.iter()) {
        assert_eq!(orig, smoothed, "zero iterations must not move any vertex");
    }
}

#[test]
fn one_iteration_moves_vertex_toward_neighbor_mean() {
    // Triangle [0,0,0], [1,0,0], [0.5,1,0], polygon [0,1,2].
    // adj[0] = {1, 2}: mean = ([1,0,0]+[0.5,1,0])/2 = [0.75, 0.5, 0]
    // v0' = 0.5*[0,0,0] + 0.5*[0.75,0.5,0] = [0.375, 0.25, 0]
    let f = SmoothFilter::new(0.5, 1);
    let out = f
        .execute(VtkDataObject::PolyData(triangle()))
        .expect("infallible: validated precondition");
    let VtkDataObject::PolyData(p) = out else {
        panic!()
    };
    let v0 = p.points[0];
    assert!(
        (v0[0] - 0.375).abs() < 1e-5,
        "v0.x after 1 step: expected 0.375, got {}",
        v0[0]
    );
    assert!(
        (v0[1] - 0.25).abs() < 1e-5,
        "v0.y after 1 step: expected 0.25, got {}",
        v0[1]
    );
    assert!(v0[2].abs() < 1e-5, "v0.z must stay 0: got {}", v0[2]);
}

#[test]
fn relaxation_factor_one_snaps_fully_to_mean() {
    // λ=1 → v_i' = mean(neighbors(v_i)) after 1 iteration.
    let f = SmoothFilter::new(1.0, 1);
    let out = f
        .execute(VtkDataObject::PolyData(triangle()))
        .expect("infallible: validated precondition");
    let VtkDataObject::PolyData(p) = out else {
        panic!()
    };
    let v0 = p.points[0];
    // mean of neighbors {1=[1,0,0], 2=[0.5,1,0]} = [0.75, 0.5, 0]
    assert!(
        (v0[0] - 0.75).abs() < 1e-5,
        "v0.x with λ=1: expected 0.75, got {}",
        v0[0]
    );
    assert!(
        (v0[1] - 0.5).abs() < 1e-5,
        "v0.y with λ=1: expected 0.5, got {}",
        v0[1]
    );
}

#[test]
fn topology_preserved_after_smoothing() {
    let f = SmoothFilter::default();
    let original = triangle();
    let original_polygons = original.polygons.clone();
    let out = f
        .execute(VtkDataObject::PolyData(original))
        .expect("infallible: validated precondition");
    let VtkDataObject::PolyData(p) = out else {
        panic!()
    };
    assert_eq!(
        p.polygons, original_polygons,
        "polygon connectivity must be unchanged after smoothing"
    );
}

#[test]
fn wrong_input_type_returns_err() {
    use crate::domain::vtk_data_object::VtkImageData;
    let f = SmoothFilter::default();
    let result = f.execute(VtkDataObject::ImageData(VtkImageData::default()));
    assert!(result.is_err(), "non-PolyData input must return Err");
}

#[test]
fn isolated_vertex_stays_unchanged() {
    // A mesh with one isolated vertex (no polygon neighbours) and
    // one triangle elsewhere — isolated vertex must not move.
    let poly = VtkPolyData {
        points: vec![
            [5.0, 5.0, 5.0], // index 0 — isolated
            [0.0, 0.0, 0.0], // index 1
            [1.0, 0.0, 0.0], // index 2
            [0.5, 1.0, 0.0], // index 3
        ],
        polygons: vec![vec![1, 2, 3]], // only connects 1,2,3
        ..Default::default()
    };
    let f = SmoothFilter::new(0.5, 50);
    let out = f
        .execute(VtkDataObject::PolyData(poly))
        .expect("infallible: validated precondition");
    let VtkDataObject::PolyData(p) = out else {
        panic!()
    };
    let iso = p.points[0];
    assert!(
        (iso[0] - 5.0).abs() < 1e-5 && (iso[1] - 5.0).abs() < 1e-5 && (iso[2] - 5.0).abs() < 1e-5,
        "isolated vertex must not move: got {:?}",
        iso
    );
}

#[test]
fn test_smooth_filter_parameter_change_triggers_rerun() {
    let mut sf = SmoothFilter::new(0.5, 20);
    let mtime_before = sf.get_mtime();

    sf.set_relaxation_factor(0.8);
    let mtime_after_relax = sf.get_mtime();
    assert!(
        mtime_after_relax > mtime_before,
        "set_relaxation_factor must bump mtime: before={}, after={}",
        mtime_before.value(),
        mtime_after_relax.value()
    );

    sf.set_iterations(5);
    let mtime_after_iters = sf.get_mtime();
    assert!(
        mtime_after_iters > mtime_after_relax,
        "set_iterations must bump mtime: before={}, after={}",
        mtime_after_relax.value(),
        mtime_after_iters.value()
    );
}

/// Two triangles sharing an edge plus one line segment plus one isolated
/// vertex: the CSR layout must reproduce the sorted, deduplicated neighbor
/// *set* of the jagged reference for every vertex.
#[test]
fn adjacency_matches_sorted_jagged_reference() {
    let poly = VtkPolyData {
        points: vec![
            [0.0, 0.0, 0.0], // 0
            [1.0, 0.0, 0.0], // 1
            [0.5, 1.0, 0.0], // 2
            [1.5, 1.0, 0.0], // 3
            [9.0, 9.0, 9.0], // 4 — isolated
        ],
        polygons: vec![vec![0, 1, 2], vec![1, 3, 2]],
        lines: vec![vec![0, 2, 3]],
        ..Default::default()
    };
    let adj = Adjacency::build(&poly);

    // CSR invariants: n + 1 offsets, zero start, sentinel == flat length,
    // monotonically non-decreasing offsets.
    assert_eq!(adj.offsets.len(), poly.points.len() + 1);
    assert_eq!(adj.offsets[0], 0);
    assert_eq!(*adj.offsets.last().unwrap(), adj.neighbors.len());
    assert!(adj.offsets.windows(2).all(|w| w[0] <= w[1]));

    // Parity: the sorted deduplicated neighbor set per vertex.
    let expected: [&[u32]; 5] = [
        &[1, 2],    // 0: polygons (0,1),(0,2) + line (0,2)
        &[0, 2, 3], // 1: polygons (0,1),(1,2),(1,3)
        &[0, 1, 3], // 2: polygons (0,2),(1,2),(2,3) + line (2,3)
        &[1, 2],    // 3: polygon (1,3),(2,3) + line (2,3)
        &[],        // 4: isolated
    ];
    for (v, want) in expected.iter().enumerate() {
        assert_eq!(
            adj.neighbors_of(v),
            *want,
            "vertex {v} neighbor run must match the sorted jagged reference"
        );
    }
}

/// On a quad grid every interior vertex is shared by four cells and must get
/// exactly its four edge-sharing neighbors, sorted and deduplicated.
#[test]
fn adjacency_interior_grid_vertex_has_sorted_edge_neighbors() {
    // 3×3 vertex grid, 2×2 quad cells; center vertex is index 4.
    let rows = 3usize;
    let cols = 3usize;
    let mut points = Vec::with_capacity(rows * cols);
    for r in 0..rows {
        for c in 0..cols {
            points.push([c as f32, r as f32, 0.0]);
        }
    }
    let mut polygons = Vec::new();
    for r in 0..rows - 1 {
        for c in 0..cols - 1 {
            let tl = r * cols + c;
            polygons.push(vec![
                tl as u32,
                (tl + 1) as u32,
                (tl + cols + 1) as u32,
                (tl + cols) as u32,
            ]);
        }
    }
    let poly = VtkPolyData {
        points,
        polygons,
        ..Default::default()
    };
    let adj = Adjacency::build(&poly);
    assert_eq!(
        adj.neighbors_of(4),
        &[1, 3, 5, 7],
        "interior vertex must reach its four edge-sharing neighbors"
    );
}

/// The sorted, deduplicated CSR layout is fully deterministic: two builds of
/// the same mesh must produce byte-identical `Adjacency` values.
#[test]
fn adjacency_build_is_deterministic() {
    let poly = VtkPolyData {
        points: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0]],
        polygons: vec![vec![0, 1, 2]],
        ..Default::default()
    };
    assert_eq!(Adjacency::build(&poly), Adjacency::build(&poly));
}
