use super::*;

// ── Structural tests ──────────────────────────────────────────────────────

#[test]
fn test_spatial_metadata_preserved() {
    let tensor = Tensor::<f32, TestBackend>::from_slice([3, 3, 3], &[100.0_f32; 27]);
    let origin = Point::new([1.0, 2.0, 3.0]);
    let spacing = Spacing::new([0.5, 1.0, 2.0]);
    let direction = Direction::identity();
    let image = Image::new(tensor, origin, spacing, direction)
        .expect("invariant: fixture tensor has the declared rank");

    let result = neighborhood_connected(&image, [0, 0, 0], 50.0, 150.0, [1, 1, 1]);
    assert_eq!(result.origin(), &origin);
    assert_eq!(result.spacing(), &spacing);
    assert_eq!(result.direction(), &direction);
}

#[test]
fn test_filter_struct_matches_function() {
    let values: Vec<f32> = (0..27).map(|i| i as f32 * 10.0).collect();
    let image = make_image(values, [3, 3, 3]);

    let via_fn = neighborhood_connected(&image, [1, 1, 1], 50.0, 200.0, [1, 1, 1]);
    let via_struct = NeighborhoodConnectedFilter::new([1, 1, 1], 50.0, 200.0)
        .with_radius([1, 1, 1])
        .apply(&image);

    let fn_vals = get_values(&via_fn);
    let struct_vals = get_values(&via_struct);
    assert_eq!(
        fn_vals, struct_vals,
        "function and filter struct must produce identical results"
    );
}

#[test]
fn test_filter_struct_default_radius() {
    // Default radius is [1,1,1]. Verify builder without explicit radius matches.
    let image = make_image(vec![100.0_f32; 27], [3, 3, 3]);

    let via_fn = neighborhood_connected(&image, [1, 1, 1], 50.0, 150.0, [1, 1, 1]);
    let via_struct = NeighborhoodConnectedFilter::new([1, 1, 1], 50.0, 150.0).apply(&image);

    assert_eq!(
        get_values(&via_fn),
        get_values(&via_struct),
        "default radius must be [1,1,1]"
    );
}

// ── Anisotropic radius test ───────────────────────────────────────────────

#[test]
fn test_anisotropic_radius() {
    // 1×5×5 image: center 1×3×3 has intensity 200, border intensity 50.
    // Bounds [150, 255].
    //
    // With radius [0,1,1]: neighborhood in z is just the voxel itself (nz=1),
    // in y and x extends ±1. The voxels whose 1×3×3 neighborhood is fully
    // inside the 1×3×3 high-intensity region are those at y=2,x=2 (center only).
    let (nz, ny, nx) = (1, 5, 5);
    let mut values = vec![50.0_f32; nz * ny * nx];
    for iy in 1..4 {
        for ix in 1..4 {
            values[iy * nx + ix] = 200.0;
        }
    }
    let image = make_image(values, [nz, ny, nx]);
    let result = neighborhood_connected(&image, [0, 2, 2], 150.0, 255.0, [0, 1, 1]);
    // Center voxel (0,2,2): neighborhood y∈{1,2,3}, x∈{1,2,3} → all 200.0 ✓
    // Its face-neighbors in-plane: (0,1,2),(0,3,2),(0,2,1),(0,2,3).
    // (0,1,2): neighborhood y∈{0,1,2}, x∈{1,2,3} → includes (0,0,{1,2,3})=50 ✗
    // Same reasoning for (0,3,2): includes y=4 which is 50. ✗
    // (0,2,1): neighborhood x∈{0,1,2} → includes x=0 which is 50. ✗
    // (0,2,3): neighborhood x∈{2,3,4} → includes x=4 which is 50. ✗
    // So only the center voxel is admissible.
    assert_eq!(
        count_foreground(&result),
        1,
        "anisotropic radius must correctly restrict admissibility"
    );
}

// ── 3-D volumetric test ───────────────────────────────────────────────────

#[test]
fn test_3d_sphere_region_growing_radius_zero() {
    // 9×9×9 image with a sphere of radius 3 at center (4,4,4) with intensity 200;
    // background intensity 50; bounds [150, 255]; neighborhood radius [0,0,0].
    // This should behave exactly like connected-threshold.
    let (nz, ny, nx) = (9, 9, 9);
    let mut values = vec![50.0_f32; nz * ny * nx];
    let (cz, cy, cx) = (4isize, 4isize, 4isize);
    let r2 = 9isize; // radius squared = 9

    let mut sphere_count = 0;
    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let dz = iz as isize - cz;
                let dy = iy as isize - cy;
                let dx = ix as isize - cx;
                if dz * dz + dy * dy + dx * dx <= r2 {
                    values[iz * ny * nx + iy * nx + ix] = 200.0;
                    sphere_count += 1;
                }
            }
        }
    }

    let image = make_image(values, [nz, ny, nx]);
    let result = neighborhood_connected(&image, [4, 4, 4], 150.0, 255.0, [0, 0, 0]);
    assert_eq!(
        count_foreground(&result),
        sphere_count,
        "radius=0 sphere must match connected-threshold exactly"
    );
}

#[test]
fn test_3d_sphere_neighborhood_erodes_boundary() {
    // 9×9×9 image with a sphere of radius 3 at center (4,4,4) with intensity 200;
    // background intensity 50; bounds [150, 255]; neighborhood radius [1,1,1].
    //
    // Sphere voxels whose 3×3×3 neighborhood extends outside the sphere will be
    // rejected (some neighbors have intensity 50). This effectively erodes the
    // sphere boundary: only interior voxels (where the entire 3×3×3 neighborhood
    // is also within the sphere) are admitted.
    //
    // A voxel at (z,y,x) is admissible iff all 27 neighbors in its 3×3×3 box
    // are also within the sphere. This is equivalent to requiring
    // (z±1, y±1, x±1) all satisfy d² ≤ 9 where d² is measured from center.
    // The worst case is the corner offset (+1,+1,+1) from the voxel.
    let (nz, ny, nx) = (9, 9, 9);
    let mut values = vec![50.0_f32; nz * ny * nx];
    let (cz, cy, cx) = (4isize, 4isize, 4isize);
    let r2 = 9isize;

    // Build sphere.
    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let dz = iz as isize - cz;
                let dy = iy as isize - cy;
                let dx = ix as isize - cx;
                if dz * dz + dy * dy + dx * dx <= r2 {
                    values[iz * ny * nx + iy * nx + ix] = 200.0;
                }
            }
        }
    }

    // Count analytically: voxel (z,y,x) is admissible iff for ALL (dz,dy,dx)
    // with |dz|≤1, |dy|≤1, |dx|≤1, the neighbor (z+dz, y+dy, x+dx) is within
    // the sphere (or outside the image domain — but at the center of a 9×9×9
    // image with sphere radius 3, no neighborhood extends outside domain).
    let mut expected_count = 0;
    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let mut all_in_sphere = true;
                'outer: for dz in -1isize..=1 {
                    for dy in -1isize..=1 {
                        for dx in -1isize..=1 {
                            let nzi = iz as isize + dz;
                            let nyi = iy as isize + dy;
                            let nxi = ix as isize + dx;
                            if nzi < 0
                                || nzi >= nz as isize
                                || nyi < 0
                                || nyi >= ny as isize
                                || nxi < 0
                                || nxi >= nx as isize
                            {
                                // Outside domain: excluded from check.
                                continue;
                            }
                            let v =
                                values[nzi as usize * ny * nx + nyi as usize * nx + nxi as usize];
                            if !(150.0..=255.0).contains(&v) {
                                all_in_sphere = false;
                                break 'outer;
                            }
                        }
                    }
                }
                if all_in_sphere {
                    // Also must be 6-connected to the seed region.
                    // For a convex shape centered at (4,4,4), all admissible
                    // voxels form a 6-connected region — verified analytically
                    // for the Euclidean sphere with r²≤9.
                    // We check that the voxel is in the original sphere itself
                    // (otherwise it's background that happens to have all
                    // in-bounds neighbors in the sphere, which is not possible
                    // for interior background of a filled sphere).
                    let dz = iz as isize - cz;
                    let dy = iy as isize - cy;
                    let dx = ix as isize - cx;
                    if dz * dz + dy * dy + dx * dx <= r2 {
                        expected_count += 1;
                    }
                }
            }
        }
    }

    let image = make_image(values, [nz, ny, nx]);
    let result = neighborhood_connected(&image, [4, 4, 4], 150.0, 255.0, [1, 1, 1]);
    assert_eq!(
        count_foreground(&result),
        expected_count,
        "neighborhood radius must erode sphere boundary"
    );
    // Verify the eroded region is strictly smaller than the full sphere.
    let full_sphere = neighborhood_connected(&image, [4, 4, 4], 150.0, 255.0, [0, 0, 0]);
    assert!(
        count_foreground(&result) < count_foreground(&full_sphere),
        "radius=1 region must be strictly smaller than radius=0 region for a sphere"
    );
}
