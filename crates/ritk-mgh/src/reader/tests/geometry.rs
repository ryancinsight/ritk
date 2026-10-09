use super::*;

/// A hand-built header is the external oracle for the header↔RITK spatial
/// contract. Two reconciliations must hold, and this fixture states both
/// without calling the reader's own helpers:
///
/// 1. the header's `[x, y, z]` spacing and `Mdc` columns reach RITK as
///    `[depth, row, col] = [z, y, x]` (`docs/architecture.md` §7–§9);
/// 2. the header's RAS center and direction cosines reach RITK's LPS stored
///    model, i.e. their x and y components change sign (§5).
#[test]
fn test_read_nondefault_spatial() -> Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("spatial.mgh");
    let backend = TestBackend::default();
    let spacing = [0.5f32, 0.75, 1.25];
    let dir_cols = [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
    let data_bytes: Vec<u8> = (0..(4 * 3 * 2))
        .map(|i| (i as f32) * 0.1)
        .flat_map(|v| v.to_be_bytes())
        .collect();
    let mgh = build_mgh_bytes(
        1,
        [4, 3, 2],
        SINGLE_FRAME,
        MRI_FLOAT,
        spacing,
        dir_cols,
        [10.0, 20.0, 30.0],
        &data_bytes,
    );
    std::fs::write(&path, &mgh)?;

    let image = read_mgh::<TestBackend, _>(&path, &backend)?;
    assert_eq!(image.shape(), [2, 3, 4]);
    // The header stores spacing in `[x, y, z]` order; RITK spacing is
    // `[Δdepth, Δrow, Δcol] = [Δz, Δy, Δx]`, its reverse
    // (`docs/architecture.md` §7–§9; `docs/book/mgh_format.md`).
    let sp = image.spacing();
    assert!((sp[0] - 1.25).abs() < 1e-6, "spacing[0]={}", sp[0]);
    assert!((sp[1] - 0.75).abs() < 1e-6, "spacing[1]={}", sp[1]);
    assert!((sp[2] - 0.5).abs() < 1e-6, "spacing[2]={}", sp[2]);

    // Header `Mdc` columns `[x_ras, y_ras, z_ras]` are `[0,1,0]`, `[-1,0,0]`,
    // `[0,0,1]`. Converting to LPS negates the x and y component of each, and
    // RITK then orders the columns `[depth, row, col] = [z, y, x]`:
    //   depth = z_ras flipped   = [0, 0, 1]
    //   row   = y_ras flipped   = [1, 0, 0]
    //   col   = x_ras flipped   = [0, -1, 0]
    let direction = image.direction();
    assert!((direction[(0, 0)] - 0.0).abs() < 1e-6);
    assert!((direction[(1, 0)] - 0.0).abs() < 1e-6);
    assert!((direction[(2, 0)] - 1.0).abs() < 1e-6);
    assert!((direction[(0, 1)] - 1.0).abs() < 1e-6);
    assert!((direction[(1, 1)] - 0.0).abs() < 1e-6);
    assert!((direction[(2, 1)] - 0.0).abs() < 1e-6);
    assert!((direction[(0, 2)] - 0.0).abs() < 1e-6);
    assert!((direction[(1, 2)] - (-1.0)).abs() < 1e-6);
    assert!((direction[(2, 2)] - 0.0).abs() < 1e-6);

    // The header's center is RAS `[10, 20, 30]`; the RAS-to-LPS flip negates
    // its x and y, giving `[-10, -20, 30]`. Subtracting `Mdc · D · h` with
    // `h = [1.5, 1.0, 0.5]` (dims `[4, 3, 2]`) gives the RAS corner
    // `[10.75, 19.25, 29.375]`, whose LPS form is the stored origin.
    let origin = image.origin();
    assert!(
        (origin[0] - (-10.75)).abs() < 1e-6,
        "origin[0]={}",
        origin[0]
    );
    assert!(
        (origin[1] - (-19.25)).abs() < 1e-6,
        "origin[1]={}",
        origin[1]
    );
    assert!((origin[2] - 29.375).abs() < 1e-6, "origin[2]={}", origin[2]);
    Ok(())
}

#[test]
fn test_read_good_ras_flag_zero() -> Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("no_ras.mgh");
    let backend = TestBackend::default();
    let values: Vec<f32> = (0..8).map(|i| i as f32).collect();
    let data_bytes: Vec<u8> = values.iter().flat_map(|v: &f32| v.to_be_bytes()).collect();
    let mut buf = Vec::with_capacity(HEADER_SIZE + data_bytes.len());
    buf.extend_from_slice(&1_i32.to_be_bytes());
    buf.extend_from_slice(&2_i32.to_be_bytes());
    buf.extend_from_slice(&2_i32.to_be_bytes());
    buf.extend_from_slice(&2_i32.to_be_bytes());
    buf.extend_from_slice(&1_i32.to_be_bytes());
    buf.extend_from_slice(&MRI_FLOAT.to_be_bytes());
    buf.extend_from_slice(&0_i32.to_be_bytes());
    buf.extend_from_slice(&0_i16.to_be_bytes());
    for _ in 0..15 {
        buf.extend_from_slice(&99.9f32.to_be_bytes());
    }
    buf.resize(HEADER_SIZE, 0u8);
    buf.extend_from_slice(&data_bytes);
    std::fs::write(&path, &buf)?;

    let image = read_mgh::<TestBackend, _>(&path, &backend)?;
    assert_eq!(image.shape(), [2, 2, 2]);
    assert_eq!(image.spacing()[0], 1.0);
    assert_eq!(image.spacing()[1], 1.0);
    assert_eq!(image.spacing()[2], 1.0);
    assert_eq!(image.direction()[(0, 0)], 1.0);
    assert_eq!(image.direction()[(1, 1)], 1.0);
    assert_eq!(image.direction()[(2, 2)], 1.0);
    assert_eq!(image.origin()[0], 0.0);
    assert_eq!(image.origin()[1], 0.0);
    assert_eq!(image.origin()[2], 0.0);
    image.data_slice().map(|loaded| {
        for (i, (&got, &expected)) in loaded.iter().zip(values.iter()).enumerate() {
            assert_eq!(got, expected, "voxel[{i}]");
        }
    })?;
    Ok(())
}
