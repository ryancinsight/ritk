#![expect(clippy::unwrap_used, reason = "ratchet RITK-UNWRAP-1")]
use super::*;

#[test]
fn test_header_binary_layout() -> Result<()> {
    let dir = tempdir()?;
    let backend = TestBackend::default();
    let path = dir.path().join("header_check.mgh");
    let data_vec: Vec<f32> = (0..(2 * 3 * 5) as u32).map(|i| i as f32).collect();
    let image = make_image_with_spatial(
        data_vec,
        2,
        3,
        5,
        Point::new([0.0, 0.0, 0.0]),
        Spacing::new([0.5, 1.0, 2.0]),
        Direction::identity(),
    );
    write_mgh(&image, &path, &backend)?;

    let raw = std::fs::read(&path)?;
    assert_eq!(raw.len(), HEADER_SIZE + 2 * 3 * 5 * 4);
    assert_eq!(i32::from_be_bytes(raw[0..4].try_into().unwrap()), 1);
    assert_eq!(i32::from_be_bytes(raw[4..8].try_into().unwrap()), 5);
    assert_eq!(i32::from_be_bytes(raw[8..12].try_into().unwrap()), 3);
    assert_eq!(i32::from_be_bytes(raw[12..16].try_into().unwrap()), 2);
    assert_eq!(i32::from_be_bytes(raw[16..20].try_into().unwrap()), 1);
    assert_eq!(i32::from_be_bytes(raw[20..24].try_into().unwrap()), 3);
    assert_eq!(i32::from_be_bytes(raw[24..28].try_into().unwrap()), 0);
    assert_eq!(i16::from_be_bytes(raw[28..30].try_into().unwrap()), 1);
    // Voxel spacing is stored in header `[x, y, z]` order. The fixture's RITK
    // spacing is `[Δdepth, Δrow, Δcol] = [0.5, 1.0, 2.0]`, so the header holds
    // `[Δx, Δy, Δz] = [Δcol, Δrow, Δdepth] = [2.0, 1.0, 0.5]`
    // (`docs/book/mgh_format.md`; `docs/architecture.md` §7–§9).
    assert_eq!(f32::from_be_bytes(raw[30..34].try_into().unwrap()), 2.0);
    assert_eq!(f32::from_be_bytes(raw[34..38].try_into().unwrap()), 1.0);
    assert_eq!(f32::from_be_bytes(raw[38..42].try_into().unwrap()), 0.5);
    // `Mdc` holds the direction cosines in the header's RAS frame and `[x, y,
    // z]` axis order. The RITK identity direction's `[depth, row, col]`
    // columns are `[e_depth, e_row, e_col]`; reversing them into header order
    // gives `[e_col, e_row, e_depth]`, and converting that basis from the
    // stored LPS frame to RAS negates the x and y component of every column
    // (`docs/book/mgh_format.md`; `docs/architecture.md` §5, §7–§9):
    //   x_ras = e_col flipped   = [0, 0, 1]
    //   y_ras = e_row flipped   = [0, -1, 0]
    //   z_ras = e_depth flipped = [-1, 0, 0]
    assert_eq!(f32::from_be_bytes(raw[42..46].try_into().unwrap()), 0.0);
    assert_eq!(f32::from_be_bytes(raw[46..50].try_into().unwrap()), 0.0);
    assert_eq!(f32::from_be_bytes(raw[50..54].try_into().unwrap()), 1.0);
    assert_eq!(f32::from_be_bytes(raw[54..58].try_into().unwrap()), 0.0);
    assert_eq!(f32::from_be_bytes(raw[58..62].try_into().unwrap()), -1.0);
    assert_eq!(f32::from_be_bytes(raw[62..66].try_into().unwrap()), 0.0);
    assert_eq!(f32::from_be_bytes(raw[66..70].try_into().unwrap()), -1.0);
    assert_eq!(f32::from_be_bytes(raw[70..74].try_into().unwrap()), 0.0);
    assert_eq!(f32::from_be_bytes(raw[74..78].try_into().unwrap()), 0.0);

    // `c_ras = origin + Mdc·D·h` in header order, with
    // `h = [(5−1)/2, (3−1)/2, (2−1)/2] = [2.0, 1.0, 0.5]`,
    // `D·h = [4.0, 1.0, 0.25]`, and `Mdc·D·h = (0.25, 1.0, 4.0)`. That sum is
    // the volume center in the stored LPS frame; converting it to RAS negates
    // its x and y (`docs/architecture.md` §5).
    let c_r = f32::from_be_bytes(raw[78..82].try_into().unwrap());
    let c_a = f32::from_be_bytes(raw[82..86].try_into().unwrap());
    let c_s = f32::from_be_bytes(raw[86..90].try_into().unwrap());
    assert!((c_r - (-0.25)).abs() < 1e-6, "c_r={c_r}");
    assert!((c_a - (-1.0)).abs() < 1e-6, "c_a={c_a}");
    assert!((c_s - 4.0).abs() < 1e-6, "c_s={c_s}");

    for (i, &byte) in raw[90..HEADER_SIZE].iter().enumerate() {
        assert_eq!(byte, 0, "Padding byte {} is non-zero: {byte}", 90 + i);
    }
    assert_eq!(f32::from_be_bytes(raw[284..288].try_into().unwrap()), 0.0);
    assert_eq!(f32::from_be_bytes(raw[400..404].try_into().unwrap()), 29.0);
    Ok(())
}

#[test]
fn test_file_contains_full_payload() -> Result<()> {
    let dir = tempdir()?;
    let backend = TestBackend::default();
    let path = dir.path().join("payload.mgh");
    let image = make_image(vec![1.0f32; 2 * 3 * 4], 2, 3, 4);

    write_mgh(&image, &path, &backend)?;
    let file_size = std::fs::metadata(&path)?.len();
    let expected = (HEADER_SIZE + 2 * 3 * 4 * 4) as u64;
    assert_eq!(
        file_size, expected,
        "File size {file_size} must equal header plus payload {expected}"
    );
    Ok(())
}
