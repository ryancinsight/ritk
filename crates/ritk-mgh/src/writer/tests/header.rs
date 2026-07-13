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
    assert_eq!(f32::from_be_bytes(raw[30..34].try_into().unwrap()), 0.5);
    assert_eq!(f32::from_be_bytes(raw[34..38].try_into().unwrap()), 1.0);
    assert_eq!(f32::from_be_bytes(raw[38..42].try_into().unwrap()), 2.0);
    assert_eq!(f32::from_be_bytes(raw[42..46].try_into().unwrap()), 1.0);
    assert_eq!(f32::from_be_bytes(raw[46..50].try_into().unwrap()), 0.0);
    assert_eq!(f32::from_be_bytes(raw[50..54].try_into().unwrap()), 0.0);
    assert_eq!(f32::from_be_bytes(raw[54..58].try_into().unwrap()), 0.0);
    assert_eq!(f32::from_be_bytes(raw[58..62].try_into().unwrap()), 1.0);
    assert_eq!(f32::from_be_bytes(raw[62..66].try_into().unwrap()), 0.0);
    assert_eq!(f32::from_be_bytes(raw[66..70].try_into().unwrap()), 0.0);
    assert_eq!(f32::from_be_bytes(raw[70..74].try_into().unwrap()), 0.0);
    assert_eq!(f32::from_be_bytes(raw[74..78].try_into().unwrap()), 1.0);

    let c_r = f32::from_be_bytes(raw[78..82].try_into().unwrap());
    let c_a = f32::from_be_bytes(raw[82..86].try_into().unwrap());
    let c_s = f32::from_be_bytes(raw[86..90].try_into().unwrap());
    assert!((c_r - 1.0).abs() < 1e-6, "c_r={c_r}");
    assert!((c_a - 1.0).abs() < 1e-6, "c_a={c_a}");
    assert!((c_s - 1.0).abs() < 1e-6, "c_s={c_s}");

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
