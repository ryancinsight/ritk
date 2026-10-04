use super::*;

#[test]
fn test_rotated_direction_in_space_directions() -> Result<()> {
    let dir_tmp = tempdir()?;
    let path = dir_tmp.path().join("rotated.nrrd");
    let backend = SequentialBackend;

    // Build the rotated direction matrix explicitly
    let mut direction = Direction::zeros();
    // Column 0 = internal depth axis = physical Z.
    direction[(0, 0)] = 0.0;
    direction[(1, 0)] = 0.0;
    direction[(2, 0)] = 1.0;
    // Column 1 = internal row axis = physical -X.
    direction[(0, 1)] = -1.0;
    direction[(1, 1)] = 0.0;
    direction[(2, 1)] = 0.0;
    // Column 2 = internal column axis = physical Y.
    direction[(0, 2)] = 0.0;
    direction[(1, 2)] = 1.0;
    direction[(2, 2)] = 0.0;

    let image = make_image(
        vec![0.0f32; 2 * 2 * 2],
        [2, 2, 2],
        Point::new([0.0, 0.0, 0.0]),
        Spacing::new([2.0, 3.0, 4.0]),
        direction,
    );

    crate::write_nrrd(&path, &image, &backend)?;

    let bytes = std::fs::read(&path)?;

    assert!(
        bytes_contain(&bytes, "(0,4,0)"),
        "sd0 must be (0,4,0); file content does not match"
    );
    assert!(
        bytes_contain(&bytes, "(-3,0,0)"),
        "sd1 must be (-3,0,0); file content does not match"
    );
    assert!(
        bytes_contain(&bytes, "(0,0,2)"),
        "sd2 must be (0,0,2); file content does not match"
    );

    Ok(())
}
