use super::*;

/// A DICOM directory with multiple series requires an explicit UID, and
/// the selected series' voxels and geometry survive conversion to NRRD.
#[test]
fn test_convert_selected_dicom_series_to_nrrd() {
    let dir = tempdir().unwrap();
    let dicom_root = dir.path().join("study");
    let first_dir = dir.path().join("first-series");
    let second_dir = dir.path().join("second-series");
    let first = make_test_image();
    let second = make_test_image_with_offset(100.0);
    write_image(&first_dir, &first, ImageFormat::Dicom).unwrap();
    write_image(&second_dir, &second, ImageFormat::Dicom).unwrap();

    let single_series = ritk_io::scan_dicom_directory(&first_dir).unwrap();
    assert_eq!(single_series.len(), 1, "first directory has one series");
    let single_expected =
        ritk_io::load_native_dicom_series(&single_series[0], &Backend::default()).unwrap();
    let single_output = dir.path().join("single.nii");
    run(ConvertArgs {
        input: first_dir.clone(),
        output: single_output.clone(),
        format: None,
        series_uid: None,
    })
    .unwrap();
    let single_recovered = read_image(&single_output).unwrap();
    assert_eq!(
        single_recovered.data_slice().unwrap(),
        single_expected.data_slice().unwrap(),
        "single-series DICOM input converts without a UID"
    );

    let instance = first_dir.join("slice_0000.dcm");
    let instance_expected = read_image(&instance).unwrap();
    let instance_output = dir.path().join("instance.nii");
    run(ConvertArgs {
        input: instance,
        output: instance_output.clone(),
        format: None,
        series_uid: None,
    })
    .unwrap();
    let instance_recovered = read_image(&instance_output).unwrap();
    assert_eq!(
        instance_recovered.shape(),
        instance_expected.shape(),
        "selecting one instance loads its containing series"
    );
    assert_eq!(
        instance_recovered.data_slice().unwrap(),
        instance_expected.data_slice().unwrap(),
        "single-file DICOM input converts its containing series through RITK"
    );

    let second_series = ritk_io::scan_dicom_directory(&second_dir).unwrap();
    assert_eq!(
        second_series.len(),
        1,
        "second fixture directory has one series"
    );
    let selected_uid = second_series[0].series_instance_uid().to_owned();
    std::fs::create_dir_all(&dicom_root).unwrap();
    for (prefix, source_dir) in [("first", &first_dir), ("second", &second_dir)] {
        for entry in std::fs::read_dir(source_dir).unwrap() {
            let path = entry.unwrap().path();
            let name = path.file_name().unwrap().to_string_lossy();
            std::fs::copy(&path, dicom_root.join(format!("{prefix}-{name}"))).unwrap();
        }
    }

    let series = ritk_io::scan_dicom_directory(&dicom_root).unwrap();
    assert_eq!(series.len(), 2, "flat study directory contains two series");
    let selected = series
        .iter()
        .find(|entry| entry.series_instance_uid() == selected_uid.as_str())
        .expect("second fixture series is discoverable");
    let expected = ritk_io::load_native_dicom_series(selected, &Backend::default()).unwrap();
    let output = dir.path().join("selected.nrrd");

    let ambiguous = run(ConvertArgs {
        input: dicom_root.clone(),
        output: output.clone(),
        format: None,
        series_uid: None,
    })
    .expect_err("multiple DICOM series require explicit selection");
    let ambiguous = format!("{ambiguous:#}");
    assert!(ambiguous.contains(&selected_uid));
    assert!(
        series
            .iter()
            .filter(|entry| entry.series_instance_uid() != selected_uid.as_str())
            .all(|entry| ambiguous.contains(entry.series_instance_uid())),
        "ambiguity diagnostic lists every SeriesInstanceUID: {ambiguous}"
    );
    assert!(!output.exists(), "ambiguous input must not write output");

    run(ConvertArgs {
        input: dicom_root,
        output: output.clone(),
        format: None,
        series_uid: Some(selected_uid),
    })
    .unwrap();

    let recovered = read_image(&output).unwrap();
    assert_eq!(recovered.shape(), expected.shape());
    assert_eq!(
        recovered.data_slice().unwrap(),
        expected.data_slice().unwrap(),
        "NRRD output contains voxels from the selected DICOM series"
    );
    assert_eq!(recovered.origin().to_array(), expected.origin().to_array());
    assert_eq!(
        recovered.spacing().to_array(),
        expected.spacing().to_array()
    );
    assert_eq!(
        recovered.direction().to_row_major(),
        expected.direction().to_row_major()
    );
}

/// Explicit DICOM output creates a readable Secondary Capture series.
/// The per-slice tolerance follows RITK's u16 rescale: quantization is at
/// most slope/2, and six-decimal DS fields add at most 0.5e-6 per value.
#[test]
fn test_convert_nifti_to_dicom_series() {
    let dir = tempdir().unwrap();
    let input = dir.path().join("input.nii");
    let output = dir.path().join("dicom-output");
    let image = make_test_image();
    write_image(&input, &image, ImageFormat::NIfTI).unwrap();
    let expected = read_image(&input).unwrap();

    run(ConvertArgs {
        input,
        output: output.clone(),
        format: Some(OutputFormat::Dicom),
        series_uid: None,
    })
    .unwrap();

    let series = ritk_io::scan_dicom_directory(&output).unwrap();
    assert_eq!(series.len(), 1, "DICOM output contains one series");
    assert_eq!(
        series[0].file_paths.len(),
        expected.shape()[0],
        "one DICOM file is written for each slice"
    );
    let recovered = ritk_io::load_native_dicom_series(&series[0], &Backend::default()).unwrap();
    assert_eq!(recovered.shape(), expected.shape());

    let expected_data = expected.data_slice().unwrap();
    let recovered_data = recovered.data_slice().unwrap();
    let slice_len = expected.shape()[1] * expected.shape()[2];
    let ds_half_ulp = 0.5e-6_f32;
    for (slice_index, (original, decoded)) in expected_data
        .chunks_exact(slice_len)
        .zip(recovered_data.chunks_exact(slice_len))
        .enumerate()
    {
        let minimum = original.iter().copied().fold(f32::INFINITY, f32::min);
        let maximum = original.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let slope = (maximum - minimum) / 65535.0_f32;
        let tolerance = 65535.0_f32 * ds_half_ulp + ds_half_ulp + slope / 2.0_f32;
        for (voxel_index, (&source, &actual)) in original.iter().zip(decoded).enumerate() {
            let error = (source - actual).abs();
            assert!(
                error <= tolerance,
                "slice {slice_index} voxel {voxel_index}: error {error} exceeds derived bound {tolerance}"
            );
        }
    }
    assert_eq!(recovered.origin().to_array(), expected.origin().to_array());
    assert_eq!(
        recovered.spacing().to_array(),
        expected.spacing().to_array()
    );
    assert_eq!(
        recovered.direction().to_row_major(),
        expected.direction().to_row_major()
    );
}
