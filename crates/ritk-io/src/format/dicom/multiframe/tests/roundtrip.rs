use arrayvec::ArrayString;
use super::*;

#[test]
fn test_multiframe_info_and_roundtrip_writer_read_consistency() {
    let device = <B as Backend>::Device::default();
    let tmp = tempfile::tempdir().expect("tempdir");
    let out_path = tmp.path().join("multiframe.dcm");
    let n_frames = 2_usize;
    let rows = 3_usize;
    let cols = 4_usize;

    let mut data: Vec<f32> = Vec::with_capacity(n_frames * rows * cols);
    for frame in 0..n_frames {
        for row in 0..rows {
            for col in 0..cols {
                data.push((frame * 100 + row * 10 + col) as f32);
            }
        }
    }

    let tensor = Tensor::<B, 3>::from_data(
        TensorData::new(data.clone(), Shape::new([n_frames, rows, cols])),
        &device,
    );
    let image = Image::new(
        tensor,
        Point::new([0.0, 0.0, 0.0]),
        Spacing::new([1.0, 1.0, 1.0]),
        Direction::identity(),
    );

    write_dicom_multiframe(&out_path, &image).expect("write_dicom_multiframe");
    assert!(out_path.exists(), "output file must exist after write");

    let info = read_multiframe_info(&out_path).expect("read_multiframe_info");
    assert_eq!(info.n_frames, n_frames, "n_frames");
    assert_eq!(info.rows, rows, "rows");
    assert_eq!(info.cols, cols, "cols");
    assert_eq!(info.bits_allocated, 16, "bits_allocated");
    assert_eq!(info.modality.as_deref(), Some("OT"), "modality");
    assert_eq!(
        info.sop_class_uid.as_deref(),
        Some("1.2.840.10008.5.1.4.1.1.7.3"),
        "sop_class_uid"
    );

    let loaded = load_dicom_multiframe::<B, _>(&out_path, &device).expect("load_dicom_multiframe");
    let [frames, loaded_rows, loaded_cols] = loaded.shape();
    assert_eq!(frames, n_frames, "frames");
    assert_eq!(loaded_rows, rows, "loaded_rows");
    assert_eq!(loaded_cols, cols, "loaded_cols");

    loaded.with_data_slice(|recovered: &[f32]| {
        assert_eq!(recovered.len(), data.len(), "recovered pixel count");
        let min_val = data.iter().copied().fold(f32::INFINITY, f32::min);
        let max_val = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let slope = if (max_val - min_val).abs() <= f32::EPSILON {
            1.0_f32
        } else {
            (max_val - min_val) / 65535.0_f32
        };
        let tolerance = slope + 1.0_f32;
        for (idx, (&orig, &rec)) in data.iter().zip(recovered.iter()).enumerate() {
            let diff = (rec - orig).abs();
            assert!(
                diff <= tolerance,
                "pixel {idx}: original={orig:.4} recovered={rec:.4} diff={diff:.6} > tol={tolerance:.6}"
            );
        }
    });
}

#[test]
fn test_write_read_multiframe_roundtrip() {
    let device = <B as Backend>::Device::default();
    let tmp = tempfile::tempdir().expect("tempdir");
    let out_path = tmp.path().join("multiframe.dcm");
    let n_frames = 3_usize;
    let rows = 4_usize;
    let cols = 5_usize;
    let mut data: Vec<f32> = Vec::with_capacity(n_frames * rows * cols);
    for frame in 0..n_frames {
        for row in 0..rows {
            for col in 0..cols {
                data.push((frame * 100 + row * 10 + col) as f32);
            }
        }
    }
    let tensor = Tensor::<B, 3>::from_data(
        TensorData::new(data.clone(), Shape::new([n_frames, rows, cols])),
        &device,
    );
    let image = Image::new(
        tensor,
        Point::new([0.0, 0.0, 0.0]),
        Spacing::new([1.0, 1.0, 1.0]),
        Direction::identity(),
    );
    write_dicom_multiframe(&out_path, &image).expect("write_dicom_multiframe");
    assert!(out_path.exists(), "output file must exist after write");
    let loaded =
        load_dicom_multiframe::<B, _>(&out_path, &device).expect("load_dicom_multiframe roundtrip");
    let [lf, lr, lc] = loaded.shape();
    assert_eq!(lf, n_frames, "recovered n_frames");
    assert_eq!(lr, rows, "recovered rows");
    assert_eq!(lc, cols, "recovered cols");
    let min_val = data.iter().copied().fold(f32::INFINITY, f32::min);
    let max_val = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let slope = if (max_val - min_val).abs() <= f32::EPSILON {
        1.0_f32
    } else {
        (max_val - min_val) / 65535.0_f32
    };
    let tolerance = slope + 1.0_f32;
    loaded.with_data_slice(|recovered: &[f32]| {
        assert_eq!(recovered.len(), data.len(), "recovered pixel count");
        for (i, (&orig, &rec)) in data.iter().zip(recovered.iter()).enumerate() {
            let diff = (rec - orig).abs();
            assert!(
                diff <= tolerance,
                "pixel {i}: original={orig:.4} recovered={rec:.4} diff={diff:.6} > tol={tolerance:.6}"
            );
        }
    });
}

#[test]
fn test_read_multiframe_info_reports_scalar_defaults_for_single_frame() {
    let device = <B as Backend>::Device::default();
    let tmp = tempfile::tempdir().expect("tempdir");
    let out_path = tmp.path().join("single_frame.dcm");

    let tensor = Tensor::<B, 3>::from_data(
        TensorData::new(vec![7.0_f32; 2 * 3], Shape::new([1_usize, 2, 3])),
        &device,
    );
    let image = Image::new(
        tensor,
        Point::new([0.0, 0.0, 0.0]),
        Spacing::new([1.0, 1.0, 1.0]),
        Direction::identity(),
    );

    write_dicom_multiframe(&out_path, &image).expect("write_dicom_multiframe");
    let info = read_multiframe_info(&out_path).expect("read_multiframe_info");
    assert_eq!(info.n_frames, 1, "single-frame file must report one frame");
    assert_eq!(info.rows, 2, "rows");
    assert_eq!(info.cols, 3, "cols");
    assert_eq!(info.bits_allocated, 16, "bits_allocated");
    assert_eq!(info.modality.as_deref(), Some("OT"), "modality");
    assert_eq!(
        info.sop_class_uid.as_deref(),
        Some("1.2.840.10008.5.1.4.1.1.7.3"),
        "sop_class_uid"
    );
}

#[test]
fn test_write_multiframe_with_spatial_metadata_round_trip() {
    // Analytical invariants:
    // - IPP round-trip: |read_ipp[i] - written_ipp[i]| < 1e-4 (DS string precision)
    // - IOP round-trip: |read_iop[i] - written_iop[i]| < 1e-4
    // - Modality round-trip: exact string match
    // - origin in loaded Image equals IPP to ±1e-4
    let device = <B as Backend>::Device::default();
    let tmp = tempfile::tempdir().expect("tempdir");
    let out_path = tmp.path().join("mf_spatial.dcm");
    let n_frames = 2_usize;
    let rows = 3_usize;
    let cols = 4_usize;
    let data: Vec<f32> = (0..n_frames * rows * cols).map(|i| i as f32).collect();
    let tensor = Tensor::<B, 3>::from_data(
        TensorData::new(data.clone(), Shape::new([n_frames, rows, cols])),
        &device,
    );
    let image = Image::new(
        tensor,
        Point::new([0.0; 3]),
        Spacing::new([1.0; 3]),
        Direction::identity(),
    );
    let spatial = MultiFrameSpatialMetadata {
        origin: [10.0, 20.0, -50.0],
        pixel_spacing: [0.8, 0.8],
        slice_thickness: 2.5,
        image_orientation: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        modality: ArrayString::from("CT").unwrap(),
    };
        write_dicom_multiframe_with_options(&out_path, &image, Some(&spatial))
        .expect("write_dicom_multiframe_with_options");
    assert!(out_path.exists(), "output file must exist");

    // Verify via read_multiframe_info
    let info = read_multiframe_info(&out_path).expect("read_multiframe_info");
    assert_eq!(
        info.modality.as_deref(),
        Some("CT"),
        "modality must round-trip"
    );
    assert_eq!(
        info.sop_class_uid.as_deref(),
        Some("1.2.840.10008.5.1.4.1.1.7.3"),
        "SOP class must be Multi-Frame Grayscale Word SC"
    );
    let ipp = info.image_position.expect("image_position must be Some");
    assert!((ipp[0] - 10.0).abs() < 1e-4, "IPP x round-trip");
    assert!((ipp[1] - 20.0).abs() < 1e-4, "IPP y round-trip");
    assert!((ipp[2] - (-50.0)).abs() < 1e-4, "IPP z round-trip");
    let iop = info
        .image_orientation
        .expect("image_orientation must be Some");
    assert!((iop[0] - 1.0).abs() < 1e-4, "IOP[0] round-trip");
    assert!((iop[4] - 1.0).abs() < 1e-4, "IOP[4] round-trip");

    // Verify via load_dicom_multiframe
    let loaded = load_dicom_multiframe::<B, _>(&out_path, &device).expect("load_dicom_multiframe");
    let loaded_origin = loaded.origin();
    assert!((loaded_origin[0] - 10.0).abs() < 1e-4, "loaded origin x");
    assert!((loaded_origin[1] - 20.0).abs() < 1e-4, "loaded origin y");
    assert!((loaded_origin[2] - (-50.0)).abs() < 1e-4, "loaded origin z");

    // Shape invariant
    let [lf, lr, lc] = loaded.shape();
    assert_eq!(lf, n_frames, "frame count");
    assert_eq!(lr, rows, "rows");
    assert_eq!(lc, cols, "cols");

    // Pixel round-trip within quantization tolerance
    let min_val = data.iter().copied().fold(f32::INFINITY, f32::min);
    let max_val = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let slope = if (max_val - min_val).abs() <= f32::EPSILON {
        1.0_f32
    } else {
        (max_val - min_val) / 65535.0_f32
    };
    let tolerance = slope + 1.0_f32;
    loaded.with_data_slice(|recovered: &[f32]| {
        for (i, (&orig, &rec)) in data.iter().zip(recovered.iter()).enumerate() {
            let diff = (rec - orig).abs();
            assert!(
                diff <= tolerance,
                "pixel {i}: orig={orig:.4} rec={rec:.4} diff={diff:.6} > tol={tolerance:.6}"
            );
        }
    });
}

#[test]
fn test_round_trip_negative_intensity_image() {
    let device = <B as Backend>::Device::default();
    let tmp = tempfile::tempdir().expect("tempdir");
    let out_path = tmp.path().join("mf_neg.dcm");
    let n = 2_usize * 3 * 4;
    let data: Vec<f32> = (0..n)
        .map(|i| -1024.0_f32 + (i as f32) * (1524.0_f32 / (n as f32 - 1.0_f32)))
        .collect();
    let tensor = Tensor::<B, 3>::from_data(
        TensorData::new(data.clone(), Shape::new([2_usize, 3, 4])),
        &device,
    );
    let image = Image::new(
        tensor,
        Point::new([0.0; 3]),
        Spacing::new([1.0; 3]),
        Direction::identity(),
    );
    write_dicom_multiframe(&out_path, &image).expect("write");
    let loaded = load_dicom_multiframe::<B, _>(&out_path, &device).expect("load");

    let min_val = data.iter().copied().fold(f32::INFINITY, f32::min);
    let max_val = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let slope = (max_val - min_val) / 65535.0_f32;
    let tolerance = slope + 1.0_f32;

    loaded.with_data_slice(|recovered: &[f32]| {
        assert_eq!(recovered.len(), data.len(), "pixel count must be preserved");
        for (i, (&orig, &rec)) in data.iter().zip(recovered.iter()).enumerate() {
            let diff = (rec - orig).abs();
            assert!(
                diff <= tolerance,
                "pixel {i}: orig={orig:.4} rec={rec:.4} diff={diff:.6} > tol={tolerance:.6}"
            );
        }
    });
}

#[test]
fn test_round_trip_flat_image_exact() {
    let device = <B as Backend>::Device::default();
    let tmp = tempfile::tempdir().expect("tempdir");
    let out_path = tmp.path().join("mf_flat.dcm");
    let constant = 42.75_f32; // exactly representable; "{:.6}" -> "42.750000" -> 42.75
    let data: Vec<f32> = vec![constant; 2 * 3 * 4];
    let tensor = Tensor::<B, 3>::from_data(
        TensorData::new(data.clone(), Shape::new([2_usize, 3, 4])),
        &device,
    );
    let image = Image::new(
        tensor,
        Point::new([0.0; 3]),
        Spacing::new([1.0; 3]),
        Direction::identity(),
    );
    write_dicom_multiframe(&out_path, &image).expect("write");
    let loaded = load_dicom_multiframe::<B, _>(&out_path, &device).expect("load");
    loaded.with_data_slice(|recovered: &[f32]| {
        assert_eq!(recovered.len(), data.len(), "pixel count must be preserved");
        for (i, &rec) in recovered.iter().enumerate() {
            assert!(
                (rec - constant).abs() <= f32::EPSILON,
                "pixel {i}: expected {constant} got {rec} (diff {})",
                (rec - constant).abs()
            );
        }
    });
}

#[test]
fn test_multiframe_info_rescale_slope_intercept_populated() {
    let device = <B as Backend>::Device::default();
    let tmp = tempfile::tempdir().expect("tempdir");
    let out_path = tmp.path().join("rescale.dcm");

    // Analytically derive expected slope/intercept
    let n_frames = 1_usize;
    let rows = 5_usize;
    let cols = 5_usize;
    let data: Vec<f32> = (0..n_frames * rows * cols).map(|i| i as f32).collect();
    let min_val = 0.0_f32;
    let max_val = 24.0_f32;
    let expected_slope = (max_val - min_val) / 65535.0_f32;
    let expected_intercept = min_val;

    let tensor = Tensor::<B, 3>::from_data(
        TensorData::new(data, Shape::new([n_frames, rows, cols])),
        &device,
    );
    let image = Image::new(
        tensor,
        Point::new([0.0; 3]),
        Spacing::new([1.0; 3]),
        Direction::identity(),
    );
    write_dicom_multiframe(&out_path, &image).expect("write");
    let info = read_multiframe_info(&out_path).expect("read_multiframe_info");

    // DS precision is 6 decimal places => tolerance is 5e-7
    let tol = 5e-7_f64;
    assert!(
        (info.rescale_slope - expected_slope as f64).abs() < tol,
        "rescale_slope: expected {:.8} got {:.8}",
        expected_slope,
        info.rescale_slope
    );
    assert!(
        (info.rescale_intercept - expected_intercept as f64).abs() < tol,
        "rescale_intercept: expected {:.8} got {:.8}",
        expected_intercept,
        info.rescale_intercept
    );
}

#[test]
fn test_load_multiframe_spacing_from_slice_thickness() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("mf.dcm");

    // 3 frames of 4×4 pixels, spacing_z = 2.5 mm
    let data: Vec<f32> = (0..3 * 4 * 4).map(|i| i as f32).collect();
    let device = <NdArray as burn::tensor::backend::Backend>::Device::default();
    let tensor = burn::tensor::Tensor::<NdArray, 3>::from_data(
        burn::tensor::TensorData::new(data, burn::tensor::Shape::new([3, 4, 4])),
        &device,
    );
    let image = Image::new(
        tensor,
        Point::new([0.0, 0.0, 0.0]),
        Spacing::new([2.5, 1.0, 1.0]),
        Direction::identity(),
    );
    let opts = MultiFrameSpatialMetadata {
        origin: [0.0, 0.0, 0.0],
        pixel_spacing: [1.0, 1.0],
        slice_thickness: 2.5,
        image_orientation: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        modality: ArrayString::from("CT").unwrap(),
    };
        write_dicom_multiframe_with_options(&path, &image, Some(&opts))
        .expect("write_dicom_multiframe_with_options");

    let out: Image<NdArray, 3> =
        load_dicom_multiframe(&path, &device).expect("load_dicom_multiframe");
    let sp = out.spacing();
    // SliceThickness round-trip: z-spacing must equal 2.5 mm within f32 rescale tolerance
    assert!(
        (sp[0] - 2.5).abs() < 0.01,
        "spacing_z={}, expected 2.5",
        sp[0]
    );
    // Shape must be [3, 4, 4]
    let shape = out.shape();
    assert_eq!(shape[0], 3, "frames");
    assert_eq!(shape[1], 4, "rows");
    assert_eq!(shape[2], 4, "cols");
}
