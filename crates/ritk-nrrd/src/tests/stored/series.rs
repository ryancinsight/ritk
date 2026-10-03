//! Stored NRRD acquisition-series tests.

use super::*;

fn write_series(
    path: &std::path::Path,
    element_type: &str,
    first: &[u8],
    second: &[u8],
    sample_width: usize,
    interleaved: bool,
) -> Result<()> {
    let (sizes, directions, kinds) = if interleaved {
        (
            "sizes: 2 2 1 1",
            "space directions: none (0.5,0,0) (0,1.5,0) (0,0,2)",
            "kinds: list domain domain domain",
        )
    } else {
        (
            "sizes: 2 1 1 2",
            "space directions: (0.5,0,0) (0,1.5,0) (0,0,2) none",
            "kinds: domain domain domain list",
        )
    };
    let fields = [
        format!("type: {element_type}"),
        "dimension: 4".to_owned(),
        sizes.to_owned(),
        "space: left-posterior-superior".to_owned(),
        directions.to_owned(),
        kinds.to_owned(),
        "endian: little".to_owned(),
        "encoding: raw".to_owned(),
    ];
    let fields = fields.iter().map(String::as_str).collect::<Vec<_>>();
    let mut bytes = Vec::new();
    if interleaved {
        for (first_sample, second_sample) in first
            .chunks_exact(sample_width)
            .zip(second.chunks_exact(sample_width))
        {
            bytes.extend_from_slice(first_sample);
            bytes.extend_from_slice(second_sample);
        }
    } else {
        bytes.extend_from_slice(first);
        bytes.extend_from_slice(second);
    }
    write_header(path, &fields, &bytes)
}

#[test]
fn stored_series_preserves_each_sample_type_and_both_acquisition_axis_layouts() -> Result<()> {
    let directory = tempdir()?;
    for (sample_type, element_type, samples) in sample_cases() {
        let first = samples.encode(ByteOrder::LeastSignificantByteFirst)?;
        let sample_width = sample_type.byte_width();
        let second = first
            .chunks_exact(sample_width)
            .rev()
            .flatten()
            .copied()
            .collect::<Vec<_>>();

        for interleaved in [true, false] {
            let layout = if interleaved { "fast" } else { "slow" };
            let path = directory
                .path()
                .join(format!("{sample_type:?}-{layout}.nrrd"));
            write_series(
                &path,
                element_type,
                &first,
                &second,
                sample_width,
                interleaved,
            )?;

            let actual = read_nrrd_stored_series(&path)?;
            assert_eq!(actual.volumes().len(), 2);
            assert_eq!(actual.axis(), &SeriesAxis::List);
            for (volume, expected_bytes) in actual.volumes().iter().zip([first.as_slice(), &second])
            {
                assert_eq!(volume.shape(), [1, 1, 2]);
                assert_eq!(volume.samples().sample_type(), sample_type);
                assert_eq!(
                    volume
                        .samples()
                        .encode(ByteOrder::LeastSignificantByteFirst)?,
                    expected_bytes
                );
                assert_eq!(volume.metadata().spacing().to_array(), [2.0, 1.5, 0.5]);
                assert_eq!(volume.coordinate_map(), &CoordinateMap::Cartesian);
                assert_eq!(volume.calibration(), &IntensityCalibration::Identity);
            }
        }
    }
    Ok(())
}

#[test]
fn diffusion_metadata_rejects_non_list_acquisition_kinds() -> Result<()> {
    let directory = tempdir()?;
    for kind in ["time", "3-vector"] {
        let path = directory.path().join(format!("{kind}.nrrd"));
        let fields = [
            "type: unsigned char",
            "dimension: 4",
            "sizes: 1 1 1 1",
            &format!("kinds: {kind} domain domain domain"),
            "space directions: none (1,0,0) (0,1,0) (0,0,1)",
            "space: LPS",
            "modality:=DWMRI",
            "DWMRI_b-value:=0",
            "DWMRI_gradient_0000:=0 0 0",
            "data file: missing-payload.raw",
        ];
        write_header(&path, &fields, &[])?;

        assert!(matches!(
            read_nrrd_stored_series(&path),
            Err(NrrdStoredReadError::UnsupportedAcquisitionKind { kind: rejected })
                if rejected == kind
        ));
    }
    Ok(())
}

#[test]
fn stored_volume_rejects_diffusion_metadata_before_opening_payload() -> Result<()> {
    let directory = tempdir()?;
    let path = directory.path().join("diffusion-volume.nrrd");
    let fields = [
        "type: unsigned char",
        "dimension: 3",
        "sizes: 1 1 1",
        "space: LPS",
        "space directions: (1,0,0) (0,1,0) (0,0,1)",
        "modality:=DWMRI",
        "DWMRI_b-value:=1000",
        "DWMRI_gradient_0000:=1 0 0",
        "data file: missing-payload.raw",
    ];
    write_header(&path, &fields, &[])?;

    assert!(matches!(
        read_nrrd_stored(&path),
        Err(NrrdStoredReadError::DiffusionRequiresAcquisitionAxis)
    ));
    Ok(())
}
