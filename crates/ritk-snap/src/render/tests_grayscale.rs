use super::{GrayscalePresentation, GrayscalePresentationError, VoiLutFunction, WindowLevel};
use ritk_io::{
    literal_arraystring, DicomObjectNode, DicomReadMetadata, DicomSliceMetadata, DicomTag,
};

#[test]
fn default_linear_uses_dicom_half_sample_boundaries() {
    let window = WindowLevel::new(4.0, 8.0);
    let values = (0..=8).map(f64::from).collect::<Vec<_>>();
    let actual = values
        .iter()
        .map(|&value| window.apply(value))
        .collect::<Vec<_>>();
    assert_eq!(actual, [0, 36, 73, 109, 146, 182, 219, 255, 255]);
}

#[test]
fn linear_exact_preserves_exact_window_boundaries() {
    let window = WindowLevel::new(4.0, 8.0);
    let values = (0..=8).map(f64::from).collect::<Vec<_>>();
    let actual = values
        .iter()
        .map(|&value| window.apply_linear_exact(value))
        .collect::<Vec<_>>();
    assert_eq!(actual, [0, 32, 64, 96, 128, 159, 191, 223, 255]);
}

#[test]
fn linear_width_one_is_a_step_at_the_half_sample_boundary() {
    let window = WindowLevel::new(5.0, 1.0);
    assert_eq!(window.apply(4.5), 0);
    assert_eq!(window.apply(5.0), 255);
    assert_eq!(window.apply(5.5), 255);
}

#[test]
fn sigmoid_is_monotone_symmetric_and_centred() {
    let window = WindowLevel::new(10.0, 4.0);
    let low = window.apply_with_function(6.0, VoiLutFunction::Sigmoid);
    let centre = window.apply_with_function(10.0, VoiLutFunction::Sigmoid);
    let high = window.apply_with_function(14.0, VoiLutFunction::Sigmoid);
    assert!(low < centre && centre < high);
    assert_eq!(centre, 128);
    assert_eq!(u16::from(low) + u16::from(high), 255);
}

#[test]
fn monochrome1_and_voi_function_are_resolved_from_preserved_tags() {
    let mut metadata = DicomReadMetadata {
        photometric_interpretation: Some(literal_arraystring::<16>("MONOCHROME1")),
        slices: vec![DicomSliceMetadata::default()],
        ..DicomReadMetadata::default()
    };
    metadata.slices[0]
        .preservation
        .object
        .insert(DicomObjectNode::text(
            DicomTag::new(0x0028, 0x1056),
            "CS",
            "SIGMOID",
        ));
    let presentation = GrayscalePresentation::from_metadata(&metadata)
        .expect("admitted monochrome metadata must resolve");
    assert_eq!(presentation.voi_function, VoiLutFunction::Sigmoid);
    assert!(presentation.invert);
}

#[test]
fn unsupported_voi_function_fails_with_the_declared_value() {
    let mut metadata = DicomReadMetadata {
        slices: vec![DicomSliceMetadata::default()],
        ..DicomReadMetadata::default()
    };
    metadata.slices[0]
        .preservation
        .object
        .insert(DicomObjectNode::text(
            DicomTag::new(0x0028, 0x1056),
            "CS",
            "POLYNOMIAL",
        ));
    let error = GrayscalePresentation::from_metadata(&metadata)
        .expect_err("unsupported VOI function must be rejected");
    assert_eq!(
        error,
        GrayscalePresentationError::UnsupportedVoiFunction {
            value: "POLYNOMIAL".to_owned()
        }
    );
}

#[test]
fn unsupported_voi_lut_sequence_fails_before_rendering() {
    let mut metadata = DicomReadMetadata {
        slices: vec![DicomSliceMetadata::default()],
        ..DicomReadMetadata::default()
    };
    metadata.slices[0]
        .preservation
        .object
        .insert(DicomObjectNode::text(
            DicomTag::new(0x0028, 0x3010),
            "SQ",
            "table",
        ));
    assert_eq!(
        GrayscalePresentation::from_metadata(&metadata),
        Err(GrayscalePresentationError::UnsupportedVoiLutSequence)
    );
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn gpu_presentation_bitfield_matches_cpu_selection() {
    let linear = GrayscalePresentation {
        voi_function: VoiLutFunction::Linear,
        invert: false,
    };
    let exact_inverted = GrayscalePresentation {
        voi_function: VoiLutFunction::LinearExact,
        invert: true,
    };
    let sigmoid = GrayscalePresentation {
        voi_function: VoiLutFunction::Sigmoid,
        invert: false,
    };
    assert_eq!(linear.gpu_code(), 0);
    assert_eq!(exact_inverted.gpu_code(), 5);
    assert_eq!(sigmoid.gpu_code(), 2);
}
