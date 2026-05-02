//! First-slice DICOM metadata rows for the viewer tag inspector.

use ritk_io::DicomSliceMetadata;

use super::{format_f64_2, format_f64_3, format_f64_6, push_slice_opt, MetadataRow};

pub(super) fn push_first_slice_rows(slice: &DicomSliceMetadata, rows: &mut Vec<MetadataRow>) {
    push_slice_opt(
        rows,
        "0008,0018",
        "SOPInstanceUID",
        "UI",
        slice.sop_instance_uid.as_deref(),
    );
    if let Some(value) = slice.instance_number {
        rows.push(MetadataRow::first_slice(
            "0020,0013",
            "InstanceNumber",
            "IS",
            value.to_string(),
        ));
    }
    if let Some(value) = slice.slice_location {
        rows.push(MetadataRow::first_slice(
            "0020,1041",
            "SliceLocation",
            "DS",
            format!("{value:.6}"),
        ));
    }
    if let Some(value) = slice.image_position_patient {
        rows.push(MetadataRow::first_slice(
            "0020,0032",
            "ImagePositionPatient",
            "DS",
            format_f64_3(value),
        ));
    }
    if let Some(value) = slice.image_orientation_patient {
        rows.push(MetadataRow::first_slice(
            "0020,0037",
            "ImageOrientationPatient",
            "DS",
            format_f64_6(value),
        ));
    }
    if let Some(value) = slice.pixel_spacing {
        rows.push(MetadataRow::first_slice(
            "0028,0030",
            "PixelSpacing",
            "DS",
            format_f64_2(value),
        ));
    }
    if let Some(value) = slice.slice_thickness {
        rows.push(MetadataRow::first_slice(
            "0018,0050",
            "SliceThickness",
            "DS",
            format!("{value:.6}"),
        ));
    }
    rows.push(MetadataRow::first_slice(
        "0028,1053",
        "RescaleSlope",
        "DS",
        format!("{:.6}", slice.rescale_slope),
    ));
    rows.push(MetadataRow::first_slice(
        "0028,1052",
        "RescaleIntercept",
        "DS",
        format!("{:.6}", slice.rescale_intercept),
    ));
    push_slice_opt(
        rows,
        "0008,0016",
        "SOPClassUID",
        "UI",
        slice.sop_class_uid.as_deref(),
    );
    push_slice_opt(
        rows,
        "meta:transfer-syntax",
        "TransferSyntaxUID",
        "UI",
        slice.transfer_syntax_uid.as_deref(),
    );
    rows.push(MetadataRow::first_slice(
        "0028,0103",
        "PixelRepresentation",
        "US",
        slice.pixel_representation.to_string(),
    ));
    rows.push(MetadataRow::first_slice(
        "0028,0100",
        "BitsAllocated",
        "US",
        slice.bits_allocated.to_string(),
    ));
    if let Some(value) = slice.window_center {
        rows.push(MetadataRow::first_slice(
            "0028,1050",
            "WindowCenter",
            "DS",
            format!("{value:.6}"),
        ));
    }
    if let Some(value) = slice.window_width {
        rows.push(MetadataRow::first_slice(
            "0028,1051",
            "WindowWidth",
            "DS",
            format!("{value:.6}"),
        ));
    }
    if let Some(value) = slice.gantry_tilt {
        rows.push(MetadataRow::first_slice(
            "0018,1120",
            "GantryDetectorTilt",
            "DS",
            format!("{value:.6}"),
        ));
    }
}
