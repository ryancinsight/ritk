#![expect(clippy::print_stderr, reason = "ratchet RITK-LINT-1")]
#![expect(clippy::print_stdout, reason = "ratchet RITK-LINT-1")]
use ritk_dicom::tags;
use ritk_dicom::{
    decode_frame_with, DecodeFrameRequest, DicomAttributeRead, DicomTag, NativeBackend,
    PixelLayout, PixelSignedness, TransferSyntaxKind,
};
use ritk_io::read_dicom_seg;
use std::env;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: {} <dicom_file>", args[0]);
        std::process::exit(1);
    }

    let obj = ritk_dicom::dataset::DicomFile::open(&args[1])?;

    let tags_to_check = [
        (tags::SERIES_INSTANCE_UID, "SeriesInstanceUID"),
        (tags::STUDY_INSTANCE_UID, "StudyInstanceUID"),
        (tags::MODALITY, "Modality"),
        (tags::SERIES_DESCRIPTION, "SeriesDescription"),
        (tags::IMAGE_ORIENTATION_PATIENT, "ImageOrientationPatient"),
        (tags::IMAGE_POSITION_PATIENT, "ImagePositionPatient"),
        (tags::PIXEL_SPACING, "PixelSpacing"),
        (tags::ROWS, "Rows"),
        (tags::COLUMNS, "Columns"),
        (tags::INSTANCE_NUMBER, "InstanceNumber"),
        (tags::SLICE_THICKNESS, "SliceThickness"),
        (tags::SLICE_LOCATION, "SliceLocation"),
        (tags::SAMPLES_PER_PIXEL, "SamplesPerPixel"),
        (tags::BITS_ALLOCATED, "BitsAllocated"),
        (
            tags::PHOTOMETRIC_INTERPRETATION,
            "PhotometricInterpretation",
        ),
    ];

    for (tag, name) in &tags_to_check {
        match obj.element(*tag) {
            Ok(elem) => {
                println!("{}: {}", name, elem.to_str().unwrap_or("?".into()));
            }
            _ => {
                println!("{}: <missing>", name);
            }
        }
    }

    let modality = obj
        .element(tags::MODALITY)
        .ok()
        .and_then(|e| e.to_str().ok().map(|s| s.trim().to_owned()));

    if modality.as_deref() == Some("SEG") {
        let seg = read_dicom_seg(&args[1])?;
        println!(
            "SEG: {} segment(s), {} frame(s), {}x{} pixels/frame, {} bits",
            seg.segments.len(),
            seg.n_frames,
            seg.rows,
            seg.cols,
            seg.bits_allocated
        );
        for segment in &seg.segments {
            println!(
                "Segment {}: {} ({})",
                segment.segment_number,
                segment.segment_label,
                segment
                    .algorithm_type
                    .as_ref()
                    .map(|a| a.as_dicom_str())
                    .unwrap_or("algorithm unspecified")
            );
        }
        if let Some(spacing) = seg.pixel_spacing {
            println!("SEG PixelSpacing: {}\\{}", spacing[0], spacing[1]);
        }
        if let Some(thickness) = seg.slice_thickness {
            println!("SEG SliceThickness: {}", thickness);
        }
        println!(
            "SEG FramePositionsPresent: {}",
            seg.image_position_per_frame
                .iter()
                .filter(|p| p.is_some())
                .count()
        );
        return Ok(());
    }

    let layout = PixelLayout {
        rows: usize::from(obj.required_unsigned(tags::ROWS, "Rows")?),
        cols: usize::from(obj.required_unsigned(tags::COLUMNS, "Columns")?),
        samples_per_pixel: usize::from(
            obj.required_unsigned(tags::SAMPLES_PER_PIXEL, "SamplesPerPixel")?,
        ),
        bits_allocated: obj.required_unsigned(tags::BITS_ALLOCATED, "BitsAllocated")?,
        bits_stored: obj.required_unsigned(tags::BITS_STORED, "BitsStored")?,
        pixel_representation: match obj
            .optional_unsigned(tags::PIXEL_REPRESENTATION, "PixelRepresentation")?
        {
            Some(1) => PixelSignedness::Signed,
            _ => PixelSignedness::Unsigned,
        },
        rescale_slope: obj
            .optional_decimal(tags::RESCALE_SLOPE, "RescaleSlope")?
            .unwrap_or(1.0) as f32,
        rescale_intercept: obj
            .optional_decimal(tags::RESCALE_INTERCEPT, "RescaleIntercept")?
            .unwrap_or(0.0) as f32,
    };
    let frames = obj
        .get(DicomTag::new(0x0028, 0x0008))
        .map_or(Ok(1), |element| element.to_num::<u32>())?;
    let transfer_syntax = TransferSyntaxKind::from_uid(obj.transfer_syntax_uid());
    let mut count = 0usize;
    for frame_index in 0..frames {
        let frame = decode_frame_with::<NativeBackend>(
            &obj,
            DecodeFrameRequest {
                frame_index,
                transfer_syntax: transfer_syntax.clone(),
                layout,
            },
        )?;
        count = count
            .checked_add(frame.pixels.len())
            .ok_or_else(|| anyhow::anyhow!("sample count overflows"))?;
    }
    println!(
        "PixelData: {count} decoded samples, {} bits, {frames} frames",
        layout.bits_allocated
    );
    Ok(())
}
