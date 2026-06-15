use dicom::dictionary_std::tags;
use dicom::object::open_file;
use dicom::pixeldata::PixelDecoder;
use ritk_io::read_dicom_seg;
use std::env;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: {} <dicom_file>", args[0]);
        std::process::exit(1);
    }

    let obj = open_file(&args[1])?;

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
        if let Ok(elem) = obj.element(*tag) {
            println!("{}: {}", name, elem.to_str().unwrap_or("?".into()));
        } else {
            println!("{}: <missing>", name);
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

    // Pixel data info
    if let Ok(pd) = obj.decode_pixel_data() {
        println!(
            "PixelData: {} samples, {} bits, {} frames",
            pd.samples_per_pixel(),
            pd.bits_allocated(),
            pd.number_of_frames()
        );
        let vec_f32 = pd.to_vec::<f32>()?;
        println!("Decoded f32 length: {}", vec_f32.len());
    } else {
        println!("PixelData: <failed to decode>");
    }

    Ok(())
}
