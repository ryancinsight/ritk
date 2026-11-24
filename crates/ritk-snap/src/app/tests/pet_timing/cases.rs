use ritk_io::format::dicom::classic_pet_fixture::ClassicPetImage;

// Decimal reference for 2^(3600/6584.04) rounded to pointer precision.
// At zero time, (5,000 Bq/mL * 70,000 g)/(350,000,000 Bq) is exactly 1 g/mL.
const ONE_HOUR_SUV: f32 = 1.460_818_3;
const ONE_HOUR: ClassicPetImage = ClassicPetImage::ONE_HOUR;

pub(super) struct TimingCase {
    pub name: &'static str,
    pub image: ClassicPetImage,
    pub second: Option<ClassicPetImage>,
    pub first_pixel: f32,
    pub suv: Option<f32>,
}

pub(super) fn timing_cases() -> Vec<TimingCase> {
    let midnight = ClassicPetImage {
        series_date: "20261122",
        series_time: Some("003000"),
        acquisition_time: Some("003000"),
        administration_time: Some("233000"),
        administration_datetime: Some("20261121233000"),
        ..ONE_HOUR
    };
    let cases = [
        ("same day", ONE_HOUR, Some(ONE_HOUR_SUV)),
        (
            "valid zero",
            ClassicPetImage {
                series_time: Some("100000"),
                acquisition_time: Some("100000"),
                ..ONE_HOUR
            },
            Some(1.0),
        ),
        (
            "missing administration",
            ClassicPetImage {
                administration_time: None,
                administration_datetime: None,
                ..ONE_HOUR
            },
            None,
        ),
        (
            "missing series reference",
            ClassicPetImage {
                series_time: None,
                ..ONE_HOUR
            },
            None,
        ),
        (
            "malformed administration",
            ClassicPetImage {
                administration_time: Some("XX0000"),
                administration_datetime: Some("20261121XX0000"),
                ..ONE_HOUR
            },
            None,
        ),
        (
            "malformed series reference",
            ClassicPetImage {
                series_time: Some("XX0000"),
                ..ONE_HOUR
            },
            None,
        ),
        (
            "invalid minutes",
            ClassicPetImage {
                series_time: Some("106000"),
                acquisition_time: Some("106000"),
                ..ONE_HOUR
            },
            None,
        ),
        ("dated midnight", midnight, Some(ONE_HOUR_SUV)),
        (
            "time-only midnight",
            ClassicPetImage {
                administration_datetime: None,
                ..midnight
            },
            None,
        ),
        (
            "same-date reversal",
            ClassicPetImage {
                administration_datetime: Some("20261122233000"),
                ..midnight
            },
            None,
        ),
        (
            "only administration datetime",
            ClassicPetImage {
                administration_time: None,
                ..ONE_HOUR
            },
            Some(ONE_HOUR_SUV),
        ),
        (
            "conflicting administration clocks",
            ClassicPetImage {
                administration_datetime: Some("20261121090000"),
                ..ONE_HOUR
            },
            None,
        ),
    ];
    let mut timing = Vec::new();
    for mode in ["START", "ADMIN", "NONE"] {
        for (name, image, suv) in cases {
            let expected = if mode == "ADMIN" { Some(1.0) } else { suv };
            timing.push(activity_case(
                name,
                ClassicPetImage {
                    decay_correction: Some(mode),
                    ..image
                },
                expected,
            ));
        }
        let image = ClassicPetImage {
            decay_correction: Some(mode),
            ..ONE_HOUR
        };
        let mut case = activity_case(
            "different references across slices",
            image,
            (mode == "ADMIN").then_some(1.0),
        );
        case.second = Some(ClassicPetImage {
            series_time: Some("120000"),
            acquisition_time: Some("120000"),
            ..image
        });
        timing.push(case);
    }
    let offset_adjusted = ClassicPetImage {
        series_time: Some("100000"),
        acquisition_time: Some("100000"),
        administration_datetime: Some("20261121100000+0100"),
        utc_offset: Some("+0000"),
        ..ONE_HOUR
    };
    for (name, image, suv) in [
        (
            "frame reference differs from acquisition start",
            ClassicPetImage {
                series_time: Some("105900"),
                acquisition_time: Some("105930"),
                frame_reference_ms: "60000",
                ..ONE_HOUR
            },
            Some(ONE_HOUR_SUV),
        ),
        (
            "START with scanner decay factor two",
            ClassicPetImage {
                decay_correction: Some("START"),
                frame_reference_ms: "6584040",
                frame_duration_ms: "7200000",
                start_decay_factor: "2",
                ..ONE_HOUR
            },
            Some(ONE_HOUR_SUV),
        ),
        ("explicit UTC offset", offset_adjusted, Some(ONE_HOUR_SUV)),
        (
            "padded UTC offset",
            ClassicPetImage {
                utc_offset: Some(" +0000 "),
                ..offset_adjusted
            },
            Some(ONE_HOUR_SUV),
        ),
        (
            "padded ADMIN mode",
            ClassicPetImage {
                decay_correction: Some(" ADMIN "),
                ..ONE_HOUR
            },
            Some(1.0),
        ),
        (
            "padded START mode",
            ClassicPetImage {
                decay_correction: Some(" START "),
                ..ONE_HOUR
            },
            Some(ONE_HOUR_SUV),
        ),
        (
            "padded frame duration",
            ClassicPetImage {
                frame_duration_ms: " +120000 ",
                ..ONE_HOUR
            },
            Some(ONE_HOUR_SUV),
        ),
        (
            "frame duration exceeds DICOM integer range",
            ClassicPetImage {
                frame_duration_ms: "2147483648",
                ..ONE_HOUR
            },
            None,
        ),
        (
            "incomparable UTC offsets",
            ClassicPetImage {
                utc_offset: None,
                ..offset_adjusted
            },
            None,
        ),
        (
            "frame reference before acquisition",
            ClassicPetImage {
                acquisition_time: Some("110030"),
                ..ONE_HOUR
            },
            None,
        ),
        (
            "missing acquisition",
            ClassicPetImage {
                acquisition_time: None,
                ..ONE_HOUR
            },
            None,
        ),
        (
            "malformed acquisition",
            ClassicPetImage {
                acquisition_time: Some("110099"),
                ..ONE_HOUR
            },
            None,
        ),
        (
            "multiple frame references",
            ClassicPetImage {
                frame_reference_ms: r"0\1000",
                ..ONE_HOUR
            },
            None,
        ),
        (
            "multiple scanner decay factors",
            ClassicPetImage {
                decay_correction: Some("START"),
                start_decay_factor: r"1\2",
                ..ONE_HOUR
            },
            None,
        ),
        (
            "unknown decay mode",
            ClassicPetImage {
                decay_correction: Some("OTHER"),
                ..ONE_HOUR
            },
            None,
        ),
        (
            "absent decay mode",
            ClassicPetImage {
                decay_correction: None,
                ..ONE_HOUR
            },
            None,
        ),
    ] {
        timing.push(activity_case(name, image, suv));
    }
    timing.push(TimingCase {
        name: "pre-normalized body-weight SUV",
        image: ClassicPetImage {
            decay_correction: None,
            units: "GML",
            ..ONE_HOUR
        },
        second: None,
        first_pixel: 2.0,
        suv: Some(2.0),
    });
    let mut uniform = activity_case(
        "common valid reference across slices",
        ONE_HOUR,
        Some(ONE_HOUR_SUV),
    );
    uniform.second = Some(ONE_HOUR);
    timing.push(uniform);
    timing
}

fn activity_case(name: &'static str, image: ClassicPetImage, suv: Option<f32>) -> TimingCase {
    TimingCase {
        name,
        image,
        second: None,
        first_pixel: 5_000.0,
        suv,
    }
}
