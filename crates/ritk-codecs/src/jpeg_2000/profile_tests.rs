//! Compare complete frames with OpenJPEG and OpenJPH, independently of RITK's writer.
use super::*;
use crate::{ImageReadBudget, PixelLayout, PixelSignedness};

macro_rules! fixture {
    ($file:literal) => {
        include_bytes!(concat!("../../tests/fixtures/jpeg2000/profiles/", $file)).as_slice()
    };
}

fn layout() -> PixelLayout {
    PixelLayout {
        rows: 33,
        cols: 35,
        samples_per_pixel: 1,
        bits_allocated: 16,
        bits_stored: 12,
        pixel_representation: PixelSignedness::Unsigned,
        rescale_slope: 1.,
        rescale_intercept: 0.,
    }
}

fn decode(
    data: &[u8],
    layout: PixelLayout,
    profile: Option<HtJ2kProfile>,
) -> anyhow::Result<Vec<i64>> {
    let budget = ImageReadBudget::DEFAULT;
    Ok(match profile {
        Some(profile) => decode_jpeg2000_stored_with_profile(
            data,
            layout,
            budget,
            Jpeg2000Profile::HighThroughput(profile),
        )?,
        None => decode_jpeg2000_stored_fragment(data, layout, budget)?,
    })
}

fn gray(i: usize) -> i64 {
    let (x, y) = (i % 35, i / 35);
    i64::try_from((x * 31 + y * 73 + (x ^ y) * 17) % 4096).expect("fixture invariant")
}

#[test]
fn independent_packet_organizations_and_progressions() -> anyhow::Result<()> {
    let expected: Vec<_> = (0..35 * 33).map(gray).collect();
    for source in [
        fixture!("classic-LRCP.j2c"),
        fixture!("classic-RLCP.j2c"),
        fixture!("classic-RPCL.j2c"),
        fixture!("classic-PCRL.j2c"),
        fixture!("classic-CPRL.j2c"),
        fixture!("classic-ppm.j2c"),
        fixture!("classic-ppt.j2c"),
        fixture!("classic-poc.j2c"),
        fixture!("classic-tileparts.j2c"),
        fixture!("classic-tile-override.j2c"),
        fixture!("classic-component-override.j2c"),
    ] {
        assert_eq!(decode(source, layout(), None)?, expected);
    }
    for source in [
        fixture!("ht-LRCP.j2c"),
        fixture!("ht-RLCP.j2c"),
        fixture!("ht-RPCL.j2c"),
        fixture!("ht-PCRL.j2c"),
        fixture!("ht-CPRL.j2c"),
        fixture!("ht-tiles.j2c"),
    ] {
        assert_eq!(
            decode(source, layout(), Some(HtJ2kProfile::Lossless))?,
            expected
        );
    }
    assert_eq!(
        decode(
            fixture!("ht-rpcl-parts.j2c"),
            layout(),
            Some(HtJ2kProfile::LosslessRpcl)
        )?,
        expected
    );
    Ok(())
}

#[test]
fn lossless_wide_coefficients_and_unsigned_dc_level_shift() -> anyhow::Result<()> {
    for source in [
        fixture!("ht-deep-basic.j2c"),
        fixture!("ht-deep-advanced.j2c"),
    ] {
        let reference: Vec<_> = (0..1155i64)
            .map(|i| {
                if (i % 35 + i / 35) % 2 == 0 {
                    (1 << 27) + i
                } else {
                    -((1 << 27) + i)
                }
            })
            .collect();
        let profile = Some(HtJ2kProfile::Lossless);
        let signed = PixelLayout {
            bits_allocated: 32,
            bits_stored: 30,
            pixel_representation: PixelSignedness::Signed,
            ..layout()
        };
        assert_eq!(decode(source, signed, profile)?, reference);
        let mut wide = source.to_vec();
        wide[42] = 37; // Ssiz changes the output DC offset only for the same reversible coefficients.
        let expected: Vec<_> = reference.iter().map(|&v| v + (1 << 37)).collect();
        assert_eq!(
            decode(
                &wide,
                PixelLayout {
                    bits_allocated: 40,
                    bits_stored: 38,
                    ..layout()
                },
                profile
            )?,
            expected
        );
    }
    let mut part1 = fixture!("classic-CPRL.j2c").to_vec();
    part1[42] = 37;
    let expected: Vec<_> = (0..1155).map(|i| gray(i) + (1 << 37) - (1 << 11)).collect();
    assert_eq!(
        decode(
            &part1,
            PixelLayout {
                bits_allocated: 40,
                bits_stored: 38,
                ..layout()
            },
            None
        )?,
        expected
    );
    Ok(())
}

#[test]
fn independent_rgb_planes_and_reversible_color_transforms() -> anyhow::Result<()> {
    let expected: Vec<_> = (0..1155usize)
        .flat_map(|i| {
            (0..3).map(move |c| {
                let (x, y) = (i % 35, i / 35);
                i64::try_from((11 * x + 17 * y + 57 * c + (x ^ y) * (c + 1)) % 256)
                    .expect("fixture invariant")
            })
        })
        .collect();
    let color = PixelLayout {
        bits_stored: 8,
        bits_allocated: 8,
        samples_per_pixel: 3,
        ..layout()
    };
    for (src, profile) in [
        (fixture!("classic-color-rgb.j2c"), None),
        (fixture!("classic-color-rct.j2c"), None),
        (fixture!("ht-color-rgb.j2c"), Some(HtJ2kProfile::Lossless)),
        (fixture!("ht-color-rct.j2c"), Some(HtJ2kProfile::Lossless)),
    ] {
        assert_eq!(decode(src, color, profile)?, expected);
    }
    Ok(())
}

#[test]
fn irreversible_matches_openjph_within_one_stored_unit() -> anyhow::Result<()> {
    let actual = decode(
        fixture!("ht-irreversible.j2c"),
        layout(),
        Some(HtJ2kProfile::Unrestricted),
    )?;
    let expected = fixture!("ht-irreversible.u16le").chunks_exact(2);
    assert_eq!(actual.len(), expected.len());
    for (i, (v, r)) in actual.into_iter().zip(expected).enumerate() {
        assert!(
            (v - i64::from(u16::from_le_bytes([r[0], r[1]]))).abs() <= 1,
            "OpenJPH disagreement at sample {i}"
        );
    }
    Ok(())
}

#[test]
fn irreversible_color_transform_agrees_with_openjph_rgb() -> anyhow::Result<()> {
    let color = PixelLayout {
        samples_per_pixel: 3,
        bits_allocated: 8,
        bits_stored: 8,
        ..layout()
    };
    let values = decode(
        fixture!("ht-color-ict.j2c"),
        color,
        Some(HtJ2kProfile::Unrestricted),
    )?;
    let reference = fixture!("ht-color-ict.rgb");
    assert_eq!(values.len(), reference.len());
    for (&sample, &channel) in values.iter().zip(reference) {
        assert!(
            (sample - i64::from(channel)).abs() <= 1,
            "OpenJPH RGB {channel}, RITK {sample}"
        );
    }
    Ok(())
}

#[test]
fn incorrect_syntax_and_progressive_profile_fail_before_output() {
    for (src, profile, reason) in [
        (
            fixture!("classic-LRCP.j2c"),
            Some(HtJ2kProfile::Lossless),
            "capabilities",
        ),
        (fixture!("ht-LRCP.j2c"), None, "capabilities"),
        (
            fixture!("ht-irreversible.j2c"),
            Some(HtJ2kProfile::Lossless),
            "reversible",
        ),
        (
            fixture!("ht-LRCP.j2c"),
            Some(HtJ2kProfile::LosslessRpcl),
            "RPCL progression",
        ),
        (
            fixture!("ht-RPCL.j2c"),
            Some(HtJ2kProfile::LosslessRpcl),
            "tile length markers",
        ),
    ] {
        let error = decode(src, layout(), profile).expect_err("invalid profile or tile index");
        assert!(format!("{error:#}").contains(reason), "{error:#}");
    }
    // Both axes exceed the .202 thumbnail bound when COD specifies no decomposition.
    let mut large = fixture!("ht-rpcl-parts.j2c").to_vec();
    for start in [8, 12, 24, 28] {
        large[start..start + 4].copy_from_slice(&128u32.to_be_bytes());
    }
    let cod = large
        .windows(2)
        .position(|bytes| bytes == [0xff, 0x52])
        .expect("COD");
    large[cod + 9] = 0;
    let oversized = PixelLayout {
        rows: 128,
        cols: 128,
        ..layout()
    };
    assert!(format!(
        "{:#}",
        decode(&large, oversized, Some(HtJ2kProfile::LosslessRpcl))
            .expect_err("thumbnail dimensions")
    )
    .contains("base resolution"));
    let mut wrong_wavelet = fixture!("classic-LRCP.j2c").to_vec();
    let cod = wrong_wavelet
        .windows(2)
        .position(|bytes| bytes == [0xff, 0x52])
        .expect("COD");
    wrong_wavelet[cod + 13] = 0;
    let error = decode_jpeg2000_with_profile(
        &wrong_wavelet,
        layout(),
        ImageReadBudget::DEFAULT,
        Jpeg2000Profile::Part1 {
            lossless_only: true,
        },
    )
    .expect_err("irreversible .90");
    assert!(format!("{error:#}").contains("reversible wavelet"));
    let mut damaged = fixture!("ht-rpcl-parts.j2c").to_vec();
    let tlm = damaged
        .windows(2)
        .position(|b| b == [255, 85])
        .expect("fixture invariant");
    damaged[tlm + 9] ^= 1;
    let error = decode(&damaged, layout(), Some(HtJ2kProfile::LosslessRpcl))
        .expect_err("invalid profile or tile index");
    assert!(format!("{error:#}").contains("TLM entry"), "{error:#}");
}

#[test]
fn ht_writer_round_trips_scalar_frames_and_progressive_thumbnail() -> anyhow::Result<()> {
    let (rows, cols) = (131u16, 147u16);
    let values: Vec<u16> = (0..rows)
        .flat_map(|y| (0..cols).map(move |x| x.wrapping_mul(371) ^ y.wrapping_mul(197)))
        .collect();
    let layout = PixelLayout {
        rows: usize::from(rows),
        cols: usize::from(cols),
        bits_stored: 16,
        ..layout()
    };
    for profile in [
        HtJ2kProfile::Lossless,
        HtJ2kProfile::LosslessRpcl,
        HtJ2kProfile::Unrestricted,
    ] {
        let frame = encode_grayscale_htj2k(&values, rows.into(), cols.into(), profile)?;
        let decoded = decode(&frame, layout, Some(profile))?;
        let error = decoded
            .iter()
            .zip(&values)
            .map(|(&a, &b)| (a - i64::from(b)).abs())
            .max()
            .expect("fixture invariant");
        if profile == HtJ2kProfile::Unrestricted {
            assert!(error < 256, "maximum irreversible error {error}");
        } else {
            assert_eq!(error, 0);
        }
    }
    Ok(())
}
