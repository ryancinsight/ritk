//! JPEG 2000 interoperability tests against a captured OpenJPEG corpus.
//!
//! The corpus was generated once with the OpenJPEG 2.5.4 command-line tools
//! from the deterministic inputs below. OpenJPEG encoded every reference
//! codestream and decoded every RITK-produced codestream before capture.
//! Hosted tests execute only the RITK-native Rust codec; they do not link or
//! run OpenJPEG.
//!
//! # Evidence tier
//!
//! - OpenJPEG 5/3 encode → RITK decode: 72 exact lossless cases.
//! - RITK 5/3 encode → OpenJPEG decode: 54 externally accepted streams, pinned
//!   byte-for-byte so the accepted encoder output cannot drift silently.
//! - OpenJPEG 9/7 encode → RITK decode: 54 cases whose reconstruction PSNR must
//!   remain within 1 dB of the captured OpenJPEG decoder baseline.
//! - Ten byte-exact MQ/EBCOT regression patterns compare RITK and OpenJPEG tile
//!   bodies and decode to exact source samples.

use ritk_codecs::jpeg_2000::encoder::{encode_grayscale_j2k, WaveletTransform};
use ritk_codecs::{decode_jpeg2000_fragment, PixelLayout, PixelSignedness};
use std::collections::BTreeSet;
use std::sync::OnceLock;

const CORPUS_BYTES: &[u8] = include_bytes!("fixtures/jpeg2000/openjpeg-2.5.4.corpus");
const CORPUS_MAGIC: [u8; 8] = *b"RITKJ2K1";

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum Producer {
    OpenJpegLossless,
    OpenJpegLossy,
    RitkLossless,
    OpenJpegEscalation,
}

impl Producer {
    fn from_byte(value: u8) -> Self {
        match value {
            0 => Self::OpenJpegLossless,
            1 => Self::OpenJpegLossy,
            2 => Self::RitkLossless,
            3 => Self::OpenJpegEscalation,
            _ => panic!("fixture corpus contains unknown producer tag {value}"),
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Fixture<'a> {
    producer: Producer,
    pattern: u8,
    rows: u16,
    cols: u16,
    precision: u8,
    levels: u8,
    reference_psnr: f64,
    codestream: &'a [u8],
}

fn take<const N: usize>(bytes: &mut &'static [u8]) -> [u8; N] {
    let (head, tail) = bytes
        .split_at_checked(N)
        .expect("fixture corpus record must not be truncated");
    *bytes = tail;
    head.try_into()
        .expect("fixture field length is fixed by the corpus format")
}

fn parse_corpus() -> Vec<Fixture<'static>> {
    let mut bytes = CORPUS_BYTES;
    assert_eq!(take::<8>(&mut bytes), CORPUS_MAGIC, "fixture corpus magic");
    let count = u32::from_le_bytes(take::<4>(&mut bytes)) as usize;
    let mut fixtures = Vec::with_capacity(count);
    for _ in 0..count {
        let producer = Producer::from_byte(take::<1>(&mut bytes)[0]);
        let pattern = take::<1>(&mut bytes)[0];
        let rows = u16::from_le_bytes(take::<2>(&mut bytes));
        let cols = u16::from_le_bytes(take::<2>(&mut bytes));
        let precision = take::<1>(&mut bytes)[0];
        let levels = take::<1>(&mut bytes)[0];
        let reference_psnr = f64::from_bits(u64::from_le_bytes(take::<8>(&mut bytes)));
        let codestream_len = u32::from_le_bytes(take::<4>(&mut bytes)) as usize;
        let (codestream, tail) = bytes
            .split_at_checked(codestream_len)
            .expect("fixture codestream must not be truncated");
        bytes = tail;
        fixtures.push(Fixture {
            producer,
            pattern,
            rows,
            cols,
            precision,
            levels,
            reference_psnr,
            codestream,
        });
    }
    assert!(
        bytes.is_empty(),
        "fixture corpus must not contain trailing bytes"
    );
    fixtures
}

fn corpus() -> &'static [Fixture<'static>] {
    static CORPUS: OnceLock<Vec<Fixture<'static>>> = OnceLock::new();
    CORPUS.get_or_init(parse_corpus)
}

fn synthetic(rows: u32, cols: u32, precision: u32) -> Vec<i32> {
    let mut state = 0xC0FF_EE00_DEAD_F00Du64;
    let amplitude = 1i64 << precision;
    (0..(rows * cols) as usize)
        .map(|index| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let noise = ((state >> 33) % 64) as i64;
            (((index as i64 * 5) + noise) % amplitude) as i32
        })
        .collect()
}

fn escalation_pattern(pattern: u8) -> Vec<i32> {
    let mut pixels = vec![128; 64];
    match pattern {
        1 => pixels[0] = 129,
        2 => pixels[4 * 8 + 4] = 129,
        3 => pixels[0] = 131,
        4 => pixels[4 * 8 + 3] = 125,
        5 => pixels[0] = 138,
        6 => pixels[3 * 8 + 1] = 100,
        7 => {
            pixels[0] = 200;
            pixels[9] = 100;
        }
        8 => {
            for (index, pixel) in pixels.iter_mut().enumerate() {
                *pixel = ((index % 8) * 30) as i32;
            }
        }
        9 => {
            for (index, pixel) in pixels.iter_mut().enumerate() {
                *pixel = ((index / 8) * 30) as i32;
            }
        }
        10 => return synthetic(8, 8, 8),
        _ => panic!("unknown escalation fixture pattern {pattern}"),
    }
    pixels
}

fn pixels(fixture: Fixture<'_>) -> Vec<i32> {
    if fixture.pattern == 0 {
        synthetic(
            u32::from(fixture.rows),
            u32::from(fixture.cols),
            u32::from(fixture.precision),
        )
    } else {
        assert_eq!(
            fixture.producer,
            Producer::OpenJpegEscalation,
            "only escalation fixtures carry a pattern tag"
        );
        escalation_pattern(fixture.pattern)
    }
}

fn layout(fixture: Fixture<'_>) -> PixelLayout {
    PixelLayout {
        rows: usize::from(fixture.rows),
        cols: usize::from(fixture.cols),
        samples_per_pixel: 1,
        bits_allocated: if fixture.precision <= 8 { 8 } else { 16 },
        pixel_representation: PixelSignedness::Unsigned,
        rescale_slope: 1.0,
        rescale_intercept: 0.0,
    }
}

fn decode(fixture: Fixture<'_>) -> Vec<f32> {
    decode_jpeg2000_fragment(fixture.codestream, layout(fixture)).unwrap_or_else(|error| {
        panic!(
            "RITK decode failed for {:?} {}x{} precision {} levels {}: {error:#}",
            fixture.producer, fixture.rows, fixture.cols, fixture.precision, fixture.levels
        )
    })
}

fn encode(fixture: Fixture<'_>, source: &[i32]) -> Vec<u8> {
    encode_grayscale_j2k(
        source,
        u32::from(fixture.rows),
        u32::from(fixture.cols),
        u32::from(fixture.precision),
        PixelSignedness::Unsigned,
        fixture.levels,
        WaveletTransform::Reversible,
    )
    .expect("captured valid fixture must encode")
}

fn tile_body(codestream: &[u8]) -> &[u8] {
    let sod = codestream
        .windows(2)
        .position(|window| window == [0xFF, 0x93])
        .expect("fixture codestream must contain SOD");
    let eoc = codestream
        .windows(2)
        .rposition(|window| window == [0xFF, 0xD9])
        .expect("fixture codestream must contain EOC");
    &codestream[sod + 2..eoc]
}

fn mse_vs_original(reconstruction: impl Iterator<Item = f64>, original: &[i32]) -> f64 {
    reconstruction
        .zip(original)
        .map(|(actual, &expected)| {
            let error = actual - f64::from(expected);
            error * error
        })
        .sum::<f64>()
        / original.len() as f64
}

fn psnr(mse: f64, precision: u8) -> f64 {
    if mse <= 0.0 {
        return f64::INFINITY;
    }
    let peak = f64::from((1u32 << precision) - 1);
    10.0 * (peak * peak / mse).log10()
}

fn fixtures_for(producer: Producer) -> impl Iterator<Item = &'static Fixture<'static>> {
    corpus()
        .iter()
        .filter(move |fixture| fixture.producer == producer)
}

#[test]
fn corpus_covers_the_complete_previous_interop_matrix() {
    let actual: BTreeSet<_> = corpus()
        .iter()
        .map(|fixture| {
            (
                fixture.producer,
                fixture.pattern,
                fixture.rows,
                fixture.cols,
                fixture.precision,
                fixture.levels,
            )
        })
        .collect();
    let mut expected = BTreeSet::new();
    for &(producer, sizes) in &[
        (
            Producer::OpenJpegLossless,
            &[(64u16, 64u16), (64, 80), (80, 64), (100, 150)][..],
        ),
        (
            Producer::OpenJpegLossy,
            &[(64u16, 64u16), (64, 80), (100, 150)][..],
        ),
        (
            Producer::RitkLossless,
            &[(64u16, 64u16), (64, 80), (100, 150)][..],
        ),
    ] {
        for &(rows, cols) in sizes {
            for precision in [8u8, 12, 16] {
                for levels in 0..=5u8 {
                    expected.insert((producer, 0, rows, cols, precision, levels));
                }
            }
        }
    }
    for pattern in 1..=10u8 {
        expected.insert((Producer::OpenJpegEscalation, pattern, 8, 8, 8, 0));
    }
    assert_eq!(actual, expected, "captured matrix must remain complete");
    assert_eq!(corpus().len(), 190, "captured matrix case count");
}

#[test]
fn openjpeg_lossless_corpus_decodes_exactly() {
    for &fixture in fixtures_for(Producer::OpenJpegLossless) {
        let source = pixels(fixture);
        let expected: Vec<f32> = source.iter().map(|&sample| sample as f32).collect();
        assert_eq!(decode(fixture), expected, "lossless OpenJPEG fixture");
    }
}

#[test]
fn ritk_lossless_encoder_remains_openjpeg_accepted() {
    for &fixture in fixtures_for(Producer::RitkLossless) {
        let source = pixels(fixture);
        let encoded = encode(fixture, &source);
        let first_difference = encoded
            .iter()
            .zip(fixture.codestream)
            .position(|(actual, expected)| actual != expected)
            .or((encoded.len() != fixture.codestream.len())
                .then(|| encoded.len().min(fixture.codestream.len())));
        assert_eq!(
            first_difference,
            None,
            "RITK encoder output diverged from the stream accepted and exactly \
             decoded by OpenJPEG 2.5.4 at byte {first_difference:?}: {}x{} \
             precision {} levels {}, lengths {} vs {}",
            fixture.rows,
            fixture.cols,
            fixture.precision,
            fixture.levels,
            encoded.len(),
            fixture.codestream.len()
        );
        let expected: Vec<f32> = source.iter().map(|&sample| sample as f32).collect();
        assert_eq!(decode(fixture), expected, "captured RITK fixture");
    }
}

#[test]
fn openjpeg_escalation_corpus_matches_tile_bodies() {
    for &fixture in fixtures_for(Producer::OpenJpegEscalation) {
        let source = pixels(fixture);
        let expected: Vec<f32> = source.iter().map(|&sample| sample as f32).collect();
        assert_eq!(decode(fixture), expected, "escalation fixture decode");
        assert_eq!(
            tile_body(&encode(fixture, &source)),
            tile_body(fixture.codestream),
            "MQ/EBCOT tile body for escalation pattern {}",
            fixture.pattern
        );
    }
}

#[test]
fn openjpeg_lossy_corpus_tracks_reference_psnr() {
    for &fixture in fixtures_for(Producer::OpenJpegLossy) {
        assert!(
            fixture.reference_psnr.is_finite(),
            "lossy fixture must carry a finite OpenJPEG PSNR baseline"
        );
        let source = pixels(fixture);
        let reconstruction = decode(fixture);
        assert_eq!(
            reconstruction.len(),
            source.len(),
            "lossy fixture sample count"
        );
        let ritk_psnr = psnr(
            mse_vs_original(
                reconstruction.iter().map(|&sample| f64::from(sample)),
                &source,
            ),
            fixture.precision,
        );
        assert!(
            ritk_psnr >= fixture.reference_psnr - 1.0,
            "RITK lossy PSNR {ritk_psnr:.2} dB is more than 1 dB below \
             OpenJPEG 2.5.4 baseline {:.2} dB for {}x{} precision {} levels {}",
            fixture.reference_psnr,
            fixture.rows,
            fixture.cols,
            fixture.precision,
            fixture.levels
        );
    }
}
