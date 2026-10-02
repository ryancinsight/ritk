//! Sample-type coverage (ADR 0053): every fixed-width type written and read in
//! its own type, both byte orders, and the `scl_slope`/`scl_inter` rescale.
//!
//! Hand-built headers write each field at its `nifti1.h` offset, so the reader
//! is checked against the specification rather than against this crate's
//! header encoder.

use super::*;
use consus_core::{write_integer, ByteOrder};
use ritk_codecs::sample::{Rescale, Sample};

/// A NIfTI-1 single-file header written field by field.
pub(super) struct RawNifti1 {
    order: ByteOrder,
    datatype: i16,
    bitpix: i16,
    dims: [u16; 3],
    scl_slope: f32,
    scl_inter: f32,
}

impl RawNifti1 {
    pub(super) fn new(order: ByteOrder, datatype: i16, bitpix: i16, dims: [u16; 3]) -> Self {
        Self {
            order,
            datatype,
            bitpix,
            dims,
            scl_slope: 0.0,
            scl_inter: 0.0,
        }
    }

    fn with_rescale(mut self, scl_slope: f32, scl_inter: f32) -> Self {
        self.scl_slope = scl_slope;
        self.scl_inter = scl_inter;
        self
    }

    /// The header, the four extension bytes, and `payload` at offset 352.
    pub(super) fn file(&self, payload: &[u8]) -> Vec<u8> {
        let order = self.order;
        let mut bytes = vec![0_u8; 352];
        put_scalar(&mut bytes[0..], 348_i32, order);
        let [nx, ny, nz] = self.dims;
        for (index, dim) in [3, nx, ny, nz, 1, 1, 1, 1].into_iter().enumerate() {
            put_scalar(&mut bytes[40 + index * 2..], dim, order);
        }
        put_scalar(&mut bytes[70..], self.datatype, order);
        put_scalar(&mut bytes[72..], self.bitpix, order);
        for index in 0..8 {
            put_scalar(&mut bytes[76 + index * 4..], 1.0_f32, order);
        }
        put_scalar(&mut bytes[108..], 352.0_f32, order);
        put_scalar(&mut bytes[112..], self.scl_slope, order);
        put_scalar(&mut bytes[116..], self.scl_inter, order);
        bytes[344..348].copy_from_slice(b"n+1\0");
        bytes.extend_from_slice(payload);
        bytes
    }
}

pub(super) fn put_scalar<T: consus_core::EndianScalar>(
    slot: &mut [u8],
    value: T,
    order: ByteOrder,
) {
    write_integer(slot, value, order).expect("invariant: the field lies inside the header");
}

/// Encode samples one scalar at a time, independent of the bulk encoder.
pub(super) fn encode<T: Sample>(values: &[T], order: ByteOrder) -> Vec<u8> {
    let width = T::TYPE.byte_width();
    let mut bytes = vec![0_u8; values.len() * width];
    for (value, slot) in values.iter().zip(bytes.chunks_exact_mut(width)) {
        put_scalar(slot, *value, order);
    }
    bytes
}

pub(super) fn image_of<T: Sample>(values: Vec<T>) -> Image<T, TestBackend, 3> {
    Image::from_flat_on(
        values,
        [2, 2, 2],
        Point::new([-11.0, 7.5, 3.25]),
        Spacing::new([2.0, 1.5, 0.75]),
        Direction::identity(),
        &SequentialBackend,
    )
    .expect("eight voxels fill a 2x2x2 grid")
}

/// Write `values` in `T` through both header versions and gzip, checking the
/// `datatype`/`bitpix` fields on disk and reading every sample back exactly.
fn round_trips<T: Sample + PartialEq + std::fmt::Debug>(values: [T; 8], code: i16, bitpix: i16) {
    let dir = tempdir().expect("temporary directory");
    let backend = SequentialBackend;
    let image = image_of(values.to_vec());

    let nifti1 = dir.path().join("typed.nii");
    crate::write_nifti(&nifti1, &image, &backend).expect("NIfTI-1 write");
    let bytes = std::fs::read(&nifti1).expect("written file");
    assert_eq!(bytes[70..72], code.to_le_bytes(), "{} datatype", T::TYPE);
    assert_eq!(bytes[72..74], bitpix.to_le_bytes(), "{} bitpix", T::TYPE);
    assert_eq!(bytes[112..116], 1.0_f32.to_le_bytes(), "scl_slope");

    let nifti2 = dir.path().join("typed2.nii.gz");
    crate::write_nifti2(&nifti2, &image, &backend).expect("NIfTI-2 write");

    for path in [&nifti1, &nifti2] {
        let loaded = crate::read_nifti::<T, _, _, _>(path, &backend, Exact).expect("typed read");
        assert_eq!(
            loaded.data_slice().expect("contiguous").to_vec(),
            values,
            "{} through {}",
            T::TYPE,
            path.display()
        );
        let (stored, rescale) =
            crate::read_nifti_stored::<T, _, _, _>(path, &backend, Exact).expect("stored read");
        assert_eq!(stored.data_slice().expect("contiguous").to_vec(), values);
        assert_eq!(rescale, Rescale::IDENTITY);
    }

    // The writer emits little-endian only; files from big-endian hosts are
    // built field by field.
    for order in [ByteOrder::LittleEndian, ByteOrder::BigEndian] {
        let file = RawNifti1::new(order, code, bitpix, [2, 2, 2]).file(&encode(&values, order));
        let loaded = crate::read_nifti_from_bytes::<T, _, _>(&file, &backend, Exact)
            .expect("hand-built file");
        assert_eq!(
            loaded.data_slice().expect("contiguous").to_vec(),
            values,
            "{} in {order:?}",
            T::TYPE
        );
    }
}

/// The `DT_*` codes of `nifti1.h`; the integer probes sit beyond `f32`'s
/// 24-bit and `f64`'s 53-bit exact-integer ranges, where a detour would round.
#[test]
fn every_sample_type_round_trips_in_its_own_type() {
    round_trips::<u8>([0, 1, 7, 127, 128, 200, 254, 255], 2, 8);
    round_trips::<i8>([-128, -64, -7, -1, 0, 1, 64, 127], 256, 8);
    round_trips::<u16>([0, 1, 7, 256, 300, 1024, 4095, 65535], 512, 16);
    round_trips::<i16>([i16::MIN, -1024, -1, 0, 1, 7, 2047, i16::MAX], 4, 16);
    round_trips::<u32>(
        [
            0,
            1,
            7,
            1 << 24,
            (1 << 24) + 1,
            1 << 31,
            u32::MAX - 1,
            u32::MAX,
        ],
        768,
        32,
    );
    round_trips::<i32>(
        [
            i32::MIN,
            -(1 << 24) - 1,
            -1,
            0,
            1,
            7,
            (1 << 24) + 1,
            i32::MAX,
        ],
        8,
        32,
    );
    round_trips::<u64>(
        [
            0,
            1,
            7,
            1 << 53,
            (1 << 53) + 1,
            1 << 63,
            u64::MAX - 1,
            u64::MAX,
        ],
        1280,
        64,
    );
    round_trips::<i64>(
        [
            i64::MIN,
            -(1 << 53) - 1,
            -1,
            0,
            1,
            7,
            (1 << 53) + 1,
            i64::MAX,
        ],
        1024,
        64,
    );
    round_trips::<f32>(
        [
            -f32::MAX,
            -0.5,
            -0.0,
            0.0,
            f32::MIN_POSITIVE,
            0.1,
            3.25,
            f32::MAX,
        ],
        16,
        32,
    );
    round_trips::<f64>(
        [
            -f64::MAX,
            -2.5,
            0.0,
            0.1,
            1.0,
            1.0 + f64::EPSILON,
            1e300,
            f64::MAX,
        ],
        64,
        64,
    );
}

#[test]
fn a_series_round_trips_in_its_own_type() {
    let dir = tempdir().expect("temporary directory");
    let path = dir.path().join("series.nii");
    let backend = SequentialBackend;
    let volumes = [
        image_of(vec![-1024_i16, -1, 0, 1, 2, 3, 4, i16::MAX]),
        image_of(vec![i16::MIN, 5, 6, 7, 8, 9, 10, 11]),
    ];
    crate::write_nifti_series(&path, &volumes, &backend).expect("series write");

    let loaded =
        crate::read_nifti_series::<i16, _, _, _>(&path, &backend, Exact).expect("series read");
    assert_eq!(loaded.len(), 2);
    for (got, want) in loaded.iter().zip(&volumes) {
        assert_eq!(
            got.data_slice().expect("contiguous"),
            want.data_slice().expect("contiguous")
        );
    }
}

#[test]
fn big_endian_samples_convert_to_a_wider_type() {
    let backend = SequentialBackend;
    let values = [
        i32::MIN,
        -(1 << 24) - 1,
        -1,
        0,
        1,
        7,
        (1 << 24) + 1,
        i32::MAX,
    ];
    let raw = RawNifti1::new(ByteOrder::BigEndian, 8, 32, [2, 2, 2]);
    let bytes = raw.file(&encode(&values, ByteOrder::BigEndian));

    let exact =
        crate::read_nifti_from_bytes::<i32, _, _>(&bytes, &backend, Exact).expect("i32 read");
    assert_eq!(exact.data_slice().expect("contiguous"), values);

    let wide =
        crate::read_nifti_from_bytes::<f64, _, _>(&bytes, &backend, Exact).expect("f64 read");
    assert_eq!(
        wide.data_slice().expect("contiguous"),
        values.map(f64::from)
    );
}

/// A CT stored as `int16` with `y = 2x - 1024`, the Hounsfield mapping shape.
fn rescaled_ct() -> (Vec<u8>, [i16; 8]) {
    let stored = [-1, 0, 1, 2, 512, 1000, 2047, i16::MAX];
    let raw = RawNifti1::new(ByteOrder::LittleEndian, 4, 16, [2, 2, 2]).with_rescale(2.0, -1024.0);
    (raw.file(&encode(&stored, ByteOrder::LittleEndian)), stored)
}

#[test]
fn rescale_applies_in_float_targets() {
    let backend = SequentialBackend;
    let (bytes, stored) = rescaled_ct();

    let single =
        crate::read_nifti_from_bytes::<f32, _, _>(&bytes, &backend, Exact).expect("f32 read");
    assert_eq!(
        single.data_slice().expect("contiguous"),
        [-1026.0, -1024.0, -1022.0, -1020.0, 0.0, 976.0, 3070.0, 64510.0]
    );

    let double =
        crate::read_nifti_from_bytes::<f64, _, _>(&bytes, &backend, Exact).expect("f64 read");
    assert_eq!(
        double.data_slice().expect("contiguous"),
        stored.map(|value| f64::from(value) * 2.0 - 1024.0)
    );

    let series = crate::read_nifti_series_from_bytes::<f32, _, _>(&bytes, &backend, Exact)
        .expect("series read");
    assert_eq!(
        series[0].data_slice().expect("contiguous"),
        single.data_slice().expect("contiguous")
    );
}

#[test]
fn rescale_into_an_integer_target_is_rejected() {
    let (bytes, _) = rescaled_ct();
    let err = crate::read_nifti_from_bytes::<i16, _, _>(&bytes, &SequentialBackend, Exact)
        .expect_err("a rescale has no faithful i16 result");
    let message = format!("{err:#}");
    assert!(message.contains("read_nifti_stored"), "{message}");
    assert!(message.contains("scl_slope 2"), "{message}");
}

#[test]
fn stored_read_returns_the_samples_and_the_rescale() {
    let dir = tempdir().expect("temporary directory");
    let path = dir.path().join("ct.nii");
    let (bytes, stored) = rescaled_ct();
    std::fs::write(&path, &bytes).expect("fixture write");

    let (image, rescale) =
        crate::read_nifti_stored::<i16, _, _, _>(&path, &SequentialBackend, Exact)
            .expect("stored read");
    assert_eq!(image.data_slice().expect("contiguous"), stored);
    assert_eq!(rescale, Rescale::new(2.0, -1024.0).expect("finite"));
}

/// `nifti1.h`: only a nonzero slope declares a rescale, whatever the intercept.
#[test]
fn zero_slope_declares_no_rescale() {
    let values = [3_i16, -4, 5, -6, 7, -8, 9, -10];
    let raw = RawNifti1::new(ByteOrder::LittleEndian, 4, 16, [2, 2, 2]).with_rescale(0.0, 100.0);
    let bytes = raw.file(&encode(&values, ByteOrder::LittleEndian));

    let image = crate::read_nifti_from_bytes::<i16, _, _>(&bytes, &SequentialBackend, Exact)
        .expect("no rescale applies");
    assert_eq!(image.data_slice().expect("contiguous"), values);
}

#[test]
fn a_valid_slope_with_a_non_finite_intercept_is_rejected() {
    let raw = RawNifti1::new(ByteOrder::LittleEndian, 2, 8, [2, 2, 2]).with_rescale(1.0, f32::NAN);
    let err =
        crate::read_nifti_from_bytes::<f32, _, _>(&raw.file(&[0; 8]), &SequentialBackend, Exact)
            .expect_err("NaN intercept beside a valid slope");
    assert!(format!("{err:#}").contains("scl_inter"), "{err:#}");
}

/// The NIfTI-2 rescale is two `f64` fields at 176 and 184; the NIfTI-1 pair is
/// two `f32` fields at 112 and 116.
#[test]
fn header_rescale_fields_sit_at_their_offsets() {
    let spatial = HeaderSpatial {
        pixdim: [1.0; 8],
        srow_x: [1.0, 0.0, 0.0, 0.0],
        srow_y: [0.0, 1.0, 0.0, 0.0],
        srow_z: [0.0, 0.0, 1.0, 0.0],
    };
    let dims = HeaderDims {
        nx: 2,
        ny: 2,
        nz: 2,
    };
    let rescale = Rescale::new(0.1, -0.25).expect("finite");

    let mut nifti2 =
        NiftiHeader::new_with_version(HeaderVersion::Two, dims, 1, SampleType::U16, spatial)
            .expect("valid header");
    nifti2.rescale = rescale;
    let encoded = nifti2.encode();
    assert_eq!(&encoded[176..184], 0.1_f64.to_le_bytes());
    assert_eq!(&encoded[184..192], (-0.25_f64).to_le_bytes());
    assert_eq!(
        NiftiHeader::parse(&encoded).expect("parses").rescale,
        rescale
    );

    let mut nifti1 = NiftiHeader::new_volume(dims, SampleType::U16, spatial).expect("valid header");
    nifti1.rescale = Rescale::new(0.5, -0.25).expect("finite");
    let encoded = nifti1.encode();
    assert_eq!(&encoded[112..116], 0.5_f32.to_le_bytes());
    assert_eq!(&encoded[116..120], (-0.25_f32).to_le_bytes());
}

#[test]
fn unsupported_datatype_codes_are_rejected() {
    // DT_COMPLEX64 = 32, 64 bits per voxel.
    let raw = RawNifti1::new(ByteOrder::LittleEndian, 32, 64, [2, 2, 2]);
    let err =
        crate::read_nifti_from_bytes::<f32, _, _>(&raw.file(&[0; 64]), &SequentialBackend, Exact)
            .expect_err("complex samples are outside the sample types");
    assert!(format!("{err:#}").contains("datatype code 32"), "{err:#}");
}

#[test]
fn a_bitpix_disagreeing_with_the_datatype_is_rejected() {
    let raw = RawNifti1::new(ByteOrder::LittleEndian, 512, 8, [2, 2, 2]);
    let err =
        crate::read_nifti_from_bytes::<f32, _, _>(&raw.file(&[0; 16]), &SequentialBackend, Exact)
            .expect_err("uint16 is 16 bits");
    assert!(format!("{err:#}").contains("bitpix 8"), "{err:#}");
}
