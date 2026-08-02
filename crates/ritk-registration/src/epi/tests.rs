//! EPI distortion model contracts.
//!
//! The oracles here are analytical: a zero field must be the identity, warp
//! followed by unwarp must return the original, reversing polarity must reverse
//! the displacement, and signal must be conserved. Each is a property of the
//! model rather than of this implementation, so none of them can be satisfied
//! by an implementation that merely runs.

use leto::Array3;

use super::*;

const SHAPE: [usize; 3] = [3, 8, 4];

fn volume_with<F: Fn(usize, usize, usize) -> f64>(builder: F) -> Array3<f64> {
    let mut values = Vec::with_capacity(SHAPE[0] * SHAPE[1] * SHAPE[2]);
    for z in 0..SHAPE[0] {
        for y in 0..SHAPE[1] {
            for x in 0..SHAPE[2] {
                values.push(builder(z, y, x));
            }
        }
    }
    Array3::from_shape_vec(SHAPE, values).expect("valid shape")
}

/// A volume with a bright band, so displacement is visible by position.
fn banded_image() -> Array3<f64> {
    volume_with(|_, y, _| if (3..=4).contains(&y) { 10.0 } else { 1.0 })
}

fn constant_field(value: f64) -> Array3<f64> {
    volume_with(|_, _, _| value)
}

/// A field varying along the phase-encode axis, so the Jacobian is non-unit.
fn ramp_field(slope: f64) -> Array3<f64> {
    volume_with(|_, y, _| slope * y as f64)
}

/// A field vanishing at both ends of the phase-encode axis.
///
/// The map is then onto the grid: nothing is displaced in from outside the
/// field of view and nothing leaves, which is the condition under which signal
/// conservation is an exact identity rather than an identity over the mapped
/// range.
fn closed_field(amplitude: f64) -> Array3<f64> {
    let last = (SHAPE[1] - 1) as f64;
    volume_with(|_, y, _| amplitude * (std::f64::consts::PI * y as f64 / last).sin())
}

/// A boundary-vanishing field along an arbitrary axis.
fn closed_field_along(axis: PhaseEncodeAxis, amplitude: f64) -> Array3<f64> {
    let last = (SHAPE[axis.index()] - 1) as f64;
    volume_with(|z, y, x| {
        let position = [z, y, x][axis.index()] as f64;
        amplitude * (std::f64::consts::PI * position / last).sin()
    })
}

/// An image linear along the phase-encode axis.
///
/// Linear interpolation reproduces a linear function exactly, so a round trip
/// over this data isolates the geometry and Jacobian bookkeeping from
/// resampling error. A step edge cannot survive two interpolations — that is a
/// property of resampling, not of the model — so it is not the right oracle for
/// an exactness claim.
fn linear_image() -> Array3<f64> {
    volume_with(|_, y, _| 2.0 + y as f64)
}

fn encoding() -> PhaseEncoding {
    PhaseEncoding::new(PhaseEncodeAxis::Row, PhaseEncodePolarity::Positive)
}

fn total_signal(volume: &Array3<f64>) -> f64 {
    volume.as_slice().expect("contiguous").iter().sum()
}

fn assert_volumes_close(actual: &Array3<f64>, expected: &Array3<f64>, tolerance: f64, what: &str) {
    let actual = actual.as_slice().expect("contiguous");
    let expected = expected.as_slice().expect("contiguous");
    for (index, (got, want)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (got - want).abs() < tolerance,
            "{what}: element {index} is {got} but expected {want}"
        );
    }
}

#[test]
fn a_zero_field_is_the_identity() {
    let image = banded_image();
    let field = constant_field(0.0);

    let distorted = distort(&image, &field, encoding()).expect("zero field is valid");
    assert_volumes_close(&distorted, &image, 1e-12, "zero-field distortion");

    let recovered = unwarp(&image, &field, encoding()).expect("zero field is valid");
    assert_volumes_close(&recovered, &image, 1e-12, "zero-field unwarp");
}

#[test]
fn unwarp_inverts_distort() {
    // The round trip is the contract: whatever the forward model does, the
    // inverse must undo it exactly, not approximately.
    let image = linear_image();
    for amplitude in [0.0, 0.4, -0.4, 0.8] {
        let field = closed_field(amplitude);
        let distorted = distort(&image, &field, encoding()).expect("non-folding");
        let recovered = unwarp(&distorted, &field, encoding()).expect("non-folding");

        assert_volumes_close(
            &recovered,
            &image,
            1e-9,
            &format!("round trip at amplitude {amplitude}"),
        );
    }
}

#[test]
fn reversing_polarity_reverses_the_displacement() {
    // The whole basis of reversed-pair field estimation: the two polarities
    // distort oppositely. Distorting with +f at positive polarity must equal
    // distorting with -f at negative polarity.
    let image = banded_image();
    let field = ramp_field(0.2);
    let negated = volume_with(|_, y, _| -0.2 * y as f64);

    let positive = distort(&image, &field, encoding()).expect("non-folding");
    let negative = distort(&image, &negated, encoding().reversed()).expect("non-folding");

    assert_volumes_close(&positive, &negative, 1e-12, "polarity reversal");
}

#[test]
fn a_reversed_pair_displaces_in_opposite_senses() {
    // With one field, the two polarities must move the band to opposite sides —
    // the observable that makes the pair informative.
    let image = banded_image();
    let field = constant_field(1.5);

    let up = distort(&image, &field, encoding()).expect("constant field is non-folding");
    let down =
        distort(&image, &field, encoding().reversed()).expect("constant field is non-folding");

    // Centre of mass along the phase-encode axis, per polarity.
    let centre = |volume: &Array3<f64>| -> f64 {
        let mut weighted = 0.0;
        let mut total = 0.0;
        for y in 0..SHAPE[1] {
            for z in 0..SHAPE[0] {
                for x in 0..SHAPE[2] {
                    let value = volume[[z, y, x]];
                    weighted += value * y as f64;
                    total += value;
                }
            }
        }
        weighted / total
    };

    assert!(
        centre(&up) < centre(&down),
        "opposite polarities must displace the band in opposite senses, got {} and {}",
        centre(&up),
        centre(&down)
    );
}

#[test]
fn a_constant_field_does_not_rescale_intensity() {
    // A pure translation has unit Jacobian everywhere: signal moves but is
    // neither compressed nor stretched, so peak intensity is unchanged.
    let image = banded_image();
    let distorted = distort(&image, &constant_field(1.0), encoding()).expect("non-folding");

    let peak = distorted
        .as_slice()
        .expect("contiguous")
        .iter()
        .fold(f64::MIN, |accumulator, value| accumulator.max(*value));
    assert!(
        (peak - 10.0).abs() < 1e-9,
        "a translation must not change peak intensity, got {peak}"
    );
}

#[test]
fn compression_raises_intensity_and_stretching_lowers_it() {
    // The Jacobian term. Omitting it leaves geometry right and intensities
    // wrong, which is invisible on inspection and biases every later fit.
    let flat = volume_with(|_, _, _| 1.0);

    let compressing = distort(&flat, &ramp_field(-0.25), encoding()).expect("non-folding");
    let stretching = distort(&flat, &ramp_field(0.25), encoding()).expect("non-folding");

    // Interior samples avoid the one-sided derivative at the ends.
    let interior = compressing[[1, 4, 2]];
    assert!(
        interior < 1.0,
        "a negative field slope compresses the map and must lower intensity, got {interior}"
    );
    let interior = stretching[[1, 4, 2]];
    assert!(
        interior > 1.0,
        "a positive field slope stretches the map and must raise intensity, got {interior}"
    );
}

#[test]
fn distortion_is_confined_to_the_phase_encode_axis() {
    // Off-resonance displaces along the readout's slow axis alone. Leakage into
    // another axis would be a real geometric error.
    let image = volume_with(|_, _, x| if x == 1 { 5.0 } else { 0.0 });
    let distorted = distort(&image, &ramp_field(0.3), encoding()).expect("non-folding");

    for z in 0..SHAPE[0] {
        for y in 0..SHAPE[1] {
            for x in 0..SHAPE[2] {
                if x != 1 {
                    assert!(
                        distorted[[z, y, x]].abs() < 1e-12,
                        "column {x} must stay empty when only column 1 has signal"
                    );
                }
            }
        }
    }
}

#[test]
fn each_axis_can_carry_the_phase_encoding() {
    // The axis is a property of the acquisition, so all three must work, not
    // just the conventional row axis.
    let image = linear_image();
    for axis in [
        PhaseEncodeAxis::Depth,
        PhaseEncodeAxis::Row,
        PhaseEncodeAxis::Column,
    ] {
        let encoding = PhaseEncoding::new(axis, PhaseEncodePolarity::Positive);
        // Boundary-vanishing so the map is onto this axis, and non-zero in the
        // interior so the case is not a disguised identity.
        let field = closed_field_along(axis, 0.3);
        let distorted = distort(&image, &field, encoding).expect("non-folding");
        let recovered = unwarp(&distorted, &field, encoding).expect("non-folding");
        assert_volumes_close(
            &recovered,
            &image,
            1e-9,
            &format!("round trip along {axis:?}"),
        );
    }
}

#[test]
fn a_folding_field_is_rejected() {
    // Where the Jacobian reaches zero, distinct true positions map to one
    // observed position and their signal is summed. No unwarping separates
    // that, so clamping and continuing would return a confident wrong answer.
    let image = banded_image();
    let folding = ramp_field(-1.5);

    let error = distort(&image, &folding, encoding()).expect_err("slope -1.5 folds the grid");
    let message = format!("{error}");
    assert!(
        message.contains("folds the voxel grid") && message.contains("Jacobian"),
        "error must name the folding and its Jacobian, got {message}"
    );
}

#[test]
fn unwarp_also_rejects_a_folding_field() {
    let image = banded_image();
    assert!(unwarp(&image, &ramp_field(-1.5), encoding()).is_err());
}

#[test]
fn a_mismatched_field_shape_is_rejected() {
    let image = banded_image();
    let wrong = Array3::from_shape_vec([2, 2, 2], vec![0.0; 8]).expect("valid");

    let error = distort(&image, &wrong, encoding()).expect_err("shapes differ");
    assert!(
        format!("{error}").contains("does not match image shape"),
        "error must name the shape disagreement"
    );
}

#[test]
fn a_non_finite_field_is_rejected() {
    // A failed field estimation can emit NaN; propagating it would silently
    // blank the corrected volume.
    let image = banded_image();
    let mut values = vec![0.0; SHAPE[0] * SHAPE[1] * SHAPE[2]];
    values[5] = f64::NAN;
    let poisoned = Array3::from_shape_vec(SHAPE, values).expect("valid");

    assert!(distort(&image, &poisoned, encoding()).is_err());
}

#[test]
fn interior_signal_is_conserved_under_distortion() {
    // The Jacobian exists to conserve signal: the intensity term is exactly the
    // change of variables in the integral identity.
    //
    // The identity holds over the *mapped* range, so it is only a statement
    // about the grid when the map is onto it. A field with a non-zero boundary
    // value stretches the domain and legitimately changes the grid total — a
    // ramp of slope 0.1 raises it by exactly 10%, which is the Jacobian working
    // rather than failing. `closed_field` vanishes at both ends, so the map is
    // onto and conservation is exact.
    let flat = volume_with(|_, _, _| 2.0);
    let distorted = distort(&flat, &closed_field(0.5), encoding()).expect("non-folding");

    let before = total_signal(&flat);
    let after = total_signal(&distorted);
    let relative = (after - before).abs() / before;
    assert!(
        relative < 1e-3,
        "a boundary-vanishing field must conserve signal, moved by {relative}"
    );
}

#[test]
fn polarity_reversal_is_an_involution() {
    assert_eq!(encoding().reversed().reversed(), encoding());
    assert_eq!(PhaseEncodePolarity::Positive.sign(), 1.0);
    assert_eq!(PhaseEncodePolarity::Negative.sign(), -1.0);
}

#[test]
fn axis_indices_match_the_tensor_order() {
    // RITK tensors are [z, y, x]; a mismatch here would distort the wrong axis
    // while every test that used one axis still passed.
    assert_eq!(PhaseEncodeAxis::Depth.index(), 0);
    assert_eq!(PhaseEncodeAxis::Row.index(), 1);
    assert_eq!(PhaseEncodeAxis::Column.index(), 2);
}
