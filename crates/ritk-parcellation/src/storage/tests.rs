use super::label_from_stored;

/// Exact integers must survive, which is the ordinary case: a label volume
/// written and read back without resampling.
#[test]
fn exact_integers_are_themselves() {
    for label in [0_u32, 1, 2, 17, 1000, 65_535] {
        assert_eq!(label_from_stored(label as f32), label);
    }
}

/// The reason this is rounding and not truncation. A label displaced downward
/// by interpolation or by the format's precision must come back as the label it
/// was written as, not the one below it.
#[test]
fn a_value_just_under_its_label_recovers_that_label() {
    assert_eq!(label_from_stored(16.999_998), 17);
    assert_eq!(label_from_stored(0.999_999_9), 1);
}

#[test]
fn a_value_just_over_its_label_recovers_that_label() {
    assert_eq!(label_from_stored(17.000_002), 17);
}

/// Halfway rounds away from zero, matching `f32::round`. The case is stated
/// rather than left implicit because a volume carrying half-integers is not a
/// label volume, and a reader that silently picked either neighbour would hide
/// that.
#[test]
fn a_halfway_value_rounds_up() {
    assert_eq!(label_from_stored(2.5), 3);
    assert_eq!(label_from_stored(0.5), 1);
}

/// Below the halfway point there is no label to recover.
#[test]
fn a_fraction_below_one_half_is_background() {
    assert_eq!(label_from_stored(0.4), 0);
    assert_eq!(label_from_stored(f32::MIN_POSITIVE), 0);
}

/// A negative sample cannot be a label. Casting it directly would wrap it into
/// a large positive one and invent a region that no atlas contains.
#[test]
fn negative_samples_are_background_rather_than_wrapped() {
    assert_eq!(label_from_stored(-1.0), 0);
    assert_eq!(label_from_stored(-17.0), 0);
    assert_eq!(label_from_stored(-0.0), 0);
    assert_eq!(label_from_stored(f32::NEG_INFINITY), 0);
}

/// Every comparison against NaN is false, so the sign test alone would let one
/// through to the cast. It is rejected explicitly.
#[test]
fn not_a_number_is_background() {
    assert_eq!(label_from_stored(f32::NAN), 0);
}

/// Infinity carries no label either, and a finite sample beyond the label range
/// saturates rather than wrapping to a small one. Stating both is what makes
/// the cast's behaviour a decision rather than an accident.
#[test]
fn samples_outside_the_label_range_do_not_wrap() {
    assert_eq!(label_from_stored(f32::INFINITY), 0);
    assert_eq!(label_from_stored(1.0e30), u32::MAX);
}
