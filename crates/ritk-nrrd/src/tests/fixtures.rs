pub(super) fn sample_value(index: usize) -> f32 {
    f32::from(u8::try_from(index).expect("invariant: test sample index fits in u8"))
}
