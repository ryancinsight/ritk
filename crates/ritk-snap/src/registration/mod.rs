//! Rigid registration of scalar viewer volumes in physical coordinates.
mod estimate;
mod metric;
mod resample;
#[cfg(test)]
mod tests;
mod volume;

pub(crate) use resample::run;
pub(crate) use volume::scalar_input;
