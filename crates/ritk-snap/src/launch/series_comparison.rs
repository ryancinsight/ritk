use super::{AppLaunchOptions, NativePresentationSelection};

/// Launch the Métis native viewer with two selected DICOM series side by side.
///
/// `options.initial_series_uid` selects panel 1 and `comparison_series_uid`
/// selects panel 2. Both must refer to distinct series in the startup study.
///
/// # Errors
/// Returns a configuration, series-load, host, or capture error.
pub fn run_app_with_series_comparison(
    options: AppLaunchOptions,
    comparison_series_uid: String,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        options.initial_series_uid.is_some(),
        "series comparison requires --series-instance-uid for panel 1"
    );
    anyhow::ensure!(
        !comparison_series_uid.trim().is_empty(),
        "comparison SeriesInstanceUID must not be empty"
    );
    let presentation_mode = options.native_presentation_mode;
    super::run_native_selection(
        options,
        NativePresentationSelection::Fixed(presentation_mode),
        Some(comparison_series_uid),
    )
}
