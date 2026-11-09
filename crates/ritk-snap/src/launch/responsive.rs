use super::{AppLaunchOptions, NativePresentationSelection};

/// Launch the Métis native host with its responsive pane layout.
///
/// This entrypoint preserves the fixed layout enum while exposing adaptive
/// native presentation to library consumers.
///
/// # Errors
/// Returns the same window, event-loop, DICOM load, and capture errors as
/// [`super::run_app_with_options`].
pub fn run_responsive_native_app_with_options(options: AppLaunchOptions) -> anyhow::Result<()> {
    super::run_native_selection(options, NativePresentationSelection::Responsive, None)
}
