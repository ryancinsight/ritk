//! Sidebar state and optional eframe panel.

/// Active tab in the viewer sidebar.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum SidebarTab {
    /// Shows the series browser.
    #[default]
    Series,
    /// Shows DICOM metadata.
    Metadata,
    /// Shows PET SUV measurements.
    PetSuv,
}

#[cfg(feature = "eframe-shell")]
mod panel;
#[cfg(test)]
mod tests;

#[cfg(feature = "eframe-shell")]
pub use panel::SidebarPanel;
