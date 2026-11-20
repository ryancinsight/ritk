//! Enhanced PET reconstruction selection in the viewer.

use super::state::SnapApp;

impl SnapApp {
    pub(crate) fn enhanced_pet_selection(&self) -> Option<&ritk_io::EnhancedPetSelection> {
        self.loaded
            .as_ref()?
            .metadata
            .as_deref()?
            .enhanced_pet_selection
            .as_ref()
    }

    pub(crate) fn select_enhanced_pet_phase(&mut self, index: usize) -> bool {
        let Some(current) = self.loaded.as_ref() else {
            return false;
        };
        let mut volume = current.clone();
        if volume.select_enhanced_pet_volume(index).is_none() {
            return false;
        }
        let status = format!("PET volume {} selected", index + 1);
        if volume.shape == current.shape
            && volume.spacing == current.spacing
            && volume.origin == current.origin
            && volume.direction == current.direction
        {
            self.loaded = Some(volume);
            self.pointer_suv = None;
            self.pointer_intensity = 0.0;
            #[cfg(all(not(target_arch = "wasm32"), feature = "eframe-shell"))]
            {
                self.cached_histogram = None;
            }
            self.status_message = status;
            self.bump_visual_revision();
            #[cfg(all(not(target_arch = "wasm32"), feature = "eframe-shell"))]
            self.refresh_cached_histogram();
        } else {
            self.load_volume(volume, status);
        }
        true
    }

    #[cfg(all(not(target_arch = "wasm32"), feature = "eframe-shell"))]
    pub(crate) fn show_enhanced_pet_selector(&mut self, ui: &mut egui::Ui) {
        let Some(selection) = self.enhanced_pet_selection() else {
            return;
        };
        if selection.volumes().len() < 2 {
            return;
        }
        let active = selection.selected_index();
        let mut next = active;
        egui::ComboBox::from_id_source("enhanced_pet_volume")
            .selected_text(format!(
                "PET volume {} of {}",
                active + 1,
                selection.volumes().len()
            ))
            .show_ui(ui, |ui| {
                for (index, volume) in selection.volumes().iter().enumerate() {
                    ui.selectable_value(
                        &mut next,
                        index,
                        pet_volume_label(index, &volume.dimensions),
                    );
                }
            });
        if next != active {
            self.select_enhanced_pet_phase(next);
        }
    }
}

#[cfg(all(not(target_arch = "wasm32"), feature = "eframe-shell"))]
fn pet_volume_label(index: usize, dimensions: &[ritk_io::EnhancedPetDimension]) -> String {
    use ritk_io::EnhancedPetCoordinate::{
        CyclePhase, Label, RespiratoryAmplitude, TemporalPosition, TriggerDelay,
    };
    let coordinates: Vec<_> = dimensions
        .iter()
        .filter_map(|dimension| match dimension.coordinate.as_ref()? {
            TemporalPosition(value) => Some(format!("Time point {value}")),
            CyclePhase(value) => Some(format!("Phase {:.0}%", value.into_base() * 100.0)),
            RespiratoryAmplitude(value) => {
                Some(format!("Amplitude {:.0}%", value.into_base() * 100.0))
            }
            TriggerDelay(value) => Some(format!(
                "Trigger {:.1} ms",
                value.in_unit::<aequitas::systems::si::units::Millisecond>()
            )),
            Label(_) => None,
        })
        .collect();
    if coordinates.is_empty() {
        format!("Volume {}", index + 1)
    } else {
        format!("{}  {}", index + 1, coordinates.join(" / "))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ritk_io::{DicomReadMetadata, EnhancedPetSelection, EnhancedPetVolume};
    use ritk_spatial::{Direction, Point, Spacing};
    use std::sync::Arc;
    use tracing_subscriber::fmt::format::FmtSpan;

    #[test]
    fn enhanced_pet_phase_switch_changes_voxels_and_preserves_compatible_cursor() {
        let _trace = tracing::subscriber::set_default(
            tracing_subscriber::fmt()
                .with_span_events(FmtSpan::FULL)
                .with_test_writer()
                .finish(),
        );
        let (mut app, phase_data) = tracing::info_span!("arrange").in_scope(|| {
            let mut app = SnapApp::default();
            let mut loaded = super::super::tests::test_volume([2, 1, 1]);
            let phase_data = Arc::new(vec![2.5, 7.5]);
            let phase = |data, slice_provenance| EnhancedPetVolume {
                dimensions: Vec::new(),
                data,
                suv_bw: None,
                slice_provenance,
                shape: [2, 1, 1],
                origin: Point::new(loaded.origin),
                spacing: Spacing::new(loaded.spacing),
                direction: Direction::identity(),
            };
            loaded.metadata = Some(Box::new(DicomReadMetadata {
                enhanced_pet_selection: EnhancedPetSelection::new(vec![
                    phase(Arc::clone(&loaded.data), None),
                    phase(
                        Arc::clone(&phase_data),
                        Some(vec![
                            ritk_io::SliceProvenance::Acquired,
                            ritk_io::SliceProvenance::Interpolated,
                        ]),
                    ),
                ]),
                ..DicomReadMetadata::default()
            }));
            app.load_volume(loaded, String::new());
            (app, phase_data)
        });
        assert_eq!(
            app.loaded
                .as_ref()
                .expect("the study remains loaded")
                .enhanced_pet_slice_label(0, 1),
            None
        );
        let original_revision = app.visual_revision;
        let cursor = app.viewer_state.slice_index;
        let selected = tracing::info_span!("act").in_scope(|| app.select_enhanced_pet_phase(1));
        let _assert = tracing::info_span!("assert").entered();
        assert!(selected);
        let volume = app.loaded.as_ref().expect("the study remains loaded");
        assert!(Arc::ptr_eq(&volume.data, &phase_data));
        assert_eq!(volume.pixel_at(1, 0, 0), 7.5);
        assert_eq!(
            volume.enhanced_pet_slice_label(0, 1),
            Some("Interpolated PET slice")
        );
        assert_eq!(volume.enhanced_pet_slice_label(0, 0), None);
        assert_eq!(volume.enhanced_pet_slice_label(1, 1), None);
        assert_eq!(
            app.enhanced_pet_selection()
                .expect("retain the PET phase selection")
                .selected_index(),
            1
        );
        assert_eq!(app.viewer_state.slice_index, cursor);
        assert!(app.visual_revision > original_revision);
        assert!(!app.select_enhanced_pet_phase(2));
        assert_eq!(
            app.enhanced_pet_selection()
                .expect("retain the PET phase selection")
                .selected_index(),
            1
        );
    }
}
