use std::sync::Arc;

use ritk_vtk::VtkImageVolume;

use crate::LoadedVolume;

impl TryFrom<&LoadedVolume> for VtkImageVolume {
    type Error = ritk_vtk::VtkImageVolumeError;

    fn try_from(volume: &LoadedVolume) -> Result<Self, Self::Error> {
        let [depth, rows, columns] = volume.shape;
        let [depth_spacing, row_spacing, column_spacing] = volume.spacing;
        let direction = volume.direction;

        // LoadedVolume uses [depth, row, column] axes; VTK uses [x, y, z]
        // = [column, row, depth]. Reordering columns here preserves the same
        // patient-space basis without teaching the VTK crate clinical terms.
        let vtk_direction = [
            direction[2],
            direction[1],
            direction[0],
            direction[5],
            direction[4],
            direction[3],
            direction[8],
            direction[7],
            direction[6],
        ];
        VtkImageVolume::from_parts(
            [columns, rows, depth],
            volume.origin,
            [column_spacing, row_spacing, depth_spacing],
            vtk_direction,
            usize::from(volume.channels),
            Arc::clone(&volume.data),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn loaded() -> LoadedVolume {
        LoadedVolume {
            data: Arc::new((0..24).map(|value| value as f32).collect()),
            shape: [2, 2, 3],
            channels: 2,
            spacing: [3.0, 2.0, 0.5],
            origin: [10.0, 20.0, 30.0],
            direction: [
                0.0, -1.0, 0.0, // depth axis in physical rows
                1.0, 0.0, 0.0, // row axis in physical rows
                0.0, 0.0, 1.0, // column axis in physical rows
            ],
            metadata: None,
            source: None,
            modality: None,
            patient_name: None,
            patient_id: None,
            study_date: None,
            series_description: None,
            series_time: None,
            patient_weight_kg: None,
            injected_dose_bq: None,
            radionuclide_half_life_s: None,
            radiopharmaceutical_start_time: None,
            decay_correction: None,
        }
    }

    #[test]
    fn conversion_reorders_geometry_without_copying_samples() {
        let source = loaded();
        let source_pointer = source.data.as_ptr();
        let image = VtkImageVolume::try_from(&source).expect("valid loaded volume");

        assert_eq!(image.dimensions(), [3, 2, 2]);
        assert_eq!(image.whole_extent(), [0, 2, 0, 1, 0, 1]);
        assert_eq!(image.spacing(), [0.5, 2.0, 3.0]);
        assert_eq!(
            image.direction(),
            [
                0.0, -1.0, 0.0, // column, row, depth basis
                0.0, 0.0, 1.0, 1.0, 0.0, 0.0,
            ]
        );
        assert_eq!(image.scalars().as_ptr(), source_pointer);
        assert_eq!(image.scalars(), source.data.as_slice());
    }

    #[test]
    fn conversion_rejects_zero_channel_volumes() {
        let mut source = loaded();
        source.channels = 0;
        source.data = Arc::new(Vec::new());
        assert_eq!(
            VtkImageVolume::try_from(&source),
            Err(ritk_vtk::VtkImageVolumeError::ZeroChannels)
        );
    }
}
