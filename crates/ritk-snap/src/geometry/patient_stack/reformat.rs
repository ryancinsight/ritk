//! Patient geometry for the index permutations used by LoadedVolume::extract_slice.
use super::{PatientStack, Vector3};
use crate::loaded_volume::LoadedVolume;

impl PatientStack {
    pub(crate) fn displayed_planes(volume: &LoadedVolume) -> [Option<Self>; 3] {
        let acquisition = Self::from_volume(volume);
        let reformats = acquisition
            .as_ref()
            .and_then(|stack| stack.reformats(volume));
        let [row, column] = reformats.unwrap_or([None, None]);
        [acquisition, row, column]
    }

    fn reformats(&self, volume: &LoadedVolume) -> Option<[Option<Self>; 2]> {
        // Use the reader's measured uniform-stack and pixel-spacing admission.
        // A single image or an irregular stack has no qualified affine depth axis.
        let [dy, dx] = volume.measurement_pixel_spacing()?;
        let pitch = volume.measurement_slice_spacing()?;
        let [depth, rows, cols] = volume.shape;
        let extent = |n: usize| Some(f64::from(u32::try_from(n.checked_sub(1)?).ok()?));
        let first = self.centers[0];
        let last = *self.centers.last()?;
        let step = (last - first) / extent(depth)?;
        // Bound accumulated departure from the endpoint affine as well as local
        // step variation. Use the reader's relative spacing budget in millimetres.
        let limit =
            crate::loaded_volume::READER_NONUNIFORM_SPACING_THRESHOLD * pitch.min(dx).min(dy);
        for (index, center) in self.centers.iter().enumerate() {
            let residual = *center - (first + step * f64::from(u32::try_from(index).ok()?));
            if residual.norm() > limit {
                return None;
            }
        }
        let u = self.column * dx;
        let v = self.row * dy;
        let origin = first - u * (extent(cols)? / 2.) - v * (extent(rows)? / 2.);
        // Every rendered pixel (x,y) in slice i has patient position
        // origin + horizontal*x + step*y + fixed*i. The centers use pixel
        // indices, whereas clipping uses the surrounding half-pixel boundaries.
        let plane = |horizontal: Vector3, fixed: Vector3, width, count| {
            let mut normal = horizontal.cross(&step).normalized()?;
            if normal.dot(&fixed) < 0. {
                normal = -normal;
            }
            let separation = normal.dot(&fixed);
            if !separation.is_finite() || separation <= 0. {
                return None;
            }
            let center = origin + horizontal * (extent(width)? / 2.) + step * (extent(depth)? / 2.);
            let distances: Vec<_> = (0..u32::try_from(count).ok()?)
                .map(|i| separation * f64::from(i))
                .collect();
            let centers = (0..u32::try_from(count).ok()?)
                .map(|i| center + fixed * f64::from(i))
                .collect();
            Some(Self {
                study: self.study.clone(),
                frame: self.frame.clone(),
                normal,
                column: horizontal.normalized()?,
                row: step.normalized()?,
                half_extent: [(extent(width)? + 1.) / 2., (extent(depth)? + 1.) / 2.],
                half_thickness: vec![None; count],
                pixel_spacing: vec![[step.norm(), horizontal.norm()]; count],
                centers,
                distances,
                anchor: None,
            })
        };
        Some([plane(u, v, cols, rows), plane(v, u, rows, cols)])
    }
}
