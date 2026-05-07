//! RT Dose analytics and DVH rendering helpers.
//!
//! Computes ROI-linked dose statistics by rasterizing RT-STRUCT closed planar
//! contours onto the loaded volume grid and sampling the projected RT-DOSE map.

use std::collections::BTreeMap;

use egui::{Color32, Pos2, Rect, Stroke, Ui, Vec2};
use ritk_io::{RtDoseGrid, RtStructureSet};

use crate::ui::rtdose_overlay::extract_dose_slice_for_volume;

#[derive(Debug, Clone, PartialEq)]
pub struct DvhPoint {
    pub dose_gy: f32,
    pub volume_fraction_ge: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RoiDoseAnalytics {
    pub roi_number: u32,
    pub roi_name: String,
    pub voxel_count: usize,
    pub min_dose_gy: f32,
    pub mean_dose_gy: f32,
    pub max_dose_gy: f32,
    pub d95_gy: f32,
    pub curve: Vec<DvhPoint>,
}

pub fn compute_roi_dose_analytics(
    rt_struct: &RtStructureSet,
    rt_dose: &RtDoseGrid,
    roi_number: u32,
    vol_shape: [usize; 3],
    vol_origin: [f64; 3],
    vol_direction: [f64; 9],
    vol_spacing: [f64; 3],
    bins: usize,
) -> Option<RoiDoseAnalytics> {
    let roi = rt_struct.rois.iter().find(|r| r.roi_number == roi_number)?;
    let inv_phys_to_voxel = inverse_phys_to_voxel(vol_direction, vol_spacing)?;

    let [depth, rows, cols] = vol_shape;
    if depth == 0 || rows == 0 || cols == 0 || bins == 0 {
        return None;
    }

    let mut polygons_by_slice: BTreeMap<usize, Vec<Vec<[f32; 2]>>> = BTreeMap::new();
    for contour in &roi.contours {
        if contour.geometric_type.trim() != "CLOSED_PLANAR" || contour.points.len() < 3 {
            continue;
        }

        let voxels: Vec<[f64; 3]> = contour
            .points
            .iter()
            .map(|p| patient_to_voxel(*p, vol_origin, inv_phys_to_voxel))
            .collect();

        if voxels.is_empty() {
            continue;
        }

        let z_mean = voxels.iter().map(|v| v[0]).sum::<f64>() / voxels.len() as f64;
        let z_dev = voxels
            .iter()
            .map(|v| (v[0] - z_mean).abs())
            .fold(0.0_f64, f64::max);
        if z_dev > 0.75 {
            continue;
        }

        let z_idx_i64 = z_mean.round() as i64;
        if z_idx_i64 < 0 || z_idx_i64 >= depth as i64 {
            continue;
        }
        let z_idx = z_idx_i64 as usize;

        let polygon: Vec<[f32; 2]> = voxels
            .into_iter()
            .map(|v| [v[1] as f32, v[2] as f32])
            .collect();
        polygons_by_slice.entry(z_idx).or_default().push(polygon);
    }

    if polygons_by_slice.is_empty() {
        return None;
    }

    let mut dose_samples: Vec<f32> = Vec::new();
    for (z_idx, polygons) in polygons_by_slice {
        let dose_map = extract_dose_slice_for_volume(
            rt_dose,
            0,
            z_idx,
            vol_shape,
            vol_origin,
            vol_direction,
            vol_spacing,
        )?;

        for r in 0..rows {
            for c in 0..cols {
                let row = r as f32 + 0.5;
                let col = c as f32 + 0.5;
                let inside = polygons
                    .iter()
                    .any(|poly| point_in_polygon(row, col, poly));
                if !inside {
                    continue;
                }
                let dose = dose_map[r * cols + c];
                if dose.is_finite() && dose >= 0.0 {
                    dose_samples.push(dose);
                }
            }
        }
    }

    if dose_samples.is_empty() {
        return None;
    }

    dose_samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let n = dose_samples.len();
    let min_dose = dose_samples[0];
    let max_dose = *dose_samples.last().unwrap_or(&min_dose);
    let mean_dose = dose_samples.iter().copied().sum::<f32>() / n as f32;
    let d95_idx = (((n - 1) as f32) * 0.05).floor() as usize;
    let d95 = dose_samples[d95_idx.min(n - 1)];

    let mut curve: Vec<DvhPoint> = Vec::with_capacity(bins + 1);
    for i in 0..=bins {
        let t = i as f32 / bins as f32;
        let threshold = max_dose * t;
        let first_ge = dose_samples.partition_point(|v| *v < threshold);
        let ge_count = n.saturating_sub(first_ge);
        curve.push(DvhPoint {
            dose_gy: threshold,
            volume_fraction_ge: ge_count as f32 / n as f32,
        });
    }

    Some(RoiDoseAnalytics {
        roi_number,
        roi_name: roi.roi_name.clone(),
        voxel_count: n,
        min_dose_gy: min_dose,
        mean_dose_gy: mean_dose,
        max_dose_gy: max_dose,
        d95_gy: d95,
        curve,
    })
}

pub fn draw_dvh_curve(ui: &mut Ui, points: &[DvhPoint]) {
    let desired = Vec2::new(260.0, 120.0);
    let (rect, _) = ui.allocate_exact_size(desired, egui::Sense::hover());
    let painter = ui.painter_at(rect);

    painter.rect_stroke(rect, 0.0, Stroke::new(1.0, Color32::GRAY));
    if points.len() < 2 {
        return;
    }

    let max_dose = points
        .iter()
        .map(|p| p.dose_gy)
        .fold(0.0_f32, f32::max)
        .max(1e-6);

    let to_screen = |dose: f32, frac: f32, area: Rect| -> Pos2 {
        let x = area.left() + (dose / max_dose).clamp(0.0, 1.0) * area.width();
        let y = area.bottom() - frac.clamp(0.0, 1.0) * area.height();
        Pos2::new(x, y)
    };

    for pair in points.windows(2) {
        let a = &pair[0];
        let b = &pair[1];
        painter.line_segment(
            [
                to_screen(a.dose_gy, a.volume_fraction_ge, rect),
                to_screen(b.dose_gy, b.volume_fraction_ge, rect),
            ],
            Stroke::new(1.5, Color32::from_rgb(255, 128, 0)),
        );
    }
}

fn point_in_polygon(row: f32, col: f32, polygon_rc: &[[f32; 2]]) -> bool {
    let n = polygon_rc.len();
    if n < 3 {
        return false;
    }

    let mut inside = false;
    let mut j = n - 1;
    for i in 0..n {
        let yi = polygon_rc[i][0];
        let xi = polygon_rc[i][1];
        let yj = polygon_rc[j][0];
        let xj = polygon_rc[j][1];

        let intersects = ((yi > row) != (yj > row))
            && (col < (xj - xi) * (row - yi) / ((yj - yi).abs().max(1e-12)) + xi);
        if intersects {
            inside = !inside;
        }
        j = i;
    }
    inside
}

fn patient_to_voxel(point_mm: [f64; 3], origin: [f64; 3], inv_phys_to_voxel: [f64; 9]) -> [f64; 3] {
    let d = [
        point_mm[0] - origin[0],
        point_mm[1] - origin[1],
        point_mm[2] - origin[2],
    ];
    [
        inv_phys_to_voxel[0] * d[0] + inv_phys_to_voxel[1] * d[1] + inv_phys_to_voxel[2] * d[2],
        inv_phys_to_voxel[3] * d[0] + inv_phys_to_voxel[4] * d[1] + inv_phys_to_voxel[5] * d[2],
        inv_phys_to_voxel[6] * d[0] + inv_phys_to_voxel[7] * d[1] + inv_phys_to_voxel[8] * d[2],
    ]
}

fn inverse_phys_to_voxel(direction: [f64; 9], spacing: [f64; 3]) -> Option<[f64; 9]> {
    let m = [
        direction[0] * spacing[0],
        direction[1] * spacing[1],
        direction[2] * spacing[2],
        direction[3] * spacing[0],
        direction[4] * spacing[1],
        direction[5] * spacing[2],
        direction[6] * spacing[0],
        direction[7] * spacing[1],
        direction[8] * spacing[2],
    ];
    invert_3x3(m)
}

fn invert_3x3(m: [f64; 9]) -> Option<[f64; 9]> {
    let det = m[0] * (m[4] * m[8] - m[5] * m[7])
        - m[1] * (m[3] * m[8] - m[5] * m[6])
        + m[2] * (m[3] * m[7] - m[4] * m[6]);
    if det.abs() < 1e-12 {
        return None;
    }

    let inv_det = 1.0 / det;
    Some([
        (m[4] * m[8] - m[5] * m[7]) * inv_det,
        (m[2] * m[7] - m[1] * m[8]) * inv_det,
        (m[1] * m[5] - m[2] * m[4]) * inv_det,
        (m[5] * m[6] - m[3] * m[8]) * inv_det,
        (m[0] * m[8] - m[2] * m[6]) * inv_det,
        (m[2] * m[3] - m[0] * m[5]) * inv_det,
        (m[3] * m[7] - m[4] * m[6]) * inv_det,
        (m[1] * m[6] - m[0] * m[7]) * inv_det,
        (m[0] * m[4] - m[1] * m[3]) * inv_det,
    ])
}

#[cfg(test)]
mod tests {
    use super::*;
    use ritk_io::{RtContour, RtDoseGrid, RtRoiInfo, RtStructureSet};

    fn square_roi() -> RtStructureSet {
        RtStructureSet {
            structure_set_label: "RT".to_owned(),
            structure_set_name: None,
            rois: vec![RtRoiInfo {
                roi_number: 1,
                roi_name: "PTV".to_owned(),
                roi_description: None,
                roi_interpreted_type: Some("PTV".to_owned()),
                display_color: Some([255, 0, 0]),
                contours: vec![RtContour {
                    geometric_type: "CLOSED_PLANAR".to_owned(),
                    points: vec![
                        [0.0, 0.0, 0.0],
                        [0.0, 2.0, 0.0],
                        [0.0, 2.0, 2.0],
                        [0.0, 0.0, 2.0],
                    ],
                }],
            }],
        }
    }

    fn uniform_dose_3x3() -> RtDoseGrid {
        RtDoseGrid {
            rows: 3,
            cols: 3,
            n_frames: 1,
            dose_type: "PHYSICAL".to_owned(),
            dose_summation_type: "PLAN".to_owned(),
            dose_grid_scaling: 1.0,
            frame_offsets: vec![0.0],
            dose_gy: vec![2.0; 9],
            image_position: Some([0.0, 0.0, 0.0]),
            image_orientation: Some([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
            pixel_spacing: Some([1.0, 1.0]),
            referenced_rt_plan_sop_instance_uid: None,
        }
    }

    #[test]
    fn point_in_polygon_square() {
        let poly = vec![[0.0, 0.0], [0.0, 3.0], [3.0, 3.0], [3.0, 0.0]];
        assert!(point_in_polygon(1.5, 1.5, &poly));
        assert!(!point_in_polygon(4.0, 4.0, &poly));
    }

    #[test]
    fn compute_roi_dvh_uniform_dose_is_step_like() {
        let analytics = compute_roi_dose_analytics(
            &square_roi(),
            &uniform_dose_3x3(),
            1,
            [1, 3, 3],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            10,
        )
        .expect("analytics");

        assert!(analytics.voxel_count > 0);
        assert!((analytics.mean_dose_gy - 2.0).abs() < 1e-6);
        assert!((analytics.max_dose_gy - 2.0).abs() < 1e-6);
        assert!((analytics.min_dose_gy - 2.0).abs() < 1e-6);
        assert_eq!(analytics.curve.first().map(|p| p.volume_fraction_ge), Some(1.0));
        assert_eq!(analytics.curve.last().map(|p| p.volume_fraction_ge), Some(1.0));
    }

    #[test]
    fn compute_roi_dvh_missing_roi_returns_none() {
        let result = compute_roi_dose_analytics(
            &square_roi(),
            &uniform_dose_3x3(),
            99,
            [1, 3, 3],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            10,
        );
        assert!(result.is_none());
    }
}
