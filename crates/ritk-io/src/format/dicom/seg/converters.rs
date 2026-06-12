use anyhow::{bail, Result};
use ritk_annotation::LabelId;
use std::collections::HashMap;

use crate::format::dicom::reader::types::literal_arraystring;

use super::types::{DicomSegmentInfo, DicomSegmentation};

/// Bit-depth encoding for a DICOM SEG pixel data frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SegEncoding {
    /// 1 bit per pixel (binary: 0 or 1).
    Binary,
    /// 8 bits per pixel (fractional: 0–255).
    Fractional,
}

/// Convert a `ritk_core` `LabelMap` with spatial metadata to a `DicomSegmentation`.
///
/// # Mathematical Specification
///
/// Creates one 2D frame per Z-slice per segment. Given a 3D label map L: Z³ → N
/// with shape [nz, ny, nx], each unique foreground label ID produces nz frames
/// (one per Z-slice), each with geometry rows=ny, cols=nx.
///
/// Total frames = (number of foreground labels) × nz.
/// Frame f for segment s and Z-index z contains pixel 1 where label equals s, else 0.
///
/// # Errors
/// - `label_map.shape` has zero dimension.
/// - No foreground labels exist.
pub fn label_map_to_dicom_seg(
    label_map: &ritk_annotation::LabelMap,
    origin: [f64; 3],
    spacing: [f64; 3],
    direction: [f64; 9],
    encoding: SegEncoding,
) -> Result<DicomSegmentation> {
    if label_map.shape.0[0] == 0 || label_map.shape.0[1] == 0 || label_map.shape.0[2] == 0 {
        bail!("label_map has zero dimension: {:?}", label_map.shape);
    }

    let mut foreground_labels = label_map.present_labels();
    foreground_labels.retain(|&id| id != 0);

    if foreground_labels.is_empty() {
        bail!("no foreground labels found in label_map");
    }

    let nz = label_map.shape.0[0];
    let ny = label_map.shape.0[1];
    let nx = label_map.shape.0[2];
    let rows = ny;
    let cols = nx;
    let n_pixels_per_frame = rows * cols;
    let n_frames = foreground_labels.len() * nz;

    let n_segments = foreground_labels.len();
    let mut segments = Vec::with_capacity(n_segments);
    let mut frame_segment_numbers = Vec::with_capacity(n_frames);
    let mut pixel_data = Vec::with_capacity(n_frames);
    let mut image_position_per_frame = Vec::with_capacity(n_frames);

    for (segment_idx, &label_id) in foreground_labels.iter().enumerate() {
        let segment_number = (segment_idx + 1) as u16;

        let label_name = label_map
            .table
            .get_label(label_id)
            .map(|e| e.name.clone())
            .unwrap_or_else(|| format!("Label {}", label_id));

        segments.push(DicomSegmentInfo {
            segment_number,
            segment_label: label_name,
            segment_description: None,
            algorithm_type: None,
        });

        for z in 0..nz {
            frame_segment_numbers.push(segment_number);

            let mut frame_pixels = Vec::with_capacity(n_pixels_per_frame);
            for y in 0..ny {
                for x in 0..nx {
                    let at_label = label_map.label_at([z, y, x]);
                    frame_pixels.push(if at_label == label_id { 1u8 } else { 0u8 });
                }
            }
            assert_eq!(
                frame_pixels.len(),
                n_pixels_per_frame,
                "frame pixel count mismatch at z={}: {} != {}",
                z,
                frame_pixels.len(),
                n_pixels_per_frame
            );
            pixel_data.push(frame_pixels);

            let dir_z_col = [direction[2], direction[5], direction[8]];
            let z_offset = z as f64 * spacing[0];
            let pos = [
                origin[0] + z_offset * dir_z_col[0],
                origin[1] + z_offset * dir_z_col[1],
                origin[2] + z_offset * dir_z_col[2],
            ];
            image_position_per_frame.push(Some(pos));
        }
    }

    let bits_allocated = match encoding {
        SegEncoding::Binary => 1u16,
        SegEncoding::Fractional => 8u16,
    };
    let segmentation_type = match encoding {
        SegEncoding::Binary => literal_arraystring("BINARY"),
        SegEncoding::Fractional => literal_arraystring("FRACTIONAL"),
    };

    // Convert 3×3 direction matrix to 6-element image orientation (row, then column direction)
    let image_orientation_6: [f64; 6] = [
        direction[0],
        direction[3],
        direction[6],
        direction[1],
        direction[4],
        direction[7],
    ];

    Ok(DicomSegmentation {
        rows,
        cols,
        n_frames,
        bits_allocated,
        segmentation_type,
        segments,
        frame_segment_numbers,
        pixel_data,
        image_position_per_frame,
        image_orientation: Some(image_orientation_6),
        pixel_spacing: Some([spacing[1], spacing[2]]),
        slice_thickness: Some(spacing[0]),
    })
}

/// Convert a [`DicomSegmentation`] into a dense `ritk_core` `LabelMap`.
///
/// # Mathematical specification
///
/// Reconstructs a dense tensor `L[z, y, x]` with shape `[nz, rows, cols]`.
/// Depth `nz` is inferred from unique per-frame physical slice positions (primary)
/// or maximal frame count across segments (sparse fallback).
///
/// Overlap policy: later writes in frame order win.
///
/// # Invariants checked
/// - `rows > 0`, `cols > 0`, `n_frames > 0`, `segments` non-empty.
/// - `frame_segment_numbers.len() == n_frames`.
/// - `pixel_data.len() == n_frames` and each frame length is `rows * cols`.
/// - Every frame must reference a defined segment.
pub fn dicom_seg_to_label_map(seg: &DicomSegmentation) -> Result<ritk_annotation::LabelMap> {
    if seg.rows == 0 || seg.cols == 0 {
        bail!(
            "DICOM-SEG has invalid frame geometry: rows={}, cols={}",
            seg.rows,
            seg.cols
        );
    }
    if seg.n_frames == 0 {
        bail!("DICOM-SEG has zero frames");
    }
    if seg.segments.is_empty() {
        bail!("DICOM-SEG has no segments");
    }
    if seg.frame_segment_numbers.len() != seg.n_frames {
        bail!(
            "frame_segment_numbers length {} != n_frames {}",
            seg.frame_segment_numbers.len(),
            seg.n_frames
        );
    }
    if seg.pixel_data.len() != seg.n_frames {
        bail!(
            "pixel_data frame count {} != n_frames {}",
            seg.pixel_data.len(),
            seg.n_frames
        );
    }

    let n_pixels_per_frame = seg.rows * seg.cols;
    for (idx, frame) in seg.pixel_data.iter().enumerate() {
        if frame.len() != n_pixels_per_frame {
            bail!(
                "pixel_data[{idx}] length {} != rows*cols {}",
                frame.len(),
                n_pixels_per_frame
            );
        }
    }

    let mut table = ritk_annotation::LabelTable::new();
    let mut segment_to_index: HashMap<u16, usize> = HashMap::with_capacity(seg.segments.len());
    let mut labels_by_index = Vec::with_capacity(seg.segments.len());
    for (segment_idx, s) in seg.segments.iter().enumerate() {
        if s.segment_number == 0 {
            bail!("segment_number 0 is invalid for DICOM-SEG");
        }
        let label_id = LabelId::from(u32::from(s.segment_number));
        let color = segment_color(label_id);
        table
            .add_label(label_id, s.segment_label.clone(), color)
            .map_err(|e| {
                anyhow::anyhow!("invalid or duplicate segment label id {}: {}", label_id, e)
            })?;
        if segment_to_index
            .insert(s.segment_number, segment_idx)
            .is_some()
        {
            bail!("duplicate segment_number {} in segments", s.segment_number);
        }
        labels_by_index.push(label_id);
    }

    let mut seen_per_segment = vec![0usize; seg.segments.len()];

    const POSITION_EPS: f64 = 1e-4;

    let dot3 = |a: [f64; 3], b: [f64; 3]| -> f64 { a[0] * b[0] + a[1] * b[1] + a[2] * b[2] };
    let cross3 = |a: [f64; 3], b: [f64; 3]| -> [f64; 3] {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    };
    let normalize3 = |v: [f64; 3]| -> Option<[f64; 3]> {
        let n2 = dot3(v, v);
        if n2 <= 0.0 {
            None
        } else {
            let inv = n2.sqrt().recip();
            Some([v[0] * inv, v[1] * inv, v[2] * inv])
        }
    };

    let all_positions_present = seg.image_position_per_frame.len() == seg.n_frames
        && seg.image_position_per_frame.iter().all(|p| p.is_some());

    let mut frame_to_z: Vec<usize> = vec![0usize; seg.n_frames];
    let nz: usize;

    if all_positions_present {
        let projection_axis = seg.image_orientation.and_then(|iop| {
            let row = [iop[0], iop[1], iop[2]];
            let col = [iop[3], iop[4], iop[5]];
            normalize3(cross3(row, col))
        });

        let mut ordered: Vec<(usize, f64)> = Vec::with_capacity(seg.n_frames);
        if let Some(nhat) = projection_axis {
            for (frame_idx, pos) in seg.image_position_per_frame.iter().enumerate() {
                let p = pos.expect("checked Some above");
                ordered.push((frame_idx, dot3(p, nhat)));
            }
        } else {
            let (mut min_x, mut max_x) = (f64::INFINITY, f64::NEG_INFINITY);
            let (mut min_y, mut max_y) = (f64::INFINITY, f64::NEG_INFINITY);
            let (mut min_z, mut max_z) = (f64::INFINITY, f64::NEG_INFINITY);
            for pos in &seg.image_position_per_frame {
                let p = pos.expect("checked Some above");
                min_x = min_x.min(p[0]);
                max_x = max_x.max(p[0]);
                min_y = min_y.min(p[1]);
                max_y = max_y.max(p[1]);
                min_z = min_z.min(p[2]);
                max_z = max_z.max(p[2]);
            }
            let span_x = max_x - min_x;
            let span_y = max_y - min_y;
            let span_z = max_z - min_z;
            let axis = if span_x >= span_y && span_x >= span_z {
                0usize
            } else if span_y >= span_z {
                1usize
            } else {
                2usize
            };
            for (frame_idx, pos) in seg.image_position_per_frame.iter().enumerate() {
                let p = pos.expect("checked Some above");
                ordered.push((frame_idx, p[axis]));
            }
        }
        ordered.sort_by(|a, b| a.1.total_cmp(&b.1));

        let mut z_bins: Vec<f64> = Vec::with_capacity(seg.n_frames);
        for (frame_idx, scalar) in ordered {
            if z_bins
                .last()
                .map(|last| (scalar - *last).abs() > POSITION_EPS)
                .unwrap_or(true)
            {
                z_bins.push(scalar);
            }
            frame_to_z[frame_idx] = z_bins.len() - 1;
        }
        nz = z_bins.len();
        if nz == 0 {
            bail!("unable to derive non-zero depth from frame positions");
        }
    } else {
        let mut frames_per_segment = vec![0usize; seg.segments.len()];
        for &segment_number in &seg.frame_segment_numbers {
            let Some(&segment_idx) = segment_to_index.get(&segment_number) else {
                bail!(
                    "frame references undefined segment_number {}",
                    segment_number
                );
            };
            frames_per_segment[segment_idx] += 1;
        }
        nz = frames_per_segment.into_iter().max().unwrap_or(0);
        if nz == 0 {
            bail!("unable to derive non-zero depth from frame layout");
        }
    }

    let mut data = vec![0u32; nz * n_pixels_per_frame];
    for (frame_idx, frame) in seg.pixel_data[..seg.n_frames].iter().enumerate() {
        let segment_number = seg.frame_segment_numbers[frame_idx];
        let Some(&segment_idx) = segment_to_index.get(&segment_number) else {
            bail!(
                "frame {} references undefined segment_number {}",
                frame_idx,
                segment_number
            );
        };
        let label_id = labels_by_index[segment_idx];

        let z = if all_positions_present {
            frame_to_z[frame_idx]
        } else {
            if seen_per_segment[segment_idx] >= nz {
                bail!(
                    "segment {} has more than {} inferred slices",
                    segment_number,
                    nz
                );
            }
            let z = seen_per_segment[segment_idx];
            seen_per_segment[segment_idx] += 1;
            z
        };

        for (i, &v) in frame.iter().enumerate() {
            if v != 0 {
                let flat = z * n_pixels_per_frame + i;
                data[flat] = u32::from(label_id);
            }
        }
    }

    ritk_annotation::LabelMap::from_data([nz, seg.rows, seg.cols], data, table)
        .map_err(|e| anyhow::anyhow!("failed to build LabelMap from DICOM-SEG: {e}"))
}

pub(super) fn segment_color(label_id: LabelId) -> ritk_annotation::RgbaBytes {
    let seed = u32::from(label_id).wrapping_mul(0x9E37_79B9);
    let r = 40 + ((seed & 0x7F) as u8);
    let g = 40 + (((seed >> 8) & 0x7F) as u8);
    let b = 40 + (((seed >> 16) & 0x7F) as u8);
    ritk_annotation::RgbaBytes::new(r, g, b, 180)
}
