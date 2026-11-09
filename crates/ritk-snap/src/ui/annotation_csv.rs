//! Annotation table text: deterministic CSV export and row labels.
//!
//! # CSV format
//!
//! ```text
//! index,type,primary_value,unit,extra
//! 0,Length,12.30,mm,
//! 1,Patient Length,5.00,mm,
//! 2,Angle,45.00,deg,
//! 3,ROI Rect,100.00,HU,σ=10.00 area=50.00mm²
//! 4,ROI Ellipse,80.00,HU,σ=5.00 area=25.00mm²
//! 5,HU Point,200.00,HU,
//! ```

use crate::tools::interaction::Annotation;

/// Build a CSV string for `annotations`.
///
/// Schema: `index,type,primary_value,unit,extra`
///
/// | Variant      | primary_value      | unit | extra                          |
/// |--------------|--------------------|------|-------------------------------|
/// | Length       | `length_mm`        | mm   | _(empty)_                     |
/// | Patient Length | three-dimensional distance | mm | _(empty)_                |
/// | Angle        | `angle_deg`        | deg  | _(empty)_                     |
/// | ROI Rect     | `mean`             | HU   | `σ=N area=Nmm²`               |
/// | ROI Ellipse  | `mean`             | HU   | `σ=N area=Nmm²`               |
/// | HU Point     | `value`            | HU   | _(empty)_                     |
///
/// The empty extra field keeps the column count constant at 5 across all
/// variants so spreadsheet applications can parse the file without special
/// handling.
pub(crate) fn csv_for(annotations: &[Annotation]) -> String {
    let mut out = String::from("index,type,primary_value,unit,extra\n");
    for (i, ann) in annotations.iter().enumerate() {
        let row = match ann {
            Annotation::Length { length_mm, .. } => {
                format!("{i},Length,{length_mm:.2},mm,\n")
            }
            Annotation::PatientLength(measurement) => {
                let length_mm = measurement.length_mm();
                format!("{i},Patient Length,{length_mm:.2},mm,\n")
            }
            Annotation::Angle { angle_deg, .. } => {
                format!("{i},Angle,{angle_deg:.2},deg,\n")
            }
            Annotation::RoiRect {
                mean,
                std_dev,
                area_mm2,
                ..
            } => {
                format!("{i},ROI Rect,{mean:.2},HU,σ={std_dev:.2} area={area_mm2:.2}mm²\n")
            }
            Annotation::RoiEllipse {
                mean,
                std_dev,
                area_mm2,
                ..
            } => {
                format!("{i},ROI Ellipse,{mean:.2},HU,σ={std_dev:.2} area={area_mm2:.2}mm²\n")
            }
            Annotation::HuPoint { value, .. } => {
                format!("{i},HU Point,{value:.2},HU,\n")
            }
        };
        out.push_str(&row);
    }
    out
}

/// Format a short human-readable label for one annotation row in the panel.
pub(crate) fn annotation_label(i: usize, ann: &Annotation) -> String {
    match ann {
        Annotation::Length { length_mm, .. } => {
            format!("#{i}  Length: {length_mm:.1} mm")
        }
        Annotation::PatientLength(measurement) => {
            let length_mm = measurement.length_mm();
            format!("#{i}  Patient length: {length_mm:.1} mm")
        }
        Annotation::Angle { angle_deg, .. } => {
            format!("#{i}  Angle: {angle_deg:.2}°")
        }
        Annotation::RoiRect {
            mean,
            std_dev,
            min,
            max,
            area_mm2,
            ..
        } => {
            format!(
                "#{i}  ROI Rect  μ={mean:.1} σ={std_dev:.1} [{min:.0},{max:.0}] {area_mm2:.1}mm²"
            )
        }
        Annotation::RoiEllipse {
            mean,
            std_dev,
            min,
            max,
            area_mm2,
            ..
        } => {
            format!(
                "#{i}  ROI Ellipse  μ={mean:.1} σ={std_dev:.1} [{min:.0},{max:.0}] {area_mm2:.1}mm²"
            )
        }
        Annotation::HuPoint { value, pos } => {
            format!("#{i}  HU ({:.0},{:.0}): {value:.0}", pos[1], pos[0])
        }
    }
}

#[cfg(test)]
mod tests;
