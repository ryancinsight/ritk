//! Orient a DICOM reading pane from its patient-space edges, before user controls.
use super::orientation::orientation_labels;
use crate::ui::ViewTransform;

pub(crate) fn reading_orientation(
    axis: usize,
    direction: &[f64; 9],
    user: ViewTransform,
) -> ViewTransform {
    let labels = orientation_labels(axis, direction);
    // Desired (right, top) edges for axial, coronal and sagittal views.
    for (right, top) in [("L", "A"), ("L", "S"), ("P", "S")] {
        if [labels.left, labels.right].contains(&right)
            && [labels.top, labels.bottom].contains(&top)
        {
            return ViewTransform {
                flip_h: (labels.left == right) ^ user.flip_h,
                flip_v: (labels.bottom == top) ^ user.flip_v,
                rotation: user.rotation,
            };
        }
        if [labels.top, labels.bottom].contains(&right)
            && [labels.left, labels.right].contains(&top)
        {
            // A clockwise quarter turn takes the old top to the right. User
            // flips apply in the resulting display axes, so their roles swap.
            return ViewTransform {
                flip_h: (labels.right == top) ^ user.flip_v,
                flip_v: (labels.bottom == right) ^ user.flip_h,
                rotation: user.rotation.rotate_cw(),
            };
        }
    }
    // Both image directions can have the same dominant LPS component in an oblique plane.
    user
}

#[cfg(test)]
mod tests;
