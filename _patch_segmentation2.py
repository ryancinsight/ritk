path = r'crates/ritk-python/src/segmentation.rs'
with open(path, 'r', encoding='utf-8') as f:
    text = f.read()

# Step 1: Update ritk_core::segmentation import block
old_import = (
    'use ritk_core::segmentation::{\n'
    '    connected_components as core_connected_components,\n'
    '    connected_threshold as core_connected_threshold, BinaryClosing, BinaryDilation, BinaryErosion,\n'
    '    BinaryOpening, ChanVeseSegmentation, GeodesicActiveContourSegmentation, KMeansSegmentation,\n'
    '    MorphologicalOperation, LaplacianLevelSet, ShapeDetectionSegmentation,\n'
    '    BinaryFillHoles, ConfidenceConnectedFilter, MorphologicalGradient,\n'
    '    NeighborhoodConnectedFilter, Skeletonization,\n'
    '    ThresholdLevelSet, WatershedSegmentation, \n'
    '};'
)
new_import = (
    'use ritk_core::segmentation::{\n'
    '    connected_components as core_connected_components,\n'
    '    connected_threshold as core_connected_threshold, BinaryClosing, BinaryDilation, BinaryErosion,\n'
    '    BinaryOpening, ChanVeseSegmentation, ConnectedComponentsFilter, GeodesicActiveContourSegmentation,\n'
    '    KMeansSegmentation, LabelStatistics,\n'
    '    MorphologicalOperation, LaplacianLevelSet, ShapeDetectionSegmentation,\n'
    '    BinaryFillHoles, ConfidenceConnectedFilter, MorphologicalGradient,\n'
    '    NeighborhoodConnectedFilter, Skeletonization,\n'
    '    ThresholdLevelSet, WatershedSegmentation,\n'
    '};\n'
    'use pyo3::types::{PyDict, PyList};'
)
assert old_import in text, f"Old import block not found"
text = text.replace(old_import, new_import, 1)
print("Step 1 done: import updated")

# Step 2: Insert label_shape_statistics before the submodule registration comment
# Anchor is the line: // -- Submodule registration (with Unicode box chars)
register_anchor = '// ── Submodule registration ────────────────────────────────────────────────────────\n\n/// Register the `segmentation` submodule'
assert register_anchor in text, f"Register anchor not found; snippet around Submodule: {text[text.find('Submodule'):text.find('Submodule')+120]!r}"

lss_func = (
    '// -- label_shape_statistics ----------------------------------------------\n\n'
    '/// Compute per-label shape statistics from a binary mask.\n'
    '///\n'
    '/// Applies `ConnectedComponentsFilter` and returns per-component spatial\n'
    '/// statistics (voxel count, centroid in index space, bounding box).\n'
    '/// Background (label 0) is excluded from results.\n'
    '///\n'
    '/// Args:\n'
    '///     mask:        Binary mask image (foreground > 0.5).\n'
    '///     connectivity: Adjacency model (6 or 26; default 6).\n'
    '///\n'
    '/// Returns:\n'
    '///     list of dicts, one per component, sorted by label ascending, each with keys:\n'
    '///     ``label`` (int), ``voxel_count`` (int),\n'
    '///     ``centroid`` (list[float]: [z, y, x] in index coordinates),\n'
    '///     ``bounding_box_min`` (list[int]: [z, y, x]),\n'
    '///     ``bounding_box_max`` (list[int]: [z, y, x]).\n'
    '///\n'
    '/// Raises:\n'
    '///     ValueError: if connectivity is not 6 or 26.\n'
    '#[pyfunction]\n'
    '#[pyo3(signature = (mask, connectivity=6_u32))]\n'
    'pub fn label_shape_statistics(\n'
    '    py: Python<\'_>,\n'
    '    mask: &PyImage,\n'
    '    connectivity: u32,\n'
    ') -> PyResult<Py<PyList>> {\n'
    '    if connectivity != 6 && connectivity != 26 {\n'
    '        return Err(pyo3::exceptions::PyValueError::new_err(format!(\n'
    '            "connectivity must be 6 or 26, got {connectivity}"\n'
    '        )));\n'
    '    }\n'
    '    let mask_arc = Arc::clone(&mask.inner);\n'
    '    let (_label_image, stats) = py.allow_threads(|| {\n'
    '        ConnectedComponentsFilter::with_connectivity(connectivity).apply(mask_arc.as_ref())\n'
    '    });\n'
    '    let list = PyList::empty_bound(py);\n'
    '    for s in &stats {\n'
    '        let dict = PyDict::new_bound(py);\n'
    '        dict.set_item("label", s.label)?;\n'
    '        dict.set_item("voxel_count", s.voxel_count)?;\n'
    '        let centroid: Vec<f64> = s.centroid.to_vec();\n'
    '        dict.set_item("centroid", centroid)?;\n'
    '        let (bb_min, bb_max) = s.bounding_box;\n'
    '        let bb_min_list: Vec<i64> = bb_min.iter().map(|&v| v as i64).collect();\n'
    '        let bb_max_list: Vec<i64> = bb_max.iter().map(|&v| v as i64).collect();\n'
    '        dict.set_item("bounding_box_min", bb_min_list)?;\n'
    '        dict.set_item("bounding_box_max", bb_max_list)?;\n'
    '        list.append(dict)?;\n'
    '    }\n'
    '    Ok(list.into())\n'
    '}\n\n'
)

text = text.replace(register_anchor, lss_func + register_anchor, 1)
print("Step 2 done: label_shape_statistics inserted")

# Step 3: Register label_shape_statistics after connected_components in register()
old_reg = '    // Labeling\n    m.add_function(wrap_pyfunction!(connected_components, &m)?)?;'
new_reg = (
    '    // Labeling\n'
    '    m.add_function(wrap_pyfunction!(connected_components, &m)?)?;\n'
    '    m.add_function(wrap_pyfunction!(label_shape_statistics, &m)?)?;'
)
assert old_reg in text, f"Labeling registration anchor not found"
text = text.replace(old_reg, new_reg, 1)
print("Step 3 done: registered label_shape_statistics")

with open(path, 'w', encoding='utf-8') as f:
    f.write(text)

print("\nsegmentation.rs patched successfully")
print("  ConnectedComponentsFilter:", 'ConnectedComponentsFilter' in text)
print("  LabelStatistics:", 'LabelStatistics' in text)
print("  PyDict/PyList:", 'PyDict, PyList' in text)
print("  fn label_shape_statistics:", 'pub fn label_shape_statistics(' in text)
print("  registered:", 'wrap_pyfunction!(label_shape_statistics' in text)
