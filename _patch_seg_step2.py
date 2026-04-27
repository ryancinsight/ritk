path = r"crates/ritk-python/src/segmentation.rs"
with open(path, "r", encoding="utf-8") as f:
    text = f.read()

# Build the label_shape_statistics function text
lss_lines = [
    "// -- label_shape_statistics ------------------------------------------
",
    "
",
    "/// Compute per-label shape statistics from a binary mask.
",
    "///
",
    "/// Applies ConnectedComponentsFilter and returns per-component spatial
",
    "/// statistics (voxel count, centroid in index space, bounding box).
",
    "/// Background (label 0) is excluded from results.
",
    "///
",
    "/// Args:
",
    "///     mask:         Binary mask image (foreground > 0.5).
",
    "///     connectivity: Adjacency model (6 or 26; default 6).
",
    "///
",
    "/// Returns:
",
    "///     list of dicts, one per component, sorted by label ascending.
",
    "///
",
    "/// Raises:
",
    "///     ValueError: if connectivity is not 6 or 26.
",
    "#[pyfunction]
",
    "#[pyo3(signature = (mask, connectivity=6_u32))]
",
    "pub fn label_shape_statistics(
",
    "    py: Python<\'_>,\n",
    "    mask: &PyImage,
",
    "    connectivity: u32,
",
    ") -> PyResult<Py<PyList>> {
",
    "    if connectivity != 6 && connectivity != 26 {
",
    "        return Err(pyo3::exceptions::PyValueError::new_err(format!(
",
    "            \"connectivity must be 6 or 26, got {connectivity}\"\n",
    "        )));
",
    "    }
",
    "    let mask_arc = Arc::clone(&mask.inner);
",
    "    let (_label_image, stats) = py.allow_threads(|| {
",
    "        ConnectedComponentsFilter::with_connectivity(connectivity).apply(mask_arc.as_ref())
",
    "    });
",
    "    let list = PyList::empty_bound(py);
",
    "    for s in &stats {
",
    "        let dict = PyDict::new_bound(py);
",
    "        dict.set_item(\"label\", s.label)?;\n",
    "        dict.set_item(\"voxel_count\", s.voxel_count)?;\n",
    "        let centroid: Vec<f64> = s.centroid.to_vec();
",
    "        dict.set_item(\"centroid\", centroid)?;\n",
    "        let (bb_min, bb_max) = s.bounding_box;
",
    "        let bb_min_list: Vec<i64> = bb_min.iter().map(|&v| v as i64).collect();
",
    "        let bb_max_list: Vec<i64> = bb_max.iter().map(|&v| v as i64).collect();
",
    "        dict.set_item(\"bounding_box_min\", bb_min_list)?;\n",
    "        dict.set_item(\"bounding_box_max\", bb_max_list)?;\n",
    "        list.append(dict)?;
",
    "}
",
    "    Ok(list.into())
",
"}
",
    "
",
]
lss = "".join(lss_lines)
