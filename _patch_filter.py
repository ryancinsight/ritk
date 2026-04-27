import sys

path = r'crates/ritk-python/src/filter.rs'
with open(path, 'r', encoding='utf-8') as f:
    text = f.read()

# Step 1: Add import after TranslationTransform line
old_import = 'use ritk_core::transform::translation::TranslationTransform;'
new_import = ('use ritk_core::transform::translation::TranslationTransform;\n'
              'use ritk_core::segmentation::DistanceTransform;')
assert old_import in text, "Import anchor not found"
text = text.replace(old_import, new_import, 1)

# Step 2: Insert distance_transform function before register()
register_anchor = 'Register the `filter` submodule.'
assert register_anchor in text, "Register anchor not found"

dt_func = (
    '// -- distance_transform --------------------------------------------------\n\n'
    '/// Compute the Euclidean (or squared Euclidean) distance transform of a binary image.\n'
    '///\n'
    '/// For each background voxel the output is the distance to the nearest foreground\n'
    '/// voxel (in physical units, respecting image spacing).  Foreground voxels receive 0.0.\n'
    '/// Implements the exact O(N) Meijster et al. (2000) algorithm.\n'
    '///\n'
    '/// Args:\n'
    '///     image:                Input binary image (foreground > foreground_threshold).\n'
    '///     foreground_threshold: Threshold above which a voxel is foreground (default 0.5).\n'
    '///     squared:              If True, return squared distances (no sqrt; default False).\n'
    '///\n'
    '/// Returns:\n'
    '///     Distance image with identical shape and spatial metadata.\n'
    '#[pyfunction]\n'
    '#[pyo3(signature = (image, foreground_threshold=0.5_f32, squared=false))]\n'
    'pub fn distance_transform(\n'
    '    py: Python<\'_>,\n'
    '    image: &PyImage,\n'
    '    foreground_threshold: f32,\n'
    '    squared: bool,\n'
    ') -> PyResult<PyImage> {\n'
    '    let arc = std::sync::Arc::clone(&image.inner);\n'
    '    let result = py.allow_threads(|| {\n'
    '        if squared {\n'
    '            DistanceTransform::squared(arc.as_ref(), foreground_threshold)\n'
    '        } else {\n'
    '            DistanceTransform::transform(arc.as_ref(), foreground_threshold)\n'
    '        }\n'
    '    });\n'
    '    Ok(into_py_image(result))\n'
    '}\n\n'
)

text = text.replace(register_anchor, dt_func + register_anchor, 1)

# Step 3: Register in register() - add before parent.add_submodule
reg_line = '    m.add_function(wrap_pyfunction!(resample_image, &m)?)?;\n    parent.add_submodule(&m)?;'
new_reg_line = ('    m.add_function(wrap_pyfunction!(resample_image, &m)?)?;\n'
                '    m.add_function(wrap_pyfunction!(distance_transform, &m)?)?;\n'
                '    parent.add_submodule(&m)?;')
assert reg_line in text, f"Register line not found"
text = text.replace(reg_line, new_reg_line, 1)

with open(path, 'w', encoding='utf-8') as f:
    f.write(text)

print("filter.rs patched successfully")
print("  import added:", 'use ritk_core::segmentation::DistanceTransform;' in text)
print("  fn distance_transform:", 'pub fn distance_transform(' in text)
print("  registered:", 'wrap_pyfunction!(distance_transform' in text)
