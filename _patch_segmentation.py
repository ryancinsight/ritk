path = r'crates/ritk-python/src/segmentation.rs'
with open(path, 'r', encoding='utf-8') as f:
    text = f.read()

# Step 1: Update ritk_core::segmentation import block to add ConnectedComponentsFilter + LabelStatistics
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
assert old_import in text, "segmentation import block not found"
text = text.replace(old_import, new_import, 1)

# Step 2: Insert label_shape_statistics before register() anchor
register_anchor = '// -- Submodule registration'
# Check if that's the exact text
if register_anchor not in text:
    # Try alternate
    register_anchor2 = 'Submodule registration'
    idx = text.find(register_anchor2)
    print(f"Alternate anchor at index: {idx}")
    print(repr(text[max(0,idx-10):idx+60]))
else:
    print(f"Found anchor: {register_anchor!r}")
