import re

with open(r'D:\ritk\crates\ritk-python\src\io.rs', 'r', encoding='utf-8') as f:
    c = f.read()

# === 1. Module doc table ===
old_t = '//! | `.nrrd`               | \u2713    | \u2713     | ritk-io NRRD         |'
new_t = (old_t + '\n'
    '//! | `.tif`, `.tiff`       | \u2713    | \u2713     | ritk-io TIFF         |\n'
    '//! | `.vtk`                | \u2713    | \u2713     | ritk-io VTK          |\n'
    '//! | `.mgh`, `.mgz`        | \u2713    | \u2713     | ritk-io MGH          |\n'
    '//! | `.hdr`, `.img`        | \u2713    | \u2713     | ritk-io Analyze      |\n'
    '//! | `.jpg`, `.jpeg`       | \u2713    | \u2713     | ritk-io JPEG         |')
assert old_t in c
c = c.replace(old_t, new_t, 1)

# === 2. read_image doc: add new formats ===
old_rd = '/// - NRRD       (.nrrd)         '
idx = c.find(old_rd)
assert idx >= 0
line_end = c.find('\n', idx)
c = (c[:line_end+1] +
     '/// - TIFF       (.tif, .tiff)   \u2014 full affine + voxel data\n'
     '/// - VTK        (.vtk)          \u2014 full affine + voxel data\n'
     '/// - MGH        (.mgh, .mgz)    \u2014 full affine + voxel data\n'
     '/// - Analyze    (.hdr, .img)    \u2014 full affine + voxel data\n'
     '/// - JPEG       (.jpg, .jpeg)   \u2014 single-slice image data\n' +
     c[line_end+1:])

# === 3. Insert new read branches before final Err in read_image ===
# Find read_dicom_series, then find the "\n        Err(PyIOError::new_err" after it
dicom_pos = c.find('read_dicom_series')
assert dicom_pos >= 0
err_anchor = '\n        Err(PyIOError::new_err'
err_pos = c.find(err_anchor, dicom_pos)
assert err_pos >= 0, "read err anchor not found after dicom"

new_read_branches = (
    '\n'
    '        if path_lower.ends_with(".tif") || path_lower.ends_with(".tiff") {\n'
    '            let image = ritk_io::read_tiff::<Backend, _>(p, &device)\n'
    '                .map_err(|e| PyIOError::new_err(format!("TIFF read error: {e}")))?;\n'
    '            return Ok(into_py_image(image));\n'
    '        }\n'
    '\n'
    '        if path_lower.ends_with(".vtk") {\n'
    '            let image = ritk_io::read_vtk::<Backend, _>(p, &device)\n'
    '                .map_err(|e| PyIOError::new_err(format!("VTK read error: {e}")))?;\n'
    '            return Ok(into_py_image(image));\n'
    '        }\n'
    '\n'
    '        if path_lower.ends_with(".mgh") || path_lower.ends_with(".mgz") {\n'
    '            let image = ritk_io::read_mgh::<Backend, _>(p, &device)\n'
    '                .map_err(|e| PyIOError::new_err(format!("MGH read error: {e}")))?;\n'
    '            return Ok(into_py_image(image));\n'
    '        }\n'
    '\n'
    '        if path_lower.ends_with(".hdr") || path_lower.ends_with(".img") {\n'
    '            let image = ritk_io::read_analyze::<Backend, _>(p, &device)\n'
    '                .map_err(|e| PyIOError::new_err(format!("Analyze read error: {e}")))?;\n'
    '            return Ok(into_py_image(image));\n'
    '        }\n'
    '\n'
    '        if path_lower.ends_with(".jpg") || path_lower.ends_with(".jpeg") {\n'
    '            let image = ritk_io::read_jpeg::<Backend, _>(p, &device)\n'
    '                .map_err(|e| PyIOError::new_err(format!("JPEG read error: {e}")))?;\n'
    '            return Ok(into_py_image(image));\n'
    '        }'
)
# Insert at err_pos (before the '\n        Err(PyIOError::new_err')
c = c[:err_pos] + new_read_branches + c[err_pos:]

# === 4. write_image doc ===
old_wd = '/// - NRRD      (.nrrd)\n'
idx = c.find(old_wd)
assert idx >= 0
c = (c[:idx + len(old_wd)] +
     '/// - TIFF      (.tif, .tiff)\n'
     '/// - VTK       (.vtk)\n'
     '/// - MGH       (.mgh, .mgz)\n'
     '/// - Analyze   (.hdr)\n'
     '/// - JPEG      (.jpg, .jpeg)\n' +
     c[idx + len(old_wd):])

# === 5. Insert new write branches before final Err in write_image ===
# Find write_nrrd, then find "\n        Err(PyIOError::new_err" after it
nrrd_pos = c.find('write_nrrd(')
assert nrrd_pos >= 0
err_anchor2 = '\n        Err(PyIOError::new_err'
err_pos2 = c.find(err_anchor2, nrrd_pos)
assert err_pos2 >= 0, "write err anchor not found after nrrd"

new_write_branches = (
    '\n'
    '        if path_lower.ends_with(".tif") || path_lower.ends_with(".tiff") {\n'
    '            ritk_io::write_tiff(image.as_ref(), &path_owned)\n'
    '                .map_err(|e| PyIOError::new_err(format!("TIFF write error: {e}")))?;\n'
    '            return Ok(());\n'
    '        }\n'
    '\n'
    '        if path_lower.ends_with(".vtk") {\n'
    '            ritk_io::write_vtk(&path_owned, image.as_ref())\n'
    '                .map_err(|e| PyIOError::new_err(format!("VTK write error: {e}")))?;\n'
    '            return Ok(());\n'
    '        }\n'
    '\n'
    '        if path_lower.ends_with(".mgh") || path_lower.ends_with(".mgz") {\n'
    '            ritk_io::write_mgh(image.as_ref(), &path_owned)\n'
    '                .map_err(|e| PyIOError::new_err(format!("MGH write error: {e}")))?;\n'
    '            return Ok(());\n'
    '        }\n'
    '\n'
    '        if path_lower.ends_with(".hdr") {\n'
    '            ritk_io::write_analyze(&path_owned, image.as_ref())\n'
    '                .map_err(|e| PyIOError::new_err(format!("Analyze write error: {e}")))?;\n'
    '            return Ok(());\n'
    '        }\n'
    '\n'
    '        if path_lower.ends_with(".jpg") || path_lower.ends_with(".jpeg") {\n'
    '            ritk_io::write_jpeg(&path_owned, image.as_ref())\n'
    '                .map_err(|e| PyIOError::new_err(format!("JPEG write error: {e}")))?;\n'
    '            return Ok(());\n'
    '        }'
)
c = c[:err_pos2] + new_write_branches + c[err_pos2:]

with open(r'D:\ritk\crates\ritk-python\src\io.rs', 'w', encoding='utf-8') as f:
    f.write(c)
print("io.rs (ritk-python) done")
