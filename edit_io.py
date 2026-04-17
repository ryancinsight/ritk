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
     '/// - TIFF       (.tif, .tiff)   ' + '\u2014 full affine + voxel data\n'
     '/// - VTK        (.vtk)          ' + '\u2014 full affine + voxel data\n'
     '/// - MGH        (.mgh, .mgz)    ' + '\u2014 full affine + voxel data\n'
     '/// - Analyze    (.hdr, .img)    ' + '\u2014 full affine + voxel data\n'
     '/// - JPEG       (.jpg, .jpeg)   ' + '\u2014 single-slice image data\n' +
     c[line_end+1:])

# === 3. Insert new read branches ===
# Find the end of the DICOM if block, then Err(
m = re.search(r'([ \t]+return Ok\(into_py_image\(image\)\);\n[ \t]+\}\n\n[ \t]+)Err\(PyIOError::new_err\(format!\(',
              c, re.DOTALL)
assert m, "read_image insertion anchor not found"
new_read_branches = (
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
    '        }\n'
    '\n'
    '        '
)
insert_pos = m.end(1) - 8  # before "        Err("
c = c[:insert_pos] + new_read_branches + c[insert_pos:]

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

# === 5. Insert new write branches ===
# Find end of write_nrrd if block, then Err(
m2 = re.search(r'write_nrrd.*?return Ok\(\);\n[ \t]+\}\n\n[ \t]+Err\(',
               c, re.DOTALL)
assert m2, "write_image insertion anchor not found"
# Find the position just before the final "        Err(" in write_image
# Use the nrrd block end as anchor
nrrd_match = re.search(r'(write_nrrd\(&path_owned, image\.as_ref\(\)\).*?return Ok\(\);\n[ \t]+\}\n)(\n[ \t]+Err\()',
                        c, re.DOTALL)
assert nrrd_match, "write_nrrd anchor not found"
insert_pos2 = nrrd_match.end(1)  # after the nrrd } block closing newline

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
    '        }\n'
)
c = c[:insert_pos2] + new_write_branches + c[insert_pos2:]

with open(r'D:\ritk\crates\ritk-python\src\io.rs', 'w', encoding='utf-8') as f:
    f.write(c)
print("io.rs (ritk-python) done")
