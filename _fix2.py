path = r"D:/ritk/crates/ritk-core/src/statistics/image_statistics.rs"
bt = chr(96)
fn_name = bt + "select_nth_unstable_by" + bt
old_line = "/// - Phase 2: O(N) amortized percentile selection via 
///   (introselect / pdqselect)"
new_line = "/// - Phase 2: O(N) amortized percentile selection via " + fn_name + "
///   (introselect / pdqselect)"
with open(path, "r", encoding="utf-8") as f:
    src = f.read()
if old_line in src:
    src = src.replace(old_line, new_line, 1)
    with open(path, "w", encoding="utf-8") as f:
        f.write(src)
    print("Patched.")
else:
    print("Pattern not found.")
