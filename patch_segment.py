path = r"crates/ritk-cli/src/commands/segment.rs"

with open(path, "r", encoding="utf-8") as f:
    content = f.read()

# Restore the corrupted Default impl body:
# The replacement in Change 6 incorrectly replaced max_iterations/neighborhood_radius
# inside the Default impl, which must keep those as explicit field values.
old_broken = (
    "            multiplier: 2.5,\n"
    "            ..Default::default()\n"
    "            initial_phi: None,\n"
)
new_fixed = (
    "            multiplier: 2.5,\n"
    "            max_iterations: 15,\n"
    "            neighborhood_radius: 1,\n"
    "            initial_phi: None,\n"
)
assert old_broken in content, "broken Default impl pattern not found"
content = content.replace(old_broken, new_fixed, 1)
print("Default impl restored")

# Verify the Default impl looks correct now
idx = content.find("impl Default for SegmentArgs")
print(repr(content[idx:idx+500]))

with open(path, "w", encoding="utf-8") as f:
    f.write(content)
print(f"Saved: {len(content)} chars")
