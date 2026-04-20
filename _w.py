import base64,sys
b=sys.stdin.read().strip()
with open(r"crates/ritk-core/src/segmentation/level_set/shape_detection.rs","w",encoding="utf-8",newline="
") as f:
    f.write(base64.b64decode(b).decode("utf-8"))
print("OK")
