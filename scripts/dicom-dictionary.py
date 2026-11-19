#!/usr/bin/env python3
"""Generate the numeric implicit-VR lookup from DICOM PS3.6 (2026c).

Source: https://dicom.nema.org/medical/dicom/2026c/source/docbook/part06/part06.xml
Only the tag and transfer representation facts are retained. Regeneration does
not run during Cargo builds.
"""

import argparse
import hashlib
from pathlib import Path
import re
import struct
import xml.etree.ElementTree as ET

SOURCE_SHA256 = "ff1dcdfb557d57db96420614fcaf6d739bb76aa74b73eba77f367be9fab0be3e"
TABLES = {"6-1", "7-1", "8-1"}
VIRTUAL = {"US or SS": "XS", "OB or OW": "OX", "US or OW": "LX",
           "US or SS or OW": "LW", "See Note": "UN", "": "UN"}
TAG = re.compile(r"\(([0-9a-fx]{4}),([0-9a-fx]{4})\)", re.I)


def text(cell):
    return "".join(cell.itertext()).replace("\u200b", "").strip()


def records(source):
    for table in source.findall(".//{*}table"):
        if table.get("label") not in TABLES:
            continue
        for row in table.findall(".//{*}tbody/{*}tr"):
            cells = [text(cell) for cell in row.findall("{*}td")]
            match = TAG.fullmatch(cells[0])
            if match is None:
                raise ValueError(f"unrecognized tag {cells[0]!r}")
            digits = "".join(match.groups()).lower()
            mask = int("".join("0" if digit == "x" else "f" for digit in digits), 16)
            value = int(digits.replace("x", "0"), 16)
            code = VIRTUAL.get(cells[3], cells[3]).encode("ascii")
            if len(code) != 2:
                raise ValueError(f"unrecognized VR {cells[3]!r}")
            yield mask, value, code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    raw = args.source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA256:
        raise ValueError("DICOM PS3.6 source hash differs from the pinned edition")
    rows = list(records(ET.fromstring(raw)))
    if len({(mask, value) for mask, value, _ in rows}) != len(rows):
        raise ValueError("the PS3.6 registry repeats a tag pattern")
    output = Path(__file__).resolve().parents[1] / "crates/ritk-dicom/src/dataset/wire/dictionary"
    output.mkdir(exist_ok=True)
    exact = sorted((value, code) for mask, value, code in rows if mask == 0xffffffff)
    patterns = sorted((mask, value, code) for mask, value, code in rows if mask != 0xffffffff)
    patterns.sort(key=lambda row: -row[0].bit_count())
    (output / "exact.bin").write_bytes(b"".join(struct.pack(">I2s", *row) for row in exact))
    (output / "patterns.bin").write_bytes(b"".join(struct.pack(">II2s", *row) for row in patterns))


if __name__ == "__main__":
    main()
