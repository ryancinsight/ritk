"""Exercise paired Classic CT stacks at known patient positions through the file picker."""
import struct
import tempfile
from pathlib import Path

from browser_navigation import verify_navigation


def write_stack(directory, series, locations, *, mixed_frame=False):
    def element(group, tag, vr, value):
        if isinstance(value, str):
            value = value.encode("ascii")
        if len(value) % 2:
            value += b"\0" if vr == "UI" else b" "
        header = struct.pack("<HH", group, tag) + vr.encode("ascii")
        if vr in {"OB", "OW"}:
            return header + b"\0\0" + struct.pack("<I", len(value)) + value
        return header + struct.pack("<H", len(value)) + value

    root = "2.25.20261123"
    ct = "1.2.840.10008.5.1.4.1.1.2"
    files = []
    for index, (z, stored) in enumerate(locations):
        sop = f"{root}.{series}.{index + 1}"
        meta = b"".join(element(2, tag, vr, value) for tag, vr, value in [
            (1, "OB", b"\0\1"), (2, "UI", ct), (3, "UI", sop),
            (16, "UI", "1.2.840.10008.1.2.1"), (18, "UI", root)])
        frame = 4 if mixed_frame and index == 0 else 3
        dataset = [(8, 22, "UI", ct), (8, 24, "UI", sop), (8, 96, "CS", "CT"),
                   (16, 32, "LO", "GEOMETRY"), (32, 13, "UI", root + ".1"),
                   (32, 14, "UI", f"{root}.{series}"), (32, 17, "IS", str(series)),
                   (32, 19, "IS", str(index + 1)), (32, 50, "DS", f"0\\0\\{z}"),
                   (32, 55, "DS", "1\\0\\0\\0\\1\\0"), (32, 82, "UI", f"{root}.{frame}"),
                   (40, 2, "US", struct.pack("<H", 1)), (40, 4, "CS", "MONOCHROME2"),
                   (40, 16, "US", struct.pack("<H", 2)), (40, 17, "US", struct.pack("<H", 2)),
                   (40, 48, "DS", "1\\1")]
        dataset += [(40, tag, "US", struct.pack("<H", value))
                    for tag, value in [(256, 16), (257, 16), (258, 15), (259, 0)]]
        dataset.append((0x7fe0, 16, "OW", struct.pack("<4H", *(stored + i for i in range(4)))))
        path = directory / f"series-{series}-slice-{index + 1}.dcm"
        path.write_bytes(bytes(128) + b"DICM" + element(2, 0, "UL", struct.pack("<I", len(meta)))
                         + meta + b"".join(element(*entry) for entry in dataset))
        files.append(path)
    return files


def verify_patient_position(client, canvas, state, wait):
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        # The second stack is delivered in reverse spatial order. Its middle
        # slice and the first stack's middle slice are both at z=4 mm.
        first = write_stack(directory, 11, [(2*i, 200+i) for i in range(5)])
        second = write_stack(directory, 12, [(i, 100+i) for i in reversed(range(9))])
        for incompatible in (False, True):
            if incompatible:
                second = write_stack(directory, 12, [(i, 100+i) for i in reversed(range(9))],
                                     mixed_frame=True)
            client.send_file_paths(client.find("#file-input"), first + second)
            wait(state + ".series === 2 && " + state + ".panels.length === 1")
            if not incompatible:
                verify_navigation(client, canvas, state, wait)
            client.send_keys(canvas, chr(0xe034))
            client.send_keys(canvas, " " + chr(0xe015) + " " + chr(0xe007))
            wait(state + ".panels.length === 2")
            panels = client.execute("return " + state + ".panels")
            source = next(i for i, (_, count) in enumerate(panels) if count == 5)
            if client.execute("return " + state + ".panel") != source:
                client.send_keys(canvas, chr(0xe004))
            target = 1 - source
            client.send_keys(canvas, chr(0xe015))  # 4 -> 6 mm
            wait(f"{state}.panels[{source}][0] === 3")
            expected = 4 if incompatible else 6
            actual = client.execute(f"return {state}.panels[{target}][0]")
            if actual != expected:
                raise AssertionError(f"patient position: expected target {expected}, got {actual}")
            if not incompatible:
                client.send_keys(canvas, chr(0xe035))  # F5 disables automatic linking
                client.send_keys(canvas, chr(0xe013))
                wait(f"{state}.panels[{source}][0] === 2")
                if client.execute(f"return {state}.panels[{target}][0]") != 6:
                    raise AssertionError("disabled synchronization changed the target")
                client.send_keys(canvas, chr(0xe035))
