"""Exercise paired Classic CT stacks at known patient positions through the file picker."""
import struct
import tempfile
from pathlib import Path

from browser_comparison_display import verify_display
from browser_navigation import chord, verify_navigation


def write_stack(directory, series, locations, *, mixed_frame=False, study=1, frame_uid=3, sagittal=False):
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
        frame = 4 if mixed_frame and index == 0 else frame_uid
        dataset = [(8, 22, "UI", ct), (8, 24, "UI", sop), (8, 96, "CS", "CT"),
                   (16, 32, "LO", "GEOMETRY"), (32, 13, "UI", f"{root}.{study}"),
                   (32, 14, "UI", f"{root}.{series}"), (32, 17, "IS", str(series)),
                   (32, 19, "IS", str(index + 1)), (32, 50, "DS", (f"{z}\\0\\0" if sagittal else f"0\\0\\{z}")),
                   (32, 55, "DS", "0\\1\\0\\0\\0\\1" if sagittal else "1\\0\\0\\0\\1\\0"), (32, 82, "UI", f"{root}.{frame}"),
                   (40, 2, "US", struct.pack("<H", 1)), (40, 4, "CS", "MONOCHROME2"),
                   (40, 16, "US", struct.pack("<H", 2)), (40, 17, "US", struct.pack("<H", 2)),
                   (40, 48, "DS", "8\\1" if sagittal else "1\\1")]
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
            source, target = open_pair(client, canvas, state, wait, first + second, navigate=not incompatible)
            if not incompatible:
                verify_display(client, canvas, state, wait, source, target)
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

        # The two series have unrelated origins and distinct frame identifiers.
        shifted = write_stack(directory, 13, [(100+i, i) for i in range(9)], study=5, frame_uid=7)
        source, target = open_pair(client, canvas, state, wait, first + shifted)
        client.send_keys(canvas, chr(0xe015))
        wait(f"{state}.panels[{source}][0] === 3 && {state}.panels[{target}][0] === 4")
        chord(client, chr(0xe008), chr(0xe035))  # Shift+F5 anchors 6 mm to 104 mm.
        client.send_keys(canvas, chr(0xe013))
        wait(f"{state}.panels[{source}][0] === 2 && {state}.panels[{target}][0] === 2")
        client.send_keys(canvas, chr(0xe004) + chr(0xe015) + chr(0xe015))
        wait(f"{state}.panels[{target}][0] === 4 && {state}.panels[{source}][0] === 3")

        sagittal = write_stack(directory, 14, [(i, i) for i in range(3)], sagittal=True)
        source, target = open_pair(client, canvas, state, wait, first + sagittal)
        verify_cursor(client, canvas, state, wait, source, target)
        # Source z=4 intersects the target halfway through its 16-mm field of view.
        verify_reference_pixel(client, target, [.4, .5], True)
        client.send_keys(canvas, chr(0xe015))
        verify_reference_pixel(client, target, [.4, .625], True)
        verify_reference_pixel(client, target, [.4, .5], False)
        chord(client, chr(0xe009), chr(0xe03c))  # Ctrl+F12
        verify_reference_pixel(client, target, [.4, .625], False)
        chord(client, chr(0xe009), chr(0xe03c))
        client.send_keys(canvas, chr(0xe004))
        verify_reference_pixel(client, source, [.75, .4], True)
        verify_reference_pixel(client, target, [.4, .625], False)
        client.send_keys(canvas, "r" + chr(0xe004))
        verify_reference_pixel(client, target, [.375, .4], True)



def verify_cursor(client, canvas, state, wait, source, target):
    client.send_keys(canvas, "q")
    wait(state + ".tool === 'Crosshair'")
    def move(fraction):
        x,y = client.execute(f"const [x,y,w,h]={state}.images[{source}]; return [Math.round(x+w*{fraction}), Math.round(y+h*.75)]")
        return {"type":"pointerMove","x":x,"y":y,"origin":"viewport"}
    def pointer(*actions):
        client.perform_actions([{"type":"pointer","id":"cursor","parameters":{"pointerType":"mouse"},"actions":list(actions)}])
    pointer(move(.25), {"type":"pointerDown","button":0})
    wait(f"{state}.panels[{target}][0] === 0")
    verify_pixel(client, target, [.4,.5], [0,229,255,255], True)
    pointer(move(.75))
    wait(f"{state}.panels[{target}][0] === 1")
    pointer({"type":"pointerUp","button":0})
    client.send_keys(canvas, "m")
    modifiers = {"type":"key","id":"cursor-chord","actions":[
        {"type":"keyDown","value":chr(0xe009)}, {"type":"keyDown","value":chr(0xe008)}]}
    client.perform_actions([modifiers])
    pointer(move(.25), {"type":"pointerDown","button":0})
    wait(f"{state}.panels[{target}][0] === 0")
    verify_pixel(client, target, [.4,.5], [0,229,255,255], True)
    pointer({"type":"pointerUp","button":0})
    modifiers["actions"] = [{"type":"keyUp","value":k} for k in [chr(0xe008),chr(0xe009)]]
    client.perform_actions([modifiers])
    wait(state + ".tool === 'Pan'")
    client.send_keys(canvas, chr(0xe004) + chr(0xe015) + chr(0xe004))
    verify_pixel(client, target, [.4,.5], [0,229,255,255], False)


def verify_reference_pixel(client, panel, fraction, expected):
    verify_pixel(client, panel, fraction, [255,255,0,255], expected)


def verify_pixel(client, panel, fraction, color, expected):
    # Read the actually presented framebuffer, including WebGPU canvas presentation.
    value = client.execute_async(f"""
        const done=arguments[arguments.length-1], c=document.querySelector('#workstation');
        const [x,y,w,h]=window.asterSlice.state().images[{panel}];
        requestAnimationFrame(() => requestAnimationFrame(() => c.toBlob(async blob => {{
            const image=await createImageBitmap(blob);
            const capture=document.createElement('canvas'); capture.width=c.width;capture.height=c.height;
            const ctx=capture.getContext('2d');ctx.drawImage(image,0,0);image.close();
            done(Array.from(ctx.getImageData(Math.round(x+w*{fraction[0]}),Math.round(y+h*{fraction[1]}),1,1).data));
        }})));
    """)
    if (value == color) != expected:
        raise AssertionError(f"reference line at {fraction} in panel {panel}: expected {color} present={expected}, got {value}")


def open_pair(client, canvas, state, wait, files, *, navigate=False):
    client.send_file_paths(client.find("#file-input"), files)
    wait(state + ".series === 2 && " + state + ".panels.length === 1")
    if navigate:
        verify_navigation(client, canvas, state, wait)
    client.send_keys(canvas, chr(0xe034))
    client.send_keys(canvas, " " + chr(0xe015) + " " + chr(0xe007))
    wait(state + ".panels.length === 2")
    panels = client.execute("return " + state + ".panels")
    source = next(i for i, (_, count) in enumerate(panels) if count == 5)
    if client.execute("return " + state + ".panel") != source:
        client.send_keys(canvas, chr(0xe004))
    return source, 1 - source
