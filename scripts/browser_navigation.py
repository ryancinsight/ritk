"""Check workstation drags against screen pixels at different fits and rotations."""


def chord(client, modifier, key):
    client.perform_actions([{
        "type": "key", "id": "workstation-shortcuts",
        "actions": [{"type": phase, "value": value} for phase, value in [
            ("keyDown", modifier), ("keyDown", key), ("keyUp", key), ("keyUp", modifier)
        ]],
    }])


def select_grid_plane(client, plane):
    # View menu in a populated, non-PET comparison grid.
    row = 8 + ("Axial", "Coronal", "Sagittal").index(plane)
    for x, y in [(88, 15), (120, 44 + 28 * row)]:
        client.perform_actions([{"type": "pointer", "id": "plane-menu", "parameters": {"pointerType": "mouse"},
            "actions": [{"type": "pointerMove", "origin": "viewport", "x": x, "y": y},
                        {"type": "pointerDown", "button": 0}, {"type": "pointerUp", "button": 0}]}])


def verify_navigation(client, canvas, state, wait):
    client.send_keys(canvas, "m")
    chord(client, chr(0xe008), "o")
    wait(state + ".orientationIdentity")
    for width, height in [(1440, 900), (960, 720)]:
        client.set_window_rect(width, height)
        wait(state + ".width === innerWidth && " + state + ".height === innerHeight")
        for turn in range(4):
            chord(client, chr(0xe009), "0")
            client.send_keys(canvas, "m")
            wait(state + ".tool === 'Pan' && " + state + ".zoom === 1")
            image, pane, size = client.execute(
                f"return [{state}.images[0], {state}.panes[0], [innerWidth, innerHeight]]"
            )
            x, y = (pane[i] + pane[i+2] // 2 - size[i] // 2 for i in range(2))
            client.pointer_drag(canvas, (x, y), (x + 24, y - 12))
            after = client.execute("return " + state + ".images[0]")
            delta = [after[i] - image[i] for i in range(2)]
            if delta != [24, -12]:
                raise AssertionError(f"window {width}x{height}, turn {turn}: pan moved {delta}")
            client.send_keys(canvas, "z")
            client.pointer_drag(canvas, (x, y), (x, y + 24))
            wait(f"Math.fround({state}.zoom) === Math.fround(0.76)")
            chord(client, chr(0xe009), "0")
            client.send_keys(canvas, "r")
    client.set_window_rect(1440, 900)
    wait(state + ".width === innerWidth && " + state + ".orientationIdentity")
