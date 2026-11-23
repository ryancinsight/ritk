"""Check workstation drags against screen pixels at different fits and rotations."""


def verify_navigation(client, canvas, state, wait):
    client.send_keys(canvas, chr(0xe008) + "o" + chr(0xe000))
    wait(state + ".orientationIdentity")
    fit = chr(0xe009) + "0" + chr(0xe000)
    for width, height in [(1440, 900), (960, 720)]:
        client.set_window_rect(width, height)
        wait(state + ".width === innerWidth && " + state + ".height === innerHeight")
        for turn in range(4):
            client.send_keys(canvas, fit + "m")
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
            client.send_keys(canvas, fit + "r")
    client.set_window_rect(1440, 900)
    wait(state + ".width === innerWidth && " + state + ".orientationIdentity")
