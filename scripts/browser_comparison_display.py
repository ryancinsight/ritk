"""Exercise relative linked transforms and window presets through workstation input."""
from browser_navigation import chord


def select_ct_one_to_one(client):
    # Browser Display menu, CT table: 14 presets, invert, 14 maps, then
    # seven display controls before 1:1. Four custom slots follow the controls.
    height = client.execute("return innerHeight")
    ct_presets, colormaps, controls, custom_slots = 14, 14, 9, 4
    first_control = ct_presets + 1 + colormaps
    row_height = min(28, (height - 30) // (first_control + controls + custom_slots))
    for y in [15, 30 + (first_control + 7) * row_height + row_height // 2]:
        client.perform_actions([{"type": "pointer", "id": "display-menu",
            "parameters": {"pointerType": "mouse"}, "actions": [
                {"type": "pointerMove", "origin": "viewport", "x": 395, "y": y},
                {"type": "pointerDown", "button": 0}, {"type": "pointerUp", "button": 0}]}])


def verify_display(client, canvas, state, wait, source, target):
    def images():
        return client.execute(f"return {state}.images")

    def drag(tool, dx, dy):
        client.send_keys(canvas, tool)
        x, y = client.execute(f"""const [x,y,w,h]={state}.panes[{source}];
            return [Math.round(x+w/2-innerWidth/2),Math.round(y+h/2-innerHeight/2)]""")
        client.pointer_drag(canvas, (x, y), (x+dx, y+dy))

    # Set just the receiving CT pane to rotated 1:1. Its 2 rows at 2 mm
    # and 8 columns at 1 mm occupy 2 by 4 screen pixels after a quarter turn.
    # An odd width assigns the grid remainder to the first column.
    client.set_window_rect(1439, 900)
    wait(f"{state}.width === innerWidth && {state}.panes[0][2] === {state}.panes[1][2] + 1")
    tab, link = chr(0xe004), chr(0xe036)
    # Firefox uses F6 to focus browser chrome. The viewer must claim the key
    # on keydown before dispatching subsequent Tab and R to the image pane.
    client.execute("""
        window.addEventListener("keydown", function probe(event) {
            if (event.key !== "F6") return;
            window.viewerHandledF6 = event.defaultPrevented;
            window.removeEventListener("keydown", probe);
        });
    """)
    client.send_keys(canvas, link + tab + "r")
    if client.execute("return window.viewerHandledF6") is not True:
        raise AssertionError("F6 left the browser's focus-navigation shortcut active")
    wait(f"{state}.panel === {target} && !{state}.orientationIdentity")
    select_ct_one_to_one(client)
    wait(f"{state}.zoom > 0 && {state}.zoom < 0.05")
    # Store the actual binary64 values in the browser. Some WebDriver JSON
    # encoders round decimal extents in transit, before Python can double them.
    unit = client.execute(f"return window.unitExtent = {state}.images[{target}].slice(2)")

    def exact_extent(multiplier):
        return client.execute(f"""return {state}.images[{target}].slice(2).every(
            (value, index) => value === window.unitExtent[index] * arguments[0])""", [multiplier])

    # 1:1 is the reciprocal fit rounded once to binary32. These intervals
    # allow one binary32 epsilon for that rounding and binary64 placement.
    for actual, expected in zip(unit, [2, 4]):
        if abs(actual - expected) > expected * 2**-23:
            raise AssertionError(f"rotated CT 1:1 dimensions: {unit}")
    client.send_keys(canvas, tab + link)
    drag("z", 0, 50)  # source 1 -> 0.5; target is already below drag minimum
    if not exact_extent(1):
        raise AssertionError(f"linked zoom-out enlarged 1:1 pane: {unit} -> {images()[target][2:]}")
    chord(client, chr(0xe009), "0")  # source 0.5 -> 1; target must double
    if not exact_extent(2):
        raise AssertionError("linked zoom-in lost the receiving pane's 1:1 scale")

    drag("z", 0, 50)
    if not exact_extent(1):
        raise AssertionError("reversing linked zoom did not recover the rotated 1:1 extent")
    client.send_keys(canvas, link)
    chord(client, chr(0xe009), "0")
    client.send_keys(canvas, tab)
    chord(client, chr(0xe009), "0")
    client.send_keys(canvas, tab + link)
    before = images()
    drag("m", 24, -12)
    after = images()
    for i in [source, target]:
        delta = [after[i][j]-before[i][j] for j in range(2)]
        if delta != [24, -12]:
            raise AssertionError(f"linked screen pan in pane {i}: {delta}")
    drag("z", 0, -25)
    scaled = images()
    for i in [source, target]:
        if abs(scaled[i][2] - 1.25 * after[i][2]) > 1:
            raise AssertionError(f"linked zoom in pane {i}: {after[i]} -> {scaled[i]}")
    client.send_keys(canvas, chr(0xe036))  # F6 toggles linked navigation.
    drag("m", -15, 6)
    drag("z", 0, 20)
    independent = images()
    if independent[target] != scaled[target]:
        raise AssertionError("disabled navigation changed the comparison image")
    client.send_keys(canvas, chr(0xe036))
    if images() != independent:
        raise AssertionError("enabling navigation changed the existing alignment")
    drag("z", 0, -20)
    relinked = images()
    for i in [source, target]:
        if abs(relinked[i][2] - 1.2 * independent[i][2]) > 1:
            raise AssertionError(f"pane {i} lost its independent fit when linking resumed")
    chord(client, chr(0xe009), "0")
    client.send_keys(canvas, tab)
    initial_window = client.execute(f"return {state}.window")
    client.send_keys(canvas, tab + "2" + tab)
    if client.execute(f"return {state}.window") != initial_window:
        raise AssertionError("independent window changed in the other pane")
    client.send_keys(canvas, tab)
    chord(client, chr(0xe008), chr(0xe036))  # Shift+F6 toggles window linking.
    client.send_keys(canvas, "1")
    selected = client.execute(f"return {state}.window")
    client.send_keys(canvas, tab)
    if client.execute(f"return {state}.window") != selected or selected == initial_window:
        raise AssertionError(f"linked window did not follow the source preset {selected}")
    client.send_keys(canvas, tab + link)  # Window-only linking.
    target_rect = images()[target]
    drag("m", 7, -3)
    if images()[target] != target_rect:
        raise AssertionError("window-only linking moved the receiving pane")
    client.send_keys(canvas, "2")
    window_only = client.execute(f"return {state}.window")
    client.send_keys(canvas, tab)
    if window_only == selected or client.execute(f"return {state}.window") != window_only:
        raise AssertionError("window preset failed with navigation linking disabled")
    chord(client, chr(0xe008), chr(0xe036))
    client.send_keys(canvas, tab + link)
    client.set_window_rect(1440, 900)
    wait(f"{state}.width === innerWidth")
