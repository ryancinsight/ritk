"""Exercise relative linked transforms and window presets through workstation input."""
from browser_navigation import chord


def verify_display(client, canvas, state, wait, source, target):
    def images():
        return client.execute(f"return {state}.images")

    def drag(tool, dx, dy):
        client.send_keys(canvas, tool)
        x, y = client.execute(f"""const [x,y,w,h]={state}.panes[{source}];
            return [Math.round(x+w/2-innerWidth/2),Math.round(y+h/2-innerHeight/2)]""")
        client.pointer_drag(canvas, (x, y), (x+dx, y+dy))

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
    if images()[target] != scaled[target]:
        raise AssertionError("disabled navigation changed the comparison image")
    client.send_keys(canvas, chr(0xe036))
    chord(client, chr(0xe009), "0")
    tab = chr(0xe004)
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
    chord(client, chr(0xe008), chr(0xe036))
    client.send_keys(canvas, tab)
