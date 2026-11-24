"""Verify anatomical plane selection and the actual CT imagery through the workstation menus."""
import argparse
from pathlib import Path
from browser_patient_position import verify_reference_pixel
from browser_navigation import chord
from browser_protocol import WebDriverClient
from verify_anonymized_mri import wait
from verify_study_folder import choose
from verify_web_ct_export import click


def state(client):
    return client.execute("return asterSlice.state()")


def select(client, plane, grid=False):
    click(client, 88, 15)
    index = ("Axial", "Coronal", "Sagittal").index(plane)
    click(client, 120, 44 + 28 * ((8 if grid else 6) + index))


def center_hash(client, pane):
    # Only the central image area is sampled to exclude menu, title and ruler pixels.
    return client.execute_async("""
       const [pane, done] = arguments;
       requestAnimationFrame(() => requestAnimationFrame(async () => {
          const [x,y,w,h] = asterSlice.state().images[pane];
          const ctx = document.querySelector('#workstation').getContext('2d');
          const pixels = ctx.getImageData(x+w/2-32,y+h/2-32,64,64).data;
          done(Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', pixels))).join(','));
       }));
    """, [pane])


def verify(url, source, driver):
    client = WebDriverClient(driver, 90)
    client.create_session("chrome", 1000, headless=True)
    try:
        client.set_window_rect(1440, 900)
        client.navigate(url)
        wait(client, "!!window.asterSlice")
        choose(client, source)
        wait(client, "asterSlice.state().series === 5")
        before = center_hash(client, 0)
        select(client, "Coronal")
        assert center_hash(client, 0) != before, "first pane still renders the acquisition plane"
        assert state(client)["planes"] == ["Coronal", "Axial", "Sagittal"]
        select(client, "Axial")
        assert center_hash(client, 0) == before
        click(client, 900, 53)  # Split screen
        click(client, 795, 142)  # first grid cell
        assert len(state(client)["panes"]) == 1, "single panel selection"
        frames = {}
        for plane in ("Axial", "Sagittal", "Coronal", "Axial"):
            select(client, plane, grid=True)
            current = state(client)
            assert current["planes"] == [plane]
            assert current["panels"][0][1] == (34 if plane == "Axial" else 512)
            frame = center_hash(client, 0)
            if plane in frames:
                assert frames[plane] == frame
            frames[plane] = frame
        assert len(set(frames.values())) == 3
        canvas = client.find("#workstation")
        client.send_keys(canvas, chr(0xe034))
        down = chr(0xe015)
        client.send_keys(canvas, down + " " + down + " " + chr(0xe007))
        wait(client, "asterSlice.state().panels.length === 2")
        initial = state(client)
        assert sorted(panel[1] for panel in initial["panels"]) == [34,67]
        # Hide locator overlays while comparing only the underlying CT pixels.
        chord(client, chr(0xe009), chr(0xe03c))
        untouched = center_hash(client, 0)
        x,y,w,h = initial["panes"][1]
        click(client, x+w//2, y+h//2)
        select(client, "Sagittal", grid=True)
        assert state(client)["planes"] == ["Axial","Sagittal"]
        assert center_hash(client, 0) == untouched
        second = center_hash(client, 1)
        x,y,w,h = initial["panes"][0]
        click(client, x+w//2, y+h//2)
        select(client, "Coronal", grid=True)
        assert state(client)["planes"] == ["Coronal","Sagittal"]
        assert center_hash(client, 1) == second and center_hash(client, 0) != untouched
        chord(client, chr(0xe009), chr(0xe03c))
        # These reconstructions share their first voxel and in-plane pixel spacing.
        # Fixed source row i cuts the sagittal target at its column center i + 1/2.
        index = state(client)["panels"][0][0]
        verify_reference_pixel(client, 1, [(index + .5) / 512, .4], True)
        for plane in ("Coronal", "Sagittal"):
            for panel in (0, 1):
                x,y,w,h = initial["panes"][panel]
                click(client, x+w//2, y+h//2)
                select(client, plane, grid=True)
            for panel in (1, 0):
                x,y,w,h = initial["panes"][panel]
                click(client, x+w//2, y+h//2)
                before = state(client)["panels"][panel][0]
                client.send_keys(canvas, down)
                assert [p[0] for p in state(client)["panels"]] == [before+1]*2
        # The active axial reconstruction has double the target's depth pitch.
        select(client, "Axial", grid=True)
        for _ in range(2):
            i = state(client)["panels"][0][0]
            verify_reference_pixel(client, 1, [.4, (2*i + .5)/67], True)
            client.send_keys(canvas, down)
        print("Verified linked sagittal and coronal scrolling and moving cross-reference pixels")
        print("Verified three distinct CT planes, reversible MPR assignment and two independent comparison panels", flush=True)
    finally:
        client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--url", default="http://127.0.0.1:8108/")
    parser.add_argument("--driver-url", required=True)
    args = parser.parse_args()
    verify(args.url, args.source, args.driver_url)
