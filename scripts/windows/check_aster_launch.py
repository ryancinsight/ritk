"""Exercise Windows Explorer-style launch and redirected CLI output."""
import ctypes
from ctypes import wintypes
from pathlib import Path
import struct
import subprocess
import sys
import time


def viewer_window(pid):
    user = ctypes.WinDLL("user32", use_last_error=True)
    callback = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    user.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
    user.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    user.GetClassNameW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    user.IsWindowVisible.argtypes = [wintypes.HWND]
    user.EnumWindows.argtypes = [callback, wintypes.LPARAM]
    found = False

    @callback
    def visit(hwnd, _):
        nonlocal found
        owner = wintypes.DWORD()
        user.GetWindowThreadProcessId(hwnd, ctypes.byref(owner))
        if owner.value == pid and user.IsWindowVisible(hwnd):
            title = ctypes.create_unicode_buffer(128)
            user.GetWindowTextW(hwnd, title, len(title))
            class_name = ctypes.create_unicode_buffer(128)
            user.GetClassNameW(hwnd, class_name, len(class_name))
            found = title.value == "AsterSlice" and class_name.value != "#32770"
        return not found

    user.EnumWindows(visit, 0)
    return found


def check_windows(exe):
    # This probe has no parent console, matching an Explorer launch. Let
    # Windows choose the child's console policy from the executable itself.
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    if kernel.GetConsoleCP():
        raise RuntimeError("test probe must start detached")
    for args in ([], ["--console"]):
        with subprocess.Popen([exe, *args]) as child:
            try:
                deadline = time.monotonic() + 30
                while not viewer_window(child.pid):
                    if child.poll() is not None or time.monotonic() >= deadline:
                        raise RuntimeError(f"viewer window did not open with {args}")
                    time.sleep(0.1)
                attached = bool(kernel.AttachConsole(child.pid))
                if attached:
                    if not kernel.FreeConsole():
                        raise ctypes.WinError(ctypes.get_last_error())
                elif ctypes.get_last_error() != 6:  # ERROR_INVALID_HANDLE: child has no console
                    raise ctypes.WinError(ctypes.get_last_error())
                if attached != bool(args):
                    raise RuntimeError(f"unexpected console attachment with {args}")
            finally:
                child.terminate()


def main():
    exe = Path(sys.argv[1]).resolve(strict=True)
    if len(sys.argv) == 3 and sys.argv[2] == "--detached":
        check_windows(exe)
        return
    with exe.open("rb") as image:
        image.seek(0x3c)
        header = struct.unpack("<I", image.read(4))[0]
        image.seek(header)
        if image.read(4) != b"PE\0\0":
            raise ValueError("missing Windows PE header")
        image.seek(header + 24 + 68)
        if struct.unpack("<H", image.read(2))[0] != 2:
            raise ValueError("expected Windows GUI subsystem")
    for flag, code, stream, expected in [
        ("--help", 0, "stdout", "--console"),
        ("--version", 0, "stdout", "asterslice "),
        ("--unknown-option", 2, "stderr", "--unknown-option"),
    ]:
        result = subprocess.run([exe, flag], capture_output=True, text=True, timeout=20)
        if result.returncode != code or expected not in getattr(result, stream):
            raise RuntimeError(f"CLI result for {flag}: {result}")
    probe = subprocess.run([sys.executable, __file__, exe, "--detached"], timeout=90,
                           capture_output=True, text=True, creationflags=subprocess.DETACHED_PROCESS)
    if probe.returncode:
        raise RuntimeError(f"Detached launch probe failed: {probe.stderr}")


if __name__ == "__main__":
    main()
