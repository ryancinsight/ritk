//! A GUI-subsystem process has no implicit console. Preserve pipes supplied by
//! its caller and create a console only for an explicit diagnostic launch.
use anyhow::Context;
use windows::core::{w, HSTRING};
use windows::Win32::System::Console::{
    AllocConsole, AttachConsole, GetConsoleCP, GetStdHandle, ATTACH_PARENT_PROCESS,
    STD_ERROR_HANDLE, STD_OUTPUT_HANDLE,
};
use windows::Win32::UI::WindowsAndMessaging::{
    MessageBoxW, MB_ICONERROR, MB_ICONINFORMATION, MB_OK,
};

pub(super) fn attach_parent() {
    // SAFETY: The constants designate this process's stdout/stderr and its parent.
    // AttachConsole never creates a window. Inherited standard handles (including
    // pipes for --version or trace output) remain owned by the caller.
    unsafe {
        if GetStdHandle(STD_OUTPUT_HANDLE).is_err() || GetStdHandle(STD_ERROR_HANDLE).is_err() {
            let _ = AttachConsole(ATTACH_PARENT_PROCESS);
        }
    }
}

pub(super) fn enable_diagnostics() -> anyhow::Result<()> {
    // SAFETY: Both calls address the current process. An existing terminal is reused.
    unsafe {
        if GetConsoleCP() == 0 && AttachConsole(ATTACH_PARENT_PROCESS).is_err() {
            AllocConsole().context("open the AsterSlice diagnostic console")?;
        }
    }
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .with_writer(std::io::stderr)
        .try_init()
        .map_err(|e| anyhow::anyhow!("initialize AsterSlice console logging: {e}"))?;
    Ok(())
}

pub(super) fn show_if_detached(message: &str, is_error: bool) {
    let (stream, icon) = if is_error {
        (STD_ERROR_HANDLE, MB_ICONERROR)
    } else {
        (STD_OUTPUT_HANDLE, MB_ICONINFORMATION)
    };
    // SAFETY: The owned UTF-16 message lives for the synchronous dialog call.
    unsafe {
        if GetStdHandle(stream).is_err() {
            let _ = MessageBoxW(
                None,
                &HSTRING::from(message),
                w!("AsterSlice"),
                MB_OK | icon,
            );
        }
    }
}
