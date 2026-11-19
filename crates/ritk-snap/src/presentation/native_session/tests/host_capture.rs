//! Real Windows host pointer capture through the native viewer session.

use super::super::window_controls::WindowAction;
use super::*;
use crate::presentation::PointerButton;
use crate::tools::{interaction::ToolState, kind::ToolKind};
use metis_platform::native::{
    run_native_application, NativeApplication, NativeFlow, WindowConfig, WindowEvent,
    WindowVisibility,
};
use metis_platform::Framebuffer;
use std::cell::Cell;
use std::time::Duration;
use windows::core::PCWSTR;
use windows::Win32::Foundation::{HWND, LPARAM, WPARAM};
use windows::Win32::System::SystemServices::MK_LBUTTON;
use windows::Win32::System::Threading::{GetCurrentProcessId, GetCurrentThreadId};
use windows::Win32::UI::Input::KeyboardAndMouse::GetCapture;
use windows::Win32::UI::WindowsAndMessaging::{
    FindWindowW, GetWindowThreadProcessId, SendMessageW, WM_LBUTTONDOWN, WM_LBUTTONUP,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HostStage {
    Created,
    DownQueued,
    UpQueued,
    Complete,
}

struct HostCaptureApplication {
    session: NativeViewerSession,
    title: Box<[u16]>,
    hwnd: Cell<HWND>,
    point: [i32; 2],
    stage: Cell<HostStage>,
}

impl HostCaptureApplication {
    fn new(session: NativeViewerSession, title: &str) -> Self {
        let (x, y) = session.viewports[0].center();
        Self {
            session,
            title: title
                .encode_utf16()
                .chain(std::iter::once(0))
                .collect::<Vec<_>>()
                .into_boxed_slice(),
            hwnd: Cell::new(HWND::default()),
            point: [x, y],
            stage: Cell::new(HostStage::Created),
        }
    }
}

impl NativeApplication for HostCaptureApplication {
    type Error = NativeViewerError;

    fn framebuffer(&self) -> &Framebuffer {
        assert_eq!(self.stage.get(), HostStage::Created);
        // SAFETY: the title is a live NUL-terminated UTF-16 allocation, and
        // the host has created its window before requesting this framebuffer.
        let hwnd = unsafe { FindWindowW(PCWSTR::null(), PCWSTR::from_raw(self.title.as_ptr())) }
            .expect("invariant: unique hidden host window exists before its first frame");
        assert!(
            capture_owner().is_invalid(),
            "host thread starts uncaptured"
        );
        assert_window_owner(hwnd);
        self.hwnd.set(hwnd);
        send_pointer(
            hwnd,
            WM_LBUTTONDOWN,
            WPARAM(usize::try_from(MK_LBUTTON.0).expect("button word fits usize")),
            self.point[0],
            self.point[1],
        );
        assert_eq!(capture_owner(), hwnd, "real press captures the host HWND");
        self.stage.set(HostStage::DownQueued);
        NativeApplication::framebuffer(&self.session)
    }

    fn handle_events(
        &mut self,
        events: &[WindowEvent],
    ) -> std::result::Result<NativeFlow, NativeViewerError> {
        match self.stage.get() {
            HostStage::DownQueued => {
                assert!(events.iter().any(|event| {
                    matches!(
                        event,
                        WindowEvent::PointerDown {
                            x,
                            y,
                            button: metis_platform::native::MouseButton::Left,
                        } if [*x, *y] == self.point
                    )
                }));
                let flow = self.session.handle_events(events)?;
                assert!(matches!(flow, NativeFlow::Continue { .. }));
                assert!(self.session.captured_buttons.contains(PointerButton::Left));
                assert_eq!(self.session.active_view, Some(0));
                let owner = self
                    .session
                    .app
                    .pointer
                    .owner
                    .expect("Pan press owns the viewer gesture");
                assert_eq!(owner.button, PointerButton::Left);
                assert_eq!(owner.tool, ToolKind::Pan);
                assert!(matches!(
                    self.session.app.tool_state,
                    ToolState::Panning { .. }
                ));

                send_pointer(self.hwnd.get(), WM_LBUTTONUP, WPARAM(0), -5, -10);
                assert!(
                    capture_owner().is_invalid(),
                    "outside release ends operating-system capture"
                );
                self.stage.set(HostStage::UpQueued);
                Ok(NativeFlow::Continue { repaint: false })
            }
            HostStage::UpQueued => {
                assert!(events.iter().any(|event| {
                    matches!(
                        event,
                        WindowEvent::PointerUp {
                            x: -5,
                            y: -10,
                            button: metis_platform::native::MouseButton::Left,
                        }
                    )
                }));
                let flow = self.session.handle_events(events)?;
                assert!(matches!(flow, NativeFlow::Continue { .. }));
                assert!(self.session.captured_buttons.is_empty());
                assert_eq!(self.session.active_view, None);
                assert_eq!(self.session.app.pointer.owner, None);
                assert_eq!(self.session.app.pointer.navigation_drag_viewport, None);
                assert!(self.session.app.tool_state.is_idle());
                self.stage.set(HostStage::Complete);
                Ok(NativeFlow::Exit)
            }
            stage => panic!("unexpected host stage {stage:?}"),
        }
    }
}

fn assert_window_owner(hwnd: HWND) {
    let mut process_id = 0;
    // SAFETY: `hwnd` is the live window found by this test, and the process-id
    // output points to initialized writable storage for this call.
    let thread_id = unsafe { GetWindowThreadProcessId(hwnd, Some(&mut process_id)) };
    // SAFETY: these functions read identifiers for the calling thread only.
    let expected_process = unsafe { GetCurrentProcessId() };
    // SAFETY: this function reads the identifier for the calling thread only.
    let expected_thread = unsafe { GetCurrentThreadId() };
    assert_eq!(process_id, expected_process);
    assert_eq!(thread_id, expected_thread);
}

fn pointer_lparam(x: i32, y: i32) -> LPARAM {
    let word = |value: i32| {
        u32::from(u16::from_ne_bytes(
            i16::try_from(value)
                .expect("pointer coordinate fits a signed word")
                .to_ne_bytes(),
        ))
    };
    LPARAM(isize::try_from((word(y) << 16) | word(x)).expect("coordinate pair fits isize"))
}

fn send_pointer(hwnd: HWND, message: u32, buttons: WPARAM, x: i32, y: i32) {
    // SAFETY: the message targets the live test HWND and carries only immediate
    // scalar button state and signed client coordinates.
    unsafe { SendMessageW(hwnd, message, Some(buttons), Some(pointer_lparam(x, y))) };
}

fn capture_owner() -> HWND {
    // SAFETY: `GetCapture` reads the calling thread's input state only.
    unsafe { GetCapture() }
}

#[test]
fn real_windows_host_releases_session_capture_outside_client() {
    let (mut session, _root) = session();
    session
        .apply_window_action(WindowAction::SelectTool(ToolKind::Pan))
        .expect("select Pan");
    let width = session.framebuffer.width();
    let height = session.framebuffer.height();
    // SAFETY: these functions read identifiers for the calling thread only.
    let process_id = unsafe { GetCurrentProcessId() };
    // SAFETY: this function reads the identifier for the calling thread only.
    let thread_id = unsafe { GetCurrentThreadId() };
    let title = format!("RITK host capture {process_id} {thread_id}");
    let config = WindowConfig::with_visibility(&title, width, height, WindowVisibility::Hidden)
        .expect("hidden host config");

    run_native_application(
        &config,
        HostCaptureApplication::new(session, &title),
        Duration::ZERO,
    )
    .expect("real hidden native host closes after the outside release");
}
