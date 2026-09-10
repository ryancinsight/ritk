//! Windows host adapter for one RITK presentation frame.

use super::PresentationFrame;
use anyhow::{anyhow, bail, Result};
use metis_platform::native::{
    run_native_application, NativeApplication, NativeFlow, WindowConfig, WindowEvent,
    WindowVisibility,
};
use metis_platform::{Color, Framebuffer};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

/// Observable result of one native frame presentation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NativeFrameOutcome {
    width: u32,
    height: u32,
    pixel_count: usize,
    frame_requests: usize,
    event_batches: usize,
}

impl NativeFrameOutcome {
    /// Width of the frame requested by the host.
    #[must_use]
    pub const fn width(self) -> u32 {
        self.width
    }

    /// Height of the frame requested by the host.
    #[must_use]
    pub const fn height(self) -> u32 {
        self.height
    }

    /// Number of RGBA pixels transferred to the host framebuffer.
    #[must_use]
    pub const fn pixel_count(self) -> usize {
        self.pixel_count
    }

    /// Number of framebuffer requests made by the host loop.
    #[must_use]
    pub const fn frame_requests(self) -> usize {
        self.frame_requests
    }

    /// Number of bounded event batches processed before exit.
    #[must_use]
    pub const fn event_batches(self) -> usize {
        self.event_batches
    }
}

/// Presents one RITK frame through the bounded native host and exits.
///
/// The host receives only pixels. DICOM loading and display mapping have
/// already completed in RITK before this function is called.
///
/// # Errors
/// Returns a frame conversion, native surface, or host transition error.
pub fn run_native_frame(frame: PresentationFrame, title: &str) -> Result<NativeFrameOutcome> {
    let framebuffer = to_framebuffer(&frame)?;
    let config = WindowConfig::with_visibility(
        title,
        frame.width(),
        frame.height(),
        WindowVisibility::Hidden,
    )?;
    let observation = Arc::new(HostObservation::default());
    run_native_application(
        &config,
        SingleFrameApplication {
            framebuffer,
            observation: Arc::clone(&observation),
        },
        Duration::ZERO,
    )
    .map_err(|error| anyhow!("RITK presentation host failed: {error}"))?;
    let frame_requests = observation.frame_requests.load(Ordering::Relaxed);
    let event_batches = observation.event_batches.load(Ordering::Relaxed);
    if observation.destroyed.load(Ordering::Relaxed) {
        bail!("native presentation surface was destroyed before close");
    }
    if frame_requests != 1 || event_batches != 1 {
        bail!(
            "native host completed an unexpected one-frame trace: {} frame requests, {} event batches",
            frame_requests,
            event_batches
        );
    }
    Ok(NativeFrameOutcome {
        width: frame.width(),
        height: frame.height(),
        pixel_count: frame.rgba().len() / 4,
        frame_requests,
        event_batches,
    })
}

fn to_framebuffer(frame: &PresentationFrame) -> Result<Framebuffer> {
    let mut framebuffer = Framebuffer::new(frame.width(), frame.height())
        .map_err(|error| anyhow!("allocate native presentation framebuffer: {error}"))?;
    let width = usize::try_from(frame.width()).map_err(|_| anyhow!("frame width exceeds usize"))?;
    for (index, pixel) in frame.rgba().chunks_exact(4).enumerate() {
        let x = i32::try_from(index % width).map_err(|_| anyhow!("frame x exceeds i32"))?;
        let y = i32::try_from(index / width).map_err(|_| anyhow!("frame y exceeds i32"))?;
        framebuffer.set_pixel(x, y, Color::rgba(pixel[0], pixel[1], pixel[2], pixel[3]));
    }
    if !frame.rgba().chunks_exact(4).remainder().is_empty() {
        return Err(anyhow!("presentation frame has an incomplete RGBA pixel"));
    }
    Ok(framebuffer)
}

struct SingleFrameApplication {
    framebuffer: Framebuffer,
    observation: Arc<HostObservation>,
}

#[derive(Default)]
struct HostObservation {
    frame_requests: AtomicUsize,
    event_batches: AtomicUsize,
    destroyed: AtomicBool,
}

impl NativeApplication for SingleFrameApplication {
    type Error = std::convert::Infallible;

    fn framebuffer(&self) -> &Framebuffer {
        self.observation
            .frame_requests
            .fetch_add(1, Ordering::Relaxed);
        &self.framebuffer
    }

    fn handle_events(
        &mut self,
        events: &[WindowEvent],
    ) -> std::result::Result<NativeFlow, Self::Error> {
        self.observation
            .event_batches
            .fetch_add(1, Ordering::Relaxed);
        if events
            .iter()
            .any(|event| matches!(event, WindowEvent::Destroyed))
        {
            self.observation.destroyed.store(true, Ordering::Relaxed);
        }
        Ok(NativeFlow::Exit)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_conversion_preserves_rgba_channels() {
        let image =
            egui::ColorImage::from_rgba_unmultiplied([2, 1], &[0, 0, 0, 0, 200, 150, 100, 255]);
        let frame = PresentationFrame::from_color_image(&image).expect("valid frame");
        let framebuffer = to_framebuffer(&frame).expect("framebuffer");
        assert_eq!(framebuffer.get_pixel(0, 0), Color::rgba(0, 0, 0, 0));
        assert_eq!(framebuffer.get_pixel(1, 0), Color::rgba(200, 150, 100, 255));
    }

    #[test]
    fn native_host_presents_one_frame_and_closes() {
        let image = egui::ColorImage::from_rgba_unmultiplied([1, 1], &[12, 34, 56, 255]);
        let frame = PresentationFrame::from_color_image(&image).expect("valid frame");
        let outcome =
            run_native_frame(frame, "RITK presentation frame test").expect("native frame");
        assert_eq!((outcome.width(), outcome.height()), (1, 1));
        assert_eq!(outcome.pixel_count(), 1);
        assert_eq!(outcome.frame_requests(), 1);
        assert_eq!(outcome.event_batches(), 1);
    }
}
