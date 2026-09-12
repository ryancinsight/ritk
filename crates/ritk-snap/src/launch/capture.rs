//! Native window capture using the host's rendered-frame response.

use std::cell::Cell;
use std::path::PathBuf;
use std::rc::Rc;
use std::time::{Duration, Instant};

use crate::app::SnapApp;
use anyhow::{bail, Context, Result};

// The normal native-test slow threshold bounds an absent screenshot response.
// The outer demonstration runner independently bounds the whole process at 60s.
const RESPONSE_DEADLINE: Duration = Duration::from_secs(30);
const LOAD_DEADLINE: Duration = Duration::from_secs(30);
const LOAD_REPAINT_INTERVAL: Duration = Duration::from_millis(8);

pub(super) enum Requirement {
    Application,
    Study,
}

enum Phase {
    Draw(Instant),
    Requested(Instant),
    Finished,
}

struct Capture {
    output: PathBuf,
    requirement: Requirement,
    phase: Phase,
}

pub(super) struct CaptureApp {
    app: SnapApp,
    capture: Option<Capture>,
    // The event loop and launch function share one result on the same thread.
    completion: Rc<Cell<Option<Result<()>>>>,
}

impl CaptureApp {
    pub(super) fn new(
        app: SnapApp,
        output: Option<PathBuf>,
        requirement: Requirement,
        completion: Rc<Cell<Option<Result<()>>>>,
    ) -> Self {
        Self {
            app,
            capture: output.map(|output| Capture {
                output,
                requirement,
                phase: Phase::Draw(Instant::now()),
            }),
            completion,
        }
    }

    fn capture(&mut self, ctx: &egui::Context) -> Result<()> {
        let Some(capture) = self.capture.as_mut() else {
            return Ok(());
        };
        match capture.phase {
            Phase::Draw(start) => {
                if matches!(capture.requirement, Requirement::Study) && self.app.loaded.is_none() {
                    if !self.app.primary_load_active() {
                        bail!("initial study did not load: {}", self.app.status_message);
                    }
                    if start.elapsed() >= LOAD_DEADLINE {
                        bail!(
                            "initial study load exceeded 30-second deadline: {}",
                            self.app.status_message
                        );
                    }
                    ctx.request_repaint_after(LOAD_REPAINT_INTERVAL);
                    return Ok(());
                }
                capture.phase = Phase::Requested(Instant::now());
                ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot);
                ctx.request_repaint_after(RESPONSE_DEADLINE);
            }
            Phase::Requested(start) => {
                let screenshot = ctx.input(|input| {
                    input.events.iter().find_map(|event| {
                        if let egui::Event::Screenshot { viewport_id, image } = event {
                            (*viewport_id == egui::ViewportId::ROOT)
                                .then(|| std::sync::Arc::clone(image))
                        } else {
                            None
                        }
                    })
                });
                if let Some(image) = screenshot {
                    let rgba: Vec<_> = image
                        .pixels
                        .iter()
                        .flat_map(egui::Color32::to_srgba_unmultiplied)
                        .collect();
                    let pixels = image::RgbaImage::from_raw(
                        u32::try_from(image.width())?,
                        u32::try_from(image.height())?,
                        rgba,
                    )
                    .context("native screenshot pixel dimensions")?;
                    pixels
                        .save_with_format(&capture.output, image::ImageFormat::Png)
                        .context("write native screenshot")?;
                    capture.phase = Phase::Finished;
                    self.completion.set(Some(Ok(())));
                    ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                } else if start.elapsed() >= RESPONSE_DEADLINE {
                    bail!("native screenshot response exceeded 30-second deadline");
                }
            }
            Phase::Finished => {}
        }
        Ok(())
    }
}

impl eframe::App for CaptureApp {
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        self.app.update(ctx, frame);
        if let Err(error) = self.capture(ctx) {
            if let Some(capture) = &mut self.capture {
                capture.phase = Phase::Finished;
            }
            self.completion.set(Some(Err(error)));
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
        }
    }
}
