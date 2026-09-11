//! Bounded background volume loads for the viewer shell.
//!
//! The task owns only a [`VolumeInput`] and returns a decoded RITK
//! [`LoadedVolume`].  A generation and cooperative cancellation token are
//! checked at both sides of decoding; the application remains the sole owner
//! that can publish the result into viewer state.

use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

#[cfg(not(target_arch = "wasm32"))]
use crate::app::volume_input::VolumeInput;
#[cfg(not(target_arch = "wasm32"))]
use crate::LoadedVolume;
#[cfg(not(target_arch = "wasm32"))]
use anyhow::{anyhow, Result};

/// Identifies the viewer state that receives a completed load.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LoadTarget {
    Primary,
    Secondary,
}

/// Cooperative cancellation shared with one decoding task.
#[derive(Clone, Default)]
pub(crate) struct LoadCancellation(Arc<AtomicBool>);

impl LoadCancellation {
    // Release publishes cancellation before the worker's Acquire check; this
    // gives close/supersession a happens-before edge without shared locks.
    pub(crate) fn cancel(&self) {
        self.0.store(true, Ordering::Release);
    }

    pub(crate) fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }
}

/// One in-flight Moirai blocking load and its publication generation.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) struct LoadTask {
    pub(crate) target: LoadTarget,
    pub(crate) generation: u64,
    pub(crate) cancellation: LoadCancellation,
    pub(crate) handle: moirai::TaskHandle<Result<LoadedVolume>>,
}

#[cfg(not(target_arch = "wasm32"))]
impl LoadTask {
    pub(crate) fn spawn(target: LoadTarget, generation: u64, input: VolumeInput) -> Self {
        let cancellation = LoadCancellation::default();
        let worker_cancellation = cancellation.clone();
        let handle = moirai::global().spawn_blocking(move || {
            if worker_cancellation.is_cancelled() {
                return Err(anyhow!("volume load cancelled before decode"));
            }
            let result = input.load();
            if worker_cancellation.is_cancelled() {
                return Err(anyhow!("volume load cancelled after decode"));
            }
            result
        });
        Self {
            target,
            generation,
            cancellation,
            handle,
        }
    }
}
