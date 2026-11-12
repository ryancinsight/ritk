//! Packaged Métis WebView2 host; no network service or patient-data logging.
use anyhow::{bail, Context, Result};
use metis_platform::native::{
    pick, DialogSelection, WebViewConfig, WebViewEvent, WebViewHostEvent, WebViewSurface,
    WindowConfig, WindowEvent,
};
use ritk_io::format::dicom::anonymize::editor::{export_batch, EditorFields};
use serde::Deserialize;
use std::{path::PathBuf, sync::mpsc, time::Duration};

#[derive(Deserialize)]
#[serde(tag = "action", deny_unknown_fields)]
enum Command {
    #[serde(rename = "input")]
    Input,
    #[serde(rename = "output")]
    Output,
    #[serde(rename = "export")]
    Export { fields: Fields },
    #[serde(rename = "close")]
    Close,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Fields {
    patient_id: String,
    patient_name: String,
    sex: String,
    age_years: String,
    birth_date: String,
    height_cm: String,
    weight_kg: String,
    study_date: String,
    study_time: String,
    study_id: String,
    accession_number: String,
}
impl From<Fields> for EditorFields {
    fn from(f: Fields) -> Self {
        Self {
            patient_id: f.patient_id,
            patient_name: f.patient_name,
            sex: f.sex,
            age_years: f.age_years,
            birth_date: f.birth_date,
            height_cm: f.height_cm,
            weight_kg: f.weight_kg,
            study_date: f.study_date,
            study_time: f.study_time,
            study_id: f.study_id,
            accession_number: f.accession_number,
        }
    }
}

fn notify(surface: &mut WebViewSurface, kind: &str, message: &str) -> Result<()> {
    surface.post_json(serde_json::json!({"kind":kind,"message":message}).to_string())?;
    Ok(())
}

pub fn run() -> Result<()> {
    let resources = tempfile::tempdir()?;
    let page = resources.path().join("index.html");
    std::fs::write(&page, include_str!("editor.html"))?;
    let uri = url::Url::from_file_path(&page).map_err(|()| anyhow::anyhow!("invalid page path"))?;
    let mut surface = WebViewSurface::new(
        &WindowConfig::new("RITK DICOM Metadata Editor", 960, 860)?,
        WebViewConfig::new(uri.as_str())?,
    )?;
    let mut input = None::<PathBuf>;
    let mut output = None::<PathBuf>;
    let mut pending = None::<mpsc::Receiver<std::result::Result<usize, String>>>;
    let mut worker = None::<std::thread::JoinHandle<()>>;
    let loop_result = (|| -> Result<()> {
        'host: while !surface.is_closed() {
            if let Some(receiver) = &pending {
                match receiver.try_recv() {
                    Ok(result) => {
                        pending = None;
                        if let Some(handle) = worker.take() {
                            let _: std::thread::Result<()> = handle.join();
                        }
                        match result {
                            Ok(count) => notify(
                                &mut surface,
                                "done",
                                &format!(
                                    "Exported {count} files. Review pixel data before sharing."
                                ),
                            )?,
                            Err(error) => notify(&mut surface, "error", &error)?,
                        }
                    }
                    Err(mpsc::TryRecvError::Disconnected) => {
                        pending = None;
                        notify(
                            &mut surface,
                            "error",
                            "Export worker stopped. Do not use the output until inspected.",
                        )?;
                    }
                    Err(mpsc::TryRecvError::Empty) => {}
                }
            }
            for event in surface.wait_events(Duration::from_millis(50))? {
                match event {
                    WebViewHostEvent::Window(
                        WindowEvent::CloseRequested | WindowEvent::Destroyed,
                    ) => {
                        surface.close()?;
                        break 'host;
                    }
                    WebViewHostEvent::Window(WindowEvent::Resized { width, height })
                        if width > 0 && height > 0 =>
                    {
                        surface.resize(width, height)?
                    }
                    WebViewHostEvent::WebView(WebViewEvent::Message { json, .. }) => {
                        let command = match serde_json::from_str::<Command>(&json) {
                            Ok(command) => command,
                            Err(_) => {
                                notify(&mut surface, "error", "Invalid editor command")?;
                                continue;
                            }
                        };
                        if pending.is_some() {
                            notify(
                                &mut surface,
                                "busy",
                                "Export in progress; wait for completion.",
                            )?;
                            continue;
                        }
                        match command {
                            Command::Input => {
                                if let Some(path) = pick(DialogSelection::Folder)? {
                                    notify(&mut surface, "input", &path.to_string_lossy())?;
                                    input = Some(path);
                                }
                            }
                            Command::Output => {
                                if let Some(path) = pick(DialogSelection::Folder)? {
                                    let path = path.join("RITK-anonymized");
                                    notify(&mut surface, "output", &path.to_string_lossy())?;
                                    output = Some(path);
                                }
                            }
                            Command::Close => {
                                surface.close()?;
                                break 'host;
                            }
                            Command::Export { fields } => {
                                let fields = EditorFields::from(fields);
                                if let Err(error) = fields.validate() {
                                    notify(&mut surface, "error", &error.to_string())?;
                                    continue;
                                }
                                let (Some(input), Some(output)) = (input.clone(), output.clone())
                                else {
                                    notify(
                                        &mut surface,
                                        "error",
                                        "Select input and output folders first.",
                                    )?;
                                    continue;
                                };
                                let (sender, receiver) = mpsc::channel();
                                worker = Some(std::thread::spawn(move || {
                                    let result = export_directory(input, output, fields)
                                        .map_err(|error| error.to_string());
                                    let _ = sender.send(result);
                                }));
                                pending = Some(receiver);
                                notify(
                                    &mut surface,
                                    "busy",
                                    "Exporting copies; originals remain unchanged.",
                                )?;
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
        Ok(())
    })();
    // A native close waits for any in-flight export instead of killing a partially written batch.
    if let Some(handle) = worker {
        let _ = handle.join();
    }
    loop_result
}

fn export_directory(input: PathBuf, output: PathBuf, fields: EditorFields) -> Result<usize> {
    let mut files = Vec::new();
    for entry in walkdir::WalkDir::new(&input).follow_links(false) {
        let entry = entry.context("Cannot enumerate input folder")?;
        if entry.file_type().is_symlink() {
            bail!("Input contains a symbolic link; select a regular-file study folder");
        }
        if entry.file_type().is_file() {
            if entry
                .file_name()
                .to_string_lossy()
                .eq_ignore_ascii_case("DICOMDIR")
            {
                continue;
            }
            if files.len() >= 100_000 {
                bail!("Input exceeds 100,000 files");
            }
            files.push(entry.path().to_path_buf());
        }
    }
    files.sort();
    let mut random = [0u8; 32];
    getrandom::fill(&mut random)
        .map_err(|_| anyhow::anyhow!("Cannot generate a fresh UID salt"))?;
    let salt: String = random.iter().map(|byte| format!("{byte:02x}")).collect();
    Ok(export_batch(&files, &output, &fields, &salt)?.file_count)
}
