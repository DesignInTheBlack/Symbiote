use std::process::{Command, Stdio, Child};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use tauri::{AppHandle, Manager};
use serde_json::json;
use crate::core::system_log;
use crate::db::Db;

pub struct VoiceManager {
    process: Arc<Mutex<Option<Child>>>,
}

impl VoiceManager {
    pub fn new() -> Self {
        Self {
            process: Arc::new(Mutex::new(None)),
        }
    }

    pub fn start(&self, app_handle: &AppHandle) -> Result<(), String> {
        let started_at = Instant::now();
        let mut process_guard = self.process.lock().map_err(|e| e.to_string())?;
        
        if process_guard.is_some() {
            return Ok(()); // Already running
        }

        // Resolve path to python script or executable
        // Assuming dev environment: relative path from CWD or specific absolute path
        // In prod, this would need resource path resolution.
        // For now, assume CWD is project root.
        
        // In dev, CWD is src-tauri, so we go up one level.
        let script_path = "../voice_service_v2.py";
        
        // Kill any existing instances first (Safety)
        #[cfg(target_os = "windows")]
        let _ = Command::new("taskkill")
            .args(&["/F", "/IM", "voice_service_v2.py"]) // Logic might fail if python.exe is the name.
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn();
            
        // We actually need to kill python processes running this script.
        // But for non-intrusive start, purely relying on port usage or robust startup in python is better.
        // Python uvicorn will fail to bind port if taken.
        
        println!("[VOICE] Spawning voice_service_v2.py...");

        let mut child = Command::new("python")
            .arg(script_path)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| format!("Failed to spawn python: {}", e))?;

        let stdout = child.stdout.take();
        let stderr = child.stderr.take();

        if let Some(out) = stdout {
            std::thread::spawn(move || {
                let reader = std::io::BufReader::new(out);
                use std::io::BufRead;
                for line in reader.lines() {
                    if let Ok(l) = line {
                        println!("[VOICE-PY] {}", l);
                    }
                }
            });
        }

        if let Some(err) = stderr {
            std::thread::spawn(move || {
                let reader = std::io::BufReader::new(err);
                use std::io::BufRead;
                for line in reader.lines() {
                    if let Ok(l) = line {
                        println!("[VOICE-PY ERR] {}", l);
                    }
                }
            });
        }

        *process_guard = Some(child);
        log_voice_event(app_handle, "voice_service_start", started_at.elapsed().as_millis() as i64, None);
        Ok(())
    }

    pub fn stop(&self, app_handle: Option<&AppHandle>) {
        let started_at = Instant::now();
        let mut process_guard = match self.process.lock() {
            Ok(g) => g,
            Err(_) => return,
        };

        if let Some(mut child) = process_guard.take() {
            println!("[VOICE] Killing voice service...");
            let _ = child.kill();
        }
        if let Some(app_handle) = app_handle {
            log_voice_event(app_handle, "voice_service_stop", started_at.elapsed().as_millis() as i64, None);
        }
    }
}

fn log_voice_event(app_handle: &AppHandle, event: &str, duration_ms: i64, detail: Option<String>) {
    let db = app_handle.state::<Arc<Db>>().inner().clone();
    let event = event.to_string();
    let detail = detail.unwrap_or_default();
    tauri::async_runtime::spawn(async move {
        let _ = system_log::log_event(
            &db.pool,
            None,
            "info",
            "voice",
            None,
            None,
            json!({
                "event": event,
                "duration_ms": duration_ms,
                "detail": detail,
            }),
        )
        .await;
    });
}
