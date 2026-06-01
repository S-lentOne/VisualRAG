// Learn more about Tauri commands at https://tauri.app/develop/calling-rust/
use std::io::Write;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::Mutex;
use tauri::Emitter;

static PYTHON_STDIN: std::sync::OnceLock<Mutex<std::process::ChildStdin>> =
    std::sync::OnceLock::new();

fn project_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

#[tauri::command]
fn analyze_image(path: String) -> Result<(), String> {
    let stdin = PYTHON_STDIN.get().ok_or("python not running")?;

    let mut stdin = stdin.lock().map_err(|_| "stdin lock failed")?;

    writeln!(stdin, r#"{{"command":"analyze_image","path":"{}"}}"#, path)
        .map_err(|e| e.to_string())?;

    Ok(())
}

#[tauri::command]
fn analyze_video(path: String) -> String {
    println!("VIDEO: {}", path);

    "ok".to_string()
}

#[tauri::command]
fn start_backend(app: tauri::AppHandle) -> String {
    let app_handle = app.clone();

    let root = project_root();

    #[cfg(target_os = "windows")]
    let python_path = root.join("bin/python.exe");

    #[cfg(not(target_os = "windows"))]
    let python_path = root.join("bin/python");
    let bridge_path = root.join("backend/bridge.py");
    let mut child = Command::new(python_path)
        .arg(bridge_path)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("failed to start python");

    let stdin = child.stdin.take().unwrap();

    PYTHON_STDIN
        .set(Mutex::new(stdin))
        .expect("stdin already initialized");

    let stdout = child.stdout.take().unwrap();

    std::thread::spawn(move || {
        let reader = BufReader::new(stdout);

        for line in reader.lines() {
            let payload = line.unwrap();

            if !payload.trim_start().starts_with('{') {
                println!("PYTHON LOG: {}", payload);
                continue;
            }

            let _ = app_handle.emit("backend-event", payload);
        }
    });

    "started".to_string()
}

#[tauri::command]
fn update_settings(settings: String) -> Result<(), String> {
    let stdin = PYTHON_STDIN.get().ok_or("python not running")?;

    let mut stdin = stdin.lock().map_err(|_| "stdin lock failed")?;

    writeln!(
        stdin,
        r#"{{"command":"update_settings","settings":{}}}"#,
        settings
    )
    .map_err(|e| e.to_string())?;

    Ok(())
}

#[tauri::command]
fn send_chat(message: String) -> Result<(), String> {
    let stdin = PYTHON_STDIN.get().ok_or("python not running")?;

    let mut stdin = stdin.lock().map_err(|_| "stdin lock failed")?;

    if let Err(e) = writeln!(stdin, r#"{{"message":"{}"}}"#, message) {
        return Err(e.to_string());
    }

    Ok(())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_dialog::init())
        .invoke_handler(tauri::generate_handler![
            start_backend,
            send_chat,
            analyze_image,
            analyze_video,
            update_settings
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
