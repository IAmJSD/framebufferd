use std::os::unix::fs::PermissionsExt;
use std::sync::Arc;
use std::sync::mpsc::Sender;
use http_body_util::Full;
use hyper::{Request, server::conn::http2, service::service_fn};
use hyper::body::{Bytes, Incoming};
use image::ImageEncoder;
use image::codecs::png::PngEncoder;
use serde::Serialize;
use tokio::net::UnixListener;
use tokio::sync::{Mutex, RwLock};

mod prompter;
mod data_storage;

#[derive(Clone)]
pub struct TokioExecutor;

impl<F> hyper::rt::Executor<F> for TokioExecutor
where
    F: std::future::Future + Send + 'static,
    F::Output: Send + 'static,
{
    fn execute(&self, fut: F) {
        tokio::task::spawn(fut);
    }
}

async fn use_thread_pool<'a, ReturnType: Send + 'static>(handler: impl FnOnce() -> ReturnType + Send + 'a, pool: &'a rayon::ThreadPool) -> ReturnType {
    let (tx, rx) = tokio::sync::oneshot::channel();

    // SAFETY: This is safe because the termination of the spawned task is after this method.
    // The pool run will never overrun this function's lifetime.
    let handler_static = unsafe {
        std::mem::transmute::<Box<dyn FnOnce() -> ReturnType + Send + 'a>,
        Box<dyn FnOnce() -> ReturnType + Send + 'static>>(Box::new(handler))
    };

    pool.spawn(move || {
        let ret = handler_static();
        let _ = tx.send(ret);
    });

    rx.await.expect("Thread pool task panicked")
}

async fn pid_to_fp(pid: i32) -> Option<String> {
    let path = format!("/proc/{}/exe", pid);
    match tokio::fs::read_link(path).await {
        Ok(fp) => Some(fp.to_string_lossy().to_string()),
        Err(_) => None,
    }
}

async fn user_is_logged_into_desktop_environment(uid: u32) -> bool {
    let path = format!("/proc/self/loginuid");
    match tokio::fs::read_to_string(path).await {
        Ok(contents) => {
            if let Ok(loginuid) = contents.trim().parse::<u32>() {
                loginuid == uid
            } else {
                false
            }
        }
        Err(_) => false
    }
}

async fn authorize(
    pid: i32,
    uid: u32,
    prompter_send: &'static std::sync::mpsc::Sender<prompter::PrompterRequest>,
    storage: &data_storage::DataStorage,
    pool: &rayon::ThreadPool,
) -> Result<hyper::Response<Full<Bytes>>, hyper::Error> {
    let fp = match pid_to_fp(pid).await {
        Some(f) => f,
        None => {
            return Ok(
                hyper::Response::builder()
                    .status(hyper::StatusCode::INTERNAL_SERVER_ERROR)
                    .body(Full::from("Unable to determine executable path"))
                    .unwrap()
            );
        }
    };

    let user_logged_in = user_is_logged_into_desktop_environment(uid).await;
    if !user_logged_in {
        return Ok(
            hyper::Response::builder()
                .status(hyper::StatusCode::FORBIDDEN)
                .body(Full::from("User not logged into desktop environment"))
                .unwrap()
        );
    }

    let existed_in_past = match storage.check_if_path_and_uid_used_before(uid, fp.clone()).await {
        Ok(existed) => existed,
        Err(e) => {
            eprintln!("Failed to check past authorizations: {}", e);
            return Ok(
                hyper::Response::builder()
                    .status(hyper::StatusCode::INTERNAL_SERVER_ERROR)
                    .body(Full::from("Data storage error"))
                    .unwrap()
            );
        }
    };

    if !existed_in_past {
        // If it didn't exist before, prompt the user
        let (reply_tx, reply_rx) = tokio::sync::oneshot::channel();
        let prompt = prompter::PrompterRequest {
            message: format!("Authorize process {} (UID {}) to access the framebuffer? This will let this application always see what is on your screen.", fp, uid),
            reply_channel: reply_tx,
        };
        if let Err(e) = use_thread_pool(|| prompter_send.send(prompt), pool).await {
            eprintln!("Failed to send prompt to prompter thread: {}", e);
            return Ok(
                hyper::Response::builder()
                    .status(hyper::StatusCode::INTERNAL_SERVER_ERROR)
                    .body(Full::from("Prompter error"))
                    .unwrap()
            );
        }
        let authorized = match reply_rx.await {
            Ok(answer) => answer,
            Err(e) => {
                eprintln!("Failed to receive reply from prompter thread: {}", e);
                return Ok(
                    hyper::Response::builder()
                        .status(hyper::StatusCode::INTERNAL_SERVER_ERROR)
                        .body(Full::from("Prompter error"))
                        .unwrap()
                );
            }
        };
        if !authorized {
            return Ok(
                hyper::Response::builder()
                    .status(hyper::StatusCode::FORBIDDEN)
                    .body(Full::from("Unauthorized"))
                    .unwrap()
            );
        }
    }

    // Create a new token
    match storage.create_token(uid as i32, fp).await {
        Ok((token, expiration)) => {
            let body = format!("{{\"token\": \"{}\", \"expiration\": {}}}", token, expiration.and_utc().timestamp_millis());
            Ok(
                hyper::Response::builder()
                    .status(hyper::StatusCode::OK)
                    .header(hyper::header::CONTENT_TYPE, "application/json")
                    .body(Full::from(body))
                    .unwrap()
            )
        }
        Err(e) => {
            eprintln!("Failed to create token: {}", e);
            Ok(
                hyper::Response::builder()
                    .status(hyper::StatusCode::INTERNAL_SERVER_ERROR)
                    .body(Full::from("Data storage error"))
                    .unwrap()
            )
        }
    }
}

async fn patch(
    uri: &hyper::Uri,
    storage: &data_storage::DataStorage,
) -> Result<hyper::Response<Full<Bytes>>, hyper::Error> {
    // Get the token from the query parameters
    let query_pairs = uri.query().iter().
        flat_map(|q| url::form_urlencoded::parse(q.as_bytes()))
        .collect::<Vec<(std::borrow::Cow<str>, std::borrow::Cow<str>)>>();
    let token_opt = query_pairs.iter()
        .find(|(key, _)| key == "token")
        .map(|(_, value)| value.to_string());
    let token = match token_opt {
        Some(t) => t,
        None => {
            return Ok(
                hyper::Response::builder()
                    .status(hyper::StatusCode::BAD_REQUEST)
                    .body(Full::from("Missing token parameter"))
                    .unwrap()
            );
        }
    };

    // Renew the token
    match storage.renew_token_expiration(token).await {
        Ok(Some(new_expiration)) => {
            let body = format!("{{\"new_expiration\": {}}}", new_expiration.and_utc().timestamp_millis());
            Ok(
                hyper::Response::builder()
                    .status(hyper::StatusCode::OK)
                    .header(hyper::header::CONTENT_TYPE, "application/json")
                    .body(Full::from(body))
                    .unwrap()
            )
        }
        Ok(None) => {
            Ok(
                hyper::Response::builder()
                    .status(hyper::StatusCode::UNAUTHORIZED)
                    .body(Full::from("Invalid token"))
                    .unwrap()
            )
        }
        Err(e) => {
            eprintln!("Failed to renew token: {}", e);
            Ok(
                hyper::Response::builder()
                    .status(hyper::StatusCode::INTERNAL_SERVER_ERROR)
                    .body(Full::from("Data storage error"))
                    .unwrap()
            )
        }
    }
}

async fn check_authorization(
    pid: i32,
    uid: u32,
    headers: &hyper::HeaderMap,
    storage: &'static data_storage::DataStorage,
) -> Option<hyper::Response<Full<Bytes>>> {
    let fp = match pid_to_fp(pid).await {
        Some(f) => f,
        None => return Some(
            hyper::Response::builder()
                .status(hyper::StatusCode::INTERNAL_SERVER_ERROR)
                .body(Full::from("Unable to determine executable path"))
                .unwrap()
        )
    };

    let user_logged_in = user_is_logged_into_desktop_environment(uid).await;
    if !user_logged_in {
        return Some(
            hyper::Response::builder()
                .status(hyper::StatusCode::FORBIDDEN)
                .body(Full::from("User not logged into desktop environment"))
                .unwrap()
        );
    }

    let token = match headers.get("X-Auth-Token") {
        Some(t) => match t.to_str() {
            Ok(s) => s,
            Err(_) => {
                return Some(
                    hyper::Response::builder()
                        .status(hyper::StatusCode::BAD_REQUEST)
                        .body(Full::from("Invalid X-Auth-Token header"))
                        .unwrap()
                );
            }
        },
        None => {
            return Some(
                hyper::Response::builder()
                    .status(hyper::StatusCode::BAD_REQUEST)
                    .body(Full::from("Missing X-Auth-Token header"))
                    .unwrap()
            );
        }
    };

    match storage.get_token(token.to_string()).await {
        Ok(Some((stored_uid, stored_fp, expiration))) => {
            if stored_uid != uid as i32 || stored_fp != fp {
                return Some(
                    hyper::Response::builder()
                        .status(hyper::StatusCode::FORBIDDEN)
                        .body(Full::from("Unauthorized"))
                        .unwrap()
                );
            }

            let now = chrono::Utc::now().naive_utc();
            if expiration < now {
                return Some(
                    hyper::Response::builder()
                        .status(hyper::StatusCode::UNAUTHORIZED)
                        .body(Full::from("Token expired"))
                        .unwrap()
                );
            }

            None
        }
        Ok(None) => {
            Some(
                hyper::Response::builder()
                    .status(hyper::StatusCode::UNAUTHORIZED)
                    .body(Full::from("Invalid token"))
                    .unwrap()
            )
        }
        Err(e) => {
            eprintln!("Failed to get token from storage: {}", e);
            Some(
                hyper::Response::builder()
                    .status(hyper::StatusCode::INTERNAL_SERVER_ERROR)
                    .body(Full::from("Data storage error"))
                    .unwrap()
            )
        }
    }
}

fn not_found() -> hyper::Response<Full<Bytes>> {
    hyper::Response::builder()
        .status(hyper::StatusCode::NOT_FOUND)
        .body(Full::from("Not Found"))
        .unwrap()
}

fn json_response<T: Serialize>(data: T) -> hyper::Response<Full<Bytes>> {
    let body = serde_json::to_vec(&data).unwrap();
    hyper::Response::builder()
        .status(hyper::StatusCode::OK)
        .header(hyper::header::CONTENT_TYPE, "application/json")
        .body(Full::from(body))
        .unwrap()
}

#[derive(Serialize)]
struct DrmDisplay {
    pub is_drm: bool,
    #[serde(flatten)]
    pub info: capturer::drm::DisplayInfo,
    #[serde(skip_serializing)]
    pub display: Arc<capturer::drm::DrmCapture>,
    pub file_path: String,
}

#[derive(Serialize)]
struct FbDisplay {
    pub is_drm: bool,
    #[serde(flatten)]
    pub info: capturer::framebuffer::FramebufferInfo,
    #[serde(skip_serializing)]
    pub display: Arc<capturer::framebuffer::FramebufferCapture>,
    pub file_path: String,
}

#[derive(Serialize)]
#[serde(untagged)]
enum DeviceInfo {
    Drm(DrmDisplay),
    Fb(FbDisplay),
}

async fn get_displays(pool: &'static rayon::ThreadPool) -> Vec<DeviceInfo> {
    let drm_displays = tokio::spawn(async move {
        use_thread_pool(capturer::DrmCapture::list_devices, pool).await
    });
    let fb_displays = tokio::spawn(async move {
        use_thread_pool(capturer::FramebufferCapture::list_devices, pool).await
    });

    let displays = Mutex::new(Some(Vec::new()));
    let mut tasks = Vec::new();
    for drm in drm_displays.await.unwrap() {
        // SAFETY: This is fine because the spawned tasks will not outlive this function.
        let displays_mutex = unsafe {
            &*( &displays as *const Mutex<Option<Vec<DeviceInfo>>> )
        };

        tasks.push(tokio::spawn(async move {
            let remapped_displays = use_thread_pool(|| {
                let opened = capturer::DrmCapture::open(drm.clone());
                match opened {
                    Ok(capture) => {
                        let info = match capture.get_display_info() {
                            Ok(i) => i,
                            Err(e) => {
                                eprintln!("Failed to get DRM display info for {}: {}", drm, e);
                                return None;
                            }
                        };
                        let arc = Arc::new(capture);
                        let remapped_displays = info.into_iter().map(|d| {
                            DeviceInfo::Drm(DrmDisplay {
                                is_drm: true,
                                info: d,
                                display: arc.clone(),
                                file_path: drm.clone(),
                            })
                        }).collect::<Vec<DeviceInfo>>();
                        Some(remapped_displays)
                    }
                    Err(e) => {
                        eprintln!("Failed to open DRM device {}: {}", drm, e);
                        None
                    }
                }
            }, pool).await;
            if let Some(devs) = remapped_displays {
                let mut disp_lock = displays_mutex.lock().await;
                disp_lock.as_mut().unwrap().extend(devs);
            }
        }));
    }
    for fb in fb_displays.await.unwrap() {
        // SAFETY: This is fine because the spawned tasks will not outlive this function.
        let displays_mutex = unsafe {
            &*( &displays as *const Mutex<Option<Vec<DeviceInfo>>> )
        };

        tasks.push(tokio::spawn(async move {
            let display = use_thread_pool(|| {
                let opened = capturer::FramebufferCapture::open(fb.clone());
                 match opened {
                    Ok(capture) => {
                        Some(FbDisplay {
                            is_drm: false,
                            info: capture.get_info(),
                            display: Arc::new(capture),
                            file_path: fb.clone(),
                        })
                    }
                    Err(e) => {
                        eprintln!("Failed to open framebuffer device {}: {}", fb, e);
                        None
                    }
                }
            }, pool).await;
            if let Some(d) = display {
                let mut disp_lock = displays_mutex.lock().await;
                disp_lock.as_mut().unwrap().push(DeviceInfo::Fb(d));
            }
        }));
    }
    for task in tasks {
        let _ = task.await;
    }

    // Extract the displays
    let mut disp_lock = displays.lock().await;
    let value = disp_lock.take().unwrap();
    drop(disp_lock);
    drop(displays);
    value
}

async fn list_displays(pool: &'static rayon::ThreadPool) -> hyper::Response<Full<Bytes>> {
    json_response(get_displays(pool).await)
}

struct DisplaysCacher {
    displays: RwLock<Option<Vec<DeviceInfo>>>,
    pool: &'static rayon::ThreadPool,
}

impl DisplaysCacher {
    // Creates a new DisplaysCacher with an empty cache.
    pub fn new(pool: &'static rayon::ThreadPool) -> Self {
        Self {
            displays: RwLock::new(None),
            pool,
        }
    }

    // Gets the framebuffer display by ID, refreshing the cache if necessary.
    pub async fn get_fb_display_by_id(&self, id: &str) -> Option<Arc<capturer::framebuffer::FramebufferCapture>> {
        // Check the cache first
        let displays_read = self.displays.read().await;
        if let Some(displays_read) = &*displays_read {
            for d in displays_read.iter() {
                if let DeviceInfo::Fb(fb) = d {
                    if fb.info.id == id {
                        return Some(Arc::clone(&fb.display));
                    }
                }
            }
        }
        drop(displays_read);

        // Refresh the cache
        let new_displays = get_displays(self.pool).await;
        let mut result = None;
        for d in new_displays.iter() {
            if let DeviceInfo::Fb(fb) = d {
                if fb.info.id == id {
                    result = Some(Arc::clone(&fb.display));
                    break;
                }
            }
        }
        self.displays.write().await.replace(new_displays);
        result
    }

    // Gets the DRM display by DRM device path, refreshing the cache if necessary.
    // Note: A single DRM device can handle multiple displays (different fb_ids),
    // so we cache by device path, not by fb_id.
    pub async fn get_drm_display_by_path(&self, path: &str) -> Option<Arc<capturer::drm::DrmCapture>> {
        // Check the cache first
        let displays_read = self.displays.read().await;
        if let Some(displays_read) = &*displays_read {
            for d in displays_read.iter() {
                if let DeviceInfo::Drm(drm) = d {
                    if drm.file_path == path {
                        return Some(Arc::clone(&drm.display));
                    }
                }
            }
        }
        drop(displays_read);

        // Refresh the cache
        let new_displays = get_displays(self.pool).await;
        let mut result = None;
        for d in new_displays.iter() {
            if let DeviceInfo::Drm(drm) = d {
                if drm.file_path == path {
                    result = Some(Arc::clone(&drm.display));
                    break;
                }
            }
        }
        self.displays.write().await.replace(new_displays);
        result
    }
}

async fn framebuffer_init(
    uri: &hyper::Uri,
    cache: &'static DisplaysCacher,
) -> Result<Arc<capturer::framebuffer::FramebufferCapture>, hyper::Response<Full<Bytes>>> {
        // Get the query parameters
    let query_pairs = uri.query().iter().
        flat_map(|q| url::form_urlencoded::parse(q.as_bytes()))
        .collect::<Vec<(std::borrow::Cow<str>, std::borrow::Cow<str>)>>();

    // Find the "id" parameter
    let id_opt = query_pairs.iter()
        .find(|(key, _)| key == "id")
        .map(|(_, value)| value.to_string());
    let id = match id_opt {
        Some(i) => i,
        None => {
            return Err(hyper::Response::builder()
                .status(hyper::StatusCode::BAD_REQUEST)
                .body(Full::from("Missing id parameter"))
                .unwrap());
        }
    };

    // Get the framebuffer display
    let fb_display = match cache.get_fb_display_by_id(&id).await {
        Some(d) => d,
        None => {
            return Err(hyper::Response::builder()
                .status(hyper::StatusCode::NOT_FOUND)
                .body(Full::from("Framebuffer display not found"))
                .unwrap());
        }
    };

    Ok(fb_display)
}

async fn framebuffer_raw_buffer_handler(
    uri: &hyper::Uri,
    pool: &'static rayon::ThreadPool,
    cache: &'static DisplaysCacher,
) -> hyper::Response<Full<Bytes>> {
    let fb_display = match framebuffer_init(uri, cache).await {
        Ok(v) => v,
        Err(resp) => return resp,
    };

    match use_thread_pool(|| fb_display.capture(), pool).await {
        Err(e) => {
            hyper::Response::builder()
                .status(hyper::StatusCode::INTERNAL_SERVER_ERROR)
                .body(Full::from(e.to_string()))
                .unwrap()
        }
        Ok(buffer) => {
            hyper::Response::builder()
                .status(hyper::StatusCode::OK)
                .header(hyper::header::CONTENT_TYPE, "application/octet-stream")
                .header("x-format", buffer.format.to_string())
                .header("x-width", buffer.width)
                .header("x-height", buffer.height)
                .header("x-pitch", buffer.pitch)
                .body(Full::from(buffer.data))
                .unwrap()
        }
    }
}

async fn framebuffer_rgba_handler(
    uri: &hyper::Uri,
    pool: &'static rayon::ThreadPool,
    cache: &'static DisplaysCacher,
) -> hyper::Response<Full<Bytes>> {
    let fb_display = match framebuffer_init(uri, cache).await {
        Ok(v) => v,
        Err(resp) => return resp,
    };

    match use_thread_pool(|| {
        match fb_display.capture() {
            Ok(captured) => Ok((
                captured.width,
                captured.height,
                captured.to_rgba(),
            )),
            Err(e) => Err(e),
        }
    }, pool).await {
        Err(e) => {
            hyper::Response::builder()
                .status(hyper::StatusCode::INTERNAL_SERVER_ERROR)
                .body(Full::from(e.to_string()))
                .unwrap()
        }
        Ok((width, height, rgba)) => {
            hyper::Response::builder()
                .status(hyper::StatusCode::OK)
                .header(hyper::header::CONTENT_TYPE, "application/octet-stream")
                .header("x-width", width)
                .header("x-height", height)
                .body(Full::from(rgba))
                .unwrap()
        }
    }
}

async fn framebuffer_png_handler(
    uri: &hyper::Uri,
    pool: &'static rayon::ThreadPool,
    cache: &'static DisplaysCacher,
) -> hyper::Response<Full<Bytes>> {
    let fb_display = match framebuffer_init(uri, cache).await {
        Ok(v) => v,
        Err(resp) => return resp,
    };

    match use_thread_pool(|| -> Result<Vec<u8>, std::io::Error> {
        match fb_display.capture() {
            Ok(captured) => {
                let w = captured.width;
                let h = captured.height;
                let rgba = captured.to_rgba();
                let mut png_data = Vec::new();
                let encoder = PngEncoder::new(&mut png_data);
                match encoder.write_image(
                    &rgba,
                    w,
                    h,
                    image::ExtendedColorType::Rgba8,
                ) {
                    Ok(_) => Ok(png_data),
                    Err(e) => Err(
                        std::io::Error::new(std::io::ErrorKind::Other, format!("Failed to encode PNG: {}", e)),
                    ),
                }
            },
            Err(e) => {
                eprintln!("Failed to capture framebuffer: {}", e);
                Err(std::io::Error::new(std::io::ErrorKind::Other, format!("Failed to capture framebuffer: {}", e)))
            },
        }
    }, pool).await {
        Err(e) => {
            hyper::Response::builder()
                .status(hyper::StatusCode::INTERNAL_SERVER_ERROR)
                .body(Full::from(e.to_string()))
                .unwrap()
        }
        Ok(png_data) => {
            hyper::Response::builder()
                .status(hyper::StatusCode::OK)
                .header(hyper::header::CONTENT_TYPE, "image/png")
                .body(Full::from(png_data))
                .unwrap()
        }
    }
}

async fn drm_init(
    uri: &hyper::Uri,
    cache: &'static DisplaysCacher,
) -> Result<(Arc<capturer::drm::DrmCapture>, u32, bool), hyper::Response<Full<Bytes>>> {
    // Get the query parameters
    let query_pairs = uri.query().iter().
        flat_map(|q| url::form_urlencoded::parse(q.as_bytes()))
        .collect::<Vec<(std::borrow::Cow<str>, std::borrow::Cow<str>)>>();

    // Find the "fb_id" and "device_path" parameters
    let fb_id_opt = query_pairs.iter()
        .find(|(key, _)| key == "fb_id")
        .map(|(_, value)| value.to_string());
    let device_path_opt = query_pairs.iter()
        .find(|(key, _)| key == "device_path")
        .map(|(_, value)| value.to_string());
    let fb_id = match fb_id_opt {
        Some(i) => match i.parse::<u32>() {
            Ok(v) => v,
            Err(_) => {
                return Err(hyper::Response::builder()
                    .status(hyper::StatusCode::BAD_REQUEST)
                    .body(Full::from("Invalid fb_id parameter"))
                    .unwrap());
            }
        },
        None => {
            return Err(hyper::Response::builder()
                .status(hyper::StatusCode::BAD_REQUEST)
                .body(Full::from("Missing fb_id parameter"))
                .unwrap());
        }
    };
    let device_path = match device_path_opt {
        Some(p) => p,
        None => {
            return Err(hyper::Response::builder()
                .status(hyper::StatusCode::BAD_REQUEST)
                .body(Full::from("Missing device_path parameter"))
                .unwrap());
        }
    };

    // Get the "vsync" parameter (default to false)
    let vsync = query_pairs.iter()
        .find(|(key, _)| key == "vsync")
        .map(|(_, value)| value.to_string())
        .unwrap_or_else(|| "false".into()) == "true";

    // Get the DRM display by device path
    // Note: We don't need fb_id for lookup since one DrmCapture handles all displays on that device
    let drm_display = match cache.get_drm_display_by_path(&device_path).await {
        Some(d) => d,
        None => {
            return Err(hyper::Response::builder()
                .status(hyper::StatusCode::NOT_FOUND)
                .body(Full::from("DRM display not found"))
                .unwrap());
        }
    };

    Ok((drm_display, fb_id, vsync))
}

async fn drm_raw_buffer_handler(
    uri: &hyper::Uri,
    pool: &'static rayon::ThreadPool,
    cache: &'static DisplaysCacher,
) -> hyper::Response<Full<Bytes>> {
    let (drm_display, fb_id, vsync) = match drm_init(uri, cache).await {
        Ok(v) => v,
        Err(resp) => return resp,
    };

    match use_thread_pool(|| {
        if vsync {
            drm_display.capture_vsync(Some(fb_id))
        } else {
            drm_display.capture(Some(fb_id))
        }
    }, pool).await {
        Err(e) => {
            hyper::Response::builder()
                .status(hyper::StatusCode::INTERNAL_SERVER_ERROR)
                .body(Full::from(e.to_string()))
                .unwrap()
        }
        Ok(buffer) => {
            hyper::Response::builder()
                .status(hyper::StatusCode::OK)
                .header(hyper::header::CONTENT_TYPE, "application/octet-stream")
                .header("x-format", buffer.format.to_string())
                .header("x-width", buffer.width)
                .header("x-height", buffer.height)
                .header("x-pitch", buffer.pitch)
                .body(Full::from(buffer.data))
                .unwrap()
        }
    }
}

async fn drm_rgba_handler(
    uri: &hyper::Uri,
    pool: &'static rayon::ThreadPool,
    cache: &'static DisplaysCacher,
) -> hyper::Response<Full<Bytes>> {
    let (drm_display, fb_id, vsync) = match drm_init(uri, cache).await {
        Ok(v) => v,
        Err(resp) => return resp,
    };

    match use_thread_pool(|| {
        if vsync {
            match drm_display.capture_vsync(Some(fb_id)) {
                Ok(c) => Ok((c.width, c.height, c.to_rgba())),
                Err(e) => return Err(e),
            }
        } else {
            drm_display.capture_rgba(Some(fb_id))
        }
    }, pool).await {
        Err(e) => {
            hyper::Response::builder()
                .status(hyper::StatusCode::INTERNAL_SERVER_ERROR)
                .body(Full::from(e.to_string()))
                .unwrap()
        }
        Ok((width, height, rgba)) => {
            hyper::Response::builder()
                .status(hyper::StatusCode::OK)
                .header(hyper::header::CONTENT_TYPE, "application/octet-stream")
                .header("x-width", width)
                .header("x-height", height)
                .body(Full::from(rgba))
                .unwrap()
        }
    }
}

async fn drm_png_handler(
    uri: &hyper::Uri,
    pool: &'static rayon::ThreadPool,
    cache: &'static DisplaysCacher,
) -> hyper::Response<Full<Bytes>> {
    let (drm_display, fb_id, vsync) = match drm_init(uri, cache).await {
        Ok(v) => v,
        Err(resp) => return resp,
    };

    match use_thread_pool(|| {
        if vsync {
            match drm_display.capture_vsync(Some(fb_id)) {
                Ok(c) => Ok((c.width, c.height, c.to_rgba())),
                Err(e) => return Err(e),
            }
        } else {
            drm_display.capture_rgba(Some(fb_id))
        }
    }, pool).await {
        Err(e) => {
            hyper::Response::builder()
                .status(hyper::StatusCode::INTERNAL_SERVER_ERROR)
                .body(Full::from(e.to_string()))
                .unwrap()
        }
        Ok((width, height, rgba)) => {
            let mut png_data = Vec::new();
            let encoder = PngEncoder::new(&mut png_data);
            match encoder.write_image(
                &rgba,
                width,
                height,
                image::ExtendedColorType::Rgba8,
            ) {
                Ok(_) => {
                    hyper::Response::builder()
                        .status(hyper::StatusCode::OK)
                        .header(hyper::header::CONTENT_TYPE, "image/png")
                        .body(Full::from(png_data))
                        .unwrap()
                }
                Err(e) => {
                    hyper::Response::builder()
                        .status(hyper::StatusCode::INTERNAL_SERVER_ERROR)
                        .body(Full::from(e.to_string()))
                        .unwrap()
                }
            }
        }
    }
}

#[tokio::main]
async fn main() {
    // Make sure the user is running as root
    if !nix::unistd::Uid::effective().is_root() {
        eprintln!("This program must be run as root.");
        std::process::exit(1);
    }

    // Get the socket path from FRAMEBUFFERD_SOCKET_PATH or use the default (/run/framebufferd.sock)
    let socket_path = std::env::var("FRAMEBUFFERD_SOCKET_PATH").unwrap_or_else(|_| "/run/framebufferd.sock".into());

    // Make sure the directory for the socket exists
    if let Some(parent) = std::path::Path::new(&socket_path).parent() {
        if let Err(e) = std::fs::create_dir_all(parent) {
            eprintln!("Failed to create socket directory {}: {}", parent.display(), e);
            std::process::exit(1);
        }
    }

    // Create the Unix socket
    let socket = match UnixListener::bind(&socket_path) {
        Ok(s) => s,
        Err(_) => {
            // Attempt to remove the existing socket file and try again
            let _ = std::fs::remove_file(&socket_path);
            match UnixListener::bind(&socket_path) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("Failed to bind to socket: {}", e);
                    std::process::exit(1);
                }
            }
        }
    };

    // Let any user connect to the socket
    if let Err(e) = std::fs::set_permissions(&socket_path, std::fs::Permissions::from_mode(0o666)) {
        eprintln!("Failed to set socket permissions: {}", e);
        std::process::exit(1);
    }

    // On exit, remove the socket file
    let socket_path_clone = socket_path.clone();
    ctrlc::set_handler(move || {
        let _ = std::fs::remove_file(&socket_path_clone);
        std::process::exit(0);
    }).expect("Error setting Ctrl-C handler");

    // Create a prompter thread
    let prompter_send = Box::leak(Box::new(prompter::start_prompter()));

    // Create a rayon thread pool
    let pool = Box::leak(Box::new(rayon::ThreadPoolBuilder::new()
        .num_threads(num_cpus::get())
        .build()
        .unwrap()));

    // Build the data storage
    let data_storage_path = std::env::var("FRAMEBUFFERD_DATA_STORAGE_PATH").unwrap_or_else(|_| "/var/lib/framebufferd/data_storage.db".into());
    let storage = match data_storage::DataStorage::new(&data_storage_path).await {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Failed to initialize data storage: {}", e);
            std::process::exit(1);
        }
    };

    // Build the displays cacher
    let displays_cacher = Box::leak(Box::new(DisplaysCacher::new(pool)));

    println!("framebufferd is listening on {}", socket_path);
    loop {
        // Accept the incoming connection
        let (stream, _) = match socket.accept().await {
            Ok(s) => s,
            Err(e) => {
                eprintln!("Failed to accept connection: {}", e);
                continue;
            }
        };
        
        // Get the PID/UID info from the client using SO_PEERCRED
        let (pid, uid) = match nix::sys::socket::getsockopt(&stream, nix::sys::socket::sockopt::PeerCredentials) {
            Ok(creds) => {
                let pid = creds.pid();
                let uid = creds.uid();
                (pid, uid)
            }
            Err(e) => {
                eprintln!("Failed to get peer credentials: {}", e);
                continue;
            }
        };

        // Use hyper to handle the connection
        let tio = hyper_util::rt::TokioIo::new(stream);

        let sender_ref: &'static Sender<prompter::PrompterRequest> = &*prompter_send;
        let pool_ref: &'static rayon::ThreadPool = &*pool;
        let cache_ref: &'static DisplaysCacher = &*displays_cacher;
        tokio::task::spawn(async move {
            if let Err(e) = http2::Builder::new(TokioExecutor)
                .serve_connection(
                    tio,
                    service_fn(move |req: Request<Incoming>| {
                        let pid = pid;
                        let uid = uid;
                        let sender_ref = sender_ref;
                        async move {
                            match req.method() {
                                &hyper::Method::GET => {
                                    let path = req.uri().path();
                                    if path == "/authorize" {
                                        // Handle authorization request
                                        return authorize(pid, uid, sender_ref, storage, pool_ref).await;
                                    }

                                    if let Some(resp) = check_authorization(pid, uid, req.headers(), storage).await {
                                        // Application needs to be authorized
                                        return Ok(resp);
                                    }

                                    // Handle other routes
                                    match path {
                                        "/list" => Ok(list_displays(pool_ref).await),
                                        "/fb/raw_buffer" => Ok(framebuffer_raw_buffer_handler(req.uri(), pool_ref, cache_ref).await),
                                        "/fb/rgba" => Ok(framebuffer_rgba_handler(req.uri(), pool_ref, cache_ref).await),
                                        "/fb/png" => Ok(framebuffer_png_handler(req.uri(), pool_ref, cache_ref).await),
                                        "/drm/raw_buffer" => Ok(drm_raw_buffer_handler(req.uri(), pool_ref, cache_ref).await),
                                        "/drm/rgba" => Ok(drm_rgba_handler(req.uri(), pool_ref, cache_ref).await),
                                        "/drm/png" => Ok(drm_png_handler(req.uri(), pool_ref, cache_ref).await),
                                        _ => Ok(not_found())
                                    }
                                }
                                &hyper::Method::PATCH => {
                                    if req.uri().path() == "/renew" {
                                        return patch(req.uri(), storage).await
                                    }
                                    Ok(not_found())
                                }
                                _ => {
                                    // Handle other methods
                                    Ok(hyper::Response::builder()
                                        .status(hyper::StatusCode::METHOD_NOT_ALLOWED)
                                        .body(Full::from("Method Not Allowed"))
                                        .unwrap())
                                }
                            }
                        }
                    })
                )
                .await
            {
                eprintln!("Error serving connection: {}", e);
            }
        });
    }
}
