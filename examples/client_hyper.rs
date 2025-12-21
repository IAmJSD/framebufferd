/// A more robust framebufferd client using hyper for HTTP/2 communication
/// This example demonstrates how to properly communicate with framebufferd
/// using the same HTTP/2 stack that the server uses.

use hyper::{Request, body::Bytes, Method, Uri};
use hyper_util::rt::TokioIo;
use http_body_util::{BodyExt, Empty};
use tokio::net::UnixStream;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct AuthResponse {
    token: String,
    expiration: i64,
}

#[derive(Debug, Deserialize)]
struct RenewResponse {
    new_expiration: i64,
}

#[derive(Debug, Deserialize)]
struct DisplayInfo {
    is_drm: bool,
    #[serde(flatten)]
    info: serde_json::Value,
}

struct FramebufferdClient {
    socket_path: String,
    token: Option<String>,
}

impl FramebufferdClient {
    pub fn new(socket_path: impl Into<String>) -> Self {
        Self {
            socket_path: socket_path.into(),
            token: None,
        }
    }

    async fn send_request(
        &self,
        method: Method,
        path: &str,
        include_auth: bool,
    ) -> Result<hyper::Response<hyper::body::Incoming>, Box<dyn std::error::Error>> {
        // Connect to Unix socket
        let stream = UnixStream::connect(&self.socket_path).await?;
        let io = TokioIo::new(stream);

        // Create HTTP/2 connection
        let (mut sender, conn) = hyper::client::conn::http2::handshake(TokioExecutor, io).await?;

        // Spawn connection task
        tokio::spawn(async move {
            if let Err(e) = conn.await {
                eprintln!("Connection error: {}", e);
            }
        });

        // Build request
        let uri = Uri::from_maybe_shared(format!("http://localhost{}", path))?;
        let mut req = Request::builder()
            .method(method)
            .uri(uri);

        if include_auth {
            if let Some(ref token) = self.token {
                req = req.header("X-Auth-Token", token);
            }
        }

        let request = req.body(Empty::<Bytes>::new())?;

        // Send request and await response
        let response = sender.send_request(request).await?;
        Ok(response)
    }

    pub async fn authorize(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let response = self.send_request(Method::GET, "/authorize", false).await?;
        
        let status = response.status();
        if status != 200 {
            let body_bytes = response.collect().await?.to_bytes();
            return Err(format!("Authorization failed: {} - {}", 
                status, 
                String::from_utf8_lossy(&body_bytes)
            ).into());
        }

        let body_bytes = response.collect().await?.to_bytes();
        let auth_response: AuthResponse = serde_json::from_slice(&body_bytes)?;
        self.token = Some(auth_response.token);
        
        println!("✓ Authorized successfully");
        println!("  Token expires at: {}", auth_response.expiration);
        Ok(())
    }

    pub async fn renew_token(&self) -> Result<i64, Box<dyn std::error::Error>> {
        let token = self.token.as_ref().ok_or("No token available")?;
        let response = self.send_request(Method::PATCH, &format!("/renew?token={}", token), false).await?;
        
        let status = response.status();
        if status != 200 {
            let body_bytes = response.collect().await?.to_bytes();
            return Err(format!("Token renewal failed: {} - {}", 
                status, 
                String::from_utf8_lossy(&body_bytes)
            ).into());
        }

        let body_bytes = response.collect().await?.to_bytes();
        let renew_response: RenewResponse = serde_json::from_slice(&body_bytes)?;
        println!("✓ Token renewed. New expiration: {}", renew_response.new_expiration);
        Ok(renew_response.new_expiration)
    }

    pub async fn list_displays(&self) -> Result<Vec<DisplayInfo>, Box<dyn std::error::Error>> {
        let response = self.send_request(Method::GET, "/list", true).await?;
        
        let status = response.status();
        if status != 200 {
            let body_bytes = response.collect().await?.to_bytes();
            return Err(format!("Failed to list displays: {} - {}", 
                status, 
                String::from_utf8_lossy(&body_bytes)
            ).into());
        }

        let body_bytes = response.collect().await?.to_bytes();
        let displays: Vec<DisplayInfo> = serde_json::from_slice(&body_bytes)?;
        Ok(displays)
    }

    pub async fn capture_fb_png(&self, id: &str) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        let response = self.send_request(Method::GET, &format!("/fb/png?id={}", id), true).await?;
        
        let status = response.status();
        if status != 200 {
            let body_bytes = response.collect().await?.to_bytes();
            return Err(format!("Failed to capture framebuffer: {} - {}", 
                status, 
                String::from_utf8_lossy(&body_bytes)
            ).into());
        }

        let body_bytes = response.collect().await?.to_bytes();
        Ok(body_bytes.to_vec())
    }

    pub async fn capture_fb_rgba(&self, id: &str) -> Result<(u32, u32, Vec<u8>), Box<dyn std::error::Error>> {
        let response = self.send_request(Method::GET, &format!("/fb/rgba?id={}", id), true).await?;
        
        let status = response.status();
        if status != 200 {
            let body_bytes = response.collect().await?.to_bytes();
            return Err(format!("Failed to capture framebuffer: {} - {}", 
                status, 
                String::from_utf8_lossy(&body_bytes)
            ).into());
        }

        let headers = response.headers();
        let width = headers.get("x-width")
            .ok_or("Missing x-width header")?
            .to_str()?
            .parse::<u32>()?;
        let height = headers.get("x-height")
            .ok_or("Missing x-height header")?
            .to_str()?
            .parse::<u32>()?;

        let body_bytes = response.collect().await?.to_bytes();
        Ok((width, height, body_bytes.to_vec()))
    }

    pub async fn capture_drm_png(&self, fb_id: u32, device_path: &str, vsync: bool) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        let vsync_param = if vsync { "true" } else { "false" };
        let encoded_path = urlencoding::encode(device_path);
        let response = self.send_request(
            Method::GET,
            &format!("/drm/png?fb_id={}&device_path={}&vsync={}", fb_id, encoded_path, vsync_param),
            true
        ).await?;
        
        let status = response.status();
        if status != 200 {
            let body_bytes = response.collect().await?.to_bytes();
            return Err(format!("Failed to capture DRM: {} - {}", 
                status, 
                String::from_utf8_lossy(&body_bytes)
            ).into());
        }

        let body_bytes = response.collect().await?.to_bytes();
        Ok(body_bytes.to_vec())
    }

    pub async fn capture_drm_rgba(&self, fb_id: u32, device_path: &str, vsync: bool) -> Result<(u32, u32, Vec<u8>), Box<dyn std::error::Error>> {
        let vsync_param = if vsync { "true" } else { "false" };
        let encoded_path = urlencoding::encode(device_path);
        let response = self.send_request(
            Method::GET,
            &format!("/drm/rgba?fb_id={}&device_path={}&vsync={}", fb_id, encoded_path, vsync_param),
            true
        ).await?;
        
        let status = response.status();
        if status != 200 {
            let body_bytes = response.collect().await?.to_bytes();
            return Err(format!("Failed to capture DRM: {} - {}", 
                status, 
                String::from_utf8_lossy(&body_bytes)
            ).into());
        }

        let headers = response.headers();
        let width = headers.get("x-width")
            .ok_or("Missing x-width header")?
            .to_str()?
            .parse::<u32>()?;
        let height = headers.get("x-height")
            .ok_or("Missing x-height header")?
            .to_str()?
            .parse::<u32>()?;

        let body_bytes = response.collect().await?.to_bytes();
        Ok((width, height, body_bytes.to_vec()))
    }
}

#[derive(Clone)]
struct TokioExecutor;

impl<F> hyper::rt::Executor<F> for TokioExecutor
where
    F: std::future::Future + Send + 'static,
    F::Output: Send + 'static,
{
    fn execute(&self, fut: F) {
        tokio::task::spawn(fut);
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Get socket path from environment or use default
    let socket_path = std::env::var("FRAMEBUFFERD_SOCKET_PATH")
        .unwrap_or_else(|_| "/run/framebufferd.sock".into());

    println!("=== framebufferd Client Example ===\n");
    println!("Connecting to: {}\n", socket_path);

    // Create client
    let mut client = FramebufferdClient::new(socket_path);

    // Step 1: Authorize
    println!("Step 1: Authorizing...");
    client.authorize().await?;

    // Step 2: List displays
    println!("\nStep 2: Listing displays...");
    let displays = client.list_displays().await?;
    println!("✓ Found {} display(s):", displays.len());
    for (i, display) in displays.iter().enumerate() {
        if display.is_drm {
            let fb_id = display.info.get("fb_id").and_then(|v| v.as_u64()).unwrap_or(0);
            let width = display.info.get("width").and_then(|v| v.as_u64()).unwrap_or(0);
            let height = display.info.get("height").and_then(|v| v.as_u64()).unwrap_or(0);
            let device_path = display.info.get("file_path").and_then(|v| v.as_str()).unwrap_or("unknown");
            println!("  [{}] DRM Display (fb_id={}, {}x{}, device={})", i, fb_id, width, height, device_path);
        } else {
            let id = display.info.get("id").and_then(|v| v.as_str()).unwrap_or("unknown");
            let width = display.info.get("width").and_then(|v| v.as_u64()).unwrap_or(0);
            let height = display.info.get("height").and_then(|v| v.as_u64()).unwrap_or(0);
            println!("  [{}] Framebuffer Display (id={}, {}x{})", i, id, width, height);
        }
    }

    if displays.is_empty() {
        println!("\n⚠ No displays found. Make sure you have framebuffer or DRM devices available.");
        return Ok(());
    }

    // Step 3: Capture screenshots from all displays
    println!("\nStep 3: Capturing screenshots from all displays...");
    for (i, display) in displays.iter().enumerate() {
        if display.is_drm {
            let fb_id = display.info["fb_id"].as_u64().ok_or("Missing fb_id")? as u32;
            let device_path = display.info["file_path"].as_str().ok_or("Missing file_path")?;
            
            println!("  [{}] Capturing DRM display (fb_id={})...", i, fb_id);
            let png_data = client.capture_drm_png(fb_id, device_path, false).await?;
            
            let output_path = format!("screenshot_drm_{}.png", i);
            tokio::fs::write(&output_path, &png_data).await?;
            println!("      ✓ Saved to: {} ({} bytes)", output_path, png_data.len());
        } else {
            let id = display.info["id"].as_str().ok_or("Missing id")?;
            
            println!("  [{}] Capturing framebuffer display (id={})...", i, id);
            let png_data = client.capture_fb_png(id).await?;
            
            let output_path = format!("screenshot_fb_{}.png", i);
            tokio::fs::write(&output_path, &png_data).await?;
            println!("      ✓ Saved to: {} ({} bytes)", output_path, png_data.len());
        }
    }

    // Step 4: Capture RGBA data from first display
    println!("\nStep 4: Capturing raw RGBA data from first display...");
    let first_display = &displays[0];
    if first_display.is_drm {
        let fb_id = first_display.info["fb_id"].as_u64().ok_or("Missing fb_id")? as u32;
        let device_path = first_display.info["file_path"].as_str().ok_or("Missing file_path")?;
        
        let (width, height, rgba_data) = client.capture_drm_rgba(fb_id, device_path, false).await?;
        println!("✓ Captured RGBA data: {}x{} ({} bytes, {} pixels)", 
            width, height, rgba_data.len(), width * height);
    } else {
        let id = first_display.info["id"].as_str().ok_or("Missing id")?;
        
        let (width, height, rgba_data) = client.capture_fb_rgba(id).await?;
        println!("✓ Captured RGBA data: {}x{} ({} bytes, {} pixels)", 
            width, height, rgba_data.len(), width * height);
    }

    // Step 5: Token renewal
    println!("\nStep 5: Renewing authentication token...");
    client.renew_token().await?;

    println!("\n=== All operations completed successfully! ===");
    Ok(())
}
