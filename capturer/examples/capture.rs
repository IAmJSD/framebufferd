//! Example: Capture the display and save to a PNG file
//!
//! Run with: sudo cargo run --example capture
//!
//! This will capture the current display and save it as `capture.png`

use capturer::{
    available_methods, list_devices, CaptureMethod, DisplayCapture, DrmCapture,
    FramebufferCapture,
};
use image::{ImageBuffer, Rgba};
use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();
    
    // Check for --vsync flag
    let use_vsync = args.iter().any(|a| a == "--vsync" || a == "-v");
    
    // Check for --bench flag with optional count
    let bench_mode = args.iter().position(|a| a == "--bench" || a == "-b");
    let bench_count: usize = bench_mode
        .and_then(|pos| args.get(pos + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);

    // Parse command line arguments (filter out flags)
    let positional_args: Vec<&str> = args.iter()
        .skip(1)
        .filter(|a| !a.starts_with('-'))
        .map(|s| s.as_str())
        .collect();

    let method = if !positional_args.is_empty() {
        match positional_args[0] {
            "drm" => CaptureMethod::Drm,
            "fb" | "framebuffer" => CaptureMethod::Framebuffer,
            "list" => {
                println!("Available capture methods:");
                for method in available_methods() {
                    println!("  - {:?}", method);
                }
                println!("\nAvailable devices:");
                for device in list_devices() {
                    println!("  - {}", device);
                }

                // Try to get more info
                println!("\nDRM Devices:");
                for device in DrmCapture::list_devices() {
                    print!("  - {}", device);
                    match DrmCapture::open(&device) {
                        Ok(drm) => {
                            if let Ok(displays) = drm.get_display_info() {
                                for d in displays {
                                    print!(
                                        " [CRTC {} - {}x{}@{}Hz {}]",
                                        d.crtc_id, d.width, d.height, d.refresh_rate, d.mode_name
                                    );
                                }
                            }
                            println!();
                        }
                        Err(e) => println!(" (error: {})", e),
                    }
                }

                println!("\nFramebuffer Devices:");
                for device in FramebufferCapture::list_devices() {
                    print!("  - {}", device);
                    match FramebufferCapture::open(&device) {
                        Ok(fb) => {
                            let info = fb.get_info();
                            println!(
                                " [{}x{} {}bpp {:?}]",
                                info.width, info.height, info.bits_per_pixel, info.pixel_format
                            );
                        }
                        Err(e) => println!(" (error: {})", e),
                    }
                }
                return;
            }
            "help" => {
                print_help(&args[0]);
                return;
            }
            _ => CaptureMethod::Auto,
        }
    } else {
        CaptureMethod::Auto
    };

    // Handle --help flag
    if args.iter().any(|a| a == "-h" || a == "--help") {
        print_help(&args[0]);
        return;
    }

    let output_file = if positional_args.len() > 1 {
        positional_args[1]
    } else if !positional_args.is_empty() && !["drm", "fb", "framebuffer", "list", "help"].contains(&positional_args[0]) {
        positional_args[0]
    } else {
        "capture.png"
    };

    println!("Opening display capture...");

    let capture = match DisplayCapture::with_method(method) {
        Ok(cap) => {
            println!("Using capture method: {:?}", cap.method());
            cap
        }
        Err(e) => {
            eprintln!("Failed to open display: {}", e);
            eprintln!();
            eprintln!("Make sure you're running as root or have video group permissions:");
            eprintln!("  sudo cargo run --example capture");
            eprintln!("  # or");
            eprintln!("  sudo usermod -a -G video $USER  (then log out and back in)");
            std::process::exit(1);
        }
    };

    // Check for --fused flag (use fused detile+convert)
    let use_fused = args.iter().any(|a| a == "--fused" || a == "-f");

    // Benchmark mode: run multiple captures and report statistics
    if bench_mode.is_some() {
        println!("Benchmark mode: running {} captures...", bench_count);
        if use_fused {
            println!("  Using FUSED detile+convert path");
        }
        
        let mut times = Vec::with_capacity(bench_count);
        let mut rgba_times = Vec::with_capacity(bench_count);
        let mut fused_times = Vec::with_capacity(bench_count);
        
        for i in 0..bench_count {
            if use_fused {
                // Fused path: capture directly to RGBA
                let start = std::time::Instant::now();
                let result = capture.capture_rgba();
                let fused_time = start.elapsed();
                
                match result {
                    Ok((w, h, _rgba)) => {
                        fused_times.push(fused_time);
                        if i == 0 {
                            println!("  Frame: {}x{} (fused to RGBA)", w, h);
                        }
                    }
                    Err(e) => {
                        eprintln!("Capture {} failed: {}", i, e);
                    }
                }
            } else {
                // Separate path: capture then convert
                let start = std::time::Instant::now();
                let frame = if use_vsync {
                    capture.capture_vsync()
                } else {
                    capture.capture()
                };
                let frame = match frame {
                    Ok(f) => f,
                    Err(e) => {
                        eprintln!("Capture {} failed: {}", i, e);
                        continue;
                    }
                };
                let capture_time = start.elapsed();
                times.push(capture_time);
                
                // Also benchmark RGBA conversion
                let rgba_start = std::time::Instant::now();
                let _rgba = frame.to_rgba();
                let rgba_time = rgba_start.elapsed();
                rgba_times.push(rgba_time);
                
                if i == 0 {
                    println!("  Frame: {}x{}, format: {:?}", frame.width, frame.height, frame.format);
                }
            }
        }
        
        if use_fused && !fused_times.is_empty() {
            fused_times.sort();
            
            let avg: std::time::Duration = fused_times.iter().sum::<std::time::Duration>() / fused_times.len() as u32;
            let min = fused_times.first().unwrap();
            let max = fused_times.last().unwrap();
            let median = &fused_times[fused_times.len() / 2];
            
            println!("\nFused Capture+RGBA Statistics ({} samples):", fused_times.len());
            println!("  Min:    {:?}", min);
            println!("  Max:    {:?}", max);
            println!("  Avg:    {:?}", avg);
            println!("  Median: {:?}", median);
            println!("\nTheoretical max FPS: {:.1}", 1.0 / avg.as_secs_f64());
        } else if !times.is_empty() {
            times.sort();
            rgba_times.sort();
            
            let avg: std::time::Duration = times.iter().sum::<std::time::Duration>() / times.len() as u32;
            let min = times.first().unwrap();
            let max = times.last().unwrap();
            let median = &times[times.len() / 2];
            
            let rgba_avg: std::time::Duration = rgba_times.iter().sum::<std::time::Duration>() / rgba_times.len() as u32;
            let rgba_min = rgba_times.first().unwrap();
            let rgba_max = rgba_times.last().unwrap();
            let rgba_median = &rgba_times[rgba_times.len() / 2];
            
            println!("\nCapture Statistics ({} samples):", times.len());
            println!("  Min:    {:?}", min);
            println!("  Max:    {:?}", max);
            println!("  Avg:    {:?}", avg);
            println!("  Median: {:?}", median);
            
            println!("\nRGBA Conversion Statistics:");
            println!("  Min:    {:?}", rgba_min);
            println!("  Max:    {:?}", rgba_max);
            println!("  Avg:    {:?}", rgba_avg);
            println!("  Median: {:?}", rgba_median);
            
            let total_avg = avg + rgba_avg;
            println!("\nTotal (capture + RGBA): {:?}", total_avg);
            println!("Theoretical max FPS: {:.1}", 1.0 / total_avg.as_secs_f64());
        }
        
        return;
    }

    if use_vsync {
        println!("Capturing display (with VSync)...");
    } else {
        println!("Capturing display...");
    }

    let start_time = std::time::Instant::now();
    let frame = if use_vsync {
        capture.capture_vsync()
    } else {
        capture.capture()
    };
    let frame = match frame {
        Ok(f) => f,
        Err(e) => {
            eprintln!("Failed to capture: {}", e);
            std::process::exit(1);
        }
    };
    let end_time = std::time::Instant::now();
    let duration = end_time.duration_since(start_time);
    println!("Capture time: {:?}", duration);

    println!(
        "Captured frame: {}x{}, format: {:?}, pitch: {}",
        frame.width, frame.height, frame.format, frame.pitch
    );

    // Convert to RGBA and save
    println!("Converting to RGBA...");
    let rgba_data = frame.to_rgba();

    println!("Saving to {}...", output_file);
    let img: ImageBuffer<Rgba<u8>, Vec<u8>> =
        ImageBuffer::from_raw(frame.width, frame.height, rgba_data)
            .expect("Failed to create image buffer");

    img.save(output_file).expect("Failed to save image");

    println!("Done! Saved to {}", output_file);
}

fn print_help(program: &str) {
    eprintln!("Usage: {} [OPTIONS] [drm|fb|list|help] [output.png]", program);
    eprintln!();
    eprintln!("Commands:");
    eprintln!("  drm         Use DRM/KMS capture");
    eprintln!("  fb          Use framebuffer capture");
    eprintln!("  list        List available devices");
    eprintln!("  help        Show this help");
    eprintln!();
    eprintln!("Options:");
    eprintln!("  -v, --vsync       Wait for VBlank before capture (reduces tearing)");
    eprintln!("  -b, --bench [N]   Benchmark mode: run N captures (default 10) and show stats");
    eprintln!("  -f, --fused       Use fused detile+convert path (faster for NVIDIA)");
    eprintln!("  -h, --help        Show this help");
    eprintln!();
    eprintln!("Must be run as root (sudo) or with video group permissions.");
}
