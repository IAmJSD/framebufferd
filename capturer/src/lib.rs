//! Direct Buffer Capture for Linux
//!
//! This library provides direct display buffer capture capabilities that bypass
//! Wayland and X11 entirely, accessing the display hardware at a lower level.
//!
//! # Capture Methods
//!
//! - **DRM/KMS** (`drm` module): Modern approach using the Direct Rendering Manager.
//!   Works with all modern GPUs and is the backend used by Wayland/X11 themselves.
//!   Requires root or membership in the `video` group.
//!
//! - **Framebuffer** (`framebuffer` module): Legacy Linux framebuffer interface via `/dev/fb*`.
//!   May not be available on all systems, especially those with GPU-only rendering.
//!   Requires root or membership in the `video` group.
//!
//! # Example
//!
//! ```no_run
//! use direct_buffer_capture_linux::{DisplayCapture, CapturedFrame};
//!
//! // Auto-detect and use the best available method
//! let capture = DisplayCapture::new().expect("Failed to open display");
//!
//! // Capture the screen
//! let frame = capture.capture().expect("Failed to capture");
//!
//! println!("Captured {}x{} frame", frame.width, frame.height);
//! ```
//!
//! # Permissions
//!
//! This library requires elevated permissions to access display hardware:
//!
//! - Run as root (`sudo`)
//! - Or add user to the `video` group: `sudo usermod -a -G video $USER`
//! - Or set appropriate permissions on `/dev/dri/*` or `/dev/fb*`

#[cfg(feature = "parallel")]
use rayon::prelude::*;

pub mod drm;
pub mod framebuffer;

pub use drm::{DisplayInfo, DrmCapture};
pub use framebuffer::{FramebufferCapture, FramebufferInfo};

use serde::Serialize;
use thiserror::Error;

/// Errors that can occur during capture operations
#[derive(Error, Debug)]
pub enum CaptureError {
    #[error("Invalid device path")]
    InvalidPath,

    #[error("Failed to open device: {0}")]
    DeviceOpen(#[source] std::io::Error),

    #[error("No capture device found")]
    NoDevice,

    #[error("No active display found")]
    NoActiveDisplay,

    #[error("ioctl failed: {0}")]
    IoctlFailed(&'static str),

    #[error("Failed to mmap: {0}")]
    MmapFailed(#[source] std::io::Error),

    #[error("Buffer is not mappable to userspace")]
    BufferNotMappable,

    #[error("Capture method not available: {0}")]
    MethodNotAvailable(&'static str),
}

/// Result type for capture operations
pub type Result<T> = std::result::Result<T, CaptureError>;

/// Pixel format of captured data
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum PixelFormat {
    /// BGRA, 8 bits per component (common on Linux/X11)
    Bgra8888,
    /// RGBA, 8 bits per component
    Rgba8888,
    /// BGRX (BGR with unused alpha byte)
    Bgrx8888,
    /// RGBX (RGB with unused alpha byte)
    Rgbx8888,
    /// BGR, 8 bits per component (24-bit)
    Bgr888,
    /// RGB, 8 bits per component (24-bit)
    Rgb888,
    /// RGB 565 (16-bit)
    Rgb565,
    /// Unknown format with raw bpp/fourcc value
    Unknown(u32),
}

impl ToString for PixelFormat {
    fn to_string(&self) -> String {
        match self {
            PixelFormat::Bgra8888 => "bgra8888".to_string(),
            PixelFormat::Rgba8888 => "rgba8888".to_string(),
            PixelFormat::Bgrx8888 => "bgrx8888".to_string(),
            PixelFormat::Rgbx8888 => "rgbx8888".to_string(),
            PixelFormat::Bgr888 => "bgr888".to_string(),
            PixelFormat::Rgb888 => "rgb888".to_string(),
            PixelFormat::Rgb565 => "rgb565".to_string(),
            PixelFormat::Unknown(val) => format!("unknown({})", val),
        }
    }
}

impl PixelFormat {
    /// Get bytes per pixel for this format
    pub fn bytes_per_pixel(&self) -> usize {
        match self {
            PixelFormat::Bgra8888
            | PixelFormat::Rgba8888
            | PixelFormat::Bgrx8888
            | PixelFormat::Rgbx8888 => 4,
            PixelFormat::Bgr888 | PixelFormat::Rgb888 => 3,
            PixelFormat::Rgb565 => 2,
            PixelFormat::Unknown(bpp) => (*bpp as usize + 7) / 8,
        }
    }

    /// Check if this format has an alpha channel
    pub fn has_alpha(&self) -> bool {
        matches!(self, PixelFormat::Bgra8888 | PixelFormat::Rgba8888)
    }
}

/// A captured frame from the display
#[derive(Debug, Clone)]
pub struct CapturedFrame {
    /// Width in pixels
    pub width: u32,
    /// Height in pixels
    pub height: u32,
    /// Bytes per row (may include padding)
    pub pitch: u32,
    /// Pixel format of the data
    pub format: PixelFormat,
    /// Raw pixel data
    pub data: Vec<u8>,
}

impl CapturedFrame {
    /// Get a pixel at the given coordinates
    ///
    /// Returns (R, G, B, A) tuple, converting from the native format
    pub fn get_pixel(&self, x: u32, y: u32) -> Option<(u8, u8, u8, u8)> {
        if x >= self.width || y >= self.height {
            return None;
        }

        let bpp = self.format.bytes_per_pixel();
        let offset = (y * self.pitch) as usize + (x as usize * bpp);

        if offset + bpp > self.data.len() {
            return None;
        }

        let pixel = &self.data[offset..offset + bpp];

        Some(match self.format {
            PixelFormat::Bgra8888 => {
                (pixel[2], pixel[1], pixel[0], pixel[3])
            }
            PixelFormat::Bgrx8888 => {
                (pixel[2], pixel[1], pixel[0], 255) // X means alpha is unused
            }
            PixelFormat::Rgba8888 => {
                (pixel[0], pixel[1], pixel[2], pixel[3])
            }
            PixelFormat::Rgbx8888 => {
                (pixel[0], pixel[1], pixel[2], 255) // X means alpha is unused
            }
            PixelFormat::Bgr888 => (pixel[2], pixel[1], pixel[0], 255),
            PixelFormat::Rgb888 => (pixel[0], pixel[1], pixel[2], 255),
            PixelFormat::Rgb565 => {
                let val = u16::from_le_bytes([pixel[0], pixel[1]]);
                let r = ((val >> 11) & 0x1F) as u8 * 255 / 31;
                let g = ((val >> 5) & 0x3F) as u8 * 255 / 63;
                let b = (val & 0x1F) as u8 * 255 / 31;
                (r, g, b, 255)
            }
            PixelFormat::Unknown(_) => (pixel[0], pixel.get(1).copied().unwrap_or(0), pixel.get(2).copied().unwrap_or(0), 255),
        })
    }

    /// Convert the frame to RGBA format (optimized with parallel processing)
    pub fn to_rgba(&self) -> Vec<u8> {
        let total_pixels = (self.width * self.height) as usize;
        let mut rgba = vec![0u8; total_pixels * 4];
        
        // Use optimized paths for common formats
        match self.format {
            PixelFormat::Bgrx8888 | PixelFormat::Bgra8888 => {
                self.convert_bgrx_to_rgba_parallel(&mut rgba);
            }
            PixelFormat::Rgbx8888 | PixelFormat::Rgba8888 => {
                self.convert_rgbx_to_rgba_parallel(&mut rgba);
            }
            _ => {
                // Fallback for other formats
                self.convert_generic_to_rgba(&mut rgba);
            }
        }
        
        rgba
    }
    
    /// Parallel BGRx/BGRa to RGBA conversion
    #[cfg(feature = "parallel")]
    fn convert_bgrx_to_rgba_parallel(&self, rgba: &mut [u8]) {
        let src = &self.data;
        let width = self.width as usize;
        let height = self.height as usize;
        let pitch = self.pitch as usize;
        let has_alpha = matches!(self.format, PixelFormat::Bgra8888);
        
        // Check for AVX2 for wider SIMD (8 pixels at once)
        #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
        {
            if is_x86_feature_detected!("avx2") {
                unsafe { self.convert_bgrx_to_rgba_avx2_parallel(rgba); }
                return;
            }
        }
        
        // Split into row chunks for parallel processing
        let chunk_size = 32; // Rows per chunk for good cache locality
        let _num_chunks = (height + chunk_size - 1) / chunk_size;
        
        rgba.par_chunks_mut(chunk_size * width * 4)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let start_y = chunk_idx * chunk_size;
                let end_y = (start_y + chunk_size).min(height);
                
                #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
                {
                    if is_x86_feature_detected!("ssse3") {
                        unsafe {
                            Self::convert_rows_ssse3(
                                src, chunk, width, pitch, start_y, end_y, has_alpha
                            );
                        }
                        return;
                    }
                }
                
                // Scalar fallback per chunk
                for y in start_y..end_y {
                    let src_row = y * pitch;
                    let dst_row = (y - start_y) * width * 4;
                    
                    for x in 0..width {
                        let src_off = src_row + x * 4;
                        let dst_off = dst_row + x * 4;
                        
                        if src_off + 4 <= src.len() && dst_off + 4 <= chunk.len() {
                            chunk[dst_off] = src[src_off + 2];
                            chunk[dst_off + 1] = src[src_off + 1];
                            chunk[dst_off + 2] = src[src_off];
                            chunk[dst_off + 3] = if has_alpha { src[src_off + 3] } else { 255 };
                        }
                    }
                }
            });
    }
    
    /// AVX2 parallel BGR to RGBA conversion (8 pixels at a time!)
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    #[target_feature(enable = "avx2")]
    unsafe fn convert_bgrx_to_rgba_avx2_parallel(&self, rgba: &mut [u8]) {
        #[cfg(target_arch = "x86_64")]
        use std::arch::x86_64::*;
        #[cfg(target_arch = "x86")]
        use std::arch::x86::*;
        
        let src = &self.data;
        let width = self.width as usize;
        let height = self.height as usize;
        let pitch = self.pitch as usize;
        let has_alpha = matches!(self.format, PixelFormat::Bgra8888);
        
        // AVX2 shuffle mask for 8 pixels: swap R and B in each 128-bit lane
        let shuffle = _mm256_setr_epi8(
            2, 1, 0, 3, 6, 5, 4, 7, 10, 9, 8, 11, 14, 13, 12, 15,
            2, 1, 0, 3, 6, 5, 4, 7, 10, 9, 8, 11, 14, 13, 12, 15
        );
        let alpha_mask = _mm256_set1_epi32(0xFF000000u32 as i32);
        
        let chunk_size = 32;
        
        rgba.par_chunks_mut(chunk_size * width * 4)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let start_y = chunk_idx * chunk_size;
                let end_y = (start_y + chunk_size).min(height);
                
                for y in start_y..end_y {
                    let src_row = y * pitch;
                    let dst_row = (y - start_y) * width * 4;
                    let mut x = 0;
                    
                    // Process 8 pixels (32 bytes) at a time with AVX2
                    while x + 8 <= width {
                        let src_off = src_row + x * 4;
                        let dst_off = dst_row + x * 4;
                        
                        if src_off + 32 <= src.len() && dst_off + 32 <= chunk.len() {
                            let pixels = _mm256_loadu_si256(src.as_ptr().add(src_off) as *const __m256i);
                            let swapped = _mm256_shuffle_epi8(pixels, shuffle);
                            
                            let result = if has_alpha {
                                swapped
                            } else {
                                _mm256_or_si256(swapped, alpha_mask)
                            };
                            
                            _mm256_storeu_si256(chunk.as_mut_ptr().add(dst_off) as *mut __m256i, result);
                        }
                        
                        x += 8;
                    }
                    
                    // Handle remaining with SSSE3 (4 pixels)
                    let shuffle128 = _mm_setr_epi8(2, 1, 0, 3, 6, 5, 4, 7, 10, 9, 8, 11, 14, 13, 12, 15);
                    let alpha_mask128 = _mm_set1_epi32(0xFF000000u32 as i32);
                    
                    while x + 4 <= width {
                        let src_off = src_row + x * 4;
                        let dst_off = dst_row + x * 4;
                        
                        if src_off + 16 <= src.len() && dst_off + 16 <= chunk.len() {
                            let pixels = _mm_loadu_si128(src.as_ptr().add(src_off) as *const __m128i);
                            let swapped = _mm_shuffle_epi8(pixels, shuffle128);
                            let result = if has_alpha { swapped } else { _mm_or_si128(swapped, alpha_mask128) };
                            _mm_storeu_si128(chunk.as_mut_ptr().add(dst_off) as *mut __m128i, result);
                        }
                        
                        x += 4;
                    }
                    
                    // Scalar for remainder
                    while x < width {
                        let src_off = src_row + x * 4;
                        let dst_off = dst_row + x * 4;
                        
                        if src_off + 4 <= src.len() && dst_off + 4 <= chunk.len() {
                            chunk[dst_off] = src[src_off + 2];
                            chunk[dst_off + 1] = src[src_off + 1];
                            chunk[dst_off + 2] = src[src_off];
                            chunk[dst_off + 3] = if has_alpha { src[src_off + 3] } else { 255 };
                        }
                        x += 1;
                    }
                }
            });
    }
    
    /// Helper to convert rows with SSSE3
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    #[target_feature(enable = "ssse3")]
    unsafe fn convert_rows_ssse3(
        src: &[u8], 
        chunk: &mut [u8], 
        width: usize, 
        pitch: usize, 
        start_y: usize, 
        end_y: usize, 
        has_alpha: bool
    ) {
        #[cfg(target_arch = "x86_64")]
        use std::arch::x86_64::*;
        #[cfg(target_arch = "x86")]
        use std::arch::x86::*;
        
        let shuffle = _mm_setr_epi8(2, 1, 0, 3, 6, 5, 4, 7, 10, 9, 8, 11, 14, 13, 12, 15);
        let alpha_mask = _mm_set1_epi32(0xFF000000u32 as i32);
        
        for y in start_y..end_y {
            let src_row = y * pitch;
            let dst_row = (y - start_y) * width * 4;
            let mut x = 0;
            
            while x + 4 <= width {
                let src_off = src_row + x * 4;
                let dst_off = dst_row + x * 4;
                
                if src_off + 16 <= src.len() && dst_off + 16 <= chunk.len() {
                    let pixels = _mm_loadu_si128(src.as_ptr().add(src_off) as *const __m128i);
                    let swapped = _mm_shuffle_epi8(pixels, shuffle);
                    let result = if has_alpha { swapped } else { _mm_or_si128(swapped, alpha_mask) };
                    _mm_storeu_si128(chunk.as_mut_ptr().add(dst_off) as *mut __m128i, result);
                }
                
                x += 4;
            }
            
            while x < width {
                let src_off = src_row + x * 4;
                let dst_off = dst_row + x * 4;
                
                if src_off + 4 <= src.len() && dst_off + 4 <= chunk.len() {
                    chunk[dst_off] = src[src_off + 2];
                    chunk[dst_off + 1] = src[src_off + 1];
                    chunk[dst_off + 2] = src[src_off];
                    chunk[dst_off + 3] = if has_alpha { src[src_off + 3] } else { 255 };
                }
                x += 1;
            }
        }
    }
    
    /// Sequential fallback when parallel feature is disabled
    #[cfg(not(feature = "parallel"))]
    fn convert_bgrx_to_rgba_parallel(&self, rgba: &mut [u8]) {
        self.convert_bgrx_to_rgba_fast(rgba);
    }
    
    /// Fast BGRx/BGRa to RGBA conversion using SIMD (sequential)
    #[inline]
    #[allow(dead_code)]
    fn convert_bgrx_to_rgba_fast(&self, rgba: &mut [u8]) {
        let src = &self.data;
        let width = self.width as usize;
        let height = self.height as usize;
        let pitch = self.pitch as usize;
        let has_alpha = matches!(self.format, PixelFormat::Bgra8888);
        
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("ssse3") {
                unsafe { self.convert_bgrx_to_rgba_ssse3(rgba); }
                return;
            }
        }
        
        #[cfg(target_arch = "x86")]
        {
            if is_x86_feature_detected!("ssse3") {
                unsafe { self.convert_bgrx_to_rgba_ssse3(rgba); }
                return;
            }
        }
        
        #[cfg(target_arch = "aarch64")]
        {
            unsafe { self.convert_bgrx_to_rgba_neon(rgba); }
            return;
        }
        
        // Scalar fallback
        for y in 0..height {
            let src_row = y * pitch;
            let dst_row = y * width * 4;
            
            for x in 0..width {
                let src_off = src_row + x * 4;
                let dst_off = dst_row + x * 4;
                
                if src_off + 4 <= src.len() && dst_off + 4 <= rgba.len() {
                    rgba[dst_off] = src[src_off + 2];     // R from B position
                    rgba[dst_off + 1] = src[src_off + 1]; // G
                    rgba[dst_off + 2] = src[src_off];     // B from R position
                    rgba[dst_off + 3] = if has_alpha { src[src_off + 3] } else { 255 };
                }
            }
        }
    }
    
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    #[target_feature(enable = "ssse3")]
    #[allow(dead_code)]
    unsafe fn convert_bgrx_to_rgba_ssse3(&self, rgba: &mut [u8]) {
        #[cfg(target_arch = "x86_64")]
        use std::arch::x86_64::*;
        #[cfg(target_arch = "x86")]
        use std::arch::x86::*;
        
        let src = &self.data;
        let width = self.width as usize;
        let height = self.height as usize;
        let pitch = self.pitch as usize;
        let has_alpha = matches!(self.format, PixelFormat::Bgra8888);
        
        // Shuffle mask to convert BGRx to RGBx (swap R and B)
        // Input:  B0 G0 R0 A0 B1 G1 R1 A1 B2 G2 R2 A2 B3 G3 R3 A3
        // Output: R0 G0 B0 A0 R1 G1 B1 A1 R2 G2 B2 A2 R3 G3 B3 A3
        let shuffle = _mm_setr_epi8(2, 1, 0, 3, 6, 5, 4, 7, 10, 9, 8, 11, 14, 13, 12, 15);
        let alpha_mask = _mm_set1_epi32(0xFF000000u32 as i32);
        
        for y in 0..height {
            let src_row = y * pitch;
            let dst_row = y * width * 4;
            let mut x = 0;
            
            // Process 4 pixels (16 bytes) at a time
            while x + 4 <= width {
                let src_off = src_row + x * 4;
                let dst_off = dst_row + x * 4;
                
                if src_off + 16 <= src.len() && dst_off + 16 <= rgba.len() {
                    let pixels = _mm_loadu_si128(src.as_ptr().add(src_off) as *const __m128i);
                    let swapped = _mm_shuffle_epi8(pixels, shuffle);
                    
                    // Force alpha to 255 if not using alpha channel
                    let result = if has_alpha {
                        swapped
                    } else {
                        _mm_or_si128(swapped, alpha_mask)
                    };
                    
                    _mm_storeu_si128(rgba.as_mut_ptr().add(dst_off) as *mut __m128i, result);
                }
                
                x += 4;
            }
            
            // Handle remaining pixels
            while x < width {
                let src_off = src_row + x * 4;
                let dst_off = dst_row + x * 4;
                
                if src_off + 4 <= src.len() && dst_off + 4 <= rgba.len() {
                    rgba[dst_off] = src[src_off + 2];
                    rgba[dst_off + 1] = src[src_off + 1];
                    rgba[dst_off + 2] = src[src_off];
                    rgba[dst_off + 3] = if has_alpha { src[src_off + 3] } else { 255 };
                }
                x += 1;
            }
        }
    }
    
    #[cfg(target_arch = "aarch64")]
    #[allow(dead_code)]
    unsafe fn convert_bgrx_to_rgba_neon(&self, rgba: &mut [u8]) {
        use std::arch::aarch64::*;
        
        let src = &self.data;
        let width = self.width as usize;
        let height = self.height as usize;
        let pitch = self.pitch as usize;
        let has_alpha = matches!(self.format, PixelFormat::Bgra8888);
        
        for y in 0..height {
            let src_row = y * pitch;
            let dst_row = y * width * 4;
            let mut x = 0;
            
            // Process 4 pixels (16 bytes) at a time using NEON
            while x + 4 <= width {
                let src_off = src_row + x * 4;
                let dst_off = dst_row + x * 4;
                
                if src_off + 16 <= src.len() && dst_off + 16 <= rgba.len() {
                    // Load 4 BGRx pixels
                    let bgra = vld4_u8(src.as_ptr().add(src_off));
                    
                    // Reorder: BGRA -> RGBA (swap lanes 0 and 2)
                    let result = uint8x8x4_t(
                        bgra.2, // R (was B)
                        bgra.1, // G
                        bgra.0, // B (was R)
                        if has_alpha { bgra.3 } else { vdup_n_u8(255) }, // A
                    );
                    
                    vst4_u8(rgba.as_mut_ptr().add(dst_off), result);
                }
                
                x += 4;
            }
            
            // Handle remaining pixels
            while x < width {
                let src_off = src_row + x * 4;
                let dst_off = dst_row + x * 4;
                
                if src_off + 4 <= src.len() && dst_off + 4 <= rgba.len() {
                    rgba[dst_off] = src[src_off + 2];
                    rgba[dst_off + 1] = src[src_off + 1];
                    rgba[dst_off + 2] = src[src_off];
                    rgba[dst_off + 3] = if has_alpha { src[src_off + 3] } else { 255 };
                }
                x += 1;
            }
        }
    }
    
    /// Parallel RGBx/RGBa to RGBA conversion
    #[cfg(feature = "parallel")]
    fn convert_rgbx_to_rgba_parallel(&self, rgba: &mut [u8]) {
        let src = &self.data;
        let width = self.width as usize;
        let height = self.height as usize;
        let pitch = self.pitch as usize;
        let has_alpha = matches!(self.format, PixelFormat::Rgba8888);
        let row_bytes = width * 4;
        
        let chunk_size = 32;
        
        rgba.par_chunks_mut(chunk_size * row_bytes)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let start_y = chunk_idx * chunk_size;
                let end_y = (start_y + chunk_size).min(height);
                
                for y in start_y..end_y {
                    let src_start = y * pitch;
                    let dst_start = (y - start_y) * row_bytes;
                    let copy_len = row_bytes.min(src.len().saturating_sub(src_start));
                    
                    if copy_len > 0 && dst_start + copy_len <= chunk.len() {
                        chunk[dst_start..dst_start + copy_len]
                            .copy_from_slice(&src[src_start..src_start + copy_len]);
                        
                        if !has_alpha {
                            for x in 0..width {
                                let off = dst_start + x * 4 + 3;
                                if off < chunk.len() {
                                    chunk[off] = 255;
                                }
                            }
                        }
                    }
                }
            });
    }
    
    #[cfg(not(feature = "parallel"))]
    fn convert_rgbx_to_rgba_parallel(&self, rgba: &mut [u8]) {
        self.convert_rgbx_to_rgba_fast(rgba);
    }
    
    /// Fast RGBx/RGBa to RGBA conversion (just copy, fix alpha if needed)
    #[inline]
    #[allow(dead_code)]
    fn convert_rgbx_to_rgba_fast(&self, rgba: &mut [u8]) {
        let src = &self.data;
        let width = self.width as usize;
        let height = self.height as usize;
        let pitch = self.pitch as usize;
        let has_alpha = matches!(self.format, PixelFormat::Rgba8888);
        let row_bytes = width * 4;
        
        for y in 0..height {
            let src_start = y * pitch;
            let dst_start = y * row_bytes;
            let copy_len = row_bytes.min(src.len().saturating_sub(src_start));
            
            if copy_len > 0 && dst_start + copy_len <= rgba.len() {
                rgba[dst_start..dst_start + copy_len]
                    .copy_from_slice(&src[src_start..src_start + copy_len]);
                
                // Set alpha to 255 if format doesn't have alpha
                if !has_alpha {
                    for x in 0..width {
                        let off = dst_start + x * 4 + 3;
                        if off < rgba.len() {
                            rgba[off] = 255;
                        }
                    }
                }
            }
        }
    }
    
    /// Generic conversion for other formats
    fn convert_generic_to_rgba(&self, rgba: &mut [u8]) {
        let width = self.width as usize;
        
        #[cfg(feature = "parallel")]
        {
            let pitch = self.pitch;
            let format = self.format;
            let data = &self.data;
            
            rgba.par_chunks_mut(width * 4)
                .enumerate()
                .for_each(|(y, row)| {
                    for x in 0..width {
                        let bpp = format.bytes_per_pixel();
                        let offset = (y as u32 * pitch) as usize + (x * bpp);
                        
                        if offset + bpp <= data.len() {
                            let pixel = &data[offset..offset + bpp];
                            let (r, g, b, a) = match format {
                                PixelFormat::Bgra8888 => (pixel[2], pixel[1], pixel[0], pixel[3]),
                                PixelFormat::Bgrx8888 => (pixel[2], pixel[1], pixel[0], 255),
                                PixelFormat::Rgba8888 => (pixel[0], pixel[1], pixel[2], pixel[3]),
                                PixelFormat::Rgbx8888 => (pixel[0], pixel[1], pixel[2], 255),
                                PixelFormat::Bgr888 => (pixel[2], pixel[1], pixel[0], 255),
                                PixelFormat::Rgb888 => (pixel[0], pixel[1], pixel[2], 255),
                                PixelFormat::Rgb565 => {
                                    let val = u16::from_le_bytes([pixel[0], pixel[1]]);
                                    let r = ((val >> 11) & 0x1F) as u8 * 255 / 31;
                                    let g = ((val >> 5) & 0x3F) as u8 * 255 / 63;
                                    let b = (val & 0x1F) as u8 * 255 / 31;
                                    (r, g, b, 255)
                                }
                                PixelFormat::Unknown(_) => {
                                    (pixel[0], pixel.get(1).copied().unwrap_or(0), 
                                     pixel.get(2).copied().unwrap_or(0), 255)
                                }
                            };
                            
                            let off = x * 4;
                            row[off] = r;
                            row[off + 1] = g;
                            row[off + 2] = b;
                            row[off + 3] = a;
                        }
                    }
                });
        }
        
        #[cfg(not(feature = "parallel"))]
        {
            let height = self.height as usize;
            for y in 0..height {
                for x in 0..width {
                    if let Some((r, g, b, a)) = self.get_pixel(x as u32, y as u32) {
                        let off = (y * width + x) * 4;
                        if off + 4 <= rgba.len() {
                            rgba[off] = r;
                            rgba[off + 1] = g;
                            rgba[off + 2] = b;
                            rgba[off + 3] = a;
                        }
                    }
                }
            }
        }
    }

    /// Convert the frame to RGB format (no alpha)
    pub fn to_rgb(&self) -> Vec<u8> {
        let total_pixels = (self.width * self.height) as usize;
        let mut rgb = vec![0u8; total_pixels * 3];
        
        let width = self.width as usize;
        let height = self.height as usize;

        for y in 0..height {
            for x in 0..width {
                if let Some((r, g, b, _)) = self.get_pixel(x as u32, y as u32) {
                    let off = (y * width + x) * 3;
                    if off + 3 <= rgb.len() {
                        rgb[off] = r;
                        rgb[off + 1] = g;
                        rgb[off + 2] = b;
                    }
                }
            }
        }

        rgb
    }
}

/// Capture method to use
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaptureMethod {
    /// Use DRM/KMS (Direct Rendering Manager)
    Drm,
    /// Use legacy framebuffer (/dev/fb*)
    Framebuffer,
    /// Auto-detect best available method
    Auto,
}

/// Unified display capture interface
///
/// This provides a high-level API that automatically selects the best
/// available capture method.
pub enum DisplayCapture {
    Drm(DrmCapture),
    Framebuffer(FramebufferCapture),
}

impl DisplayCapture {
    /// Create a new display capture using auto-detection
    pub fn new() -> Result<Self> {
        Self::with_method(CaptureMethod::Auto)
    }

    /// Create a new display capture with a specific method
    pub fn with_method(method: CaptureMethod) -> Result<Self> {
        match method {
            CaptureMethod::Drm => Ok(DisplayCapture::Drm(DrmCapture::open_default()?)),
            CaptureMethod::Framebuffer => {
                Ok(DisplayCapture::Framebuffer(FramebufferCapture::open_default()?))
            }
            CaptureMethod::Auto => {
                // Try DRM first (more reliable on modern systems)
                if let Ok(drm) = DrmCapture::open_default() {
                    return Ok(DisplayCapture::Drm(drm));
                }
                // Fall back to framebuffer
                if let Ok(fb) = FramebufferCapture::open_default() {
                    return Ok(DisplayCapture::Framebuffer(fb));
                }
                Err(CaptureError::NoDevice)
            }
        }
    }

    /// Capture the current display
    pub fn capture(&self) -> Result<CapturedFrame> {
        match self {
            DisplayCapture::Drm(drm) => drm.capture(None),
            DisplayCapture::Framebuffer(fb) => fb.capture(),
        }
    }

    /// Capture the current display with VSync synchronization
    ///
    /// This waits for the vertical blanking period before capturing to reduce
    /// screen tearing artifacts when capturing video or animations.
    ///
    /// Note: VSync is only supported with DRM capture. For framebuffer capture,
    /// this falls back to regular capture without synchronization.
    pub fn capture_vsync(&self) -> Result<CapturedFrame> {
        match self {
            DisplayCapture::Drm(drm) => drm.capture_vsync(None),
            DisplayCapture::Framebuffer(fb) => fb.capture(), // No vsync support for fbdev
        }
    }

    /// Get the capture method being used
    pub fn method(&self) -> CaptureMethod {
        match self {
            DisplayCapture::Drm(_) => CaptureMethod::Drm,
            DisplayCapture::Framebuffer(_) => CaptureMethod::Framebuffer,
        }
    }
    
    /// Capture directly to RGBA format in one fused operation
    /// 
    /// This is faster than capture() + to_rgba() because it combines
    /// detiling and color conversion in a single pass, avoiding an
    /// extra read/write cycle over the full frame data.
    /// 
    /// Returns (width, height, rgba_data)
    #[cfg(feature = "parallel")]
    pub fn capture_rgba(&self) -> Result<(u32, u32, Vec<u8>)> {
        match self {
            DisplayCapture::Drm(drm) => drm.capture_rgba(None),
            DisplayCapture::Framebuffer(fb) => {
                let frame = fb.capture()?;
                let rgba = frame.to_rgba();
                Ok((frame.width, frame.height, rgba))
            }
        }
    }
}

/// Check what capture methods are available on this system
pub fn available_methods() -> Vec<CaptureMethod> {
    let mut methods = Vec::new();

    if !DrmCapture::list_devices().is_empty() {
        methods.push(CaptureMethod::Drm);
    }

    if !FramebufferCapture::list_devices().is_empty() {
        methods.push(CaptureMethod::Framebuffer);
    }

    methods
}

/// List all available capture devices
pub fn list_devices() -> Vec<String> {
    let mut devices = DrmCapture::list_devices();
    devices.extend(FramebufferCapture::list_devices());
    devices
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pixel_format_bpp() {
        assert_eq!(PixelFormat::Bgra8888.bytes_per_pixel(), 4);
        assert_eq!(PixelFormat::Rgb888.bytes_per_pixel(), 3);
        assert_eq!(PixelFormat::Rgb565.bytes_per_pixel(), 2);
    }

    #[test]
    fn test_list_devices() {
        // This should not panic even without permissions
        let devices = list_devices();
        println!("Found devices: {:?}", devices);
    }
}
