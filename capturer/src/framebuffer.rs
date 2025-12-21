//! Linux framebuffer (fbdev) capture module
//!
//! This module provides direct access to the Linux framebuffer device (/dev/fb*)
//! for screen capture. This is the legacy interface but still works on many systems.

use crate::{CaptureError, CapturedFrame, PixelFormat, Result};
use libc::{c_int, c_ulong, c_void, close, ioctl, mmap, munmap, open, O_RDONLY};
use std::ffi::CString;
use std::fs;
use std::path::Path;
use std::ptr;
use serde::Serialize;

// Framebuffer ioctl definitions
const FBIOGET_VSCREENINFO: c_ulong = 0x4600;
const FBIOGET_FSCREENINFO: c_ulong = 0x4602;

/// Variable screen info structure
#[repr(C)]
#[derive(Default, Debug, Clone)]
struct FbVarScreenInfo {
    xres: u32,
    yres: u32,
    xres_virtual: u32,
    yres_virtual: u32,
    xoffset: u32,
    yoffset: u32,
    bits_per_pixel: u32,
    grayscale: u32,
    // Bitfield info for each color
    red: FbBitfield,
    green: FbBitfield,
    blue: FbBitfield,
    transp: FbBitfield,
    nonstd: u32,
    activate: u32,
    height: u32,  // height in mm
    width: u32,   // width in mm
    accel_flags: u32,
    // Timing info
    pixclock: u32,
    left_margin: u32,
    right_margin: u32,
    upper_margin: u32,
    lower_margin: u32,
    hsync_len: u32,
    vsync_len: u32,
    sync: u32,
    vmode: u32,
    rotate: u32,
    colorspace: u32,
    reserved: [u32; 4],
}

#[repr(C)]
#[derive(Default, Debug, Clone, Copy)]
struct FbBitfield {
    offset: u32,
    length: u32,
    msb_right: u32,
}

/// Fixed screen info structure
#[repr(C)]
#[derive(Debug, Clone)]
struct FbFixScreenInfo {
    id: [u8; 16],
    smem_start: c_ulong,
    smem_len: u32,
    type_: u32,
    type_aux: u32,
    visual: u32,
    xpanstep: u16,
    ypanstep: u16,
    ywrapstep: u16,
    _pad: u16,
    line_length: u32,
    mmio_start: c_ulong,
    mmio_len: u32,
    accel: u32,
    capabilities: u16,
    reserved: [u16; 2],
}

impl Default for FbFixScreenInfo {
    fn default() -> Self {
        Self {
            id: [0; 16],
            smem_start: 0,
            smem_len: 0,
            type_: 0,
            type_aux: 0,
            visual: 0,
            xpanstep: 0,
            ypanstep: 0,
            ywrapstep: 0,
            _pad: 0,
            line_length: 0,
            mmio_start: 0,
            mmio_len: 0,
            accel: 0,
            capabilities: 0,
            reserved: [0; 2],
        }
    }
}

/// Framebuffer capture device
pub struct FramebufferCapture {
    fd: c_int,
    var_info: FbVarScreenInfo,
    fix_info: FbFixScreenInfo,
    mapped_ptr: *mut u8,
    mapped_size: usize,
}

impl FramebufferCapture {
    /// Open a framebuffer device for capture
    ///
    /// Typically `/dev/fb0` is the primary display
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path_str = path.as_ref().to_str().ok_or(CaptureError::InvalidPath)?;
        let c_path = CString::new(path_str).map_err(|_| CaptureError::InvalidPath)?;

        let fd = unsafe { open(c_path.as_ptr(), O_RDONLY) };
        if fd < 0 {
            return Err(CaptureError::DeviceOpen(std::io::Error::last_os_error()));
        }

        let mut var_info = FbVarScreenInfo::default();
        let mut fix_info = FbFixScreenInfo::default();

        // Get variable screen info
        if unsafe { ioctl(fd, FBIOGET_VSCREENINFO, &mut var_info) } != 0 {
            unsafe { close(fd) };
            return Err(CaptureError::IoctlFailed("FBIOGET_VSCREENINFO"));
        }

        // Get fixed screen info
        if unsafe { ioctl(fd, FBIOGET_FSCREENINFO, &mut fix_info) } != 0 {
            unsafe { close(fd) };
            return Err(CaptureError::IoctlFailed("FBIOGET_FSCREENINFO"));
        }

        let mapped_size = fix_info.smem_len as usize;

        // Memory map the framebuffer
        let mapped_ptr = unsafe {
            mmap(
                ptr::null_mut(),
                mapped_size,
                libc::PROT_READ,
                libc::MAP_SHARED,
                fd,
                0,
            )
        };

        if mapped_ptr == libc::MAP_FAILED {
            unsafe { close(fd) };
            return Err(CaptureError::MmapFailed(std::io::Error::last_os_error()));
        }

        Ok(Self {
            fd,
            var_info,
            fix_info,
            mapped_ptr: mapped_ptr as *mut u8,
            mapped_size,
        })
    }

    /// Find and open the first available framebuffer device
    pub fn open_default() -> Result<Self> {
        // Try fb0 through fb7
        for i in 0..8 {
            let path = format!("/dev/fb{}", i);
            if Path::new(&path).exists() {
                match Self::open(&path) {
                    Ok(cap) => return Ok(cap),
                    Err(_) => continue,
                }
            }
        }
        Err(CaptureError::NoDevice)
    }

    /// List available framebuffer devices
    pub fn list_devices() -> Vec<String> {
        let mut devices = Vec::new();
        for i in 0..8 {
            let path = format!("/dev/fb{}", i);
            if Path::new(&path).exists() {
                devices.push(path);
            }
        }
        // Also check /sys/class/graphics
        if let Ok(entries) = fs::read_dir("/sys/class/graphics") {
            for entry in entries.flatten() {
                let name = entry.file_name().to_string_lossy().to_string();
                if name.starts_with("fb") {
                    let path = format!("/dev/{}", name);
                    if !devices.contains(&path) && Path::new(&path).exists() {
                        devices.push(path);
                    }
                }
            }
        }
        devices.sort();
        devices
    }

    /// Get the width of the framebuffer in pixels
    pub fn width(&self) -> u32 {
        self.var_info.xres
    }

    /// Get the height of the framebuffer in pixels
    pub fn height(&self) -> u32 {
        self.var_info.yres
    }

    /// Get the bits per pixel
    pub fn bits_per_pixel(&self) -> u32 {
        self.var_info.bits_per_pixel
    }

    /// Get the line length (pitch) in bytes
    pub fn pitch(&self) -> u32 {
        self.fix_info.line_length
    }

    /// Get the pixel format based on framebuffer configuration
    pub fn pixel_format(&self) -> PixelFormat {
        let var = &self.var_info;

        match var.bits_per_pixel {
            32 => {
                // Check component order based on offsets
                if var.red.offset == 16 && var.green.offset == 8 && var.blue.offset == 0 {
                    if var.transp.length > 0 {
                        PixelFormat::Bgra8888
                    } else {
                        PixelFormat::Bgrx8888
                    }
                } else if var.red.offset == 0 && var.green.offset == 8 && var.blue.offset == 16 {
                    if var.transp.length > 0 {
                        PixelFormat::Rgba8888
                    } else {
                        PixelFormat::Rgbx8888
                    }
                } else {
                    PixelFormat::Unknown(32)
                }
            }
            24 => {
                if var.red.offset == 16 {
                    PixelFormat::Bgr888
                } else {
                    PixelFormat::Rgb888
                }
            }
            16 => PixelFormat::Rgb565,
            _ => PixelFormat::Unknown(var.bits_per_pixel),
        }
    }

    /// Capture the current framebuffer contents
    pub fn capture(&self) -> Result<CapturedFrame> {
        // Refresh var_info in case display mode changed
        let mut var_info = FbVarScreenInfo::default();
        if unsafe { ioctl(self.fd, FBIOGET_VSCREENINFO, &mut var_info) } != 0 {
            return Err(CaptureError::IoctlFailed("FBIOGET_VSCREENINFO"));
        }

        let width = var_info.xres;
        let height = var_info.yres;
        let pitch = self.fix_info.line_length;

        // Calculate offset for virtual framebuffer (handles double buffering)
        let y_offset = var_info.yoffset;
        let start_offset = (y_offset * pitch) as usize;

        let frame_size = (pitch * height) as usize;
        let mut data = vec![0u8; frame_size];

        // Copy from mapped memory
        unsafe {
            let src = self.mapped_ptr.add(start_offset);
            ptr::copy_nonoverlapping(src, data.as_mut_ptr(), frame_size.min(self.mapped_size - start_offset));
        }

        Ok(CapturedFrame {
            width,
            height,
            pitch,
            format: self.pixel_format(),
            data,
        })
    }

    /// Get detailed information about the framebuffer
    pub fn get_info(&self) -> FramebufferInfo {
        let id = self
            .fix_info
            .id
            .iter()
            .take_while(|&&c| c != 0)
            .map(|&c| c as char)
            .collect();

        FramebufferInfo {
            id,
            width: self.var_info.xres,
            height: self.var_info.yres,
            virtual_width: self.var_info.xres_virtual,
            virtual_height: self.var_info.yres_virtual,
            bits_per_pixel: self.var_info.bits_per_pixel,
            pitch: self.fix_info.line_length,
            memory_size: self.fix_info.smem_len,
            pixel_format: self.pixel_format(),
        }
    }
}

impl Drop for FramebufferCapture {
    fn drop(&mut self) {
        if !self.mapped_ptr.is_null() {
            unsafe {
                munmap(self.mapped_ptr as *mut c_void, self.mapped_size);
            }
        }
        unsafe {
            close(self.fd);
        }
    }
}

// Safety: The file descriptor and mapped memory are thread-safe for reading
unsafe impl Send for FramebufferCapture {}
unsafe impl Sync for FramebufferCapture {}

/// Detailed framebuffer information
#[derive(Debug, Clone, Serialize)]
pub struct FramebufferInfo {
    pub id: String,
    pub width: u32,
    pub height: u32,
    pub virtual_width: u32,
    pub virtual_height: u32,
    pub bits_per_pixel: u32,
    pub pitch: u32,
    pub memory_size: u32,
    pub pixel_format: PixelFormat,
}
