//! DRM/KMS capture module - direct GPU framebuffer access
//!
//! This module uses the Direct Rendering Manager (DRM) subsystem to capture
//! the display buffer directly from the GPU, bypassing any display server.

use crate::{CaptureError, CapturedFrame, PixelFormat, Result};
use libc::{c_int, c_ulong, c_void, close, ioctl, mmap, munmap, open, O_RDWR};
use std::ffi::CString;
use std::fs;
use std::path::Path;
use std::ptr;
use std::sync::atomic::{AtomicUsize, Ordering};
use serde::Serialize;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

// DRM ioctl definitions
const DRM_IOCTL_BASE: c_ulong = 'd' as c_ulong;

macro_rules! drm_io {
    ($nr:expr) => {
        (2u64 << 30) | (DRM_IOCTL_BASE << 8) | ($nr as c_ulong)
    };
}

macro_rules! drm_iowr {
    ($nr:expr, $sz:expr) => {
        (3u64 << 30) | (($sz as c_ulong) << 16) | (DRM_IOCTL_BASE << 8) | ($nr as c_ulong)
    };
}

// DRM ioctl numbers
const DRM_IOCTL_SET_MASTER: c_ulong = drm_io!(0x1e);
const DRM_IOCTL_DROP_MASTER: c_ulong = drm_io!(0x1f);
const DRM_IOCTL_WAIT_VBLANK: c_ulong = drm_iowr!(0x3a, 24); // union drm_wait_vblank
const DRM_IOCTL_MODE_GETRESOURCES: c_ulong = drm_iowr!(0xA0, 64);
const DRM_IOCTL_MODE_GETCRTC: c_ulong = drm_iowr!(0xA1, 64);
const DRM_IOCTL_MODE_GETFB: c_ulong = drm_iowr!(0xAD, 24);
const DRM_IOCTL_MODE_GETFB2: c_ulong = drm_iowr!(0xCE, 128);
const DRM_IOCTL_MODE_MAP_DUMB: c_ulong = drm_iowr!(0xB3, 16);
const DRM_IOCTL_PRIME_HANDLE_TO_FD: c_ulong = drm_iowr!(0x2d, 12);

// VBlank sequence types
const DRM_VBLANK_RELATIVE: u32 = 0x1;
const DRM_VBLANK_NEXTONMISS: u32 = 0x10000000;
const DRM_VBLANK_HIGH_CRTC_SHIFT: u32 = 1;

// DRM mode resources struct
#[repr(C)]
#[derive(Default, Debug)]
struct DrmModeResources {
    fb_id_ptr: u64,
    crtc_id_ptr: u64,
    connector_id_ptr: u64,
    encoder_id_ptr: u64,
    count_fbs: u32,
    count_crtcs: u32,
    count_connectors: u32,
    count_encoders: u32,
    min_width: u32,
    max_width: u32,
    min_height: u32,
    max_height: u32,
}

// DRM CRTC struct
#[repr(C)]
#[derive(Default, Debug)]
struct DrmModeCrtc {
    set_connectors_ptr: u64,
    count_connectors: u32,
    crtc_id: u32,
    fb_id: u32,
    x: u32,
    y: u32,
    gamma_size: u32,
    mode_valid: u32,
    mode: DrmModeInfo,
}

#[repr(C)]
#[derive(Default, Debug, Clone, Copy)]
struct DrmModeInfo {
    clock: u32,
    hdisplay: u16,
    hsync_start: u16,
    hsync_end: u16,
    htotal: u16,
    hskew: u16,
    vdisplay: u16,
    vsync_start: u16,
    vsync_end: u16,
    vtotal: u16,
    vscan: u16,
    vrefresh: u32,
    flags: u32,
    type_: u32,
    name: [u8; 32],
}

// DRM framebuffer struct
#[repr(C)]
#[derive(Default, Debug)]
struct DrmModeFb {
    fb_id: u32,
    width: u32,
    height: u32,
    pitch: u32,
    bpp: u32,
    depth: u32,
    handle: u32,
}

// DRM framebuffer2 struct (newer, supports modifiers)
#[repr(C)]
#[derive(Default, Debug)]
struct DrmModeFb2 {
    fb_id: u32,
    width: u32,
    height: u32,
    pixel_format: u32,
    flags: u32,
    handles: [u32; 4],
    pitches: [u32; 4],
    offsets: [u32; 4],
    modifier: [u64; 4],
}

// DRM map dumb buffer struct
#[repr(C)]
#[derive(Default, Debug)]
struct DrmModeMapDumb {
    handle: u32,
    pad: u32,
    offset: u64,
}

// DRM prime handle to fd struct
#[repr(C)]
#[derive(Default, Debug)]
struct DrmPrimeHandle {
    handle: u32,
    flags: u32,
    fd: i32,
}

// DRM wait vblank request/reply union
// This must match the kernel's union drm_wait_vblank layout
#[repr(C)]
#[derive(Default)]
struct DrmWaitVblank {
    request_type: u32,
    sequence: u32,
    signal: u64, // unsigned long on 64-bit
}

/// Cached display state to avoid repeated ioctls
#[derive(Clone, Default)]
struct CachedDisplayState {
    #[allow(dead_code)]
    crtc_id: u32,
    fb_id: u32,
    width: u32,
    height: u32,
    pitch: u32,
    format: Option<PixelFormat>,
    modifier: u64,
    handle: u32,
}

/// DRM capture device with caching for fast repeated captures
pub struct DrmCapture {
    fd: c_int,
    is_master: bool,
    /// Cached display state to skip ioctls on repeated captures
    cached_state: Option<CachedDisplayState>,
    /// Generation counter - changes when display config changes
    config_generation: AtomicUsize,
}

impl DrmCapture {
    /// Open a DRM device for capture
    ///
    /// Typically `/dev/dri/card0` is the primary display
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path_str = path.as_ref().to_str().ok_or(CaptureError::InvalidPath)?;
        let c_path = CString::new(path_str).map_err(|_| CaptureError::InvalidPath)?;

        let fd = unsafe { open(c_path.as_ptr(), O_RDWR) };
        if fd < 0 {
            return Err(CaptureError::DeviceOpen(std::io::Error::last_os_error()));
        }

        let mut capture = Self {
            fd,
            is_master: false,
            cached_state: None,
            config_generation: AtomicUsize::new(0),
        };

        // Try to become DRM master (needed for some operations)
        // This requires root or appropriate permissions
        if unsafe { ioctl(fd, DRM_IOCTL_SET_MASTER, ptr::null_mut::<c_void>()) } == 0 {
            capture.is_master = true;
        }

        // Pre-cache the display state for fast first capture
        capture.refresh_cache();

        Ok(capture)
    }

    /// Find and open the first available DRM device
    pub fn open_default() -> Result<Self> {
        // Try card0 through card7
        for i in 0..8 {
            let path = format!("/dev/dri/card{}", i);
            if Path::new(&path).exists() {
                match Self::open(&path) {
                    Ok(cap) => return Ok(cap),
                    Err(_) => continue,
                }
            }
        }
        Err(CaptureError::NoDevice)
    }

    /// List available DRM devices
    pub fn list_devices() -> Vec<String> {
        let mut devices = Vec::new();
        if let Ok(entries) = fs::read_dir("/dev/dri") {
            for entry in entries.flatten() {
                let name = entry.file_name().to_string_lossy().to_string();
                if name.starts_with("card") {
                    devices.push(format!("/dev/dri/{}", name));
                }
            }
        }
        devices.sort();
        devices
    }

    /// Get DRM mode resources
    fn get_resources(&self) -> Result<(Vec<u32>, Vec<u32>)> {
        let mut res = DrmModeResources::default();

        // First call to get counts
        if unsafe { ioctl(self.fd, DRM_IOCTL_MODE_GETRESOURCES, &mut res) } != 0 {
            return Err(CaptureError::IoctlFailed("MODE_GETRESOURCES"));
        }

        let mut fb_ids = vec![0u32; res.count_fbs as usize];
        let mut crtc_ids = vec![0u32; res.count_crtcs as usize];
        let mut connector_ids = vec![0u32; res.count_connectors as usize];
        let mut encoder_ids = vec![0u32; res.count_encoders as usize];

        res.fb_id_ptr = fb_ids.as_mut_ptr() as u64;
        res.crtc_id_ptr = crtc_ids.as_mut_ptr() as u64;
        res.connector_id_ptr = connector_ids.as_mut_ptr() as u64;
        res.encoder_id_ptr = encoder_ids.as_mut_ptr() as u64;

        // Second call to get actual data
        if unsafe { ioctl(self.fd, DRM_IOCTL_MODE_GETRESOURCES, &mut res) } != 0 {
            return Err(CaptureError::IoctlFailed("MODE_GETRESOURCES"));
        }

        Ok((crtc_ids, fb_ids))
    }

    /// Get CRTC info
    fn get_crtc(&self, crtc_id: u32) -> Result<DrmModeCrtc> {
        let mut crtc = DrmModeCrtc::default();
        crtc.crtc_id = crtc_id;

        if unsafe { ioctl(self.fd, DRM_IOCTL_MODE_GETCRTC, &mut crtc) } != 0 {
            return Err(CaptureError::IoctlFailed("MODE_GETCRTC"));
        }

        Ok(crtc)
    }

    /// Get framebuffer info (legacy)
    fn get_fb(&self, fb_id: u32) -> Result<DrmModeFb> {
        let mut fb = DrmModeFb::default();
        fb.fb_id = fb_id;

        if unsafe { ioctl(self.fd, DRM_IOCTL_MODE_GETFB, &mut fb) } != 0 {
            return Err(CaptureError::IoctlFailed("MODE_GETFB"));
        }

        Ok(fb)
    }

    /// Get framebuffer info (newer API with format info)
    fn get_fb2(&self, fb_id: u32) -> Result<DrmModeFb2> {
        let mut fb = DrmModeFb2::default();
        fb.fb_id = fb_id;

        if unsafe { ioctl(self.fd, DRM_IOCTL_MODE_GETFB2, &mut fb) } != 0 {
            return Err(CaptureError::IoctlFailed("MODE_GETFB2"));
        }

        Ok(fb)
    }

    /// Map a dumb buffer to userspace
    fn map_dumb(&self, handle: u32, size: usize) -> Result<*mut u8> {
        let mut map = DrmModeMapDumb {
            handle,
            ..Default::default()
        };

        if unsafe { ioctl(self.fd, DRM_IOCTL_MODE_MAP_DUMB, &mut map) } != 0 {
            return Err(CaptureError::IoctlFailed("MODE_MAP_DUMB"));
        }

        let ptr = unsafe {
            mmap(
                ptr::null_mut(),
                size,
                libc::PROT_READ,
                libc::MAP_SHARED,
                self.fd,
                map.offset as i64,
            )
        };

        if ptr == libc::MAP_FAILED {
            return Err(CaptureError::MmapFailed(std::io::Error::last_os_error()));
        }

        Ok(ptr as *mut u8)
    }

    /// Wait for the next vertical blanking period on the given CRTC
    ///
    /// This synchronizes with the display refresh to avoid tearing artifacts
    /// when capturing during video playback or animations.
    ///
    /// The crtc_index is the index (0-based) of the CRTC to sync with.
    /// Use 0 for the primary display.
    pub fn wait_vblank(&self, crtc_index: u32) -> Result<()> {
        let mut vblank = DrmWaitVblank {
            // Wait for 1 vblank relative to current, with NEXTONMISS to handle timing
            request_type: DRM_VBLANK_RELATIVE 
                | DRM_VBLANK_NEXTONMISS
                | (crtc_index << DRM_VBLANK_HIGH_CRTC_SHIFT),
            sequence: 1, // Wait for next vblank
            signal: 0,
        };

        if unsafe { ioctl(self.fd, DRM_IOCTL_WAIT_VBLANK, &mut vblank) } != 0 {
            // VBlank wait failed - this can happen on some drivers/setups
            // We don't treat this as a fatal error, just continue without sync
            return Ok(());
        }

        Ok(())
    }

    /// Capture the current display framebuffer with VSync synchronization
    ///
    /// This waits for the vertical blanking period before capturing to reduce
    /// screen tearing artifacts when capturing video or animations.
    /// 
    /// If `fb_id` is provided, captures that specific framebuffer. Otherwise,
    /// captures the first active display.
    pub fn capture_vsync(&self, fb_id: Option<u32>) -> Result<CapturedFrame> {
        // Find the active CRTC index for vblank sync
        let (crtc_ids, _) = self.get_resources()?;
        
        if let Some(target_fb_id) = fb_id {
            eprintln!("capture_vsync called with fb_id={}", target_fb_id);
            // Find the CRTC that's using this fb_id
            for (index, &crtc_id) in crtc_ids.iter().enumerate() {
                if let Ok(crtc) = self.get_crtc(crtc_id) {
                    if crtc.fb_id == target_fb_id && crtc.mode_valid != 0 {
                        eprintln!("  Found CRTC {} with matching fb_id, waiting for vblank", crtc_id);
                        // Wait for vblank on this CRTC
                        let _ = self.wait_vblank(index as u32);
                        break;
                    }
                }
            }
        } else {
            // Wait for vblank on first active display
            for (index, &crtc_id) in crtc_ids.iter().enumerate() {
                if let Ok(crtc) = self.get_crtc(crtc_id) {
                    if crtc.fb_id != 0 && crtc.mode_valid != 0 {
                        // Wait for vblank on this CRTC
                        let _ = self.wait_vblank(index as u32);
                        break;
                    }
                }
            }
        }

        // Now capture
        self.capture(fb_id)
    }

    /// Refresh the cached display state
    /// 
    /// Call this if you expect the display configuration to have changed.
    pub fn refresh_cache(&mut self) {
        if let Ok((crtc_ids, _)) = self.get_resources() {
            for crtc_id in crtc_ids {
                if let Ok(crtc) = self.get_crtc(crtc_id) {
                    if crtc.fb_id != 0 && crtc.mode_valid != 0 {
                        // Try the newer FB2 API first
                        let (width, height, pitch, handle, format, modifier) = 
                            match self.get_fb2(crtc.fb_id) {
                                Ok(fb2) => {
                                    let format = fourcc_to_pixel_format(fb2.pixel_format);
                                    (fb2.width, fb2.height, fb2.pitches[0], 
                                     fb2.handles[0], format, fb2.modifier[0])
                                }
                                Err(_) => {
                                    if let Ok(fb) = self.get_fb(crtc.fb_id) {
                                        let format = match fb.bpp {
                                            32 => PixelFormat::Bgra8888,
                                            24 => PixelFormat::Bgr888,
                                            16 => PixelFormat::Rgb565,
                                            _ => PixelFormat::Unknown(fb.bpp),
                                        };
                                        (fb.width, fb.height, fb.pitch, fb.handle, format, 0u64)
                                    } else {
                                        continue;
                                    }
                                }
                            };
                        
                        self.cached_state = Some(CachedDisplayState {
                            crtc_id,
                            fb_id: crtc.fb_id,
                            width,
                            height,
                            pitch,
                            format: Some(format),
                            modifier,
                            handle,
                        });
                        self.config_generation.fetch_add(1, Ordering::Relaxed);
                        return;
                    }
                }
            }
        }
    }

    /// Capture the current display framebuffer
    ///
    /// This reads the active CRTC's framebuffer directly from GPU memory.
    /// 
    /// If `fb_id` is provided, captures that specific framebuffer. Otherwise,
    /// captures the first active display.
    pub fn capture(&self, fb_id: Option<u32>) -> Result<CapturedFrame> {
        if let Some(target_fb_id) = fb_id {
            eprintln!("capture called with fb_id={}", target_fb_id);
            // Check if cached state matches the requested fb_id
            if let Some(ref cached) = self.cached_state {
                eprintln!("  Cached state: fb_id={}", cached.fb_id);
                if cached.fb_id == target_fb_id {
                    eprintln!("  Using cached state (match!)");
                    return self.capture_with_state(cached);
                }
            }
            
            eprintln!("  Cache miss, calling capture_fb_uncached");
            // Query for the specific fb_id
            self.capture_fb_uncached(target_fb_id)
        } else {
            // Fast path: use cached state if available
            if let Some(ref cached) = self.cached_state {
                return self.capture_with_state(cached);
            }
            
            // Slow path: query DRM for display state
            self.capture_uncached()
        }
    }
    
    /// Capture using cached display state (fast path)
    #[inline]
    fn capture_with_state(&self, state: &CachedDisplayState) -> Result<CapturedFrame> {
        let format = state.format.unwrap_or(PixelFormat::Bgrx8888);
        
        if state.handle == 0 {
            return self.capture_via_prime(
                state.fb_id, state.width, state.height, 
                state.pitch, format, state.modifier
            );
        }

        let buffer_height = if !is_linear_modifier(state.modifier) {
            aligned_buffer_height(state.height, state.modifier)
        } else {
            state.height
        };

        let size = (state.pitch * buffer_height) as usize;
        let ptr = self.map_dumb(state.handle, size)?;

        let (data, pitch) = if !is_linear_modifier(state.modifier) {
            let bpp = format.bytes_per_pixel() as u32 * 8;
            let linear_data = unsafe {
                detile_from_gpu_memory_parallel(
                    ptr, size, state.width, state.height, state.pitch, bpp, state.modifier
                )
            };
            unsafe { munmap(ptr as *mut c_void, size); }
            let linear_pitch = state.width * (bpp / 8);
            (linear_data, linear_pitch)
        } else {
            let mut data = vec![0u8; size];
            unsafe {
                copy_from_gpu_memory(ptr, data.as_mut_ptr(), size);
                munmap(ptr as *mut c_void, size);
            }
            (data, state.pitch)
        };

        Ok(CapturedFrame {
            width: state.width,
            height: state.height,
            pitch,
            format,
            data,
        })
    }

    /// Capture without using cache (queries DRM each time)
    fn capture_uncached(&self) -> Result<CapturedFrame> {
        let (crtc_ids, _fb_ids) = self.get_resources()?;

        // Find an active CRTC with a framebuffer
        for crtc_id in crtc_ids {
            let crtc = self.get_crtc(crtc_id)?;

            if crtc.fb_id == 0 || crtc.mode_valid == 0 {
                continue;
            }

            return self.capture_fb_by_info(crtc.fb_id);
        }

        Err(CaptureError::NoActiveDisplay)
    }

    /// Capture a specific framebuffer without using cache
    fn capture_fb_uncached(&self, fb_id: u32) -> Result<CapturedFrame> {
        if fb_id == 0 {
            return Err(CaptureError::NoActiveDisplay);
        }
        
        self.capture_fb_by_info(fb_id)
    }

    /// Internal method to capture by framebuffer ID
    fn capture_fb_by_info(&self, fb_id: u32) -> Result<CapturedFrame> {
        // Try the newer FB2 API first, fall back to legacy
        let (width, height, pitch, handle, format, modifier) = match self.get_fb2(fb_id) {
            Ok(fb2) => {
                let format = fourcc_to_pixel_format(fb2.pixel_format);
                (
                    fb2.width,
                    fb2.height,
                    fb2.pitches[0],
                    fb2.handles[0],
                    format,
                    fb2.modifier[0],
                )
            }
            Err(_) => {
                let fb = self.get_fb(fb_id)?;
                let format = match fb.bpp {
                    32 => PixelFormat::Bgra8888,
                    24 => PixelFormat::Bgr888,
                    16 => PixelFormat::Rgb565,
                    _ => PixelFormat::Unknown(fb.bpp),
                };
                // Legacy FB doesn't report modifier, assume linear
                (fb.width, fb.height, fb.pitch, fb.handle, format, 0u64)
            }
        };

        if handle == 0 {
            // Can't map this buffer directly - it may be a GPU-only buffer
            // Try alternate method via prime fd
            return self.capture_via_prime(fb_id, width, height, pitch, format, modifier);
        }

        // For tiled formats, buffer height is aligned to block boundaries
        // We need to copy the full buffer to get all visible pixel data
        let buffer_height = if !is_linear_modifier(modifier) {
            aligned_buffer_height(height, modifier)
        } else {
            height
        };

        let size = (pitch * buffer_height) as usize;
        let ptr = self.map_dumb(handle, size)?;

        // For tiled formats, detile directly from GPU memory to avoid double copy
        let (data, pitch) = if !is_linear_modifier(modifier) {
            let bpp = format.bytes_per_pixel() as u32 * 8;
            let linear_data = unsafe {
                detile_from_gpu_memory_parallel(ptr, size, width, height, pitch, bpp, modifier)
            };
            unsafe { munmap(ptr as *mut c_void, size); }
            let linear_pitch = width * (bpp / 8);
            (linear_data, linear_pitch)
        } else {
            // Linear format - just stream copy
            let mut data = vec![0u8; size];
            unsafe {
                copy_from_gpu_memory(ptr, data.as_mut_ptr(), size);
                munmap(ptr as *mut c_void, size);
            }
            (data, pitch)
        };

        Ok(CapturedFrame {
            width,
            height,
            pitch,
            format,
            data,
        })
    }

    /// Capture via PRIME fd export (for GPU buffers that can't be mapped directly)
    fn capture_via_prime(
        &self,
        fb_id: u32,
        width: u32,
        height: u32,
        pitch: u32,
        format: PixelFormat,
        modifier: u64,
    ) -> Result<CapturedFrame> {
        // Try to get handle from fb2
        let fb2 = self.get_fb2(fb_id)?;

        if fb2.handles[0] == 0 {
            return Err(CaptureError::BufferNotMappable);
        }

        // Try to export as PRIME fd
        let mut prime = DrmPrimeHandle {
            handle: fb2.handles[0],
            flags: libc::O_RDONLY as u32,
            fd: -1,
        };

        if unsafe { ioctl(self.fd, DRM_IOCTL_PRIME_HANDLE_TO_FD, &mut prime) } != 0 {
            return Err(CaptureError::IoctlFailed("PRIME_HANDLE_TO_FD"));
        }

        // For tiled formats, buffer height is aligned to block boundaries
        let buffer_height = if !is_linear_modifier(modifier) {
            aligned_buffer_height(height, modifier)
        } else {
            height
        };

        let size = (pitch * buffer_height) as usize;

        // mmap the prime fd
        let ptr = unsafe {
            mmap(
                ptr::null_mut(),
                size,
                libc::PROT_READ,
                libc::MAP_SHARED,
                prime.fd,
                0,
            )
        };

        if ptr == libc::MAP_FAILED {
            unsafe { close(prime.fd) };
            return Err(CaptureError::MmapFailed(std::io::Error::last_os_error()));
        }

        // For tiled formats, detile directly from GPU memory to avoid double copy
        let (data, pitch) = if !is_linear_modifier(modifier) {
            let bpp = format.bytes_per_pixel() as u32 * 8;
            let linear_data = unsafe {
                detile_from_gpu_memory_parallel(ptr as *const u8, size, width, height, pitch, bpp, modifier)
            };
            unsafe {
                munmap(ptr as *mut c_void, size);
                close(prime.fd);
            }
            let linear_pitch = width * (bpp / 8);
            (linear_data, linear_pitch)
        } else {
            // Linear format - just stream copy
            let mut data = vec![0u8; size];
            unsafe {
                copy_from_gpu_memory(ptr as *const u8, data.as_mut_ptr(), size);
                munmap(ptr as *mut c_void, size);
                close(prime.fd);
            }
            (data, pitch)
        };

        Ok(CapturedFrame {
            width,
            height,
            pitch,
            format,
            data,
        })
    }

    /// Capture directly to RGBA format in one fused operation
    /// 
    /// This is faster than capture() + to_rgba() because it:
    /// - Detiles and converts BGR→RGB in a single pass
    /// - Avoids an extra memory read/write cycle
    /// 
    /// If `fb_id` is provided, captures that specific framebuffer. Otherwise,
    /// captures the first active display.
    /// 
    /// Returns (width, height, rgba_data)
    #[cfg(feature = "parallel")]
    pub fn capture_rgba(&self, fb_id: Option<u32>) -> Result<(u32, u32, Vec<u8>)> {
        if let Some(target_fb_id) = fb_id {
            eprintln!("capture_rgba called with fb_id={}", target_fb_id);
            // Check if cached state matches
            if let Some(ref cached) = self.cached_state {
                eprintln!("  Cached state: fb_id={}", cached.fb_id);
                if cached.fb_id == target_fb_id {
                    eprintln!("  Using cached RGBA path (match!)");
                    return self.capture_rgba_with_state(cached);
                }
            }
            
            eprintln!("  Cache miss, using fallback capture + conversion");
            // Fall back to capture + conversion
            let frame = self.capture(Some(target_fb_id))?;
            let rgba = frame.to_rgba();
            Ok((frame.width, frame.height, rgba))
        } else {
            // Use cached state if available
            if let Some(ref cached) = self.cached_state {
                return self.capture_rgba_with_state(cached);
            }
            self.capture_rgba_uncached()
        }
    }
    
    #[cfg(feature = "parallel")]
    fn capture_rgba_with_state(&self, state: &CachedDisplayState) -> Result<(u32, u32, Vec<u8>)> {
        let format = state.format.unwrap_or(PixelFormat::Bgrx8888);
        
        // Only use fused path for NVIDIA tiled BGRx/BGRa formats
        let is_nvidia_tiled = !is_linear_modifier(state.modifier) 
            && (state.modifier >> 56) == DRM_FORMAT_MOD_VENDOR_NVIDIA;
        let is_bgr = matches!(format, PixelFormat::Bgrx8888 | PixelFormat::Bgra8888);
        
        if is_nvidia_tiled && is_bgr && state.handle != 0 {
            let buffer_height = aligned_buffer_height(state.height, state.modifier);
            let size = (state.pitch * buffer_height) as usize;
            let ptr = self.map_dumb(state.handle, size)?;
            
            let rgba = unsafe {
                detile_nvidia_to_rgba_parallel(
                    ptr, size, state.width, state.height, 
                    state.pitch, state.modifier & 0x00ffffffffffffff
                )
            };

            unsafe { munmap(ptr as *mut c_void, size); }
            
            return Ok((state.width, state.height, rgba));
        }
        
        // Fall back to regular capture + conversion
        let frame = self.capture_with_state(state)?;
        let rgba = frame.to_rgba();
        Ok((frame.width, frame.height, rgba))
    }
    
    #[cfg(feature = "parallel")]
    fn capture_rgba_uncached(&self) -> Result<(u32, u32, Vec<u8>)> {
        let frame = self.capture_uncached()?;
        let rgba = frame.to_rgba();
        Ok((frame.width, frame.height, rgba))
    }

    /// Get information about all active displays
    pub fn get_display_info(&self) -> Result<Vec<DisplayInfo>> {
        let (crtc_ids, _) = self.get_resources()?;
        let mut displays = Vec::new();

        for crtc_id in crtc_ids {
            if let Ok(crtc) = self.get_crtc(crtc_id) {
                if crtc.mode_valid != 0 {
                    let mode_name = crtc
                        .mode
                        .name
                        .iter()
                        .take_while(|&&c| c != 0)
                        .map(|&c| c as char)
                        .collect();

                    displays.push(DisplayInfo {
                        crtc_id,
                        fb_id: crtc.fb_id,
                        width: crtc.mode.hdisplay as u32,
                        height: crtc.mode.vdisplay as u32,
                        refresh_rate: crtc.mode.vrefresh,
                        mode_name,
                    });
                }
            }
        }

        Ok(displays)
    }
}

impl Drop for DrmCapture {
    fn drop(&mut self) {
        if self.is_master {
            unsafe {
                ioctl(self.fd, DRM_IOCTL_DROP_MASTER, ptr::null_mut::<c_void>());
            }
        }
        unsafe {
            close(self.fd);
        }
    }
}

/// Information about a display/CRTC
#[derive(Debug, Clone, Serialize)]
pub struct DisplayInfo {
    pub crtc_id: u32,
    pub fb_id: u32,
    pub width: u32,
    pub height: u32,
    pub refresh_rate: u32,
    pub mode_name: String,
}

// DRM format modifier vendor ID for NVIDIA
const DRM_FORMAT_MOD_VENDOR_NVIDIA: u64 = 0x03;

// Special modifier values
const DRM_FORMAT_MOD_LINEAR: u64 = 0;
const DRM_FORMAT_MOD_INVALID: u64 = 0x00ffffffffffffff;

/// Check if modifier indicates linear layout
fn is_linear_modifier(modifier: u64) -> bool {
    modifier == DRM_FORMAT_MOD_LINEAR || modifier == DRM_FORMAT_MOD_INVALID
}

/// Calculate the aligned buffer height for tiled formats
/// Tiled buffers are padded to block boundaries, so we need to read more
/// data than just pitch * display_height to capture all visible pixels
fn aligned_buffer_height(height: u32, modifier: u64) -> u32 {
    let vendor = modifier >> 56;
    
    match vendor {
        DRM_FORMAT_MOD_VENDOR_NVIDIA => {
            // NVIDIA block-linear: GOBs are 8 rows, stacked into blocks
            // h (bits 0-3) = log2 of block height in GOBs
            const GOB_HEIGHT: u32 = 8;
            let h = (modifier & 0xf) as u32;
            let block_height_gobs = 1u32 << h;
            let block_height_rows = block_height_gobs * GOB_HEIGHT;
            
            // Round up to next block boundary
            ((height + block_height_rows - 1) / block_height_rows) * block_height_rows
        }
        _ => {
            // For other vendors, use a conservative alignment
            // Most tiling schemes use 8 or 16 row alignment
            let alignment = 16u32;
            ((height + alignment - 1) / alignment) * alignment
        }
    }
}

/// Detile directly from GPU memory using streaming reads (parallel version)
/// This avoids a double-copy by reading from GPU memory while detiling
/// 
/// # Safety
/// - src must point to valid mapped GPU memory of at least `src_len` bytes
unsafe fn detile_from_gpu_memory_parallel(
    src: *const u8,
    src_len: usize,
    width: u32,
    height: u32,
    pitch: u32,
    bpp: u32,
    modifier: u64,
) -> Vec<u8> {
    let vendor = modifier >> 56;
    
    match vendor {
        DRM_FORMAT_MOD_VENDOR_NVIDIA => {
            detile_nvidia_parallel(src, src_len, width, height, pitch, bpp, modifier & 0x00ffffffffffffff)
        }
        _ => {
            // For other formats, use generic row copy with streaming reads
            detile_generic_parallel(src, src_len, width, height, pitch, bpp)
        }
    }
}

/// Detile NVIDIA block-linear AND convert BGR→RGB in one fused pass
/// Returns RGBA data directly, saving a full memory pass
/// 
/// # Safety
/// - src must point to valid mapped GPU memory of at least `src_len` bytes
#[cfg(feature = "parallel")]
pub unsafe fn detile_nvidia_to_rgba_parallel(
    src: *const u8,
    src_len: usize,
    width: u32,
    height: u32,
    pitch: u32,
    modifier: u64,
) -> Vec<u8> {
    let width = width as usize;
    let height = height as usize;
    let pitch = pitch as usize;
    let mut rgba = vec![0u8; width * height * 4];

    const GOB_WIDTH: usize = 64;
    const GOB_HEIGHT: usize = 8;
    const GOB_SIZE: usize = GOB_WIDTH * GOB_HEIGHT;

    let h = (modifier & 0xf) as usize;
    let block_height_gobs = 1usize << h;
    let block_height_rows = block_height_gobs * GOB_HEIGHT;
    let gobs_x = pitch / GOB_WIDTH;
    let block_size = gobs_x * block_height_gobs * GOB_SIZE;
    let pixels_per_gob = GOB_WIDTH / 4; // 16 pixels per GOB row (at 4 bytes/pixel)

    let src_ptr = SendPtr(src);
    let dst_ptr = SendPtrMut(rgba.as_mut_ptr());
    let row_stride = width * 4;

    (0..height).into_par_iter().for_each(move |dst_y| {
        let src_p = src_ptr;
        let dst_p = dst_ptr;
        
        let block_y = dst_y / block_height_rows;
        let y_in_block = dst_y % block_height_rows;
        let gob_y = y_in_block / GOB_HEIGHT;
        let y_in_gob = y_in_block % GOB_HEIGHT;
        
        let block_offset = block_y * block_size;
        let dst_row_offset = dst_y * row_stride;

        let mut pixel_x = 0usize;
        let mut gob_x = 0;
        
        while gob_x < gobs_x && pixel_x < width {
            let gob_index = gob_x * block_height_gobs + gob_y;
            let gob_offset = block_offset + gob_index * GOB_SIZE;
            let src_offset = gob_offset + y_in_gob * GOB_WIDTH;
            
            // Process up to 16 pixels from this GOB row
            let pixels_this_gob = pixels_per_gob.min(width - pixel_x);
            
            if src_offset + pixels_this_gob * 4 <= src_len {
                let src_row = src_p.0.add(src_offset);
                let dst_row = dst_p.0.add(dst_row_offset + pixel_x * 4);
                
                // Use SIMD to convert BGRx→RGBA
                convert_bgrx_to_rgba_row(src_row, dst_row, pixels_this_gob);
            }
            
            pixel_x += pixels_this_gob;
            gob_x += 1;
        }
    });

    rgba
}

/// Convert a row of BGRx pixels to RGBA using SIMD
#[inline]
unsafe fn convert_bgrx_to_rgba_row(src: *const u8, dst: *mut u8, pixel_count: usize) {
    let mut i = 0;
    
    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::*;
        
        if is_x86_feature_detected!("avx2") && pixel_count >= 8 {
            let shuffle = _mm256_setr_epi8(
                2, 1, 0, 3, 6, 5, 4, 7, 10, 9, 8, 11, 14, 13, 12, 15,
                2, 1, 0, 3, 6, 5, 4, 7, 10, 9, 8, 11, 14, 13, 12, 15
            );
            let alpha_mask = _mm256_set1_epi32(0xFF000000u32 as i32);
            
            while i + 8 <= pixel_count {
                let pixels = _mm256_loadu_si256(src.add(i * 4) as *const __m256i);
                let swapped = _mm256_shuffle_epi8(pixels, shuffle);
                let result = _mm256_or_si256(swapped, alpha_mask);
                _mm256_storeu_si256(dst.add(i * 4) as *mut __m256i, result);
                i += 8;
            }
        } else if is_x86_feature_detected!("ssse3") && pixel_count >= 4 {
            let shuffle = _mm_setr_epi8(2, 1, 0, 3, 6, 5, 4, 7, 10, 9, 8, 11, 14, 13, 12, 15);
            let alpha_mask = _mm_set1_epi32(0xFF000000u32 as i32);
            
            while i + 4 <= pixel_count {
                let pixels = _mm_loadu_si128(src.add(i * 4) as *const __m128i);
                let swapped = _mm_shuffle_epi8(pixels, shuffle);
                let result = _mm_or_si128(swapped, alpha_mask);
                _mm_storeu_si128(dst.add(i * 4) as *mut __m128i, result);
                i += 4;
            }
        }
    }
    
    // Scalar fallback for remainder
    while i < pixel_count {
        let off = i * 4;
        *dst.add(off) = *src.add(off + 2);     // R
        *dst.add(off + 1) = *src.add(off + 1); // G
        *dst.add(off + 2) = *src.add(off);     // B
        *dst.add(off + 3) = 255;               // A
        i += 1;
    }
}

/// Detile directly from GPU memory using streaming reads (sequential fallback)
/// 
/// # Safety
/// - src must point to valid mapped GPU memory of at least `src_len` bytes
#[allow(dead_code)]
unsafe fn detile_from_gpu_memory(
    src: *const u8,
    src_len: usize,
    width: u32,
    height: u32,
    pitch: u32,
    bpp: u32,
    modifier: u64,
) -> Vec<u8> {
    let vendor = modifier >> 56;
    
    match vendor {
        DRM_FORMAT_MOD_VENDOR_NVIDIA => {
            detile_nvidia_from_gpu(src, src_len, width, height, pitch, bpp, modifier & 0x00ffffffffffffff)
        }
        _ => {
            // For other formats, use generic row copy with streaming reads
            detile_generic_from_gpu(src, src_len, width, height, pitch, bpp)
        }
    }
}

/// Parallel NVIDIA block-linear detiling using rayon
/// Processes row chunks in parallel for maximum throughput
#[cfg(feature = "parallel")]
unsafe fn detile_nvidia_parallel(
    src: *const u8,
    src_len: usize,
    width: u32,
    height: u32,
    pitch: u32,
    bpp: u32,
    modifier: u64,
) -> Vec<u8> {
    let bytes_per_pixel = (bpp / 8) as usize;
    let linear_pitch = (width as usize) * bytes_per_pixel;
    let pitch = pitch as usize;
    let height = height as usize;
    let mut linear = vec![0u8; linear_pitch * height];

    const GOB_WIDTH: usize = 64;
    const GOB_HEIGHT: usize = 8;
    const GOB_SIZE: usize = GOB_WIDTH * GOB_HEIGHT;

    let h = (modifier & 0xf) as usize;
    let block_height_gobs = 1usize << h;
    let block_height_rows = block_height_gobs * GOB_HEIGHT;
    let gobs_x = pitch / GOB_WIDTH;
    let block_size = gobs_x * block_height_gobs * GOB_SIZE;

    // Wrap pointers for parallel access
    let src_ptr = SendPtr(src);
    let dst_ptr = SendPtrMut(linear.as_mut_ptr());
    let linear_len = linear.len();

    // Process rows in parallel - each row is independent
    (0..height).into_par_iter().for_each(move |dst_y| {
        let src_p = src_ptr;
        let dst_p = dst_ptr;
        
        let block_y = dst_y / block_height_rows;
        let y_in_block = dst_y % block_height_rows;
        let gob_y = y_in_block / GOB_HEIGHT;
        let y_in_gob = y_in_block % GOB_HEIGHT;
        
        let block_offset = block_y * block_size;
        let dst_row_offset = dst_y * linear_pitch;

        let mut gob_x = 0;
        while gob_x < gobs_x {
            let gob_index = gob_x * block_height_gobs + gob_y;
            let gob_offset = block_offset + gob_index * GOB_SIZE;
            let src_offset = gob_offset + y_in_gob * GOB_WIDTH;
            
            let dst_x_bytes = gob_x * GOB_WIDTH;
            let dst_offset = dst_row_offset + dst_x_bytes;
            
            let copy_len = GOB_WIDTH.min(linear_pitch.saturating_sub(dst_x_bytes));
            
            if src_offset + copy_len <= src_len && dst_offset + copy_len <= linear_len && copy_len > 0 {
                if copy_len == GOB_WIDTH {
                    copy_64_from_gpu(src_p.0.add(src_offset), dst_p.0.add(dst_offset));
                } else {
                    ptr::copy_nonoverlapping(
                        src_p.0.add(src_offset), 
                        dst_p.0.add(dst_offset), 
                        copy_len
                    );
                }
            }
            
            gob_x += 1;
        }
    });

    linear
}

/// Fallback to sequential when parallel feature is disabled
#[cfg(not(feature = "parallel"))]
unsafe fn detile_nvidia_parallel(
    src: *const u8,
    src_len: usize,
    width: u32,
    height: u32,
    pitch: u32,
    bpp: u32,
    modifier: u64,
) -> Vec<u8> {
    detile_nvidia_from_gpu(src, src_len, width, height, pitch, bpp, modifier)
}

/// Detile NVIDIA block-linear directly from GPU memory (sequential version)
unsafe fn detile_nvidia_from_gpu(
    src: *const u8,
    src_len: usize,
    width: u32,
    height: u32,
    pitch: u32,
    bpp: u32,
    modifier: u64,
) -> Vec<u8> {
    let bytes_per_pixel = (bpp / 8) as usize;
    let linear_pitch = (width as usize) * bytes_per_pixel;
    let pitch = pitch as usize;
    let height = height as usize;
    let mut linear = vec![0u8; linear_pitch * height];

    const GOB_WIDTH: usize = 64;
    const GOB_HEIGHT: usize = 8;
    const GOB_SIZE: usize = GOB_WIDTH * GOB_HEIGHT;

    let h = (modifier & 0xf) as usize;
    let block_height_gobs = 1usize << h;
    let block_height_rows = block_height_gobs * GOB_HEIGHT;

    let gobs_x = pitch / GOB_WIDTH;
    let block_size = gobs_x * block_height_gobs * GOB_SIZE;
    
    let dst_ptr = linear.as_mut_ptr();

    // Process all rows
    for dst_y in 0..height {
        let block_y = dst_y / block_height_rows;
        let y_in_block = dst_y % block_height_rows;
        let gob_y = y_in_block / GOB_HEIGHT;
        let y_in_gob = y_in_block % GOB_HEIGHT;
        
        let block_offset = block_y * block_size;
        let dst_row_offset = dst_y * linear_pitch;

        let mut gob_x = 0;
        while gob_x < gobs_x {
            let gob_index = gob_x * block_height_gobs + gob_y;
            let gob_offset = block_offset + gob_index * GOB_SIZE;
            let src_offset = gob_offset + y_in_gob * GOB_WIDTH;
            
            let dst_x_bytes = gob_x * GOB_WIDTH;
            let dst_offset = dst_row_offset + dst_x_bytes;
            
            let copy_len = GOB_WIDTH.min(linear_pitch.saturating_sub(dst_x_bytes));
            
            if src_offset + copy_len <= src_len && dst_offset + copy_len <= linear.len() && copy_len > 0 {
                if copy_len == GOB_WIDTH {
                    // Full GOB - use streaming read
                    copy_64_from_gpu(src.add(src_offset), dst_ptr.add(dst_offset));
                } else {
                    // Partial - just copy
                    ptr::copy_nonoverlapping(src.add(src_offset), dst_ptr.add(dst_offset), copy_len);
                }
            }
            
            gob_x += 1;
        }
    }

    linear
}

/// Wrapper to make raw const pointers Send for parallel iteration
#[derive(Clone, Copy)]
struct SendPtr<T>(*const T);
unsafe impl<T> Send for SendPtr<T> {}
unsafe impl<T> Sync for SendPtr<T> {}

/// Wrapper to make raw mut pointers Send for parallel iteration
#[derive(Clone, Copy)]
struct SendPtrMut<T>(*mut T);
unsafe impl<T> Send for SendPtrMut<T> {}
unsafe impl<T> Sync for SendPtrMut<T> {}

/// Copy exactly 64 bytes from GPU memory using streaming loads
#[inline(always)]
unsafe fn copy_64_from_gpu(src: *const u8, dst: *mut u8) {
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    {
        #[cfg(target_arch = "x86_64")]
        use std::arch::x86_64::*;
        #[cfg(target_arch = "x86")]
        use std::arch::x86::*;
        
        // Check alignment - MOVNTDQA requires 16-byte alignment
        if (src as usize) & 0xF == 0 {
            if is_x86_feature_detected!("sse4.1") {
                let v0 = _mm_stream_load_si128(src as *const __m128i);
                let v1 = _mm_stream_load_si128(src.add(16) as *const __m128i);
                let v2 = _mm_stream_load_si128(src.add(32) as *const __m128i);
                let v3 = _mm_stream_load_si128(src.add(48) as *const __m128i);
                _mm_storeu_si128(dst as *mut __m128i, v0);
                _mm_storeu_si128(dst.add(16) as *mut __m128i, v1);
                _mm_storeu_si128(dst.add(32) as *mut __m128i, v2);
                _mm_storeu_si128(dst.add(48) as *mut __m128i, v3);
                return;
            }
        }
    }
    
    // Fallback
    copy_64_bytes(src, dst);
}

/// Parallel generic detile from GPU memory
#[cfg(feature = "parallel")]
unsafe fn detile_generic_parallel(
    src: *const u8,
    src_len: usize,
    width: u32,
    height: u32,
    pitch: u32,
    bpp: u32,
) -> Vec<u8> {
    let bytes_per_pixel = (bpp / 8) as usize;
    let linear_pitch = (width as usize) * bytes_per_pixel;
    let pitch = pitch as usize;
    let height = height as usize;

    let mut linear = vec![0u8; linear_pitch * height];
    let src_ptr = SendPtr(src);
    let dst_ptr = SendPtrMut(linear.as_mut_ptr());
    let linear_len = linear.len();

    // Process rows in parallel
    (0..height).into_par_iter().for_each(move |y| {
        let src_p = src_ptr;
        let dst_p = dst_ptr;
        
        let src_offset = y * pitch;
        let dst_offset = y * linear_pitch;
        let copy_len = linear_pitch.min(src_len.saturating_sub(src_offset));

        if copy_len > 0 && dst_offset + copy_len <= linear_len {
            copy_from_gpu_memory(src_p.0.add(src_offset), dst_p.0.add(dst_offset), copy_len);
        }
    });

    linear
}

/// Fallback when parallel feature is disabled
#[cfg(not(feature = "parallel"))]
unsafe fn detile_generic_parallel(
    src: *const u8,
    src_len: usize,
    width: u32,
    height: u32,
    pitch: u32,
    bpp: u32,
) -> Vec<u8> {
    detile_generic_from_gpu(src, src_len, width, height, pitch, bpp)
}

/// Generic detile from GPU memory with streaming reads (sequential)
unsafe fn detile_generic_from_gpu(
    src: *const u8,
    src_len: usize,
    width: u32,
    height: u32,
    pitch: u32,
    bpp: u32,
) -> Vec<u8> {
    let bytes_per_pixel = (bpp / 8) as usize;
    let linear_pitch = (width as usize) * bytes_per_pixel;
    let pitch = pitch as usize;
    let height = height as usize;

    let mut linear = vec![0u8; linear_pitch * height];
    let dst_ptr = linear.as_mut_ptr();

    for y in 0..height {
        let src_offset = y * pitch;
        let dst_offset = y * linear_pitch;
        let copy_len = linear_pitch.min(src_len.saturating_sub(src_offset));

        if copy_len > 0 && dst_offset + copy_len <= linear.len() {
            copy_from_gpu_memory(src.add(src_offset), dst_ptr.add(dst_offset), copy_len);
        }
    }

    linear
}

/// Fast copy from GPU/WC memory using non-temporal loads (streaming reads)
/// This is critical for performance when reading from mapped GPU memory
/// 
/// # Safety
/// - src must be 16-byte aligned for best performance
/// - src and dst must point to valid memory of at least `len` bytes
#[inline]
unsafe fn copy_from_gpu_memory(src: *const u8, dst: *mut u8, len: usize) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("sse4.1") {
            copy_from_wc_memory_sse41(src, dst, len);
            return;
        }
    }
    
    #[cfg(target_arch = "x86")]
    {
        if is_x86_feature_detected!("sse4.1") {
            copy_from_wc_memory_sse41(src, dst, len);
            return;
        }
    }
    
    // Fallback for non-x86 or no SSE4.1
    // Still use our SIMD copy which is faster than ptr::copy_nonoverlapping
    fast_memcpy(src, dst, len);
}

/// SSE4.1 streaming read from write-combining memory
/// Uses MOVNTDQA which is specifically designed for reading WC memory
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[target_feature(enable = "sse4.1")]
unsafe fn copy_from_wc_memory_sse41(src: *const u8, dst: *mut u8, len: usize) {
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    
    let mut offset = 0;
    
    // Handle unaligned prefix - copy until src is 16-byte aligned
    let align_offset = (src as usize) & 0xF;
    if align_offset != 0 {
        let prefix_len = (16 - align_offset).min(len);
        ptr::copy_nonoverlapping(src, dst, prefix_len);
        offset = prefix_len;
    }
    
    // Main loop: 64 bytes at a time using non-temporal loads
    while offset + 64 <= len {
        let s = src.add(offset);
        let d = dst.add(offset);
        
        // MOVNTDQA - non-temporal load, bypasses cache, designed for WC memory
        let v0 = _mm_stream_load_si128(s as *const __m128i);
        let v1 = _mm_stream_load_si128(s.add(16) as *const __m128i);
        let v2 = _mm_stream_load_si128(s.add(32) as *const __m128i);
        let v3 = _mm_stream_load_si128(s.add(48) as *const __m128i);
        
        // Store to destination (regular stores are fine since dst is in system RAM)
        _mm_storeu_si128(d as *mut __m128i, v0);
        _mm_storeu_si128(d.add(16) as *mut __m128i, v1);
        _mm_storeu_si128(d.add(32) as *mut __m128i, v2);
        _mm_storeu_si128(d.add(48) as *mut __m128i, v3);
        
        offset += 64;
    }
    
    // Handle 16-byte chunks
    while offset + 16 <= len {
        let s = src.add(offset);
        let d = dst.add(offset);
        
        let v = _mm_stream_load_si128(s as *const __m128i);
        _mm_storeu_si128(d as *mut __m128i, v);
        
        offset += 16;
    }
    
    // Copy remaining bytes
    if offset < len {
        ptr::copy_nonoverlapping(src.add(offset), dst.add(offset), len - offset);
    }
}

/// Fast 64-byte copy using SIMD when available
/// 
/// # Safety
/// Caller must ensure src and dst point to valid memory of at least 64 bytes
#[inline(always)]
unsafe fn copy_64_bytes(src: *const u8, dst: *mut u8) {
    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::*;
        // Use AVX if available (256-bit = 32 bytes), otherwise SSE2 (128-bit = 16 bytes)
        if is_x86_feature_detected!("avx") {
            let s0 = _mm256_loadu_si256(src as *const __m256i);
            let s1 = _mm256_loadu_si256(src.add(32) as *const __m256i);
            _mm256_storeu_si256(dst as *mut __m256i, s0);
            _mm256_storeu_si256(dst.add(32) as *mut __m256i, s1);
        } else {
            // SSE2 fallback (always available on x86_64)
            let s0 = _mm_loadu_si128(src as *const __m128i);
            let s1 = _mm_loadu_si128(src.add(16) as *const __m128i);
            let s2 = _mm_loadu_si128(src.add(32) as *const __m128i);
            let s3 = _mm_loadu_si128(src.add(48) as *const __m128i);
            _mm_storeu_si128(dst as *mut __m128i, s0);
            _mm_storeu_si128(dst.add(16) as *mut __m128i, s1);
            _mm_storeu_si128(dst.add(32) as *mut __m128i, s2);
            _mm_storeu_si128(dst.add(48) as *mut __m128i, s3);
        }
    }
    
    #[cfg(target_arch = "x86")]
    {
        use std::arch::x86::*;
        // SSE2 on 32-bit x86 (check at runtime since not always available)
        if is_x86_feature_detected!("sse2") {
            let s0 = _mm_loadu_si128(src as *const __m128i);
            let s1 = _mm_loadu_si128(src.add(16) as *const __m128i);
            let s2 = _mm_loadu_si128(src.add(32) as *const __m128i);
            let s3 = _mm_loadu_si128(src.add(48) as *const __m128i);
            _mm_storeu_si128(dst as *mut __m128i, s0);
            _mm_storeu_si128(dst.add(16) as *mut __m128i, s1);
            _mm_storeu_si128(dst.add(32) as *mut __m128i, s2);
            _mm_storeu_si128(dst.add(48) as *mut __m128i, s3);
        } else {
            ptr::copy_nonoverlapping(src, dst, 64);
        }
    }
    
    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;
        // NEON: 128-bit = 16 bytes per register
        let s0 = vld1q_u8(src);
        let s1 = vld1q_u8(src.add(16));
        let s2 = vld1q_u8(src.add(32));
        let s3 = vld1q_u8(src.add(48));
        vst1q_u8(dst, s0);
        vst1q_u8(dst.add(16), s1);
        vst1q_u8(dst.add(32), s2);
        vst1q_u8(dst.add(48), s3);
    }
    
    #[cfg(target_arch = "arm")]
    {
        // ARMv7 with NEON - use inline assembly for guaranteed NEON usage
        #[cfg(target_feature = "neon")]
        {
            use std::arch::arm::*;
            let s0 = vld1q_u8(src);
            let s1 = vld1q_u8(src.add(16));
            let s2 = vld1q_u8(src.add(32));
            let s3 = vld1q_u8(src.add(48));
            vst1q_u8(dst, s0);
            vst1q_u8(dst.add(16), s1);
            vst1q_u8(dst.add(32), s2);
            vst1q_u8(dst.add(48), s3);
        }
        #[cfg(not(target_feature = "neon"))]
        {
            ptr::copy_nonoverlapping(src, dst, 64);
        }
    }
    
    #[cfg(not(any(target_arch = "x86_64", target_arch = "x86", target_arch = "aarch64", target_arch = "arm")))]
    {
        ptr::copy_nonoverlapping(src, dst, 64);
    }
}

/// Fast bulk memory copy using SIMD
/// 
/// # Safety
/// Caller must ensure src and dst point to valid memory of at least `len` bytes
#[inline(always)]
unsafe fn fast_memcpy(src: *const u8, dst: *mut u8, len: usize) {
    let mut offset = 0;
    
    // Copy 64-byte chunks
    while offset + 64 <= len {
        copy_64_bytes(src.add(offset), dst.add(offset));
        offset += 64;
    }
    
    // Copy remaining bytes
    if offset < len {
        ptr::copy_nonoverlapping(src.add(offset), dst.add(offset), len - offset);
    }
}

/// Convert DRM fourcc format to PixelFormat
fn fourcc_to_pixel_format(fourcc: u32) -> PixelFormat {
    // DRM fourcc codes (little-endian)
    const DRM_FORMAT_XRGB8888: u32 = 0x34325258; // XR24
    const DRM_FORMAT_ARGB8888: u32 = 0x34325241; // AR24
    const DRM_FORMAT_XBGR8888: u32 = 0x34324258; // XB24
    const DRM_FORMAT_ABGR8888: u32 = 0x34324241; // AB24
    const DRM_FORMAT_RGB565: u32 = 0x36314752; // RG16
    const DRM_FORMAT_BGR888: u32 = 0x34324742; // BG24
    const DRM_FORMAT_RGB888: u32 = 0x34324752; // RG24

    match fourcc {
        DRM_FORMAT_XRGB8888 => PixelFormat::Bgrx8888,
        DRM_FORMAT_ARGB8888 => PixelFormat::Bgra8888,
        DRM_FORMAT_XBGR8888 => PixelFormat::Rgbx8888,
        DRM_FORMAT_ABGR8888 => PixelFormat::Rgba8888,
        DRM_FORMAT_RGB565 => PixelFormat::Rgb565,
        DRM_FORMAT_BGR888 => PixelFormat::Bgr888,
        DRM_FORMAT_RGB888 => PixelFormat::Rgb888,
        _ => PixelFormat::Unknown(fourcc),
    }
}
