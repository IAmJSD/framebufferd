# framebufferd

A Linux daemon that provides permission-controlled access to screen capture via an HTTP/2 Unix socket API. It bypasses display servers (Wayland/X11) and directly accesses the GPU framebuffer.

## Warnings

**Not for production use.** This project is experimental.

**Only tested on NVIDIA GPUs.** Other hardware may not work correctly or at all.

**Not suitable for X environments where the X server is serving multiple systems.** This daemon captures directly from the GPU framebuffer, which will not work correctly in multi-seat or remote X configurations.

## Requirements

- Linux with DRM/KMS support (or legacy framebuffer)
- Root privileges (required for `/dev/dri/*` and `/dev/fb*` access)
- User must be logged into a desktop environment

## Building

```bash
cargo build --release
```

## Running

```bash
sudo ./target/release/framebufferd
```

The daemon listens on a Unix socket (default: `/run/framebufferd.sock`).

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `FRAMEBUFFERD_SOCKET_PATH` | `/run/framebufferd.sock` | Unix socket path |
| `FRAMEBUFFERD_DATA_STORAGE_PATH` | `/var/lib/framebufferd/data_storage.db` | SQLite database path |

## API

**All clients MUST use HTTP/2 via h2c (HTTP/2 cleartext over Unix socket).** Standard HTTP/1.1 will not work.

### Authentication

All capture endpoints require an `X-Auth-Token` header. Tokens are obtained via the `/authorize` endpoint and expire after 24 hours.

### Endpoints

#### `GET /authorize`

Request authorization to use framebuffer operations. Displays a GUI dialog to the logged-in user.

**Response:**
```json
{
  "token": "550e8400-e29b-41d4-a716-446655440000",
  "expiration": 1703088000000
}
```

- `token`: UUID string for authenticating subsequent requests
- `expiration`: Token expiration time (milliseconds since epoch)

---

#### `PATCH /renew?token=<token>`

Extend token expiration by 24 hours.

**Query Parameters:**
- `token` (required): The token to renew

**Response:**
```json
{
  "new_expiration": 1703174400000
}
```

---

#### `GET /list`

List all available display devices (DRM and legacy framebuffer).

**Headers:**
- `X-Auth-Token` (required)

**Response:**
```json
[
  {
    "is_drm": true,
    "fb_id": 0,
    "width": 1920,
    "height": 1080,
    "x_offset": 0,
    "y_offset": 0,
    "file_path": "/dev/dri/card0"
  },
  {
    "is_drm": false,
    "id": "fb0",
    "width": 1024,
    "height": 768,
    "virtual_width": 1024,
    "virtual_height": 768,
    "bits_per_pixel": 32,
    "bytes_per_line": 4096
  }
]
```

---

### Framebuffer Capture (Legacy)

For capturing from `/dev/fb*` devices.

#### `GET /fb/raw_buffer?id=<device_id>`

Get raw framebuffer data.

**Query Parameters:**
- `id` (required): Framebuffer device ID (e.g., `fb0`)

**Headers:**
- `X-Auth-Token` (required)

**Response Headers:**
- `x-format`: Pixel format (e.g., `bgra8888`, `rgb565`)
- `x-width`: Width in pixels
- `x-height`: Height in pixels
- `x-pitch`: Bytes per line

**Response Body:** Raw pixel data

---

#### `GET /fb/rgba?id=<device_id>`

Get RGBA-converted framebuffer data.

**Query Parameters:**
- `id` (required): Framebuffer device ID

**Headers:**
- `X-Auth-Token` (required)

**Response Headers:**
- `x-width`: Width in pixels
- `x-height`: Height in pixels

**Response Body:** RGBA pixel data (4 bytes per pixel)

---

#### `GET /fb/png?id=<device_id>`

Get PNG-encoded screenshot.

**Query Parameters:**
- `id` (required): Framebuffer device ID

**Headers:**
- `X-Auth-Token` (required)

**Response:** PNG image

---

### DRM Capture

For capturing from modern DRM/KMS devices (`/dev/dri/*`).

#### `GET /drm/raw_buffer?fb_id=<id>&device_path=<path>&vsync=<bool>`

Get raw DRM buffer data.

**Query Parameters:**
- `fb_id` (required): Framebuffer ID from `/list`
- `device_path` (required): Device path (e.g., `/dev/dri/card0`)
- `vsync` (optional): Wait for vertical sync (`true`/`false`, default `false`)

**Headers:**
- `X-Auth-Token` (required)

**Response Headers:**
- `x-format`: Pixel format
- `x-width`: Width in pixels
- `x-height`: Height in pixels
- `x-pitch`: Bytes per line

**Response Body:** Raw pixel data

---

#### `GET /drm/rgba?fb_id=<id>&device_path=<path>&vsync=<bool>`

Get RGBA-converted DRM buffer data.

**Query Parameters:**
- `fb_id` (required): Framebuffer ID
- `device_path` (required): Device path
- `vsync` (optional): Wait for vertical sync

**Headers:**
- `X-Auth-Token` (required)

**Response Headers:**
- `x-width`: Width in pixels
- `x-height`: Height in pixels

**Response Body:** RGBA pixel data (4 bytes per pixel)

---

#### `GET /drm/png?fb_id=<id>&device_path=<path>&vsync=<bool>`

Get PNG-encoded screenshot from DRM.

**Query Parameters:**
- `fb_id` (required): Framebuffer ID
- `device_path` (required): Device path
- `vsync` (optional): Wait for vertical sync

**Headers:**
- `X-Auth-Token` (required)

**Response:** PNG image

---

### HTTP Status Codes

| Code | Meaning |
|------|---------|
| 200 | Success |
| 400 | Missing required parameters |
| 401 | Token expired or invalid |
| 403 | User not logged in or access denied |
| 404 | Device or endpoint not found |
| 405 | Wrong HTTP method |
| 500 | Capture or processing failure |

## Client Example

See `examples/client_hyper.rs` for a complete example using the Hyper HTTP/2 client.

Basic workflow:
1. Connect to the Unix socket
2. Perform HTTP/2 handshake (h2c)
3. Call `GET /authorize` to obtain a token
4. Use the token in `X-Auth-Token` header for capture requests
5. Renew token with `PATCH /renew` before expiration

## License

See LICENSE file.
