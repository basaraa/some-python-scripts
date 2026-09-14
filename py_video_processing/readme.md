# Video Batch Processor 🎬✂️

A lightweight and powerful Python Tkinter GUI tool designed for batch and individual video processing, cutting, cropping, resolution scaling, HDR tonemapping, and conversion powered by **FFmpeg**.

---

## 🚀 Features

- **Batch & Single-File Processing:** Process multiple videos simultaneously with either global settings or custom per-video overrides.
- **HDR to SDR Tonemapping:**
  - Automatic detection of HDR10, HLG, and Dolby Vision metadata.
  - High-quality tonemapping to standard SDR (BT.709) using FFmpeg's `libplacebo` (BT.2390 tonemapping algorithm).
- **Visual CROP Tool:**
  - Crop boundaries in **pixels** or **percentages** (Left, Right, Top, Bottom).
  - Real-time visual canvas preview with an exact bounding box overlay.
  - Apply globally across all videos or save specific crop overrides per file.
- **Visual CUT & Trim:**
  - Precise Start and End time trimming (`HH:MM:SS`).
  - Integrated in-window video player and timeline slider for seeking exact frames.
  - Quick-action buttons: *Set START = current frame* and *Set END = current frame*.
- **Profile-Based Resolution Scaling:**
  - Automatically groups files by their source dimensions.
  - Customize output resolutions per aspect ratio / input profile using high-quality **Lanczos** scaling.
- **Video & Audio Encoding:**
  - **Hardware Acceleration:** Auto-detects **NVIDIA NVENC** (`hevc_nvenc`) with tunable CQ quality and NVENC presets (`p1`–`p7`).
  - **Software Encoding:** High-efficiency CPU encoding with `libx265` (`CRF` + `medium` preset).
  - **Audio Conversion:** Convert incompatible EAC3 tracks to AAC (160 kbps) or perform lossless stream copying (`copy`).
  - **Audio Track Filtering & Export:** Keep specific audio tracks by index, or export raw audio directly to `.m4a`.
  - **Container Selection:** Export to `.mkv`, `.mp4`, `.mov`, or `.webm`.
- **Smart Subtitle Management:**
  - **Normalized Language Filtering:** Standardizes language tags (`cs`/`cze`/`ces`/`cz`, `sk`/`slo`/`slk`, `en`/`eng`, `zh`/`chi`/`zho`, etc.) so filtering works consistently across mixed batch files.
  - **Automatic WebVTT to SRT Conversion:** Automatically detects and converts problematic MKV `S_TEXT/WEBVTT` tracks into clean `.srt` without language swapping or desync.
  - **Subtitle Extraction:** Extract subtitles from MKV into standalone files (`.srt`, `.ass`, `.sup`).
  - **External SRT Muxing:** Automatically merges external `.srt` subtitles matching the video filename.

---

## 📦 Requirements

### 1. System Dependencies (External Tools)

These external command-line binaries must be installed and accessible in your system **PATH**:

| Tool | Purpose | Required |
| :--- | :--- | :--- |
| **FFmpeg** | Video, audio, subtitle encoding/filtering (built with `libplacebo` for HDR tonemapping) | **Yes** |
| **FFprobe** | Extracting media stream metadata and technical info | **Yes** |
| **MKVToolNix** (`mkvextract`) | Fast subtitle track extraction from MKV files | *Optional* (has built-in Python EBML parser fallback) |

> 💡 **Windows Tip:** You can quickly install full FFmpeg via winget:  
> ```powershell
> winget install Gyan.FFmpeg
> ```  
> or download full builds from [ffmpeg.org](https://ffmpeg.org/download.html) / [gyan.dev](https://www.gyan.dev/ffmpeg/builds/).

---

### 2. Python Dependencies

- **Python 3.8+**
- Standard library modules used: `tkinter`, `subprocess`, `json`, `threading`, `tempfile`, `base64`, `re`, `shutil`, `os`, `time`.

The only external Python package required (for smooth in-app frame rendering and playback):

```bash
pip install pillow
```

*(Note for Linux users: You may also need to install the system Tkinter package: `sudo apt install python3-tk`)*

---

## 🛠️ Usage
1. Clone or download this repository.
2. Place the script in your working directory or open it directly:
   ```bash
   python main.py
   ```
3. Click "Add videos..." or let it automatically load videos from the current folder via "Refresh videos".
4. Configure your desired operations (Crop, Cut, Resolution, Audio/Video convert, HDR Tonemapping, Subtitles).
5. Click "START PROCESSING". Output files will be saved with the _processed suffix in the same directory.