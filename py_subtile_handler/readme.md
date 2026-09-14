# Subtitle Toolbox All-in-One

A comprehensive desktop GUI application built with Python and Tkinter designed for batch subtitle management, extraction, time-shifting, format conversion, and text cleaning.

---

## Features

The toolbox contains 5 dedicated modules (tabs):

1. **Extractor (Video to Subtitles)**
   - Scans single video files or whole directories (`.mp4`, `.mkv`, `.avi`, `.mov`).
   - Uses `ffprobe` to detect all embedded subtitle streams and their languages.
   - Allows selective extraction of specific subtitle tracks directly into `.srt` files using `ffmpeg`.

2. **Shifter (Time Alignment)**
   - Shifts timestamps (forward `+` or backward `-`) in `.ass` and `.srt` files.
   - Supports 3 modes:
     - **All lines:** Shifts the entire file.
     - **From word:** Starts shifting only from the line containing a specific keyword/dialogue.
     - **From OP (Opening gap):** Automatically detects gaps larger than 90 seconds (typical anime/TV opening) and shifts everything after it.

3. **SRT <-> TXT (Translator Helper)**
   - **SRT → TXT:** Separates subtitle dialogues into pure text (`.txt`) and stores timing indices separately (`_timing.txt`), making it easy to translate text in batch via AI or translation tools without breaking timecode structure.
   - **TXT → SRT:** Merges translated text back with the preserved timings into proper `.srt` files.

4. **Cleaner (Tag & Formatting Stripper)**
   - Strips ASS styling tags (e.g. `{\an8}`, `{\pos(...)}`, `{\c&H...}`) and formatting codes (`\N`, `\h`, `\H`) from `.ass` and `.srt` files.
   - Removes empty lines left after cleaning.

5. **ASS to SRT Converter**
   - Batch converts Advanced SubStation Alpha (`.ass`) files to SubRip (`.srt`) format.
   - Optional recursive folder scanning.
   - Option to automatically delete source `.ass` files after successful conversion.

---

## Prerequisites & Requirements

### 1. Python
- **Python 3.8+** is recommended.
- *Note for Linux users:* You may need to install Tkinter manually:
  ```bash
  sudo apt install python3-tk
  ```

### 2. External Dependencies (Crucial for Extractor)
- **FFmpeg & FFprobe**: Required for scanning and extracting subtitles from video files.
  - **Download:** [FFmpeg Official Downloads](https://ffmpeg.org/download.html) or via package managers:
    - **Windows (Winget):** `winget install Gyan.FFmpeg`
    - **macOS (Homebrew):** `brew install ffmpeg`
    - **Linux (Ubuntu/Debian):** `sudo apt install ffmpeg`
  - Ensure `ffmpeg` and `ffprobe` are added to your system's `PATH`.

### 3. Python Libraries
The script attempts to auto-install missing packages on launch, but you can install them manually:

```bash
pip install pysubs2 asstosrt
```