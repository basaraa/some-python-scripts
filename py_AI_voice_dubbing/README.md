# Local AI Video Dubbing & Localization Studio 🎬🎙️

An advanced, offline-first local AI dubbing and multi-voice narration studio built with Python and Tkinter. It automates speech separation, character diarization, pitch-aware voice assignment, neural speech synthesis (Supertonic, Edge-TTS, Coqui XTTS-v2), and adaptive audio mastering with full sound effect (SFX) preservation.

> [!NOTE]
> **Research & Educational Exploration**: This project was developed as an experimental exploration to test the boundaries of artificial intelligence in audio-to-audio dubbing and text-to-audio speech synthesis. Please see the [Disclaimer & Ethical Use](#-disclaimer--educational--research-purpose) section before using this tool.

---

## 🌟 Key Features

### 🎬 Tab 1: Video Dubbing Studio
* **Automated Audio Extraction & Separation**:
  * Extracts audio from any container (`.mp4`, `.mkv`, `.avi`, `.mov`, `.webm`, etc.).
  * Uses **HTDemucs** (`--two-stems vocals`) to isolate clean dialogue while isolating the background music and sound effects (BGM & SFX).
* **AI Character Diarization & F0 Pitch Profiling**:
  * Offline speaker clustering via **SpeechBrain ECAPA-TDNN** or **Resemblyzer**.
  * **Dual Pitch Profiles**:
    * **Asia (China, Japan, Korea)** *(Default)*: Tailored for tonal and pitch-accent languages with higher fundamental frequencies (Male: < 160 Hz, Female: > 185 Hz, Dead-zone: 160–185 Hz).
    * **Western / Europe (US, EU)**: Tailored for lower European male pitches (Male: < 142 Hz, Female: > 168 Hz, Dead-zone: 142–168 Hz).
  * Automatically tags gender (`Male 👨`, `Female 👩`, `? Uncertain`) and assigns suitable voices.
  * Silent subtitles (e.g., text/signs on walls, graphic cards, scene breaks) automatically default to `Speaker_01` with voice `⚡ Supertonic [M1]`.
* **Character Voice Customization**:
  * Interactive dropdown for detected characters to choose any voice:
    * ⚡ **Supertonic**: Local ONNX neural voices (`M1`..`M5`, `F1`..`F5`).
    * 🎙️ **Edge-TTS**: High-fidelity neural voices.
    * 🧬 **Voice Cloning**: Any custom `.wav` / `.mp3` reference voice from `./voices/`.
  * **Preview Button**: Instantly sample any voice before rendering.
* **Full-Featured Subtitle & Voice Table**:
  * Displays timecodes, assigned character, assigned voice, and subtitle text.
  * Extended selection (hold `Ctrl` or `Shift` to multi-select lines) with sortable columns (by ID, timing, character, voice, or text).
  * Override individual lines with custom voices or reset back to character defaults.
* **Smart Incremental Caching & 3-Phase Subtitle Sync**:
  * Content-hash caching (`sub_{hash12}.wav`): editing or deleting lines in your `.srt` file does **not** cause cascading re-generation.
  * Only modified lines are synthesized; unchanged lines load instantly from disk cache.
* **Studio Master Mixer with SFX & Background Preservation**:
  * Dubbing attenuation mask only applies while speech is actively playing (using synthesized audio duration, not wide `.srt` intervals).
  * **100% Bit-Perfect Passthrough**: Background music, room ambiance, whispers, screams, spell sounds, and foley before and after each sentence are fully preserved without dead silence.
  * **Dynamic Transient Detection**: High-energy spikes (screams, crashes, glass breaking, sword clashes, magic whooshes) pass through at up to 65% volume even during speech.
  * Ducking slider (default `0%` disabled) allows fine-tuning background attenuation.
* **🚀 Batch Dubbing Mode**:
  * Dub entire directories of videos with a single click.
  * Automatically pairs video files with matching subtitles (e.g., `movie.mp4` with `movie.srt`, `movie.en.srt`, `movie.ja.srt`, or `movie.cs.srt`).
  * Automatically detects already dubbed files and skips processing if the `.srt` has not been modified.

---

### 🎙️ Tab 2: Text Narration & Multi-Voice Speech
* **Sentence-Level Multi-Voice Synthesis**:
  * Paste text or import `.txt` / `.srt` files; sentences are automatically parsed (supporting English, European punctuation `.!?`, and Japanese/CJK punctuation `。！？「」`).
  * Assign different voices to different sentences or characters in a sortable, multi-select table.
* **Companion Cache Verification**:
  * Every generated sentence chunk generates a companion `.txt` signature (`[engine:voice:speed]\ntext`).
  * If text or voice changes, only that specific chunk is regenerated; existing chunks are reused seamlessly.
* **Instant Export**:
  * Concatenates all sentence chunks with natural pauses, normalizes peak levels, and exports to 160 kbps MP3 or uncompressed WAV.

---

### 🌐 Supported Languages & TTS Engines
| Language | Code | Supertonic (Local ONNX) | Edge-TTS (Online) | XTTS-v2 (Cloning) |
|---|:---:|:---:|:---:|:---:|
| **Czech** | `cs` | ✅ (M1-M5, F1-F5) | ✅ Antonín, Vlasta | ✅ Native |
| **Slovak** | `sk` | ✅ (M1-M5, F1-F5) | ✅ Lukáš, Viktória | ✅ Native (CS/SK) |
| **Japanese** | `ja` | ✅ (M1-M5, F1-F5) | ✅ Nanami, Keita, Aoi, Naoki | ✅ Native |
| **English** | `en` | ✅ (M1-M5, F1-F5) | ✅ Guy, Jenny, Christopher | ✅ Native |
| **German** | `de` | ✅ (M1-M5, F1-F5) | ✅ Conrad, Katja | ✅ Native |
| **Polish** | `pl` | ✅ (M1-M5, F1-F5) | ✅ Marek, Zofia | ✅ Native |

---

## 💻 System Requirements

* **Operating System**: Windows 10/11, Linux, or macOS.
* **Python**: `3.10`, `3.11`, or `3.12` (Python 3.12 recommended).
* **FFmpeg**: Must be installed and accessible in your system `PATH`.
* **Hardware**:
  * **GPU** (Recommended): NVIDIA GPU with 4GB+ VRAM (CUDA 12.x or 11.8).
  * **CPU**: Multicore CPU with 8GB+ RAM (Supertonic runs locally on CPU via ONNX Runtime with zero idle thread-spinning).

---

## 📦 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/voice_dub.git
cd voice_dub
```

### 2. Install FFmpeg
* **Windows**: Download from [gyan.dev](https://www.gyan.dev/ffmpeg/builds/) or install via winget / scoop:
  ```powershell
  winget install Gyan.FFmpeg
  ```
  Ensure `ffmpeg -version` works in your terminal.
* **Linux (Ubuntu/Debian)**:
  ```bash
  sudo apt update && sudo apt install -y ffmpeg
  ```
* **macOS**:
  ```bash
  brew install ffmpeg
  ```

### 3. Install PyTorch with CUDA
For optimal performance with an NVIDIA GPU, install PyTorch with CUDA support first:
```bash
# For CUDA 12.4 / 12.6
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Or for CPU-only:
pip install torch torchvision torchaudio
```

### 4. Install Dependencies
Install all required packages using the provided `requirements.txt`:
```bash
pip install -r requirements.txt
```

Or install packages individually:
```bash
pip install supertonic edge-tts soundfile sounddevice numpy scipy demucs TTS transformers onnxruntime speechbrain resemblyzer faster-whisper
```

---

## 🚀 Running the Application

Launch the studio by running:
```bash
python dubbing.py
```

---

## 📁 Directory Structure

```text
voice_dub/
├── dubbing.py                  # Main application script
├── requirements.txt            # Python dependencies
├── README.md                   # Documentation
│
├── voices/                     # Custom reference voice cloning library
│   ├── narrator.mp3            # Reference audio sample (3-15 seconds)
│   ├── narrator.txt            # (Optional) Companion text of what is said
│   └── hero_voice.wav
│
├── output_narration/           # Output directory for Tab 2 MP3 speech exports
│
└── <video_name>_ai_dub/        # Auto-generated project directory per video (Tab 1)
    ├── project.json            # Persistent project state, character & line voice mapping
    ├── audio/
    │   └── extracted_audio.wav # Full raw audio track extracted from the video
    ├── separation/
    │   ├── vocals.wav          # Isolated dialogue track (HTDemucs)
    │   └── bgm.wav             # Isolated background music & SFX track
    ├── speakers/
    │   └── Speaker_01_sample.wav # Clean preview sample for each character
    ├── generated/
    │   ├── sub_a1b2c3d4e5f6.wav # Incremental audio chunk for each subtitle line
    │   └── sub_a1b2c3d4e5f6.txt # Cache signature for instant change detection
    └── output/
        ├── final_master_audio.wav # Master mixed audio track
        └── <video>_dubbed_<lang>_<timestamp>.mkv # Final dubbed video output
```

---

## 📖 Usage Guide

### Tab 1: Video Dubbing Workflow

1. **Select Video File**:
   * Click **Browse...** to pick your input video (`.mp4`, `.mkv`, etc.).
2. **Select Subtitles**:
   * **Mode A (Existing Subtitles)**: Pick your `.srt` / `.ass` subtitle file translated into the target language.
   * **Mode B (Auto-Transcribe)**: Auto-transcribes the original audio using Whisper if no subtitles are available.
3. **Configure Settings**:
   * **Default Engine**: Choose Supertonic (Local), Edge-TTS (Online), or Voice Cloning (XTTS-v2).
   * **Language**: Select target language (`cs`, `sk`, `ja`, `en`, `de`, `pl`).
   * **Gender / F0**: Choose `Asia (China, Japan, Korea)` for Asian films/anime, or `Western / Europe` for European/American content.
   * **Ducking**: Leave at `0%` to preserve all background music and sound effects at full volume.
4. **Step 1: Analyze & Diarize**:
   * Click **▶ 1. Analyze & Diarize**.
   * The system separates audio with HTDemucs, detects distinct characters, and estimates pitch.
   * Preview characters in the table and customize their default voices using the **Voice for character** dropdown and **✔ Assign Voice to Character**.
5. **Customize Subtitle Lines (Optional)**:
   * Select any line(s) in the subtitles table (hold `Ctrl` or `Shift`).
   * Choose a specific voice from the toolbar and click **✓ Apply to Selected**.
6. **Step 2: Generate Dub & Export Video**:
   * Click **🚀 2. Generate Dub & Export Video**.
   * Progress is shown in the progress bar and real-time log terminal.
   * The finalized dubbed video is exported to `<video>_ai_dub/output/`.

### Tab 1: Batch Dubbing
To dub multiple episodes or clips at once:
1. Place all your video files and their corresponding `.srt` files in the same folder.
   * Example: `episode_01.mkv` and `episode_01.srt` (or `episode_01.ja.srt`, `episode_01.cs.srt`).
2. Open the studio, select one of the videos, configure your desired language and voices.
3. Click **🚀 Batch Dubbing (Folder)**.
4. The studio iterates through all video pairs automatically, skipping any that are already up to date.

---

### Tab 2: Text Narration & Multi-Voice Speech

1. Choose your language in **Language** (`English`, `Czech`, `Slovak`, `Japanese`, etc.).
2. Choose a **Default Voice** and speech **Speed**.
3. Type or paste text into the main text box, or click **📂 Load from File (.txt / .srt)**.
4. Sentences appear automatically in the **Multi-Voice Phrase Assignment** table.
5. Highlight individual sentences with `Ctrl` or `Shift`, select different voices, and click **✓ Apply to Selected**.
6. Click **🎙️ Generate Speech (MP3 160 kbps)**.
7. Click **▶ Play Generated MP3** or **💾 Save MP3 As...** to export your audio.

---

### 🧬 Custom Voice Cloning (`voices/` Folder)

To clone any real person or character voice:
1. Drop a clean audio clip (`3` to `15` seconds) of the person speaking into the `./voices/` directory:
   * Supported formats: `.wav`, `.mp3`, `.flac`, `.m4a`.
   * Example: `voices/gandalf.mp3`.
2. *(Optional but Recommended)* Create a companion text file with the exact same basename:
   * Example: `voices/gandalf.txt` containing: `You shall not pass!`
   * If you don't provide a `.txt` file, the studio will automatically transcribe the audio using Whisper on first use and save the companion `.txt` for you.
3. Your custom voice will immediately appear in all voice dropdowns prefixed with `🧬 [Clone]`.

---

## 🛠️ Performance & CPU Optimizations

* **Zero Idle CPU**: ONNX Runtime thread spinning is disabled (`session.intra_op.allow_spinning = 0`), ensuring `0%` CPU utilization when idle.
* **OpenMP / Thread Limits**: Threads are throttled (`OMP_WAIT_POLICY=PASSIVE`, `OMP_NUM_THREADS=4`) to prevent background thread starvation.
* **VRAM Management**: Automatic CUDA memory cleanup (`torch.cuda.empty_cache()` and `gc.collect()`) between separation, diarization, and synthesis steps.

---

## ⚖️ Disclaimer & Educational / Research Purpose

> [!IMPORTANT]
> **Research & Boundary Testing Only**:
> This software was developed solely as an experimental project to research and test the practical capabilities and limits of modern AI in automated **audio-to-audio dubbing** and **text-to-audio voice synthesis**.
>
> **No Malicious Intent**:
> This project was created strictly for legitimate educational, creative, localization, and academic exploration. It was **not** created, designed, or intended for any harmful, deceptive, or malicious purposes (such as unauthorized voice cloning, deepfakes, impersonation, fraud, defamation, or misleading media).
>
> **Limitation of Liability**:
> The author and contributors accept **no responsibility or liability** whatsoever for any misuse, unlawful conduct, or damages resulting from the use of this software or any media generated with it. By using this tool, the user agrees that:
> 1. You are solely responsible for ensuring you have all required legal rights, consents, or copyright permissions for any media, video, audio, or reference voices processed.
> 2. You will strictly adhere to all applicable local, national, and international laws, intellectual property rights, and privacy regulations regarding voice likeness and data protection.

---

## 📄 License

This project is open-source under the MIT License.

