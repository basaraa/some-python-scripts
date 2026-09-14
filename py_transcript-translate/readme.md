# 🎬 Whisper + Meta NLLB AI Subtitle Generator & Translator

A high-performance, fully local GUI desktop application for automatic video/audio transcription and neural translation into SRT subtitles. 

Powered by **faster-whisper** (CTranslate2-optimized Whisper) for ultra-fast speech recognition and **Meta AI's NLLB-200 (No Language Left Behind)** for offline neural machine translation across 200+ languages.

---

## ✨ Features

- **⚡ Blazing Fast Transcription:** Uses `faster-whisper` with CTranslate2 backend, up to 4x faster than standard OpenAI Whisper with reduced VRAM/RAM footprint.
- **🧠 100% Local Neural AI Translation:** 
  - Direct **Whisper native translation** to English.
  - Multi-language translation (Slovak, Czech, German, Spanish, French, etc.) powered locally by **Meta NLLB-200** neural network (no third-party cloud APIs, no data sent online).
- **🚀 GPU Batch Acceleration:** Video segments are batched and translated on GPU in parallel for maximum throughput.
- **🎙️ Built-in VAD (Voice Activity Detection):** Automatically filters out silence and background noise, improving accuracy and speed while eliminating hallucinations.
- **📂 Zero-Overhead Video Processing:** Reads audio directly from containers (`.mp4`, `.mkv`, `.avi`, `.mov`, `.mp3`, `.wav`, etc.) via PyAV without writing temporary `.wav` files to disk.
- **🖥️ Non-blocking Tkinter GUI:** Clean desktop interface with live console logging, thread-safe background execution, and model selectors.
- **📝 Standard SRT Output:** Outputs synchronized subtitles with standardized naming conventions (e.g., `video.srt` or `video.sk.srt`).

---

## 🛠️ Requirements & Prerequisites

### 1. System Requirements
- **OS:** Windows 10/11, macOS, or Linux
- **Python:** Python `3.9` to `3.12`
- **Hardware:**
  - **GPU (Recommended):** NVIDIA GPU with CUDA support for near real-time processing.
  - **CPU:** Supported automatically with multi-threaded INT8 quantization.

### 2. FFmpeg (System Requirement)
`faster-whisper` and PyAV rely on FFmpeg libraries to decode audio tracks.

- **Windows:** Download from [gyan.dev](https://www.gyan.dev/ffmpeg/builds/) and add the `bin` folder to your System `PATH` (or run `winget install Gyan.FFmpeg`).
- **macOS:** `brew install ffmpeg`
- **Linux (Ubuntu/Debian):** `sudo apt update && sudo apt install ffmpeg`

---

## 📦 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/whisper-ai-subtitles.git
   cd whisper-ai-subtitles
   ```

2. **(Optional) Create a virtual environment:**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Linux/macOS:
   source venv/bin/activate
   ```

3. **Install PyTorch with CUDA support (for NVIDIA GPU users):**
   > *Note: If you only plan to use CPU, you can skip this step and proceed directly to step 4.*

   Check [pytorch.org](https://pytorch.org/) for your specific CUDA version, for example:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

4. **Install Python dependencies:**
   ```bash
   pip install faster-whisper transformers sentencepiece
   ```

---

## 🚀 How to Use

1. Launch the application:
   ```bash
   python whisper_ai_translator.py
   ```

2. **Select Directory:** Choose the folder containing your media files.
3. **Select Whisper Model:**
   - `large-v3-turbo` *(Recommended)*: Optimal balance of top-tier quality and small-model speed.
   - `medium` / `small`: Ideal for lower-end hardware or CPU-only setups.
4. **Choose Task:**
   - **Transcription only:** Leave "Enable AI Translation" unchecked to generate subtitles in the spoken language.
   - **Translation:** Check the box and select your desired target language:
     - **English (en):** Uses Whisper's built-in translation engine.
     - **Other Languages (sk, cs, de, fr, etc.):** Uses the local Meta NLLB-200 AI model.
5. **Click "Start Transcription & Translation":** The script will process all media files in the folder and save `.srt` files right next to each video.

---

## 🏗️ Architecture & How It Works

```text
┌────────────────────────────────┐
│ Video/Audio Files (.mp4, .mkv) │
└───────────────┬────────────────┘
                │ PyAV (Direct memory stream)
                ▼
┌────────────────────────────────┐
│      faster-whisper (VAD)      │ ──► Transcribes speech + timestamps
└───────────────┬────────────────┘
                │
         Is Translation Enabled?
        /                        \
      NO                          YES
      │                            │
      │                     Target Language?
      │                    /                \
      │             English (en)         Non-English (sk, cs, de...)
      │                  │                            │
      │         Whisper Native Translate       Meta NLLB-200 AI
      │                  │                   (Batched GPU Translation)
      ▼                  ▼                            ▼
┌─────────────────────────────────────────────────────────────┐
│                 Formatted .SRT Subtitle File                │
└─────────────────────────────────────────────────────────────┘
```

---

## 📋 `requirements.txt`

```text
faster-whisper>=1.0.0
torch>=2.0.0
transformers>=4.38.0
sentencepiece>=0.2.0
```

---

## ⚖️ License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- [OpenAI Whisper](https://github.com/openai/whisper) for the core speech recognition foundation.
- [SYSTRAN faster-whisper](https://github.com/SYSTRAN/faster-whisper) for CTranslate2 optimization.
- [Meta AI NLLB Team](https://github.com/facebookresearch/fairseq/tree/nllb) for the No Language Left Behind translation model.
- [Hugging Face](https://huggingface.co/) for the Transformers model hub and runtime.