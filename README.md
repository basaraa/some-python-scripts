# Some Python Scripts 🐍

A collection of open-source Python scripts, desktop GUI utilities, machine learning projects, and multimedia automation tools created for personal productivity, experimentation, and educational purposes.

---

## ⚠️ Disclaimer & Legal Notice

> [!IMPORTANT]
> **Universal Disclaimer Applicable to All Scripts in this Repository:**
>
> 1. **Personal & Non-Commercial Purpose:**  
>    Every script and project in this repository was conceived, developed, and maintained strictly for **personal, educational, research, and non-commercial** needs. None of these tools were designed, intended, or encouraged for any illicit, unauthorized, harmful, or malicious activities.
>
> 2. **No Malicious Intent:**  
>    These utilities are meant for legitimate workflows such as managing personal media archives, academic machine learning exploration, learning cryptography fundamentals, and local offline productivity. They must **not** be used for copyright infringement, unauthorized redistribution, privacy violations, unauthorized scraping, deepfake/impersonation fraud, or any illegal conduct.
>
> 3. **Limitation of Liability & User Responsibility:**  
>    The author accepts **no liability or responsibility whatsoever** for any misuse, damage, data loss, legal claims, or regulatory consequences arising from the use, modification, or distribution of this software or any media generated with it.  
>    **You, the user, are solely responsible** for ensuring that your usage complies with all applicable local, national, and international laws, intellectual property rights, data protection regulations, and third-party terms of service.
>
> 4. **"AS IS" Warranty:**  
>    All code in this repository is provided on an **"AS IS" basis**, without warranty of any kind, express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose, or non-infringement.

---

## 📂 Repository Structure & Project Overview

Below is an overview of the tools and modules included in this repository, grouped by domain:

### 🎙️ AI & Multimedia Processing (Offline & Local)

| Project | Description | Primary Technologies |
| :--- | :--- | :--- |
| [**`py_AI_voice_dubbing/`**](./py_AI_voice_dubbing) | **Local AI Video Dubbing & Narration Studio:** An offline-first desktop GUI that automates audio-vocal separation, speaker diarization, pitch-aware voice assignment, neural speech synthesis (Supertonic, Kokoro, Edge-TTS), and audio-video muxing. | Python, Tkinter, Demucs, Whisper, SoundFile |
| [**`py_transcript-translate/`**](./py_transcript-translate) | **AI Subtitle Generator & Neural Translator:** High-performance, offline transcription and translation suite powered by `faster-whisper` (CTranslate2) and Meta AI's `NLLB-200` to produce synchronized `.srt` subtitles in 200+ languages. | Python, Tkinter, faster-whisper, PyAV, Meta NLLB |

---

### 🎬 Media & Graphic Tools

| Project | Description | Primary Technologies |
| :--- | :--- | :--- |
| [**`py_video_processing/`**](./py_video_processing) | **Video Trimmer & Encoder GUI:** Lightweight desktop utility for cutting, trimming, profile-based aspect-ratio scaling, fast stream copying, and batch audio/video re-encoding. | Python, Tkinter, FFmpeg, FFprobe |
| [**`py_image_processor/`**](./py_image_processor) | **Batch Image Converter & Editor:** Non-destructive image processing pipeline supporting multi-format conversion (`PNG`, `JPEG`, `WEBP`, `BMP`, `TIFF`, `ICO`), interactive cropping, pixel/percentage resizing, and dynamic filter toggling. | Python, Tkinter, Pillow (PIL) |

---

### 💬 Subtitle Utilities

| Project | Description | Primary Technologies |
| :--- | :--- | :--- |
| [**`py_subtile_handler/`**](./py_subtile_handler) | **All-in-One Subtitle Toolbox:** Desktop GUI with 5 integrated modules: stream extraction from video containers, intelligent timestamp shifting (including anime OP gap detection), SRT↔TXT dialogue splitter for AI translations, style tag stripper, and ASS to SRT converter. | Python, Tkinter, FFmpeg, pysubs2 |
| [**`subtitles-ass-to-srt-converter/`**](./subtitles-ass-to-srt-converter) | Batch converter from Advanced SubStation Alpha (`.ass`) to SubRip (`.srt`) format with recursive directory traversal. | Python, Tkinter, asstosrt |
| [**`subtitles-shift-time/`**](./subtitles-shift-time) | Time-offset alignment tool to shift `.ass` and `.srt` subtitle timestamps forward or backward from the start or from a specific dialogue line. | Python, Tkinter, pysubs2 |
| [**`subtitles-ass-remove-styles/`**](./subtitles-ass-remove-styles) | Batch cleanup utility to strip all styling tags, bracket overrides (`{...}`), positioning, and color codes from `.ass` subtitle files. | Python, Tkinter, pysubs2 |
| [**`subtitles-ass-replace-all-fonts/`**](./subtitles-ass-replace-all-fonts) | Replaces font family names across all `.ass` subtitle files in a specified folder with a target font. | Python, Tkinter |

---

### 🧠 Machine Learning & Data Science (Academic Research)

*Developed as university assignments and exploratory projects in 2021.*

| Project | Description | Primary Technologies |
| :--- | :--- | :--- |
| [**`convolution-neural-network-types-of-food/`**](./convolution-neural-network-types-of-food) | **Food Classification CNN:** Convolutional neural network for classifying food images, featuring data augmentation, batch normalization, regularizers, checkpoints, and 2D/3D dimensionality reduction visualization. | Python, TensorFlow, Keras, Scikit-learn, Seaborn |
| [**`neural-network-wine-quality/`**](./neural-network-wine-quality) | **Wine Quality Predictor:** Neural network and logistic regression models predicting wine quality based on 13 physicochemical parameters, featuring early stopping and error curves. | Python, TensorFlow, Keras, Scikit-learn, Pandas |
| [**`SVM-regression-music-loudness/`**](./SVM-regression-music-loudness) | **Music Loudness Regressor:** Support Vector Machine (SVM) regression and ensemble learning (AdaBoost, Random Forest) to predict track loudness using Spotify audio features. | Python, Scikit-learn, SciPy, Pandas, Seaborn |

---

### 🔐 Cryptography & Security (Academic Fundamentals)

*Educational implementations demonstrating classic cryptographic ciphers and execution timing.*

| Project | Description | Primary Technologies |
| :--- | :--- | :--- |
| [**`AES-encrypt-decrypt/`**](./AES-encrypt-decrypt) | GUI utility for file encryption and decryption using the Advanced Encryption Standard (AES) with execution time measurement. | Python, Tkinter, PyCryptodome |
| [**`RSA-encrypt-decrypt/`**](./RSA-encrypt-decrypt) | GUI utility for asymmetric file encryption, decryption, and key-pair generation using the RSA algorithm with execution benchmarks. | Python, Tkinter, PyCryptodome |

---

### 🎮 Gaming & Utilities

| Project | Description | Primary Technologies |
| :--- | :--- | :--- |
| [**`naruto-online-calculator/`**](./naruto-online-calculator) | Desktop calculator for estimating coupon accumulation, savings timelines, and in-game resource management for the browser game *Naruto Online*. | Python, Tkinter |

---

## 📜 Open Source & License

This repository is **open-source** and provided free of charge for community reference, personal use, and educational learning. Individual subprojects may be governed by the [MIT License](https://opensource.org/licenses/MIT) or other compatible open-source terms. Please review the individual project directories and licenses for subproject-specific details and dependencies.
