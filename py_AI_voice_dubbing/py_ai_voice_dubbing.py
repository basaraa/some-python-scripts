"""
LOCAL AI DUBBING & LOCALIZATION STUDIO
- Optimized CPU load: Disabled ONNX Runtime thread-spinning and active polling (0% idle CPU)
- Native Supertonic TTS support (10 voices: M1-M5, F1-F5) supporting English, Czech, Slovak, Japanese, and more
- Automatic speaker diarization with adaptive gender/F0 pitch profiling (Asia vs Western modes)
- Tab 1: Full-featured subtitle/line table with extended Ctrl/Shift selection, sortable columns, and per-line voice assignment
- Tab 1: Professional audio mixer with dynamic Foley/SFX passthrough (screams, spells, impacts, and background ambiance are preserved)
- Tab 1: Batch dubbing mode for processing entire folders with automatic video/subtitle matching
- Tab 2: Multi-voice narration with sentence parsing, companion cache verification (.txt signatures), and instant playback/export
- Universal: Automatic reference voice transcription with companion voicename.txt support for custom clones

DISCLAIMER & ETHICAL USE NOTICE:
This program was developed strictly as an experimental exploration to test and research the capabilities
and boundaries of modern AI in automated audio-to-audio dubbing and text-to-audio voice synthesis.
It was NOT created for, nor should it ever be used for, any malicious, deceptive, fraudulent, or defamatory
purposes (such as unauthorized voice cloning, deepfakes, or impersonation). The author and contributors
accept no liability or responsibility for any misuse, damage, or legal infringement arising from the use
of this software. By using this tool you agree to comply with all applicable laws and copyright regulations.
"""

import os

# ==============================================================================
# 0. CPU USAGE OPTIMIZATION (ONNX Runtime & OpenMP Thread-Spinning Fix)
# Must run before importing any heavy scientific libraries!
# ==============================================================================
os.environ["OMP_WAIT_POLICY"] = "PASSIVE"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

import sys
import json
import time
import math
import shutil
import asyncio
import subprocess
import threading
import queue
import gc
import re
import random
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import soundfile as sf
import sounddevice as sd
import edge_tts
import torch
import torch.nn.functional as F

# Patch pre ONNX Runtime na elimináciu spin-loopu v pozadí
try:
    import onnxruntime as ort
    _orig_ort_init = ort.InferenceSession.__init__
    def _safe_ort_init(self, *args, **kwargs):
        sess_opt = kwargs.get("sess_options", None)
        if sess_opt is None:
            sess_opt = ort.SessionOptions()
        # Disable aggressive spin-loop (prevents idle threads from consuming 89-100% CPU)
        sess_opt.add_session_config_entry("session.intra_op.allow_spinning", "0")
        sess_opt.intra_op_num_threads = min(4, os.cpu_count() or 4)
        sess_opt.inter_op_num_threads = 1
        sess_opt.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        kwargs["sess_options"] = sess_opt
        return _orig_ort_init(self, *args, **kwargs)
    ort.InferenceSession.__init__ = _safe_ort_init
except Exception:
    pass

# Attempt to import Supertonic TTS
try:
    from supertonic import TTS as SupertonicTTS
except ImportError:
    SupertonicTTS = None

# Restrict PyTorch threads for CPU efficiency
if torch.get_num_threads() > 4:
    torch.set_num_threads(4)

# Safe PyTorch padding fix for CUDA
_original_torch_pad = F.pad
def _safe_torch_pad(input, pad, mode='constant', value=0.0):
    if mode in ('reflect', 'replicate', 'circular') and input.dim() <= 2:
        if input.dim() == 1:
            x = input.unsqueeze(0).unsqueeze(0)
            res = _original_torch_pad(x, pad, mode=mode, value=value)
            return res.squeeze(0).squeeze(0)
        elif input.dim() == 2:
            x = input.unsqueeze(1)
            res = _original_torch_pad(x, pad, mode=mode, value=value)
            return res.squeeze(1)
    return _original_torch_pad(input, pad, mode=mode, value=value)

F.pad = _safe_torch_pad
torch.nn.functional.pad = _safe_torch_pad

# Compatibility fix for XTTS / Transformers (isin_mps_friendly)
try:
    import transformers.pytorch_utils
    if not hasattr(transformers.pytorch_utils, "isin_mps_friendly"):
        transformers.pytorch_utils.isin_mps_friendly = torch.isin
except Exception:
    pass

# Fix pre PyTorch 2.6+ weights_only
_orig_torch_load = torch.load
def _safe_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = _safe_torch_load

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLE_RATE = 24000
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VOICES_DIR = os.path.join(SCRIPT_DIR, "voices")
os.makedirs(VOICES_DIR, exist_ok=True)

SUPERTONIC_MALE_VOICES = ["M1", "M2", "M3", "M4", "M5"]
SUPERTONIC_FEMALE_VOICES = ["F1", "F2", "F3", "F4", "F5"]

SUPERTONIC_VOICE_CATALOG = [
    ("M1 (Male - Deep / Battle)", "M1"),
    ("M2 (Male - Natural Narrator)", "M2"),
    ("M3 (Male - Younger / Clear)", "M3"),
    ("M4 (Male - Calm / Deliberate)", "M4"),
    ("M5 (Male - Cinematic Charismatic)", "M5"),
    ("F1 (Female - Melodic / Warm)", "F1"),
    ("F2 (Female - Crisp / Clear)", "F2"),
    ("F3 (Female - Younger / Energetic)", "F3"),
    ("F4 (Female - Storyteller)", "F4"),
    ("F5 (Female - Gentle / Elegant)", "F5")
]

EDGE_TTS_VOICE_CATALOG = {
    "cs": [
        ("cs-CZ - Antonin (Male)", "cs-CZ-AntoninNeural"),
        ("cs-CZ - Vlasta (Female)", "cs-CZ-VlastaNeural")
    ],
    "sk": [
        ("sk-SK - Lukas (Male)", "sk-SK-LukasNeural"),
        ("sk-SK - Viktoria (Female)", "sk-SK-ViktoriaNeural")
    ],
    "ja": [
        ("ja-JP - Nanami (Female - Warm / Anime)", "ja-JP-NanamiNeural"),
        ("ja-JP - Keita (Male - Deep / Natural)", "ja-JP-KeitaNeural"),
        ("ja-JP - Aoi (Female - Gentle)", "ja-JP-AoiNeural"),
        ("ja-JP - Naoki (Male - Clear)", "ja-JP-NaokiNeural")
    ],
    "en": [
        ("en-US - Guy (Male)", "en-US-GuyNeural"),
        ("en-US - Jenny (Female)", "en-US-JennyNeural"),
        ("en-US - Christopher (Male)", "en-US-ChristopherNeural")
    ],
    "de": [
        ("de-DE - Conrad (Male)", "de-DE-ConradNeural"),
        ("de-DE - Katja (Female)", "de-DE-KatjaNeural")
    ],
    "pl": [
        ("pl-PL - Marek (Male)", "pl-PL-MarekNeural"),
        ("pl-PL - Zofia (Female)", "pl-PL-ZofiaNeural")
    ]
}

# ==============================================================================
# 1. AUDIO & SYSTEM UTILITIES
# ==============================================================================

def run_cmd(cmd, check=True):
    startupinfo = None
    if os.name == 'nt':
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, startupinfo=startupinfo)
    if check and p.returncode != 0:
        raise RuntimeError(f"Chyba príkazu: {' '.join(cmd)}\nStderr: {p.stderr}")
    return p

def convert_to_mp3_160k(input_audio, output_mp3):
    cmd = [
        "ffmpeg", "-y", "-i", input_audio,
        "-codec:a", "libmp3lame",
        "-b:a", "160k",
        "-ar", "44100",
        output_mp3
    ]
    run_cmd(cmd)
    return output_mp3

def trim_audio_silence(audio_arr, sr, db_thresh=26):
    if len(audio_arr) == 0:
        return audio_arr
    energy = np.abs(audio_arr)
    max_e = np.max(energy)
    if max_e == 0:
        return audio_arr
    thresh = max_e * (10 ** (-db_thresh / 20))
    indices = np.where(energy > thresh)[0]
    if len(indices) == 0:
        return audio_arr
    pad_s = int(sr * 0.04)
    pad_e = int(sr * 0.06)
    start_idx = max(0, indices[0] - pad_s)
    end_idx = min(len(audio_arr), indices[-1] + pad_e)
    res = audio_arr[start_idx:end_idx]
    return res if len(res) > int(sr * 0.08) else audio_arr

def estimate_pitch_f0(audio_chunk, sr=SAMPLE_RATE):
    """
    Estimates the fundamental frequency (F0) of voice using autocorrelation.
    Optimized with 4096-sample window and energy gating for high accuracy
    across tonal and pitch-accent languages (Chinese, Japanese, Korean) and short utterances.
    """
    if len(audio_chunk) < int(sr * 0.15):
        return 140.0

    chunk = audio_chunk.astype(np.float32)
    chunk -= np.mean(chunk)

    # Energetický prah – preskočíme tiché úseky (nie sú vokál)
    rms = np.sqrt(np.mean(chunk ** 2))
    if rms < 5e-4:
        return 140.0

    # Okno 4096 pre spoľahlivejší odhad (256 ms @ 16kHz)
    window_len = min(len(chunk), 4096)
    # Viac krokov cez segment – 8 prekrývajúcich sa okien
    n_steps = 8
    step = max(1, (len(chunk) - window_len) // max(1, n_steps - 1))
    measured_pitches = []

    min_lag = max(1, int(sr / 350))   # ~350 Hz horný limit
    max_lag = min(window_len - 1, int(sr / 65))  # ~65 Hz spodný limit

    for start_pos in range(0, len(chunk) - window_len + 1, step):
        seg = chunk[start_pos:start_pos + window_len].copy()
        seg -= np.mean(seg)
        seg_rms = np.sqrt(np.mean(seg ** 2))
        if seg_rms < 3e-4:
            continue

        # Hann okno pre potlačenie spektrálneho úniku
        seg *= np.hanning(len(seg))

        n = len(seg)
        n_fft = 2 ** int(np.ceil(np.log2(2 * n - 1)))
        fx = np.fft.rfft(seg, n_fft)
        corr = np.fft.irfft(fx * np.conj(fx))[:n]

        if min_lag >= max_lag:
            continue

        region = corr[min_lag:max_lag]
        peak_rel = int(np.argmax(region))
        peak = min_lag + peak_rel

        # Parabolic interpolation for sub-sample accuracy
        if 0 < peak_rel < len(region) - 1:
            y0, y1, y2 = region[peak_rel - 1], region[peak_rel], region[peak_rel + 1]
            denom = (2 * y1 - y0 - y2)
            if denom > 1e-10:
                delta = 0.5 * (y0 - y2) / denom
                peak = min_lag + peak_rel + delta

        if peak > 0:
            # Pomer peak/zero-lag pre filtrovanie slabých koreláciái
            strength = corr[int(peak)] / (corr[0] + 1e-9)
            if strength > 0.15:  # minimálna sila korelácie
                pitch = float(sr / peak)
                if 65.0 <= pitch <= 350.0:
                    measured_pitches.append(pitch)

    if not measured_pitches:
        return 140.0

    # Percentil 25–75 pre robustnosť (odstráni outliere z tónových jazykov)
    arr = np.array(measured_pitches)
    p25, p75 = np.percentile(arr, 25), np.percentile(arr, 75)
    robust = arr[(arr >= p25) & (arr <= p75)]
    if len(robust) == 0:
        robust = arr

    return float(np.clip(np.median(robust), 65.0, 350.0))

def get_vram_status():
    if DEVICE == "cuda" and torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        return f"VRAM: {allocated:.2f}GB / {total:.2f}GB"
    return "Device: CPU / ONNX (Optimalizované)"

def clear_vram():
    if DEVICE == "cuda":
        gc.collect()
        torch.cuda.empty_cache()

def get_audio_duration(file_path):
    cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", file_path]
    try:
        res = run_cmd(cmd)
        return float(res.stdout.strip())
    except Exception:
        data, sr = sf.read(file_path)
        return len(data) / sr

def format_timestamp_srt(seconds):
    millis = int((seconds - int(seconds)) * 1000)
    mins, secs = divmod(int(seconds), 60)
    hours, mins = divmod(mins, 60)
    return f"{hours:02d}:{mins:02d}:{secs:02d},{millis:03d}"

def parse_timestamp_srt(ts_str):
    ts_str = ts_str.strip().replace(',', '.')
    parts = ts_str.split(':')
    if len(parts) == 3:
        h, m, s = parts
        return float(h) * 3600 + float(m) * 60 + float(s)
    elif len(parts) == 2:
        m, s = parts
        return float(m) * 60 + float(s)
    return float(ts_str)

# ==============================================================================
# SPOLOČNÉ SPRÁVCOVSKÉ FUNKCIE PRE HLASY A TRANSKRIPCIU (.txt)
# ==============================================================================

def scan_voices_directory():
    os.makedirs(VOICES_DIR, exist_ok=True)
    valid_exts = ('.wav', '.mp3', '.flac', '.ogg', '.m4a')
    voices = {}
    for f in sorted(os.listdir(VOICES_DIR)):
        if f.lower().endswith(valid_exts):
            name = os.path.splitext(f)[0]
            voices[name] = os.path.join(VOICES_DIR, f)
    return voices

def get_or_create_reference_transcription(voice_audio_path, log_cb=print):
    """
    Verifies the existence of a companion voicename.txt file for reference voice cloning (voices/*.mp3, *.wav).
    If it exists and is not empty, loads and returns its text.
    If it does not exist, automatically invokes Whisper (faster-whisper) to transcribe the speech,
    saves the transcription into voicename.txt (UTF-8 encoded), and returns the transcribed text.
    Subsequent runs load the text directly from the companion file without invoking Whisper again.
    """
    if not voice_audio_path or not os.path.exists(voice_audio_path):
        return ""

    base_path, _ = os.path.splitext(voice_audio_path)
    txt_path = base_path + ".txt"

    # 1. Existujúci .txt súbor
    if os.path.exists(txt_path):
        try:
            with open(txt_path, "r", encoding="utf-8-sig", errors="ignore") as f:
                content = f.read().strip()
            if content:
                log_cb(f"[Custom Voice] Using existing transcription from '{os.path.basename(txt_path)}'")
                return content
        except Exception as e:
            log_cb(f"[Warning] Failed to load {txt_path}: {e}")

    # 2. Whisper transcription when companion .txt is absent
    log_cb(f"[Custom Voice] Transcription for '{os.path.basename(voice_audio_path)}' not found. Transcribing with Whisper...")
    try:
        from faster_whisper import WhisperModel
        whisper = WhisperModel("medium", device=DEVICE, compute_type="float16" if "cuda" in DEVICE else "int8")
        segments, _ = whisper.transcribe(voice_audio_path, vad_filter=True)
        transcribed_text = " ".join([s.text.strip() for s in segments if s.text.strip()]).strip()
        del whisper
        clear_vram()

        if transcribed_text:
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(transcribed_text)
            log_cb(f"[Custom Voice] Transcription created and saved to '{os.path.basename(txt_path)}' ({len(transcribed_text)} characters).")
            return transcribed_text
        else:
            log_cb(f"[Warning] Whisper detected no speech in '{os.path.basename(voice_audio_path)}'.")
            return ""
    except Exception as e:
        log_cb(f"[Warning] Whisper transcription error for {voice_audio_path}: {e}")
        return ""

def get_all_available_voices(target_lang="cs"):
    """
    Returns (catalog_list, voice_map) for voice selection dropdowns:
    - ⚡ [Supertonic] M1...M5, F1...F5 (local ONNX neural voices)
    - 🎙️ [Edge] ... (high-quality Microsoft Edge neural voices for selected language)
    - 🧬 [Clone] ... (custom voice clones located in ./voices/)
    """
    catalog = []
    v_map = {}

    # 1. Supertonic voices (always available locally)
    for label, v_id in SUPERTONIC_VOICE_CATALOG:
        disp = f"⚡ [Supertonic] {label}"
        catalog.append(disp)
        v_map[disp] = {"type": "supertonic", "voice_id": v_id, "label": disp}

    # 2. Edge-TTS voices (high-quality online neural voices)
    lang_code = target_lang.split()[0].lower() if target_lang else "cs"
    if lang_code in EDGE_TTS_VOICE_CATALOG:
        for disp_name, v_id in EDGE_TTS_VOICE_CATALOG[lang_code]:
            disp = f"🎙️ [Edge] {disp_name}"
            catalog.append(disp)
            v_map[disp] = {"type": "edge", "voice_id": v_id, "label": disp}
    else:
        for disp_name, v_id in EDGE_TTS_VOICE_CATALOG.get("cs", []):
            disp = f"🎙️ [Edge] {disp_name}"
            catalog.append(disp)
            v_map[disp] = {"type": "edge", "voice_id": v_id, "label": disp}

    # 3. Custom cloned voices from ./voices/
    v_dict = scan_voices_directory()
    for vname, vpath in v_dict.items():
        disp = f"🧬 [Klon] {vname}"
        catalog.append(disp)
        v_map[disp] = {"type": "clone", "path": vpath, "voice_id": vname, "label": disp}

    return catalog, v_map

def resolve_voice_for_subtitle(assigned_voice_str, spk, state, lang_code="cs"):
    """
    Evaluates the assigned voice string for a subtitle line or falls back to the character's default voice.
    Returns a tuple: (engine_type, explicit_voice, ref_voice_wav)
    where engine_type is 'supertonic', 'edge', or 'xtts'.
    """
    speakers = state.get("data", {}).get("speakers", {})
    spk_info = speakers.get(spk, {})
    voices_dict = scan_voices_directory()
    label = (assigned_voice_str or "").strip()

    # Custom clone
    if "klon" in label.lower() or "vlastný" in label.lower():
        for vname, vpath in voices_dict.items():
            if vname.lower() in label.lower():
                return "xtts", None, vpath
        if spk_info.get("custom_voice_path") and os.path.exists(spk_info["custom_voice_path"]):
            return "xtts", None, spk_info["custom_voice_path"]

    # Supertonic
    if "supertonic" in label.lower() or re.search(r'\b([MF][1-5])\b', label):
        m = re.search(r'\b([MF][1-5])\b', label)
        st_id = m.group(1) if m else spk_info.get("supertonic_voice", "M1")
        return "supertonic", st_id, None

    # Edge-TTS
    if "edge" in label.lower() or "neural" in label.lower():
        m = re.search(r'([a-z]{2}-[A-Z]{2}-\w+Neural)', label)
        if m:
            return "edge", m.group(1), None
        avg_f0 = spk_info.get("avg_f0", 140.0)
        is_fem = avg_f0 >= 175.0
        if lang_code == "cs":
            return "edge", "cs-CZ-VlastaNeural" if is_fem else "cs-CZ-AntoninNeural", None
        elif lang_code == "sk":
            return "edge", "sk-SK-ViktoriaNeural" if is_fem else "sk-SK-LukasNeural", None
        elif lang_code == "ja":
            return "edge", "ja-JP-NanamiNeural" if is_fem else "ja-JP-KeitaNeural", None
        else:
            return "edge", "en-US-JennyNeural" if is_fem else "en-US-GuyNeural", None

    # Predvolený fallback podľa postavy (Supertonic ako default)
    if spk_info.get("custom_voice_path") and os.path.exists(spk_info["custom_voice_path"]):
        return "xtts", None, spk_info["custom_voice_path"]

    st_v = spk_info.get("supertonic_voice")
    if not st_v:
        avg_f0 = spk_info.get("avg_f0", 140.0)
        st_v = "F1" if avg_f0 >= 175.0 else "M1"
    return "supertonic", st_v, None

def sort_treeview(tree, col, sort_state_dict):
    """
    Sorts items in ttk.Treeview by column.
    Supports numerical sorting (ID), chronological sorting (time), and alphabetical sorting (text, character, voice).
    Toggles ascending / descending order on repeated clicks.
    """
    reverse = sort_state_dict["reverse"] if sort_state_dict.get("col") == col else False
    sort_state_dict["col"] = col
    sort_state_dict["reverse"] = not reverse

    items = [(tree.set(k, col), k) for k in tree.get_children('')]
    if col == "id":
        items.sort(key=lambda t: int(t[0]) if str(t[0]).isdigit() else 0, reverse=not reverse)
    elif col == "time":
        def get_start(val):
            try:
                part = val.split("-->")[0].strip()
                return parse_timestamp_srt(part)
            except Exception:
                return 0.0
        items.sort(key=lambda t: get_start(t[0]), reverse=not reverse)
    else:
        items.sort(key=lambda t: str(t[0]).lower(), reverse=not reverse)

    for index, (val, k) in enumerate(items):
        tree.move(k, '', index)

# ==============================================================================
# 2. AUDIO EXTRACTION & DEMUCS SEPARATION
# ==============================================================================

def extract_audio_from_video(video_path, output_audio_path):
    cmd = ["ffmpeg", "-y", "-i", video_path, "-vn", "-acodec", "pcm_s16le", "-ar", str(SAMPLE_RATE), "-ac", "1", output_audio_path]
    run_cmd(cmd)
    return output_audio_path

def separate_sources(audio_path, output_dir, log_cb=print):
    """
    Separates speech dialogue from background music and sound effects using HTDemucs.
    """
    log_cb("[Separation] Running HTDemucs to isolate dialogue and preserve background...")
    clear_vram()
    cmd = [sys.executable, "-m", "demucs.separate", "-n", "htdemucs", "--two-stems", "vocals", "-d", DEVICE, "-o", output_dir, audio_path]
    run_cmd(cmd)
    
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    demucs_out = os.path.join(output_dir, "htdemucs", base_name)
    vocals_src = os.path.join(demucs_out, "vocals.wav")
    no_vocals_src = os.path.join(demucs_out, "no_vocals.wav")
    
    bgm_path = os.path.join(output_dir, "bgm.wav")
    vocals_path = os.path.join(output_dir, "vocals.wav")
    
    if os.path.exists(no_vocals_src):
        shutil.copy2(no_vocals_src, bgm_path)
    if os.path.exists(vocals_src):
        shutil.copy2(vocals_src, vocals_path)
        
    clear_vram()
    log_cb("[Separation] Done. Clean speech track and background audio (BGM & SFX) ready.")
    return vocals_path, bgm_path

# ==============================================================================
# 3. SPEAKER DIARIZATION & AUTOMATIC VOICE ASSIGNMENT
# ==============================================================================

class OfflineSpeakerEncoder:
    def __init__(self, log_cb=print):
        self.backend = None
        self.encoder = None
        try:
            from speechbrain.inference.speaker import EncoderClassifier
            log_cb("[Diarization] Loading offline SpeechBrain ECAPA-TDNN model...")
            self.encoder = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": DEVICE})
            self.backend = "speechbrain"
        except Exception:
            try:
                from resemblyzer import VoiceEncoder
                log_cb("[Diarization] Loading offline Resemblyzer...")
                self.encoder = VoiceEncoder(device=DEVICE)
                self.backend = "resemblyzer"
            except Exception as e:
                log_cb(f"[Varovanie] Neurónový encoder nedostupný: {e}")
                self.backend = "fallback"

    def embed_utterance(self, wav_data, sr=SAMPLE_RATE):
        if self.backend == "speechbrain":
            if sr != 16000:
                import scipy.signal
                wav_data = scipy.signal.resample(wav_data, int(len(wav_data) * 16000 / sr))
            tensor = torch.tensor(wav_data, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                emb = self.encoder.encode_batch(tensor).squeeze().cpu().numpy()
            return emb / (np.linalg.norm(emb) + 1e-8)
        elif self.backend == "resemblyzer":
            from resemblyzer import preprocess_wav
            wav_pre = preprocess_wav(wav_data, source_sr=sr)
            emb = self.encoder.embed_utterance(wav_pre)
            return emb / (np.linalg.norm(emb) + 1e-8)
        else:
            return np.zeros(192, dtype=np.float32)

def assign_supertonic_voices_to_speakers(speakers, profile="asia"):
    """
    Assigns Supertonic voices (M1-M5 / F1-F5) strictly according to character gender and speaking duration:
      - Cutoffs:
          * Asia (China, Japan, Korea): Male < 165.0 Hz, Female >= 165.0 Hz
          * Western / Europe (US, EU): Male < 155.0 Hz, Female >= 155.0 Hz
      - Definitive spk_info["gender"] = "male" | "female" is stored for all UI views.
      - Male with longest speaking duration gets M1, second M2, etc.
      - Female with longest speaking duration gets F1, second F2, etc.
    """
    if not speakers:
        return

    is_asia = "azia" in str(profile).lower() or "ázij" in str(profile).lower() or "asia" in str(profile).lower()
    split_threshold = 165.0 if is_asia else 155.0

    male_speakers = []
    female_speakers = []

    for spk_id, spk_info in speakers.items():
        f0 = spk_info.get("avg_f0", 140.0)
        dur = spk_info.get("total_duration", 0.0)

        is_fem = (f0 >= split_threshold)
        spk_info["gender"] = "female" if is_fem else "male"
        spk_info["is_female"] = is_fem
        spk_info["gender_confidence"] = f"{'female' if is_fem else 'male'} (F0={f0:.1f}Hz, thresh={split_threshold:.0f}Hz)"

        if is_fem:
            female_speakers.append((spk_id, spk_info, dur))
        else:
            male_speakers.append((spk_id, spk_info, dur))

    # Sort each gender independently by total speaking duration (longest speaking first!)
    male_speakers.sort(key=lambda x: x[2], reverse=True)
    female_speakers.sort(key=lambda x: x[2], reverse=True)

    # Assign Male voices M1..M5 by duration
    for rank, (spk_id, spk_info, dur) in enumerate(male_speakers):
        v_idx = rank % len(SUPERTONIC_MALE_VOICES)
        chosen = SUPERTONIC_MALE_VOICES[v_idx]
        spk_info["supertonic_voice"] = chosen
        spk_info["assigned_voice"] = f"⚡ Supertonic [{chosen}]"
        for seg in spk_info.get("segments", []):
            seg["assigned_voice"] = f"⚡ Supertonic [{chosen}]"

    # Assign Female voices F1..F5 by duration
    for rank, (spk_id, spk_info, dur) in enumerate(female_speakers):
        v_idx = rank % len(SUPERTONIC_FEMALE_VOICES)
        chosen = SUPERTONIC_FEMALE_VOICES[v_idx]
        spk_info["supertonic_voice"] = chosen
        spk_info["assigned_voice"] = f"⚡ Supertonic [{chosen}]"
        for seg in spk_info.get("segments", []):
            seg["assigned_voice"] = f"⚡ Supertonic [{chosen}]"


def diarize_segments_direct(audio_path, segments, max_speakers=6, log_cb=print, profile="asia"):
    log_cb(f"[Diarization] Starting character analysis (Priority: speech duration, max {max_speakers} characters)...")
    
    audio_data, sr = sf.read(audio_path)
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)

    encoder = OfflineSpeakerEncoder(log_cb=log_cb)
    if encoder.backend == "fallback":
        for seg in segments:
            seg["speaker"] = "Speaker_01"
            seg["f0"] = 140.0
        spks = build_speakers_dict(segments)
        assign_supertonic_voices_to_speakers(spks, profile=profile)
        return segments, spks

    features = []
    valid_indices = []
    pitches = []

    for idx, seg in enumerate(segments):
        dur = seg["end"] - seg["start"]
        if dur < 0.35:
            continue

        s_samp = max(0, int(seg["start"] * sr))
        e_samp = min(len(audio_data), int(seg["end"] * sr))
        chunk = audio_data[s_samp:e_samp].astype(np.float32)
        chunk = trim_audio_silence(chunk, sr)

        if len(chunk) < int(sr * 0.25):
            continue

        try:
            emb = encoder.embed_utterance(chunk, sr=sr)
            f0 = estimate_pitch_f0(chunk, sr=sr)
            features.append(emb)
            pitches.append(f0)
            valid_indices.append(idx)
        except Exception:
            continue

    del encoder
    clear_vram()

    for seg in segments:
        seg["speaker"] = None
        seg["f0"] = 140.0

    from sklearn.cluster import AgglomerativeClustering
    if len(features) > 1:
        features_arr = np.array(features)
        norms = np.linalg.norm(features_arr, axis=1, keepdims=True)
        features_arr = features_arr / np.maximum(norms, 1e-8)

        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=0.65,
            metric="cosine",
            linkage="average"
        )
        labels = clustering.fit_predict(features_arr)

        cluster_durations = {}
        for lab, val_idx in zip(labels, valid_indices):
            seg_dur = segments[val_idx]["end"] - segments[val_idx]["start"]
            cluster_durations[lab] = cluster_durations.get(lab, 0.0) + seg_dur

        sorted_by_duration = sorted(cluster_durations.keys(), key=lambda c: cluster_durations[c], reverse=True)
        top_clusters = sorted_by_duration[:max_speakers]

        centroids = {}
        for c_id in top_clusters:
            mask = (labels == c_id)
            c_vec = np.mean(features_arr[mask], axis=0)
            centroids[c_id] = c_vec / (np.linalg.norm(c_vec) + 1e-8)

        final_cluster_ids = []
        for i, lab in enumerate(labels):
            if lab in centroids:
                final_cluster_ids.append(lab)
            else:
                feat = features_arr[i]
                best_c = max(centroids.keys(), key=lambda c_key: np.dot(feat, centroids[c_key]))
                final_cluster_ids.append(best_c)

        rank_mapping = {c_id: f"Speaker_{rank+1:02d}" for rank, c_id in enumerate(top_clusters)}

        for i, val_idx in enumerate(valid_indices):
            spk_name = rank_mapping[final_cluster_ids[i]]
            segments[val_idx]["speaker"] = spk_name
            segments[val_idx]["f0"] = pitches[i]

        last_known_spk = segments[valid_indices[0]]["speaker"]
        last_known_f0 = segments[valid_indices[0]]["f0"]
        for seg in segments:
            if seg["speaker"] is None:
                # Titulok bez reči v originálnom audiu (napr. nápis na stene, grafika, ticho pred/po).
                # Namiesto náhodného priradenia postavy z predchádzajúcej scény priradíme Speaker_01 s hlasom M1
                seg["speaker"] = "Speaker_01"
                seg["f0"] = 120.0
                seg["assigned_voice"] = "⚡ Supertonic [M1]"
            else:
                last_known_spk = seg["speaker"]
                last_known_f0 = seg["f0"]
    else:
        for seg in segments:
            seg["speaker"] = "Speaker_01"
            seg["f0"] = 140.0

    speakers = build_speakers_dict(segments)
    assign_supertonic_voices_to_speakers(speakers, profile=profile)

    for spk_id, info in sorted(speakers.items(), key=lambda x: x[1]["total_duration"], reverse=True):
        f0_val = info.get("avg_f0", 140.0)
        gender_str = "Female 👩" if info.get("is_female") else "Male 👨"
        conf = info.get("gender_confidence", "")
        log_cb(f"[Character] {spk_id}: {gender_str} ({conf}) | F0={f0_val:.0f}Hz | Duration: {info['total_duration']:.1f}s | Supertonic: {info.get('supertonic_voice')}")

    return segments, speakers
    
def build_speakers_dict(segments, existing_speakers=None):
    """
    Constructs a dictionary of character metadata, calculating total speaking duration
    and robust median F0 pitch across all segment utterances.
    """
    existing = existing_speakers or {}
    speakers = {}

    for seg in segments:
        spk = seg["speaker"]
        if not spk:
            continue

        if spk not in speakers:
            old_data = existing.get(spk, {})
            speakers[spk] = {
                "segments": [],
                "total_duration": 0.0,
                "preview_file": old_data.get("preview_file", ""),
                "voice_path": old_data.get("voice_path", ""),
                "supertonic_voice": old_data.get("supertonic_voice", ""),
                "custom_voice_path": old_data.get("custom_voice_path", ""),
                "mode": old_data.get("mode", "Auto"),
                "status": old_data.get("status", "Ready"),
                "avg_f0": old_data.get("avg_f0", 140.0)
            }
        dur = seg["end"] - seg["start"]
        speakers[spk]["segments"].append(seg)
        speakers[spk]["total_duration"] += dur

    for spk_id, spk_info in speakers.items():
        pitches = [s["f0"] for s in spk_info["segments"] if "f0" in s and 65.0 < s["f0"] < 350.0]
        if len(pitches) >= 2:
            arr = np.array(pitches, dtype=np.float32)
            # Robustný medián cez percentil 20-80 (odfiltruje outliere z tónových jazykov)
            p20, p80 = np.percentile(arr, 20), np.percentile(arr, 80)
            robust = arr[(arr >= p20) & (arr <= p80)]
            median_f0 = float(np.median(robust if len(robust) > 0 else arr))
            std_f0 = float(np.std(robust if len(robust) > 0 else arr))
            spk_info["avg_f0"] = median_f0
            spk_info["f0_std"] = std_f0
        elif pitches:
            spk_info["avg_f0"] = float(pitches[0])
            spk_info["f0_std"] = 0.0
        else:
            spk_info["avg_f0"] = 140.0
            spk_info["f0_std"] = 0.0

    return speakers

def generate_speaker_previews(audio_path, speakers, output_dir):
    """
    Extracts representative clean voice samples for each detected character for UI previews.
    """
    audio_data, sr = sf.read(audio_path)
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)

    os.makedirs(output_dir, exist_ok=True)
    all_segs = []
    for spk_id, spk_info in speakers.items():
        for s in spk_info["segments"]:
            all_segs.append((s["start"], s["end"], spk_id, s))

    all_segs.sort(key=lambda x: x[0])

    for spk_id, spk_info in speakers.items():
        candidates = [s for s in spk_info["segments"] if 2.0 <= (s["end"] - s["start"]) <= 8.0]
        candidates.sort(key=lambda x: (x["end"] - x["start"]), reverse=True)
        best_chunk = None

        for seg in candidates:
            s_samp = max(0, int(seg["start"] * sr))
            e_samp = min(len(audio_data), int(seg["end"] * sr))
            clean_c = trim_audio_silence(audio_data[s_samp:e_samp], sr)
            if len(clean_c) >= int(sr * 2.5):
                best_chunk = clean_c[:int(sr * 7.0)]
                break

        if best_chunk is None and spk_info["segments"]:
            sorted_all = sorted(spk_info["segments"], key=lambda x: (x["end"] - x["start"]), reverse=True)
            seg = sorted_all[0]
            s_samp = max(0, int(seg["start"] * sr))
            e_samp = min(len(audio_data), int(seg["end"] * sr))
            best_chunk = trim_audio_silence(audio_data[s_samp:e_samp], sr)

        if best_chunk is None or len(best_chunk) == 0:
            best_chunk = np.zeros(int(sr * 3.0), dtype=np.float32)

        out_file = os.path.join(output_dir, f"{spk_id}_clean_voice.wav")
        sf.write(out_file, best_chunk, sr)
        spk_info["preview_file"] = out_file
        if not spk_info.get("voice_path"):
            spk_info["voice_path"] = out_file

def get_valid_voice_path(spk_id, state):
    speakers = state["data"].get("speakers", {})
    spk_data = speakers.get(spk_id, {})
    vp = spk_data.get("custom_voice_path") or spk_data.get("voice_path", "")
    if vp and os.path.exists(vp) and os.path.getsize(vp) > 1000:
        return vp
    pf = spk_data.get("preview_file", "")
    if pf and os.path.exists(pf) and os.path.getsize(pf) > 1000:
        return pf
    return None

def get_effective_voice_for_sub(sub, speakers_dict=None):
    """Returns the resolved voice label for a subtitle line."""
    v = sub.get("assigned_voice")
    if v and str(v).strip():
        return str(v).strip()
    if speakers_dict:
        spk = sub.get("speaker", "Speaker_01")
        spk_data = speakers_dict.get(spk, {})
        if spk_data.get("custom_voice_path"):
            return f"🧬 [Klon] {os.path.basename(spk_data['custom_voice_path'])}"
        elif spk_data.get("supertonic_voice"):
            return f"⚡ Supertonic [{spk_data['supertonic_voice']}]"
    return "⚡ Supertonic [M1]"

def merge_continuous_subtitles(subtitles, max_gap=0.40, speakers=None):
    """
    Merges continuous subtitle lines if they share the exact same voice
    (or character) and occur consecutively within max_gap seconds.
    This produces a single unified synthesis request for natural, flowing speech.
    """
    if not subtitles:
        return []

    merged = []
    curr = dict(subtitles[0])
    curr_v = get_effective_voice_for_sub(curr, speakers)
    curr["assigned_voice"] = curr_v

    for nxt in subtitles[1:]:
        gap = nxt["start"] - curr["end"]
        nxt_v = get_effective_voice_for_sub(nxt, speakers)
        nxt["assigned_voice"] = nxt_v

        same_voice = (curr_v == nxt_v)
        same_speaker = (curr.get("speaker") == nxt.get("speaker"))

        # Merge if they share the same voice (or same speaker with non-conflicting voice) and gap is small
        if (same_voice or (same_speaker and not curr_v)) and gap <= max_gap and gap >= -0.05:
            curr["end"] = max(curr["end"], nxt["end"])
            curr["text"] = f"{curr['text'].strip()} {nxt['text'].strip()}"
            curr["assigned_voice"] = curr_v
        else:
            merged.append(curr)
            curr = dict(nxt)
            curr_v = nxt_v

    merged.append(curr)

    for idx, item in enumerate(merged):
        item["id"] = idx + 1

    return merged

# ==============================================================================
# 4. SUBTITLE PARSING & TEXT UTILITIES
# ==============================================================================

def parse_srt_file(srt_path):
    with open(srt_path, 'r', encoding='utf-8-sig') as f:
        content = f.read().strip()
    return parse_srt_content(content)

def parse_srt_content(raw_srt_text):
    blocks = re.split(r'\n\s*\n', raw_srt_text.strip())
    subtitles = []
    for block in blocks:
        lines = [l.strip() for l in block.split('\n') if l.strip()]
        if len(lines) >= 2:
            time_line = lines[1] if '-->' in lines[1] else lines[0]
            text_lines = lines[2:] if '-->' in lines[1] else lines[1:]
            if '-->' in time_line:
                times = time_line.split('-->')
                start_sec = parse_timestamp_srt(times[0])
                end_sec = parse_timestamp_srt(times[1])
                subtitles.append({
                    "id": len(subtitles) + 1,
                    "start": start_sec,
                    "end": end_sec,
                    "text": " ".join(text_lines),
                    "speaker": "Speaker_01",
                    "assigned_voice": ""
                })
    return subtitles

def export_working_file(segments, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        for seg in segments:
            v_tag = f" | {seg['assigned_voice']}" if seg.get("assigned_voice") else ""
            f.write(f"[{seg['speaker']}{v_tag}]\n")
            f.write(f"{format_timestamp_srt(seg['start'])} --> {format_timestamp_srt(seg['end'])}\n")
            f.write(f"{seg['text']}\n\n")

def import_working_file(working_path):
    with open(working_path, 'r', encoding='utf-8-sig') as f:
        content = f.read().strip()
    blocks = re.split(r'\n\s*\n', content)
    items = []
    for b in blocks:
        lines = [l.strip() for l in b.split('\n') if l.strip()]
        if len(lines) >= 3 and lines[0].startswith('[') and '-->' in lines[1]:
            header = lines[0].replace('[', '').replace(']', '').strip()
            spk = header
            assigned_voice = ""
            if '|' in header:
                parts = header.split('|', 1)
                spk = parts[0].strip()
                assigned_voice = parts[1].strip()

            times = lines[1].split('-->')
            items.append({
                "id": len(items) + 1,
                "speaker": spk,
                "assigned_voice": assigned_voice,
                "start": parse_timestamp_srt(times[0]),
                "end": parse_timestamp_srt(times[1]),
                "text": " ".join(lines[2:])
            })
    return items

def split_text_into_sentences_list(raw_text):
    """
    Splits text by punctuation (. ! ? 。 ！？ and line breaks) into discrete sentences for Tab 2 phrase table.
    """
    cleaned = raw_text.replace("\r\n", "\n")
    cleaned = re.sub(r'[„“”"«»「」]', '', cleaned)
    lines = [l.strip() for l in cleaned.split("\n") if l.strip()]
    sentences = []
    for line in lines:
        parts = re.findall(r'[^.!?。！？]+(?:[.!?。！？]+|$)', line)
        for p in parts:
            p_clean = p.strip()
            if p_clean and not re.fullmatch(r'^[.,!?:;\- 。！？、 ]+$', p_clean):
                if p_clean[-1] not in ".!?。！？":
                    p_clean += "."
                sentences.append(p_clean)
    return sentences

def sync_subtitles_with_disk(old_subs, new_subs, speakers):
    """
    Intelligently synchronizes subtitle lines loaded from .srt file with existing project state.
    Preserves assigned characters, voices, and audio_chunk_keys for both unchanged and modified lines.

    3-Phase 1-to-1 Matching:
      1. Exact text match + start time proximity < 10s (unchanged lines retain exact voice and cache)
      2. Timestamp slot alignment < 0.4s (handles edited text in the same scene slot)
      3. Proximity fallback < 1.5s (fallback for timing shifts)
    """
    if not old_subs:
        for s in new_subs:
            spk = s.get("speaker", "Speaker_01")
            spk_data = speakers.get(spk, {})
            st_v = spk_data.get("supertonic_voice", "M1")
            if not s.get("assigned_voice"):
                s["assigned_voice"] = f"⚡ Supertonic [{st_v}]"
        return new_subs

    def _norm(t):
        return " ".join(t.replace("\r", " ").replace("\n", " ").split()).lower()

    available = list(old_subs)  # 1-to-1 consumption pool
    synced = []

    for idx, new_s in enumerate(new_subs):
        ns_text = _norm(new_s.get("text", ""))
        ns_start = new_s["start"]
        matched_old = None

        # --- Fáza 1: rovnaký text + čas blízko (nezmenený riadok) ---
        best1_diff = 10.0
        best1_cand = None
        for cand in available:
            if _norm(cand.get("text", "")) == ns_text:
                diff = abs(ns_start - cand["start"])
                if diff < best1_diff:
                    best1_diff = diff
                    best1_cand = cand
        if best1_cand is not None:
            matched_old = best1_cand
            available.remove(best1_cand)

        # --- Fáza 2: rovnaký časový slot (editovaný text) ---
        if matched_old is None:
            best2_diff = 0.4
            best2_cand = None
            for cand in available:
                diff = abs(ns_start - cand["start"])
                if diff < best2_diff:
                    best2_diff = diff
                    best2_cand = cand
            if best2_cand is not None:
                matched_old = best2_cand
                available.remove(best2_cand)

        # --- Fáza 3: fallback – časový prekryv < 1.5 s ---
        if matched_old is None:
            best3_diff = 1.5
            best3_cand = None
            for cand in available:
                diff = abs(ns_start - cand["start"])
                if diff < best3_diff:
                    best3_diff = diff
                    best3_cand = cand
            if best3_cand is not None:
                matched_old = best3_cand
                available.remove(best3_cand)

        if matched_old:
            new_s["speaker"] = matched_old.get("speaker", "Speaker_01")
            new_s["assigned_voice"] = matched_old.get("assigned_voice", "")
            # Zachovaj audio_chunk_key pre cache reuse v process_dub
            if matched_old.get("audio_chunk_key"):
                new_s["audio_chunk_key"] = matched_old["audio_chunk_key"]
        else:
            new_s["speaker"] = "Speaker_01"

        if not new_s.get("assigned_voice"):
            spk_data = speakers.get(new_s["speaker"], {})
            st_v = spk_data.get("supertonic_voice", "M1")
            new_s["assigned_voice"] = f"⚡ Supertonic [{st_v}]"

        new_s["id"] = idx + 1
        synced.append(new_s)

    return synced

def is_batch_video_up_to_date(vid_path, sub_path, target_lang):
    """
    Checks whether a video has already been successfully dubbed and exported,
    and ensures the .srt subtitle file has not been modified since the export.
    """
    base_dir = os.path.splitext(vid_path)[0] + "_ai_dub"
    proj_file = os.path.join(base_dir, "project.json")
    if not os.path.exists(proj_file):
        return False

    try:
        with open(proj_file, "r", encoding="utf-8") as f:
            p_data = json.load(f)
    except Exception:
        return False

    if not p_data.get("steps_completed", {}).get("final_export", False):
        return False

    out_dir = os.path.join(base_dir, "output")
    if not os.path.exists(out_dir):
        return False

    dubbed_videos = [f for f in os.listdir(out_dir) if f.lower().endswith(('.mkv', '.mp4')) and '_dubbed_' in f.lower()]
    if not dubbed_videos:
        return False

    # Ak bol .srt súbor upravený po uložení project.json, video nie je aktuálne!
    if os.path.exists(sub_path):
        if os.path.getmtime(sub_path) > os.path.getmtime(proj_file):
            return False

    return True

def split_text_into_natural_sentences(raw_text):
    """
    Splits continuous text into natural sentence paragraphs for TTS processing.
    """
    cleaned = raw_text.replace("\r\n", "\n")
    cleaned = re.sub(r'[„“”"«»]', '', cleaned)
    cleaned = re.sub(r'[–—―]', ' - ', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned)

    paragraphs = [p.strip() for p in cleaned.split("\n") if p.strip()]
    final_chunks = []

    for p in paragraphs:
        sentences = re.findall(r'[^.!?]+(?:[.!?]+|$)', p)
        for s in sentences:
            s_clean = s.strip()
            if not s_clean or re.fullmatch(r'^[.,!?:;\- ]+$', s_clean):
                continue

            s_clean = re.sub(r'\.+', '.', s_clean)
            if len(s_clean) > 140:
                parts = re.split(r'([,;:\-]+)', s_clean)
                cur = ""
                for part in parts:
                    if len(cur) + len(part) <= 140:
                        cur += part
                    else:
                        c_strip = cur.strip()
                        if c_strip and not re.fullmatch(r'^[.,!?:;\- ]+$', c_strip):
                            final_chunks.append(c_strip)
                        cur = part
                c_strip = cur.strip()
                if c_strip and not re.fullmatch(r'^[.,!?:;\- ]+$', c_strip):
                    final_chunks.append(c_strip)
            else:
                final_chunks.append(s_clean)

        final_chunks.append("<PARAGRAPH_BREAK>")

    if final_chunks and final_chunks[-1] == "<PARAGRAPH_BREAK>":
        final_chunks.pop()

    sanitized_chunks = []
    for c in final_chunks:
        if c == "<PARAGRAPH_BREAK>":
            sanitized_chunks.append(c)
        else:
            c_final = c.strip()
            if c_final and c_final[-1] not in ".!?":
                c_final += "."
            sanitized_chunks.append(c_final)

    return sanitized_chunks

# ==============================================================================
# 5. MULTI-ENGINE SYNTHESIS (SUPERTONIC + EDGE-TTS + XTTS CLONING)
# ==============================================================================

class DynamicDubbingEngine:
    def __init__(self, target_lang="cs", engine_type="supertonic", state=None):
        self.target_lang = target_lang
        self.engine_type = engine_type  # "supertonic", "edge", "xtts"
        self.state = state or {}
        self.xtts_model = None
        self.supertonic_model = None
        self.supertonic_styles = {}

    def ensure_supertonic_loaded(self, log_cb=print):
        """Ensures the local ONNX Supertonic TTS engine is initialized."""
        if self.supertonic_model is None:
            if SupertonicTTS is None:
                raise RuntimeError("Knižnica 'supertonic' nie je nainštalovaná! Spustite: pip install supertonic")
            log_cb("[Supertonic] Initializing local ONNX model...")
            self.supertonic_model = SupertonicTTS(auto_download=True)
            log_cb("[Supertonic] Supertonic TTS engine is ready.")

    def ensure_xtts_loaded(self, log_cb=print):
        """Loads Coqui XTTS-v2 multilingual voice cloning model into GPU VRAM."""
        if self.xtts_model is None:
            try:
                import transformers.pytorch_utils
                if not hasattr(transformers.pytorch_utils, "isin_mps_friendly"):
                    transformers.pytorch_utils.isin_mps_friendly = torch.isin
                from TTS.api import TTS
                log_cb("[Voice Cloner] Loading offline XTTS-v2 model into VRAM for voice cloning...")
                self.xtts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(DEVICE)
                log_cb("[Voice Cloner] XTTS-v2 model is ready.")
            except Exception as e:
                raise RuntimeError(f"Failed to load XTTS-v2 voice cloning model: {e}")

    def get_assigned_edge_voice(self, spk_id):
        speakers = self.state.get("data", {}).get("speakers", {})
        spk_data = speakers.get(spk_id, {})
        avg_f0 = spk_data.get("avg_f0", 140.0)
        is_female = avg_f0 >= 175.0

        if self.target_lang == "cs":
            return "cs-CZ-VlastaNeural" if is_female else "cs-CZ-AntoninNeural"
        elif self.target_lang == "sk":
            return "sk-SK-ViktoriaNeural" if is_female else "sk-SK-LukasNeural"
        elif self.target_lang == "ja":
            return "ja-JP-NanamiNeural" if is_female else "ja-JP-KeitaNeural"
        else:
            return "en-US-JennyNeural" if is_female else "en-US-GuyNeural"

    def get_assigned_supertonic_voice(self, spk_id):
        speakers = self.state.get("data", {}).get("speakers", {})
        spk_data = speakers.get(spk_id, {})
        assigned = spk_data.get("supertonic_voice")
        if assigned:
            return assigned
        avg_f0 = spk_data.get("avg_f0", 140.0)
        is_female = avg_f0 >= 175.0
        return random.choice(SUPERTONIC_FEMALE_VOICES) if is_female else random.choice(SUPERTONIC_MALE_VOICES)

    async def _synth_edge(self, text, output_wav, voice_name, speed=1.0):
        speed_pct = int(round((speed - 1.0) * 100))
        rate_str = f"{speed_pct:+d}%"

        temp_mp3 = output_wav + ".tmp.mp3"
        max_retries = 3

        for attempt in range(max_retries):
            try:
                communicate = edge_tts.Communicate(text, voice_name, rate=rate_str, pitch="+0Hz")
                await communicate.save(temp_mp3)
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    raise RuntimeError(f"Edge-TTS zlyhalo po {max_retries} pokusoch: {e}")
                await asyncio.sleep(1.5 * (attempt + 1))

        cmd = [
            "ffmpeg", "-y", "-i", temp_mp3,
            "-ar", str(SAMPLE_RATE), "-ac", "1",
            output_wav
        ]
        run_cmd(cmd)

        if os.path.exists(temp_mp3):
            os.remove(temp_mp3)
        await asyncio.sleep(0.10)

    def generate_speech(self, text, spk_id, ref_voice_wav, output_wav_path, target_duration=None, log_cb=print, explicit_voice=None, explicit_engine=None):
        if not text or not text.strip():
            sf.write(output_wav_path, np.zeros(int(SAMPLE_RATE * 0.2), dtype=np.float32), SAMPLE_RATE)
            return output_wav_path, "neutral"

        calc_speed = 1.0
        if target_duration and target_duration > 0.5:
            needed = len(text) / 14.0
            if needed > target_duration:
                calc_speed = float(np.clip(needed / target_duration, 1.0, 1.35))

        active_engine = explicit_engine if explicit_engine else self.engine_type

        # 1. XTTS-v2 Klonovanie
        if active_engine in ("xtts", "clone") or (ref_voice_wav and os.path.exists(ref_voice_wav) and active_engine == "clone"):
            log_cb(f"[Voice Cloner] Cloning voice for {spk_id} from reference: {os.path.basename(ref_voice_wav)}")
            
            # Kontrola a generovanie sprievodného voicename.txt cez Whisper
            get_or_create_reference_transcription(ref_voice_wav, log_cb=log_cb)

            self.ensure_xtts_loaded(log_cb=log_cb)

            xtts_lang = "cs" if self.target_lang in ["cs", "sk"] else self.target_lang
            clean_text = text.strip()
            if not clean_text.endswith(('.', '!', '?')):
                clean_text += '.'
            clean_text += ' '

            self.xtts_model.tts_to_file(
                text=clean_text,
                speaker_wav=ref_voice_wav,
                language=xtts_lang,
                file_path=output_wav_path,
                speed=calc_speed,
                enable_text_splitting=False
            )
            return output_wav_path, "neutral"

        # 2. Supertonic (Lokálne ONNX)
        elif active_engine == "supertonic":
            self.ensure_supertonic_loaded(log_cb=log_cb)
            st_voice = explicit_voice if explicit_voice else self.get_assigned_supertonic_voice(spk_id)
            log_cb(f"[Supertonic] Synthesizing voice {st_voice} ({self.target_lang}) for {spk_id}")

            if st_voice not in self.supertonic_styles:
                self.supertonic_styles[st_voice] = self.supertonic_model.get_voice_style(voice_name=st_voice)
            style = self.supertonic_styles[st_voice]

            temp_super_wav = output_wav_path + ".st_temp.wav"
            st_lang = self.target_lang if self.target_lang in ["cs", "sk", "ja", "en", "de", "pl"] else "en"

            wav, _ = self.supertonic_model.synthesize(
                text=text.strip(),
                voice_style=style,
                lang=st_lang,
                total_steps=8,
                speed=float(np.clip(calc_speed, 0.7, 1.6))
            )
            self.supertonic_model.save_audio(wav, temp_super_wav)

            cmd = [
                "ffmpeg", "-y", "-i", temp_super_wav,
                "-ar", str(SAMPLE_RATE), "-ac", "1",
                output_wav_path
            ]
            run_cmd(cmd)

            if os.path.exists(temp_super_wav):
                os.remove(temp_super_wav)

            return output_wav_path, "neutral"

        # 3. Edge-TTS
        else:
            voice_name = explicit_voice if explicit_voice else self.get_assigned_edge_voice(spk_id)
            log_cb(f"[Edge-TTS] Synthesizing voice {voice_name} for {spk_id}")
            asyncio.run(self._synth_edge(text, output_wav_path, voice_name, speed=calc_speed))
            return output_wav_path, "neutral"

    def unload_model(self):
        clear_vram()

# ==============================================================================
# 6. COLLISION MANAGER & ADAPTIVE MASTER MIXER (PRESERVES BACKGROUND SOUNDS)
# ==============================================================================

def apply_studio_vocal_clarity(input_wav, output_wav, speed_factor=1.0, eq_persona="neutral", sr=SAMPLE_RATE):
    speed_factor = float(np.clip(speed_factor, 0.85, 1.45))
    af_chain = ["highpass=f=80"]

    if eq_persona == "deep_warm":
        af_chain.append("equalizer=f=220:t=q:w=1.0:g=3.0")
        af_chain.append("equalizer=f=3000:t=q:w=1.2:g=2.0")
    elif eq_persona == "crisp_bright":
        af_chain.append("highpass=f=110")
        af_chain.append("equalizer=f=3600:t=q:w=1.5:g=4.0")
    else:
        af_chain.append("equalizer=f=3200:t=q:w=1.2:g=3.2")

    af_chain.append(f"atempo={speed_factor:.3f}")
    filter_str = ",".join(af_chain)

    cmd = [
        "ffmpeg", "-y", "-i", input_wav,
        "-af", filter_str,
        "-ar", str(sr), "-ac", "1",
        output_wav
    ]
    run_cmd(cmd)

def fit_audio_strictly_to_slot(raw_wav, final_wav, allocated_time, eq_persona="neutral", sr=SAMPLE_RATE):
    allocated_time = max(0.5, float(allocated_time))
    data, file_sr = sf.read(raw_wav)
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)

    cleaned = trim_audio_silence(data, file_sr, db_thresh=32)
    cur_dur = len(cleaned) / file_sr

    tmp_in = final_wav + ".in.wav"
    sf.write(tmp_in, cleaned, file_sr)

    speed = 1.0
    if cur_dur > allocated_time:
        speed = min(cur_dur / allocated_time, 1.40)

    apply_studio_vocal_clarity(tmp_in, final_wav, speed_factor=speed, eq_persona=eq_persona, sr=sr)

    if os.path.exists(tmp_in):
        os.remove(tmp_in)
        
def assemble_dialogue_track_anticollision(subtitles_with_audio, total_duration, target_rms=0.075, sr=SAMPLE_RATE):
    base_samples = int(math.ceil(total_duration * sr))
    full_track = np.zeros(base_samples, dtype=np.float32)
    active_mask = np.zeros(base_samples, dtype=np.float32)

    sorted_items = sorted(subtitles_with_audio, key=lambda x: x["start"])

    for item in sorted_items:
        wav_path = item.get("final_audio")
        if not wav_path or not os.path.exists(wav_path):
            continue

        data, _ = sf.read(wav_path)
        if len(data) == 0:
            continue

        active_samples = data[np.abs(data) > 0.015]
        if len(active_samples) > 0:
            current_rms = np.sqrt(np.mean(active_samples**2))
            if current_rms > 1e-4:
                data = data * (target_rms / current_rms)

        start_s = int(item["start"] * sr)
        end_s = start_s + len(data)

        if start_s >= base_samples:
            continue
        if end_s > base_samples:
            data = data[:base_samples - start_s]
            end_s = base_samples

        seg_len = end_s - start_s
        if seg_len <= 0:
            continue

        overlap = active_mask[start_s:end_s]
        weight = np.where(overlap > 0.5, 0.80, 1.0).astype(np.float32)

        fade_len = min(int(sr * 0.010), seg_len // 4)
        if fade_len > 0:
            fade_in = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
            fade_out = np.linspace(1.0, 0.0, fade_len, dtype=np.float32)
            data[:fade_len] *= fade_in
            data[-fade_len:] *= fade_out

        full_track[start_s:end_s] += (data[:seg_len] * weight)
        active_mask[start_s:end_s] += 1.0

    peak = np.max(np.abs(full_track))
    if peak > 0.90:
        full_track = (full_track / peak) * 0.90

    return full_track

def create_adaptive_master_mix(orig_audio_path, bgm_audio_path, dialogue_track, subtitles, output_path, dlg_vol=0.90, bg_vol=1.00, ducking=0.0, sr=SAMPLE_RATE):
    """
    Combines the dubbing dialogue track, background music, and ambient sound effects.
    Features:
    1. active_dubbing_mask is built STRICTLY for the actual duration when dubbing speech is playing
       (based on synthesized audio duration), NOT the wide time intervals of the subtitle file (.srt).
       This completely eliminates dead silence before and after spoken lines.
    2. orig_vocals (sounds that HTDemucs extracted into vocals: screams, impacts, breaking glass,
       spell whooshes, and battle Foley) remain 100% untouched outside of speech. During speech,
       a 20% Foley/ambience floor is preserved, and dynamic transient detection passes loud sound
       effects (screams, crashes) up to 65% so action and fantasy scenes are never muffled or silent.
    """
    import scipy.signal

    def load_and_resample(path, target_sr):
        data, file_sr = sf.read(path)
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)
        if file_sr != target_sr:
            num_samples = int(round(len(data) * target_sr / file_sr))
            data = scipy.signal.resample(data, num_samples)
        return data.astype(np.float32)

    orig_data = load_and_resample(orig_audio_path, sr)
    bgm_data = load_and_resample(bgm_audio_path, sr)
    dlg_data = dialogue_track.astype(np.float32)

    target_len = len(orig_data)

    def fit_len(arr, l):
        if len(arr) < l:
            return np.pad(arr, (0, l - len(arr)), mode='constant')
        return arr[:l]

    bgm_data = fit_len(bgm_data, target_len)
    dlg_data = fit_len(dlg_data, target_len)
    orig_data = fit_len(orig_data, target_len)

    active_dubbing_mask = np.zeros(target_len, dtype=np.float32)
    ramp_len = int(sr * 0.035)  # 35 ms jemný prechod
    ramp_up = np.linspace(0.0, 1.0, ramp_len, dtype=np.float32)
    ramp_down = np.linspace(1.0, 0.0, ramp_len, dtype=np.float32)

    for item in subtitles:
        wav_p = item.get("final_audio")
        if not wav_p or not os.path.exists(wav_p):
            continue

        try:
            actual_dur = get_audio_duration(wav_p)
        except Exception:
            actual_dur = max(0.5, item["end"] - item["start"])

        # Ak chunk nemá zvuk alebo je nulový, nemaskujeme
        if actual_dur <= 0.05:
            continue

        # Maskujeme iba skutočný čas dabingu
        s_idx = max(0, int(item["start"] * sr))
        e_idx = min(target_len, int((item["start"] + actual_dur) * sr))
        if e_idx <= s_idx:
            continue

        active_dubbing_mask[s_idx:e_idx] = 1.0
        f_s = max(0, s_idx - ramp_len)
        l_in = s_idx - f_s
        if l_in > 0:
            active_dubbing_mask[f_s:s_idx] = np.maximum(active_dubbing_mask[f_s:s_idx], ramp_up[-l_in:])

        f_e = min(target_len, e_idx + ramp_len)
        l_out = f_e - e_idx
        if l_out > 0:
            active_dubbing_mask[e_idx:f_e] = np.maximum(active_dubbing_mask[e_idx:f_e], ramp_down[:l_out])

    orig_vocals = orig_data - bgm_data

    # Detekcia transientov (hlasné zvukové efekty, výkriky duchov/príšer, prasknutie šálky, rinčanie mečov)
    abs_vocals = np.abs(orig_vocals)
    transient_boost = np.clip((abs_vocals - 0.10) / 0.25, 0.0, 1.0)

    # Útlm pôvodného hlasu počas dabingu:
    # Mimo dabingu (mask=0): 100% priechodnosť originálu (dokonalé zachovanie všetkých zvukov)
    # Počas dabingu (mask=1): 20% základná atmosféra + až 45% transient boost pre efekty a výkriky
    vocal_retention = 0.20 + (0.45 * transient_boost)
    passthrough_factor = 1.0 - (active_dubbing_mask * (1.0 - vocal_retention))
    passthrough_vocals = orig_vocals * passthrough_factor

    # Zachovanie plného zvuku pozadia (ducking = 0.0 znamená žiadne oslabenie pozadia pri dialógoch)
    bgm_attenuation = 1.0 - (float(ducking) * active_dubbing_mask)
    processed_bgm = bgm_data * bgm_attenuation * bg_vol
    final_mix = processed_bgm + passthrough_vocals + (dlg_data * dlg_vol)

    peak = np.max(np.abs(final_mix))
    if peak > 0.95:
        final_mix = (final_mix / peak) * 0.95

    sf.write(output_path, final_mix, sr)
    return output_path
    
def mix_and_export_video(video_path, mixed_audio_track, output_video_path, target_lang="cs"):
    """
    Exports the video into an MKV container with multi-track audio:
      Track 1 (Default): Newly dubbed AI audio with language metadata
      Track 2 (Optional): Original audio from input video preserved lossless
    """
    lang_map = {
        "cs": "ces", "sk": "slk", "ja": "jpn", "en": "eng", "de": "deu", "pl": "pol"
    }
    iso_code = lang_map.get(str(target_lang).lower().strip(), "und")
    lang_title = f"AI Dubbing ({str(target_lang).upper()})"

    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", mixed_audio_track,
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-map", "0:a:0?",
        "-c:v", "copy",
        "-c:a:0", "aac", "-b:a:0", "320k",
        "-metadata:s:a:0", f"title={lang_title}",
        "-metadata:s:a:0", f"language={iso_code}",
        "-disposition:a:0", "default",
        "-c:a:1", "copy",
        "-metadata:s:a:1", "title=Original Audio",
        "-disposition:a:1", "0",
        output_video_path
    ]
    run_cmd(cmd)

# ==============================================================================
# 7. PROJECT STATE & STORAGE MANAGER
# ==============================================================================

def create_project_structure(video_path):
    base_dir = os.path.splitext(video_path)[0] + "_ai_dub"
    folders = {
        "root": base_dir,
        "audio": os.path.join(base_dir, "audio"),
        "separation": os.path.join(base_dir, "separation"),
        "speakers": os.path.join(base_dir, "speakers"),
        "generated": os.path.join(base_dir, "generated"),
        "subtitles": os.path.join(base_dir, "subtitles"),
        "output": os.path.join(base_dir, "output")
    }
    for p in folders.values():
        os.makedirs(p, exist_ok=True)
    project_file = os.path.join(base_dir, "project.json")
    return folders, project_file

def load_or_init_project(video_path, project_file):
    if os.path.exists(project_file):
        with open(project_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {
        "video_path": video_path,
        "mode": "EXISTING_SRT",
        "tts_engine": "Supertonic (Lokálne)",
        "target_lang": "cs",
        "use_separation": True,
        "speakers": {},
        "subtitles": [],
        "steps_completed": {}
    }

def save_project(project_data, project_file):
    with open(project_file, 'w', encoding='utf-8') as f:
        json.dump(project_data, f, indent=4, ensure_ascii=False)

# ==============================================================================
# 8. TKINTER GUI APPLICATION
# ==============================================================================

def find_matching_subtitle(video_path, target_lang="cs"):
    """
    Finds a matching subtitle (.srt or .ass) for a video file.
    Priority:
      1. Exact language match (e.g. video.cs.srt, video.cz.srt, video.sk.srt, video.ja.srt)
      2. Exact base stem (e.g. video.srt, video.ass)
      3. Language/General aliases (e.g. video.en.srt, video.default.srt)
      4. Single subtitle file in directory if only one exists
    """
    if not video_path or not os.path.exists(video_path):
        return None

    folder = os.path.dirname(os.path.abspath(video_path))
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    sub_exts = ('.srt', '.ass')

    lang_code = target_lang.split()[0].lower() if target_lang else "cs"
    aliases = [lang_code]
    if lang_code in ("cs", "sk"):
        aliases.extend(["cz", "cze", "ces", "sk", "slo", "slk"])
    elif lang_code == "ja":
        aliases.extend(["jp", "jpn"])
    elif lang_code == "en":
        aliases.extend(["eng", "default"])
    aliases.extend(["default", "en", "eng"])

    # 1. Check language-specific suffixes
    for ext in sub_exts:
        for al in aliases:
            p = os.path.join(folder, f"{base_name}.{al}{ext}")
            if os.path.exists(p):
                return p
            p_dash = os.path.join(folder, f"{base_name}-{al}{ext}")
            if os.path.exists(p_dash):
                return p_dash

    # 2. Check exact base name (video.srt, video.ass)
    for ext in sub_exts:
        p = os.path.join(folder, f"{base_name}{ext}")
        if os.path.exists(p):
            return p

    # 3. Fuzzy prefix match in folder
    prefix = base_name.lower() + "."
    try:
        all_subs = [f for f in os.listdir(folder) if f.lower().endswith(sub_exts)]
        for f in all_subs:
            if f.lower().startswith(prefix):
                return os.path.join(folder, f)
        # If there is only exactly 1 subtitle file in the entire video directory, use it!
        if len(all_subs) == 1:
            return os.path.join(folder, all_subs[0])
    except Exception:
        pass

    return None

def create_app():
    root = tk.Tk()
    root.title("Local AI Video Dubbing & Localization Studio")
    root.geometry("1440x960")
    root.minsize(960, 680)

    dub_state = {
        "video_path": "",
        "sub_path": "",
        "folders": {},
        "proj_file": "",
        "data": {},
        "tts_engine": None,
        "cancel_requested": False
    }

    narr_state = {
        "cancel_requested": False,
        "last_mp3": ""
    }

    narr_phrases = []

    log_queue = queue.Queue()

    style = ttk.Style()
    style.theme_use("clam")
    style.configure("Treeview", rowheight=26)
    style.configure("TButton", padding=4)
    style.configure("Accent.TButton", font=('Segoe UI', 9, 'bold'))

    notebook = ttk.Notebook(root)
    notebook.pack(fill="both", expand=True, padx=8, pady=5)

    tab_dubbing = ttk.Frame(notebook)
    tab_narration = ttk.Frame(notebook)

    notebook.add(tab_dubbing, text=" 🎬 1. Video Dubbing Studio (Supertonic, Edge & Clone) ")
    notebook.add(tab_narration, text=" 🎙️ 2. Text Narration & Multi-Voice Speech ")

    # Sort states for Treeview tables
    sub_tree_sort_state = {"col": None, "reverse": False}
    phrase_tree_sort_state = {"col": None, "reverse": False}

    # ==========================================================================
    # TAB 1: VIDEO DUBBING STUDIO
    # ==========================================================================
    top_frame = ttk.LabelFrame(tab_dubbing, text=" 1. Input Files & Configuration ", padding=8)
    top_frame.pack(fill="x", padx=10, pady=3)
    top_frame.columnconfigure(0, weight=0)
    top_frame.columnconfigure(1, weight=1)
    top_frame.columnconfigure(2, weight=0)

    ttk.Label(top_frame, text="Video/Audio:").grid(row=0, column=0, sticky="w", padx=(0, 4), pady=2)
    video_entry = ttk.Entry(top_frame)
    video_entry.grid(row=0, column=1, padx=(2, 4), pady=2, sticky="we")

    def auto_load_matching_subtitles(v_path):
        if not v_path or not os.path.exists(v_path):
            return
        tgt_l = lang_combo.get().strip() if 'lang_combo' in locals() or 'lang_combo' in globals() else "cs"
        matched_sub = find_matching_subtitle(v_path, target_lang=tgt_l)
        if matched_sub:
            sub_entry.delete(0, tk.END)
            sub_entry.insert(0, matched_sub)
            mode_var.set("EXISTING_SRT")
            dub_state["data"]["sub_path"] = matched_sub
            log(f"[Subtitles] Automatically loaded matching subtitles: {os.path.basename(matched_sub)}")

    def browse_video():
        fn = filedialog.askopenfilename(filetypes=[("Video/Audio", "*.mp4 *.mkv *.avi *.mov *.mp3 *.wav")])
        if fn:
            video_entry.delete(0, tk.END)
            video_entry.insert(0, fn)
            init_project_state(fn)
            auto_load_matching_subtitles(fn)

    def on_video_entry_change(event=None):
        v = video_entry.get().strip()
        if v and os.path.exists(v):
            init_project_state(v)
            if not sub_entry.get().strip():
                auto_load_matching_subtitles(v)

    video_entry.bind("<FocusOut>", on_video_entry_change)
    video_entry.bind("<Return>", on_video_entry_change)

    vid_btn_box = ttk.Frame(top_frame)
    vid_btn_box.grid(row=0, column=2, padx=(4, 0), pady=2, sticky="w")
    ttk.Button(vid_btn_box, text="Browse...", command=browse_video).pack(side="left")

    mode_var = tk.StringVar(value="EXISTING_SRT")
    ttk.Radiobutton(top_frame, text="Mode A: Existing subtitles in target language (.srt/.ass)", variable=mode_var, value="EXISTING_SRT").grid(row=1, column=1, sticky="w", pady=1)
    ttk.Radiobutton(top_frame, text="Mode B: No subtitles (auto-transcribe for translation)", variable=mode_var, value="NO_SRT").grid(row=2, column=1, sticky="w", pady=1)

    ttk.Label(top_frame, text="Subtitles (.srt):").grid(row=3, column=0, sticky="w", padx=(0, 4), pady=2)
    sub_entry = ttk.Entry(top_frame)
    sub_entry.grid(row=3, column=1, padx=(2, 4), pady=2, sticky="we")

    def browse_sub():
        fn = filedialog.askopenfilename(filetypes=[("Subtitles", "*.srt *.ass")])
        if fn:
            sub_entry.delete(0, tk.END)
            sub_entry.insert(0, fn)

    sub_btn_box = ttk.Frame(top_frame)
    sub_btn_box.grid(row=3, column=2, padx=(4, 0), pady=2, sticky="w")
    ttk.Button(sub_btn_box, text="Browse...", command=browse_sub).pack(side="left")
    whisper_sub_btn = ttk.Button(sub_btn_box, text="🎙 Transcribe with Whisper", command=lambda: threading.Thread(target=task_transcribe_whisper, daemon=True).start())
    whisper_sub_btn.pack(side="left", padx=(4, 0))

    settings_frame = ttk.Frame(top_frame)
    settings_frame.grid(row=4, column=1, sticky="w", pady=3)

    ttk.Label(settings_frame, text="Default Engine:").pack(side="left", padx=(0, 2))
    engine_var = tk.StringVar(value="Supertonic (Local)")
    engine_combo = ttk.Combobox(settings_frame, textvariable=engine_var, values=["Supertonic (Local)", "Edge-TTS (Online)", "Voice Cloning (XTTS-v2)"], width=18, state="readonly")
    engine_combo.pack(side="left", padx=4)

    sep_var = tk.BooleanVar(value=True)
    ttk.Checkbutton(settings_frame, text="Separate Audio (HTDemucs)", variable=sep_var).pack(side="left", padx=5)

    ttk.Label(settings_frame, text="Language:").pack(side="left", padx=(8, 2))
    lang_combo = ttk.Combobox(settings_frame, values=["cs", "sk", "ja", "en", "de", "pl"], width=5, state="readonly")
    lang_combo.set("cs")
    lang_combo.pack(side="left")

    ttk.Label(settings_frame, text="Gender / F0:").pack(side="left", padx=(8, 2))
    pitch_profile_var = tk.StringVar(value="Asia (China, Japan, Korea)")
    pitch_combo = ttk.Combobox(settings_frame, textvariable=pitch_profile_var, values=["Asia (China, Japan, Korea)", "Western / Europe (US, EU)"], width=25, state="readonly")
    pitch_combo.pack(side="left", padx=2)

    def on_pitch_profile_change(event=None):
        speakers = dub_state["data"].get("speakers", {})
        if speakers:
            cur_p = pitch_profile_var.get()
            assign_supertonic_voices_to_speakers(speakers, profile=cur_p)
            save_project(dub_state["data"], dub_state["proj_file"])
            refresh_speaker_table()
            refresh_subtitle_table()
            log(f"[Profile] Character voices re-assigned according to profile: {cur_p}")

    pitch_combo.bind("<<ComboboxSelected>>", on_pitch_profile_change)

    ttk.Label(settings_frame, text="Max Characters:").pack(side="left", padx=(10, 2))
    max_spk_spin = ttk.Spinbox(settings_frame, from_=2, to=15, width=4)
    max_spk_spin.set(6)
    max_spk_spin.pack(side="left")

    force_reanalyze_var = tk.BooleanVar(value=False)
    ttk.Checkbutton(settings_frame, text="Force re-analyze characters", variable=force_reanalyze_var).pack(side="left", padx=8)

    vol_frame = ttk.Frame(top_frame)
    vol_frame.grid(row=5, column=1, sticky="w", pady=(3, 2))

    ttk.Label(vol_frame, text="Dubbing Volume:").pack(side="left", padx=(0, 2))
    dlg_vol_var = tk.DoubleVar(value=0.90)
    dlg_scale = ttk.Scale(vol_frame, from_=0.50, to=1.40, variable=dlg_vol_var, orient="horizontal", length=75)
    dlg_scale.pack(side="left")
    dlg_lbl = ttk.Label(vol_frame, text="90%")
    dlg_lbl.pack(side="left", padx=(2, 10))
    dlg_scale.config(command=lambda v: dlg_lbl.config(text=f"{int(float(v)*100)}%"))

    ttk.Label(vol_frame, text="Background Volume:").pack(side="left", padx=(0, 2))
    bg_vol_var = tk.DoubleVar(value=1.00)
    bg_scale = ttk.Scale(vol_frame, from_=0.50, to=1.40, variable=bg_vol_var, orient="horizontal", length=75)
    bg_scale.pack(side="left")
    bg_lbl = ttk.Label(vol_frame, text="100%")
    bg_lbl.pack(side="left", padx=(2, 10))
    bg_scale.config(command=lambda v: bg_lbl.config(text=f"{int(float(v)*100)}%"))

    ttk.Label(vol_frame, text="Background Ducking:").pack(side="left", padx=(0, 2))
    duck_var = tk.DoubleVar(value=0.0)
    duck_scale = ttk.Scale(vol_frame, from_=0.0, to=0.50, variable=duck_var, orient="horizontal", length=75)
    duck_scale.pack(side="left")
    duck_lbl = ttk.Label(vol_frame, text="0% (Disabled - full background preserved)")
    duck_lbl.pack(side="left", padx=(2, 4))
    duck_scale.config(command=lambda v: duck_lbl.config(text=f"{int(float(v)*100)}%" if float(v) > 0.01 else "0% (Disabled - full background preserved)"))

    # 2. Rámec pre detegované postavy
    mid_frame = ttk.LabelFrame(tab_dubbing, text=" 2. Detected Characters (Default Voices - M1..M5 / F1..F5) ", padding=6)
    mid_frame.pack(fill="x", padx=10, pady=2)

    columns = ("speaker", "duration", "gender", "mode", "voice_ref")
    spk_tree = ttk.Treeview(mid_frame, columns=columns, show="headings", height=4, selectmode="extended")
    
    spk_tree.heading("speaker", text="Character")
    spk_tree.heading("duration", text="Speech Duration")
    spk_tree.heading("gender", text="Gender")
    spk_tree.heading("mode", text="Mode")
    spk_tree.heading("voice_ref", text="Assigned Voice / Sample")

    spk_tree.column("speaker", width=95, anchor="center")
    spk_tree.column("duration", width=85, anchor="center")
    spk_tree.column("gender", width=95, anchor="center")
    spk_tree.column("mode", width=140, anchor="w")
    spk_tree.column("voice_ref", width=380, anchor="w")
    spk_tree.pack(side="left", fill="both", expand=True)

    tree_scroll = ttk.Scrollbar(mid_frame, orient="vertical", command=spk_tree.yview)
    spk_tree.configure(yscrollcommand=tree_scroll.set)
    tree_scroll.pack(side="right", fill="y")

    tbl_btn_frame = ttk.Frame(tab_dubbing)
    tbl_btn_frame.pack(fill="x", padx=10, pady=2)

    def refresh_speaker_table():
        if 'update_spk_voices_dropdown' in locals() or 'update_spk_voices_dropdown' in globals():
            update_spk_voices_dropdown()
        for row in spk_tree.get_children():
            spk_tree.delete(row)
        speakers = dub_state["data"].get("speakers", {})
        eng = engine_var.get()

        for spk_id, d in sorted(speakers.items()):
            dur_str = f"{d.get('total_duration', 0):.1f} s"

            # Always display gender directly from the assigned character metadata
            if d.get("gender") == "female" or d.get("is_female") is True:
                gender_label = "Female 👩"
            elif d.get("gender") == "male" or d.get("is_female") is False:
                gender_label = "Male 👨"
            else:
                prof_val = pitch_profile_var.get()
                is_asia_prof = "ázia" in prof_val.lower() or "azia" in prof_val.lower() or "asia" in prof_val.lower()
                split_thresh = 165.0 if is_asia_prof else 155.0
                gender_label = "Female 👩" if d.get("avg_f0", 140.0) >= split_thresh else "Male 👨"

            if d.get("custom_voice_path"):
                v_name = f"Custom WAV: {os.path.basename(d['custom_voice_path'])}"
                mode_str = "Custom Clone"
            elif "Supertonic" in eng:
                v_name = f"⚡ Supertonic [{d.get('supertonic_voice', 'M1')}]"
                mode_str = "Supertonic"
            elif "Edge" in eng:
                edge_v = "cs-CZ-VlastaNeural" if avg_f0 >= 175.0 else "cs-CZ-AntoninNeural"
                v_name = f"🎙️ Edge: {edge_v}"
                mode_str = "Edge-TTS"
            else:
                v_name = os.path.basename(d.get("voice_path") or d.get("preview_file") or "clean_voice.wav")
                mode_str = "Voice Cloning (XTTS)"

            spk_tree.insert("", tk.END, values=(
                spk_id,
                dur_str,
                gender_label,
                mode_str,
                v_name
            ))

    def play_preview():
        sel = spk_tree.selection()
        if not sel:
            messagebox.showinfo("Info", "Please select a character from the list.")
            return
        spk_id = spk_tree.item(sel[0])["values"][0]
        spk_data = dub_state["data"].get("speakers", {}).get(spk_id, {})
        eng = engine_var.get()

        try:
            if "Supertonic" in eng and not spk_data.get("custom_voice_path"):
                v_name = spk_data.get("supertonic_voice", "M1")
                test_wav = os.path.join(VOICES_DIR, f"preview_st_{v_name}.wav")
                if not os.path.exists(test_wav):
                    engine = DynamicDubbingEngine(target_lang=lang_combo.get().strip(), engine_type="supertonic")
                    engine.ensure_supertonic_loaded(log_cb=log)
                    style = engine.supertonic_model.get_voice_style(voice_name=v_name)
                    sample_txt = "This is a sample voice for dubbing synthesized with Supertonic."
                    wav, _ = engine.supertonic_model.synthesize(text=sample_txt, voice_style=style, lang=lang_combo.get().strip(), total_steps=8, speed=1.0)
                    engine.supertonic_model.save_audio(wav, test_wav)
                data, sr = sf.read(test_wav)
                sd.stop()
                sd.play(data, sr)
                return

            v_path = get_valid_voice_path(spk_id, dub_state)
            if not v_path or not os.path.exists(v_path):
                messagebox.showwarning("Notice", f"Voice for {spk_id} was not found.")
                return
            data, sr = sf.read(v_path)
            sd.stop()
            sd.play(data, sr)
        except Exception as e:
            messagebox.showwarning("Notice", f"Playback error: {str(e)}")

    def cycle_supertonic_voice():
        sel = spk_tree.selection()
        if not sel:
            messagebox.showinfo("Info", "Please select a character from the table.")
            return
        spk_id = spk_tree.item(sel[0])["values"][0]
        spk_data = dub_state["data"]["speakers"][spk_id]
        is_fem = spk_data.get("avg_f0", 140.0) >= 165.0
        pool = SUPERTONIC_FEMALE_VOICES if is_fem else SUPERTONIC_MALE_VOICES
        curr_v = spk_data.get("supertonic_voice", pool[0])
        
        curr_idx = pool.index(curr_v) if curr_v in pool else -1
        next_v = pool[(curr_idx + 1) % len(pool)]
        spk_data["supertonic_voice"] = next_v
        
        # Aktualizujeme aj repliky danej postavy, ktoré nemajú explicitný prepis
        for sub in dub_state["data"].get("subtitles", []):
            if sub.get("speaker") == spk_id:
                sub["assigned_voice"] = f"⚡ Supertonic [{next_v}]"

        save_project(dub_state["data"], dub_state["proj_file"])
        refresh_speaker_table()
        refresh_subtitle_table()
        log(f"[Voice] {spk_id} switched to Supertonic [{next_v}]")

    def assign_custom_voice():
        sel = spk_tree.selection()
        if not sel:
            messagebox.showinfo("Info", "Please select a character from the list.")
            return
        spk_id = spk_tree.item(sel[0])["values"][0]
        fn = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav *.mp3 *.flac *.m4a")])
        if fn:
            target_wav = os.path.join(dub_state["folders"]["speakers"], f"{spk_id}_custom_{os.path.basename(fn)}.wav")
            cmd = ["ffmpeg", "-y", "-i", fn, "-ar", str(SAMPLE_RATE), "-ac", "1", target_wav]
            run_cmd(cmd)
            custom_voice_name = os.path.splitext(os.path.basename(fn))[0]
            
            # Ak pri zdrojovom súbore existuje .txt, skopírujeme ho k cieľovému
            src_txt = os.path.splitext(fn)[0] + ".txt"
            if os.path.exists(src_txt):
                shutil.copy2(src_txt, os.path.splitext(target_wav)[0] + ".txt")

            dub_state["data"]["speakers"][spk_id]["custom_voice_path"] = target_wav
            dub_state["data"]["speakers"][spk_id]["mode"] = f"Vlastný WAV ({custom_voice_name})"
            
            # Nastavíme aj pre repliky tejto postavy
            for sub in dub_state["data"].get("subtitles", []):
                if sub.get("speaker") == spk_id:
                    sub["assigned_voice"] = f"🧬 [Klon] {custom_voice_name}"

            save_project(dub_state["data"], dub_state["proj_file"])
            refresh_speaker_table()
            refresh_subtitle_table()

    ttk.Label(tbl_btn_frame, text="Character Voice:").pack(side="left", padx=(2, 2))
    spk_voice_combo = ttk.Combobox(tbl_btn_frame, width=34, state="readonly")
    spk_voice_combo.pack(side="left", padx=2)
    spk_voice_combo.bind("<Return>", lambda e: play_speaker_voice_preview())

    def update_spk_voices_dropdown():
        tgt_l = lang_combo.get().strip() if 'lang_combo' in locals() or 'lang_combo' in globals() else "cs"
        cat, vmap = get_all_available_voices(tgt_l)
        spk_voice_combo['values'] = cat
        if cat and not spk_voice_combo.get():
            spk_voice_combo.set(cat[0])

    # Populate immediately on creation!
    update_spk_voices_dropdown()

    def on_speaker_select(event=None):
        if not spk_voice_combo['values']:
            update_spk_voices_dropdown()
        sel = spk_tree.selection()
        if not sel:
            return
        spk_id = spk_tree.item(sel[0])["values"][0]
        spk_data = dub_state["data"].get("speakers", {}).get(spk_id, {})
        cur_v = spk_data.get("assigned_voice") or spk_data.get("supertonic_voice")
        if cur_v:
            cur_v_lower = str(cur_v).lower()
            for val in spk_voice_combo["values"]:
                if cur_v_lower in val.lower() or val.lower() in cur_v_lower:
                    spk_voice_combo.set(val)
                    break

    spk_tree.bind("<<TreeviewSelect>>", on_speaker_select)

    def assign_selected_voice_to_speaker():
        sel = spk_tree.selection()
        if not sel:
            messagebox.showinfo("Selection", "Please select a character from the table first.")
            return
        chosen = spk_voice_combo.get()
        if not chosen:
            messagebox.showinfo("Voice", "Please select a voice from the dropdown list.")
            return

        cat, vmap = get_all_available_voices(lang_combo.get().strip())
        v_info = vmap.get(chosen, {})

        for item_id in sel:
            spk_id = spk_tree.item(item_id)["values"][0]
            spk_data = dub_state["data"]["speakers"].setdefault(spk_id, {})

            if v_info.get("type") == "supertonic":
                v_id = v_info["voice_id"]
                spk_data["supertonic_voice"] = v_id
                spk_data["custom_voice_path"] = ""
                spk_data["mode"] = "Supertonic"
                assigned_label = f"⚡ Supertonic [{v_id}]"
                # If user manually assigns an F voice, mark as female; if M voice, mark as male
                if str(v_id).upper().startswith("F"):
                    spk_data["gender"] = "female"
                    spk_data["is_female"] = True
                elif str(v_id).upper().startswith("M"):
                    spk_data["gender"] = "male"
                    spk_data["is_female"] = False
            elif v_info.get("type") == "edge":
                v_id = v_info["voice_id"]
                spk_data["edge_voice"] = v_id
                spk_data["custom_voice_path"] = ""
                spk_data["mode"] = f"Edge-TTS ({v_id})"
                assigned_label = f"🎙️ [Edge] {v_id}"
            elif v_info.get("type") == "clone":
                v_path = v_info.get("path", "")
                spk_data["custom_voice_path"] = v_path
                spk_data["mode"] = f"Vlastný WAV ({v_info.get('voice_id')})"
                assigned_label = f"🧬 [Klon] {v_info.get('voice_id')}"
            else:
                assigned_label = chosen

            spk_data["assigned_voice"] = assigned_label

            # Aktualizujeme aj všetky doterajšie repliky danej postavy
            for sub in dub_state["data"].get("subtitles", []):
                if sub.get("speaker") == spk_id:
                    sub["assigned_voice"] = assigned_label

        save_project(dub_state["data"], dub_state["proj_file"])
        refresh_speaker_table()
        refresh_subtitle_table()
        log(f"[Character] Voice successfully assigned to character: {chosen}")

    def play_speaker_voice_preview():
        chosen = spk_voice_combo.get()
        if not chosen:
            sel = spk_tree.selection()
            if sel:
                spk_id = spk_tree.item(sel[0])["values"][0]
                spk_data = dub_state["data"].get("speakers", {}).get(spk_id, {})
                chosen = spk_data.get("assigned_voice")
        if not chosen:
            messagebox.showinfo("Info", "Select a voice in the dropdown to listen to a preview.")
            return
        ok, msg = play_any_voice_sample(chosen, target_lang=lang_combo.get().strip(), log_cb=log)
        if not ok:
            messagebox.showwarning("Notice", msg)

    def play_speaker_original_clip():
        sel = spk_tree.selection()
        if not sel:
            messagebox.showinfo("Info", "Please select a character from the table first.")
            return
        spk_id = spk_tree.item(sel[0])["values"][0]
        spk_data = dub_state["data"].get("speakers", {}).get(spk_id, {})
        clip_path = spk_data.get("preview_file")
        if not (clip_path and os.path.exists(clip_path)):
            spk_dir = dub_state.get("folders", {}).get("speakers")
            if spk_dir:
                clip_path = os.path.join(spk_dir, f"{spk_id}_clean_voice.wav")
        if not (clip_path and os.path.exists(clip_path)):
            messagebox.showwarning(
                "Notice",
                f"Original voice clip for {spk_id} was not found.\n\n"
                "Please run '1. Analyze & Diarize' first to separate audio and extract character clips."
            )
            return
        try:
            data, sr = sf.read(clip_path)
            sd.stop()
            sd.play(data, sr)
            log(f"[Playback] Playing original clip for {spk_id}: {os.path.basename(clip_path)}")
        except Exception as e:
            messagebox.showwarning("Notice", f"Playback error: {e}")

    def merge_selected_speakers():
        sel = spk_tree.selection()
        if len(sel) < 2:
            messagebox.showinfo(
                "Merge Characters",
                "Please select at least 2 characters in the table to merge (hold Ctrl or Shift while clicking)."
            )
            return

        spk_ids = [spk_tree.item(item_id)["values"][0] for item_id in sel]
        target_spk = spk_ids[0]
        source_spks = spk_ids[1:]

        speakers = dub_state["data"].setdefault("speakers", {})
        if target_spk not in speakers:
            messagebox.showerror("Error", f"Target character '{target_spk}' not found in project.")
            return

        confirm = messagebox.askyesno(
            "Confirm Character Merge",
            f"Are you sure you want to merge character(s) {', '.join(source_spks)} into '{target_spk}'?\n\n"
            f"All dialogue lines belonging to {', '.join(source_spks)} will be reassigned to '{target_spk}'."
        )
        if not confirm:
            return

        target_data = speakers[target_spk]
        target_voice = target_data.get("assigned_voice") or f"⚡ Supertonic [{target_data.get('supertonic_voice', 'M1')}]"

        # 1. Merge segments and durations
        for src_id in source_spks:
            if src_id in speakers:
                src_data = speakers[src_id]
                target_data["segments"].extend(src_data.get("segments", []))
                target_data["total_duration"] += src_data.get("total_duration", 0.0)
                del speakers[src_id]

        # 2. Recalculate robust avg_f0 for target character across combined segments
        all_pitches = [s["f0"] for s in target_data["segments"] if "f0" in s and 65.0 < s["f0"] < 350.0]
        if len(all_pitches) >= 2:
            p_arr = np.array(all_pitches, dtype=np.float32)
            p20, p80 = np.percentile(p_arr, 20), np.percentile(p_arr, 80)
            robust = p_arr[(p_arr >= p20) & (p_arr <= p80)]
            target_data["avg_f0"] = float(np.median(robust if len(robust) > 0 else p_arr))
        elif all_pitches:
            target_data["avg_f0"] = float(all_pitches[0])

        # 3. Update all subtitle lines belonging to the merged characters
        subtitles = dub_state["data"].get("subtitles", [])
        reassigned_count = 0
        for sub in subtitles:
            if sub.get("speaker") in source_spks:
                sub["speaker"] = target_spk
                sub["assigned_voice"] = target_voice
                reassigned_count += 1

        save_project(dub_state["data"], dub_state["proj_file"])
        refresh_speaker_table()
        refresh_subtitle_table()

        log(f"[Characters] Successfully merged {', '.join(source_spks)} into {target_spk} ({reassigned_count} dialogue lines reassigned).")
        msg_txt = f"Successfully merged {', '.join(source_spks)} into {target_spk}!\n\nUpdated {reassigned_count} dialogue lines to {target_spk}."
        messagebox.showinfo("Characters Merged", msg_txt)

    ttk.Button(tbl_btn_frame, text="▶ Test", width=7, command=play_speaker_voice_preview).pack(side="left", padx=2)
    ttk.Button(tbl_btn_frame, text="✔ Assign", style="Accent.TButton", command=assign_selected_voice_to_speaker).pack(side="left", padx=2)
    ttk.Separator(tbl_btn_frame, orient="vertical").pack(side="left", fill="y", padx=8, pady=2)
    ttk.Button(tbl_btn_frame, text="▶ Play Original Clip (Video)", command=play_speaker_original_clip).pack(side="left", padx=3)
    ttk.Button(tbl_btn_frame, text="🔗 Merge Characters (Ctrl/Shift)", command=merge_selected_speakers).pack(side="left", padx=3)

    # 3. Nový rámec: Zoznam replík / titulkov a priradenie hlasov (Tab 1)
    sub_frame = ttk.LabelFrame(tab_dubbing, text=" 3. Subtitles & Voice Assignment (Ctrl/Shift selection, sortable columns) ", padding=6)
    sub_frame.pack(fill="both", expand=True, padx=10, pady=3)

    sub_toolbar = ttk.Frame(sub_frame)
    sub_toolbar.pack(fill="x", pady=(0, 4))

    ttk.Label(sub_toolbar, text="Line Voice:").pack(side="left", padx=(2, 2))
    sub_voice_combo = ttk.Combobox(sub_toolbar, width=34, state="readonly")
    sub_voice_combo.pack(side="left", padx=2)
    sub_voice_combo.bind("<Return>", lambda e: play_sub_voice_preview())

    def update_tab1_voices_dropdown():
        cat, vmap = get_all_available_voices(lang_combo.get().strip())
        sub_voice_combo['values'] = cat
        if cat and not sub_voice_combo.get():
            sub_voice_combo.set(cat[0])

    def assign_voice_to_selected_subtitles():
        sel = sub_tree.selection()
        if not sel:
            messagebox.showinfo("Selection", "Select at least one subtitle line (Ctrl/Shift selection supported).")
            return
        chosen = sub_voice_combo.get()
        if not chosen:
            messagebox.showinfo("Voice", "Please choose a voice from the dropdown list.")
            return

        sub_dict = {str(s["id"]): s for s in dub_state["data"].get("subtitles", [])}
        for item_id in sel:
            row_id_val = str(sub_tree.set(item_id, "id"))
            if row_id_val in sub_dict:
                sub_dict[row_id_val]["assigned_voice"] = chosen
            sub_tree.set(item_id, "voice", chosen)

        save_project(dub_state["data"], dub_state["proj_file"])
        log(f"[Subtitles] Voice '{chosen}' assigned to {len(sel)} subtitle lines.")

    def reset_selected_subtitles_voice():
        sel = sub_tree.selection()
        if not sel:
            sel = sub_tree.get_children()

        speakers = dub_state["data"].get("speakers", {})
        sub_dict = {str(s["id"]): s for s in dub_state["data"].get("subtitles", [])}

        for item_id in sel:
            row_id_val = str(sub_tree.set(item_id, "id"))
            if row_id_val in sub_dict:
                sub = sub_dict[row_id_val]
                spk = sub.get("speaker", "Speaker_01")
                spk_data = speakers.get(spk, {})
                if spk_data.get("custom_voice_path"):
                    v_label = f"🧬 [Klon] {os.path.basename(spk_data['custom_voice_path'])}"
                elif spk_data.get("supertonic_voice"):
                    v_label = f"⚡ Supertonic [{spk_data['supertonic_voice']}]"
                else:
                    v_label = "⚡ Supertonic [M1]"
                sub["assigned_voice"] = v_label
                sub_tree.set(item_id, "voice", v_label)

        save_project(dub_state["data"], dub_state["proj_file"])
        log(f"[Subtitles] Lines reset to character default voices ({len(sel)} lines).")

    def play_sub_voice_preview():
        chosen = sub_voice_combo.get()
        if not chosen:
            messagebox.showinfo("Voice", "Please select a voice from the dropdown list.")
            return
        ok, msg = play_any_voice_sample(chosen, target_lang=lang_combo.get().strip(), log_cb=log)
        if not ok:
            messagebox.showwarning("Upozornenie", msg)

    ttk.Button(sub_toolbar, text="▶ Test", width=7, command=play_sub_voice_preview).pack(side="left", padx=2)
    ttk.Button(sub_toolbar, text="✓ Apply to Selected", style="Accent.TButton", command=assign_voice_to_selected_subtitles).pack(side="left", padx=3)
    ttk.Button(sub_toolbar, text="↺ Reset to Character", command=reset_selected_subtitles_voice).pack(side="left", padx=3)
    ttk.Button(sub_toolbar, text="Select All", command=lambda: sub_tree.selection_set(sub_tree.get_children())).pack(side="left", padx=3)

    sub_columns = ("id", "time", "speaker", "voice", "text")
    sub_tree = ttk.Treeview(sub_frame, columns=sub_columns, show="headings", height=8, selectmode="extended")

    sub_tree.heading("id", text="#", command=lambda: sort_treeview(sub_tree, "id", sub_tree_sort_state))
    sub_tree.heading("time", text="Timing", command=lambda: sort_treeview(sub_tree, "time", sub_tree_sort_state))
    sub_tree.heading("speaker", text="Character", command=lambda: sort_treeview(sub_tree, "speaker", sub_tree_sort_state))
    sub_tree.heading("voice", text="Assigned Voice", command=lambda: sort_treeview(sub_tree, "voice", sub_tree_sort_state))
    sub_tree.heading("text", text="Subtitle Text", command=lambda: sort_treeview(sub_tree, "text", sub_tree_sort_state))

    sub_tree.column("id", width=45, anchor="center")
    sub_tree.column("time", width=180, anchor="center")
    sub_tree.column("speaker", width=95, anchor="center")
    sub_tree.column("voice", width=220, anchor="w")
    sub_tree.column("text", width=480, anchor="w")

    sub_tree.pack(side="left", fill="both", expand=True)
    sub_scroll = ttk.Scrollbar(sub_frame, orient="vertical", command=sub_tree.yview)
    sub_tree.configure(yscrollcommand=sub_scroll.set)
    sub_scroll.pack(side="right", fill="y")

    sub_tree.bind("<Control-a>", lambda e: sub_tree.selection_set(sub_tree.get_children()))
    sub_tree.bind("<Double-1>", lambda e: assign_voice_to_selected_subtitles())

    def refresh_subtitle_table():
        for r in sub_tree.get_children():
            sub_tree.delete(r)

        speakers = dub_state["data"].get("speakers", {})
        subtitles = dub_state["data"].get("subtitles", [])

        for s in subtitles:
            sid = s.get("id", 1)
            spk = s.get("speaker", "Speaker_01")
            assigned_v = s.get("assigned_voice")

            # Ak replika nemá explicitne priradený hlas, získa predvolený hlas svojej postavy
            if not assigned_v:
                spk_data = speakers.get(spk, {})
                if spk_data.get("custom_voice_path"):
                    assigned_v = f"🧬 [Klon] {os.path.basename(spk_data['custom_voice_path'])}"
                elif spk_data.get("supertonic_voice"):
                    assigned_v = f"⚡ Supertonic [{spk_data['supertonic_voice']}]"
                else:
                    assigned_v = "⚡ Supertonic [M1]"
                s["assigned_voice"] = assigned_v

            time_str = f"{format_timestamp_srt(s['start'])} --> {format_timestamp_srt(s['end'])}"
            sub_tree.insert("", tk.END, iid=str(sid), values=(sid, time_str, spk, assigned_v, s.get("text", "")))

    engine_combo.bind("<<ComboboxSelected>>", lambda e: [refresh_speaker_table(), refresh_subtitle_table()])
    lang_combo.bind("<<ComboboxSelected>>", lambda e: [update_tab1_voices_dropdown(), update_spk_voices_dropdown(), refresh_speaker_table(), refresh_subtitle_table()])

    # 4. Rámec ovládania spustenia dabingu
    dub_ctrl_frame = ttk.LabelFrame(tab_dubbing, text=" 4. Dubbing Execution ", padding=8)
    dub_ctrl_frame.pack(fill="x", padx=10, pady=3)

    btn_row = ttk.Frame(dub_ctrl_frame)
    btn_row.pack(fill="x", pady=2)

    analyze_btn = ttk.Button(btn_row, text="▶ 1. Analyze & Diarize", style="Accent.TButton")
    analyze_btn.pack(side="left", padx=5)

    export_work_btn = ttk.Button(btn_row, text="Export Translation (Mode B)")
    export_work_btn.pack(side="left", padx=5)

    import_work_btn = ttk.Button(btn_row, text="Import Translation (Mode B)")
    import_work_btn.pack(side="left", padx=5)

    dub_btn = ttk.Button(btn_row, text="🚀 2. Generate Dub & Export Video", style="Accent.TButton")
    dub_btn.pack(side="left", padx=5)

    batch_dub_btn = ttk.Button(btn_row, text="🚀 Batch Dubbing (Folder)", style="Accent.TButton")
    batch_dub_btn.pack(side="left", padx=5)

    reset_proj_btn = ttk.Button(btn_row, text="🗑️ Clear Project Cache")
    reset_proj_btn.pack(side="left", padx=5)

    cancel_btn = ttk.Button(btn_row, text="⏹ Cancel", state="disabled")
    cancel_btn.pack(side="right", padx=5)

    # ==========================================================================
    # TAB 2: TEXT NARRATION & VIACHLASOVÝ PREDNES
    # ==========================================================================
    narr_top = ttk.LabelFrame(tab_narration, text=" 1. Language & Default Voice (Supertonic, Edge & Clone) ", padding=8)
    narr_top.pack(fill="x", padx=10, pady=3)

    cfg_bar = ttk.Frame(narr_top)
    cfg_bar.pack(fill="x", pady=2)

    ttk.Label(cfg_bar, text="Language:").pack(side="left", padx=(5, 2))
    narr_lang_combo = ttk.Combobox(cfg_bar, values=["cs (Czech)", "sk (Slovak)", "ja (Japanese)", "en (English)", "de (German)", "pl (Polish)"], width=15, state="readonly")
    narr_lang_combo.set("cs (Czech)")
    narr_lang_combo.pack(side="left", padx=2)

    ttk.Label(cfg_bar, text="Default Voice:").pack(side="left", padx=(12, 2))
    voice_combo = ttk.Combobox(cfg_bar, width=40, state="readonly")
    voice_combo.pack(side="left", padx=2)

    ttk.Label(cfg_bar, text="Speed:").pack(side="left", padx=(12, 2))
    speed_combo = ttk.Combobox(cfg_bar, values=["0.85", "0.90", "0.95", "1.0", "1.05", "1.10"], width=5)
    speed_combo.set("1.0")
    speed_combo.pack(side="left", padx=2)

    voice_mapping_tab2 = {}

    def update_narr_voices_dropdown(event=None):
        nonlocal voice_mapping_tab2
        cat, vmap = get_all_available_voices(narr_lang_combo.get().split()[0])
        voice_mapping_tab2 = vmap
        voice_combo['values'] = cat
        if cat and not voice_combo.get():
            voice_combo.set(cat[0])
        phrase_voice_combo['values'] = cat
        if cat and not phrase_voice_combo.get():
            phrase_voice_combo.set(cat[0])

    narr_lang_combo.bind("<<ComboboxSelected>>", lambda e: [update_narr_voices_dropdown(), parse_and_update_tab2_phrases(force_default=False)])

    btn_voice_row = ttk.Frame(narr_top)
    btn_voice_row.pack(fill="x", pady=(4, 2))

    def play_tab2_voice():
        chosen = voice_combo.get()
        if not chosen:
            messagebox.showwarning("Voice", "Please select a voice from the list.")
            return
        ok, msg = play_any_voice_sample(chosen, target_lang=narr_lang_combo.get().split()[0], log_cb=log)
        if not ok:
            messagebox.showwarning("Upozornenie", msg)

    def add_custom_voice_tab2():
        fn = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav *.mp3 *.flac *.ogg *.m4a")])
        if fn:
            clean_name = os.path.splitext(os.path.basename(fn))[0] + ".wav"
            clean_wav = os.path.join(VOICES_DIR, clean_name)
            cmd = ["ffmpeg", "-y", "-i", fn, "-ar", str(SAMPLE_RATE), "-ac", "1", clean_wav]
            run_cmd(cmd)
            
            # Kontrola a skopírovanie .txt súboru, ak existuje
            src_txt = os.path.splitext(fn)[0] + ".txt"
            if os.path.exists(src_txt):
                shutil.copy2(src_txt, os.path.splitext(clean_wav)[0] + ".txt")

            update_narr_voices_dropdown()
            update_tab1_voices_dropdown()
            update_spk_voices_dropdown()
            messagebox.showinfo("Success", f"Voice added to ./voices/{clean_name}")

    ttk.Button(btn_voice_row, text="▶ Preview Default Voice", command=play_tab2_voice).pack(side="left", padx=3)
    ttk.Button(btn_voice_row, text="➕ Add Custom Voice (.wav/.mp3)", command=add_custom_voice_tab2).pack(side="left", padx=3)
    ttk.Button(btn_voice_row, text="📁 Open voices/ Folder", command=lambda: os.startfile(VOICES_DIR) if os.name == 'nt' else subprocess.run(['xdg-open', VOICES_DIR])).pack(side="left", padx=3)

    # 2. Vstup hlavného textu
    narr_cfg = ttk.LabelFrame(tab_narration, text=" 2. Main Text Input ", padding=6)
    narr_cfg.pack(fill="x", padx=10, pady=3)

    def load_narr_file():
        fn = filedialog.askopenfilename(filetypes=[("Text & Subtitle Files", "*.txt *.srt")])
        if fn:
            try:
                with open(fn, 'r', encoding='utf-8-sig') as f:
                    content = f.read()
                narr_text_box.delete("1.0", tk.END)
                narr_text_box.insert("1.0", content)
                parse_and_update_tab2_phrases(force_default=False)
            except Exception as e:
                messagebox.showwarning("Notice", f"Error: {e}")

    top_txt_bar = ttk.Frame(narr_cfg)
    top_txt_bar.pack(fill="x", pady=2)
    ttk.Label(top_txt_bar, text="Type or paste text below (automatically split into sentences):").pack(side="left")
    ttk.Button(top_txt_bar, text="📂 Load from File (.txt / .srt)", command=load_narr_file).pack(side="right")

    narr_text_box = tk.Text(narr_cfg, height=5, wrap="word", font=("Segoe UI", 10))
    narr_text_box.pack(fill="both", expand=True, padx=4, pady=3)
    
    default_text = "Azeroth calls! Deep in the forests of Ashenvale, the drums of war thunder across the land! Thrall, Warchief of the Horde, rallies his fiercest champions at the gates of Orgrimmar!"
    narr_text_box.insert("1.0", default_text)

    # 3. Zoznam viet & priradenie hlasov jednotlivým frázam (Tab 2)
    narr_phrase_frame = ttk.LabelFrame(tab_narration, text=" 3. Multi-Voice Phrase Assignment (Ctrl/Shift selection, sortable columns) ", padding=6)
    narr_phrase_frame.pack(fill="both", expand=True, padx=10, pady=3)

    phrase_toolbar = ttk.Frame(narr_phrase_frame)
    phrase_toolbar.pack(fill="x", pady=(0, 4))

    ttk.Label(phrase_toolbar, text="Voice for selected phrases:").pack(side="left", padx=(2, 4))
    phrase_voice_combo = ttk.Combobox(phrase_toolbar, width=40, state="readonly")
    phrase_voice_combo.pack(side="left", padx=2)

    def parse_and_update_tab2_phrases(force_default=False):
        raw_text = narr_text_box.get("1.0", tk.END).strip()
        if not raw_text:
            narr_phrases.clear()
            for r in phrase_tree.get_children():
                phrase_tree.delete(r)
            return

        sentences = split_text_into_sentences_list(raw_text)
        cur_cat, cur_map = get_all_available_voices(narr_lang_combo.get().split()[0])
        def_label = voice_combo.get()
        if not def_label and cur_cat:
            def_label = cur_cat[0]
        def_info = cur_map.get(def_label, {"type": "supertonic", "voice_id": "M1", "label": def_label})

        # Zachovanie predchádzajúcich priradení hlasov pre rovnaké vety
        old_by_idx = {i: p for i, p in enumerate(narr_phrases)}
        old_by_text = {p["text"].strip(): p for p in narr_phrases}

        new_phrases = []
        for idx, s in enumerate(sentences):
            s_clean = s.strip()
            v_label = def_label
            v_info = def_info

            if not force_default:
                if idx in old_by_idx and old_by_idx[idx]["text"].strip() == s_clean:
                    v_label = old_by_idx[idx]["voice_label"]
                    v_info = old_by_idx[idx]["voice_info"]
                elif s_clean in old_by_text:
                    v_label = old_by_text[s_clean]["voice_label"]
                    v_info = old_by_text[s_clean]["voice_info"]

            new_phrases.append({
                "id": idx + 1,
                "text": s_clean,
                "voice_label": v_label,
                "voice_info": v_info
            })

        narr_phrases.clear()
        narr_phrases.extend(new_phrases)

        for r in phrase_tree.get_children():
            phrase_tree.delete(r)

        for p in narr_phrases:
            phrase_tree.insert("", tk.END, iid=str(p["id"]), values=(p["id"], p["voice_label"], p["text"]))

    def assign_voice_to_selected_phrases():
        sel = phrase_tree.selection()
        if not sel:
            messagebox.showinfo("Selection", "Select at least one sentence from the table (Ctrl/Shift selection supported).")
            return
        chosen = phrase_voice_combo.get()
        if not chosen:
            messagebox.showinfo("Voice", "Please select a voice from the dropdown list.")
            return

        cur_cat, cur_map = get_all_available_voices(narr_lang_combo.get().split()[0])
        chosen_info = cur_map.get(chosen, {"type": "supertonic", "voice_id": "M1", "label": chosen})

        phrase_dict = {str(p["id"]): p for p in narr_phrases}
        for item_id in sel:
            if str(item_id) in phrase_dict:
                phrase_dict[str(item_id)]["voice_label"] = chosen
                phrase_dict[str(item_id)]["voice_info"] = chosen_info
            phrase_tree.set(item_id, "voice", chosen)

        log(f"[Narration] Voice '{chosen}' assigned to {len(sel)} sentences.")

    def reset_all_phrases_to_default():
        parse_and_update_tab2_phrases(force_default=True)
        log("[Narration] Default voice applied to all sentences.")

    def play_tab2_phrase_voice_preview():
        chosen = phrase_voice_combo.get()
        if not chosen:
            messagebox.showinfo("Voice", "Please select a voice from the dropdown list.")
            return
        ok, msg = play_any_voice_sample(chosen, target_lang=narr_lang_combo.get().split()[0], log_cb=log)
        if not ok:
            messagebox.showwarning("Upozornenie", msg)

    ttk.Button(phrase_toolbar, text="✓ Apply to Selected (Ctrl/Shift)", style="Accent.TButton", command=assign_voice_to_selected_phrases).pack(side="left", padx=4)
    ttk.Button(phrase_toolbar, text="🔄 Parse Sentences", command=lambda: parse_and_update_tab2_phrases(force_default=False)).pack(side="left", padx=3)
    ttk.Button(phrase_toolbar, text="↺ Reset All to Default", command=reset_all_phrases_to_default).pack(side="left", padx=3)
    ttk.Button(phrase_toolbar, text="▶ Preview Voice", command=play_tab2_phrase_voice_preview).pack(side="left", padx=3)
    ttk.Button(phrase_toolbar, text="Select All", command=lambda: phrase_tree.selection_set(phrase_tree.get_children())).pack(side="left", padx=3)

    phrase_columns = ("id", "voice", "text")
    phrase_tree = ttk.Treeview(narr_phrase_frame, columns=phrase_columns, show="headings", height=8, selectmode="extended")

    phrase_tree.heading("id", text="#", command=lambda: sort_treeview(phrase_tree, "id", phrase_tree_sort_state))
    phrase_tree.heading("voice", text="Assigned Voice", command=lambda: sort_treeview(phrase_tree, "voice", phrase_tree_sort_state))
    phrase_tree.heading("text", text="Sentence / Phrase Text", command=lambda: sort_treeview(phrase_tree, "text", phrase_tree_sort_state))

    phrase_tree.column("id", width=45, anchor="center")
    phrase_tree.column("voice", width=260, anchor="w")
    phrase_tree.column("text", width=680, anchor="w")

    phrase_tree.pack(side="left", fill="both", expand=True)
    phrase_scroll = ttk.Scrollbar(narr_phrase_frame, orient="vertical", command=phrase_tree.yview)
    phrase_tree.configure(yscrollcommand=phrase_scroll.set)
    phrase_scroll.pack(side="right", fill="y")

    phrase_tree.bind("<Control-a>", lambda e: phrase_tree.selection_set(phrase_tree.get_children()))
    phrase_tree.bind("<Double-1>", lambda e: assign_voice_to_selected_phrases())

    # Debounce for text editing in Tab 2
    _typing_timer = None
    def on_narr_text_modified(event=None):
        nonlocal _typing_timer
        if _typing_timer is not None:
            root.after_cancel(_typing_timer)
        _typing_timer = root.after(700, lambda: parse_and_update_tab2_phrases(force_default=False))

    narr_text_box.bind("<KeyRelease>", on_narr_text_modified)

    # 4. Spodné akcie pre Tab 2
    narr_action_f = ttk.Frame(tab_narration)
    narr_action_f.pack(fill="x", padx=10, pady=5)

    narr_gen_btn = ttk.Button(narr_action_f, text="🎙️ Generate Speech (MP3 160 kbps)", style="Accent.TButton")
    narr_gen_btn.pack(side="left", padx=5)

    narr_play_out_btn = ttk.Button(narr_action_f, text="▶ Play Generated MP3", state="disabled")
    narr_play_out_btn.pack(side="left", padx=5)

    narr_save_btn = ttk.Button(narr_action_f, text="💾 Save MP3 As...", state="disabled")
    narr_save_btn.pack(side="left", padx=5)

    narr_cancel_btn = ttk.Button(narr_action_f, text="⏹ Cancel", state="disabled")
    narr_cancel_btn.pack(side="right", padx=5)

    # ==========================================================================
    # SPOLOČNÝ PANEL: PROGRESS & LOGS
    # ==========================================================================
    bottom_frame = ttk.LabelFrame(root, text=" System Status & Log ", padding=6)
    bottom_frame.pack(fill="both", expand=True, padx=10, pady=4)

    prog_var = ttk.Progressbar(bottom_frame, mode="determinate")
    prog_var.pack(fill="x", padx=5, pady=2)

    status_row = ttk.Frame(bottom_frame)
    status_row.pack(fill="x", padx=5)
    
    status_lbl = ttk.Label(status_row, text="Ready.", font=('Segoe UI', 9))
    status_lbl.pack(side="left")

    vram_lbl = ttk.Label(status_row, text=get_vram_status(), font=('Segoe UI', 8, 'italic'))
    vram_lbl.pack(side="right")

    log_txt = tk.Text(bottom_frame, height=5, bg="#1e1e1e", fg="#d4d4d4", font=("Consolas", 8))
    log_txt.pack(fill="both", expand=True, padx=5, pady=3)

    def log(msg):
        log_queue.put(msg)

    def process_logs():
        while not log_queue.empty():
            msg = log_queue.get_nowait()
            log_txt.insert(tk.END, msg + "\n")
            log_txt.see(tk.END)
            status_lbl.config(text=msg)
            vram_lbl.config(text=get_vram_status())
        root.after(120, process_logs)

    def init_project_state(video_path):
        folders, proj_file = create_project_structure(video_path)
        dub_state["video_path"] = video_path
        dub_state["folders"] = folders
        dub_state["proj_file"] = proj_file
        dub_state["data"] = load_or_init_project(video_path, proj_file)
        refresh_speaker_table()
        refresh_subtitle_table()
        log(f"[Project] Directory: {folders['root']}")

    # ==========================================================================
    # LOGIKA PRE TAB 1 (VIDEO DUBBING & DÁVKOVÝ DABING)
    # ==========================================================================
    def find_matching_subtitle(video_path, target_lang="cs"):
        """
        Searches for a subtitle file (.srt/.ass) associated with the given video:
        1. With target language suffix: e.g. video.en.srt / video.cs.srt / video.ja.srt
        2. Exact match: video.srt / video.ass
        3. Any language suffix: video.*.srt / video.*.ass (e.g. video.en.srt)
        """
        folder = os.path.dirname(os.path.abspath(video_path))
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        sub_exts = ('.srt', '.ass')
        
        # 1. Presná zhoda s cieľovým jazykom
        lang_code = target_lang.split()[0].lower() if target_lang else "cs"
        for ext in sub_exts:
            p = os.path.join(folder, f"{base_name}.{lang_code}{ext}")
            if os.path.exists(p):
                return p
                
        # 2. Presná zhoda s kmeňom videa
        for ext in sub_exts:
            p = os.path.join(folder, f"{base_name}{ext}")
            if os.path.exists(p):
                return p

        # 3. Zhoda s bodkou a akýmkoľvek jazykom (napr. video.en.srt)
        prefix = base_name.lower() + "."
        candidates = []
        try:
            for f in os.listdir(folder):
                if f.lower().startswith(prefix):
                    for ext in sub_exts:
                        if f.lower().endswith(ext):
                            candidates.append(os.path.join(folder, f))
        except Exception:
            pass
            
        if candidates:
            return candidates[0]
            
        return None

    def scan_directory_for_video_pairs(folder_path, target_lang="cs"):
        """
        Finds all video files in a folder that have an existing matching .srt/.ass subtitle file.
        Ignores already dubbed videos (_dubbed_) and subdirectories.
        """
        pairs = []
        if not os.path.exists(folder_path):
            return pairs
        valid_vid_exts = ('.mp4', '.mkv', '.avi', '.mov', '.webm', '.wmv', '.flv')
        for f in sorted(os.listdir(folder_path)):
            full_vid_path = os.path.join(folder_path, f)
            if not os.path.isfile(full_vid_path):
                continue
            if "_dubbed_" in f.lower():
                continue
            if not f.lower().endswith(valid_vid_exts):
                continue
            sub_path = find_matching_subtitle(full_vid_path, target_lang=target_lang)
            if sub_path:
                pairs.append((full_vid_path, sub_path))
        return pairs

    def process_analyze(v_path, srt_p=None, is_batch=False):
        try:
            if not os.path.exists(v_path):
                if not is_batch:
                    messagebox.showerror("Error", "Video file does not exist!")
                log(f"[Error] Video file does not exist: {v_path}")
                return False
            
            init_project_state(v_path)
            if not is_batch:
                dub_state["cancel_requested"] = False
                analyze_btn.config(state="disabled")
                dub_btn.config(state="disabled")
                batch_dub_btn.config(state="disabled")
                cancel_btn.config(state="normal")

            prog_var.config(value=10)

            if dub_state["cancel_requested"]:
                return False

            extracted_audio = os.path.join(dub_state["folders"]["audio"], "extracted_audio.wav")
            if not os.path.exists(extracted_audio):
                log("[Audio] Extracting audio track...")
                extract_audio_from_video(v_path, extracted_audio)
            dub_state["data"]["extracted_audio"] = extracted_audio

            if dub_state["cancel_requested"]:
                return False

            use_sep = sep_var.get()
            dub_state["data"]["use_separation"] = use_sep
            speech_audio = extracted_audio

            if use_sep:
                vocals_path = os.path.join(dub_state["folders"]["separation"], "vocals.wav")
                bgm_path = os.path.join(dub_state["folders"]["separation"], "bgm.wav")
                if not (os.path.exists(vocals_path) and os.path.exists(bgm_path)):
                    prog_var.config(value=25)
                    separate_sources(extracted_audio, dub_state["folders"]["separation"], log_cb=log)
                speech_audio = vocals_path
                dub_state["data"]["vocals_audio"] = vocals_path
                dub_state["data"]["bgm_audio"] = bgm_path

            if dub_state["cancel_requested"]:
                return False

            prog_var.config(value=50)
            m = mode_var.get()
            dub_state["data"]["mode"] = m
            
            if m == "EXISTING_SRT" or is_batch:
                actual_srt = srt_p if srt_p else sub_entry.get().strip()
                if not (actual_srt and os.path.exists(actual_srt)):
                    if not is_batch:
                        ans = messagebox.askyesno(
                            "Subtitles Missing",
                            "No subtitle (.srt) file was specified.\n\n"
                            "Would you like to automatically transcribe dialogue from the video using Whisper now?"
                        )
                        if ans:
                            log("[STT] Auto-transcribing dialogue with Whisper...")
                            from faster_whisper import WhisperModel
                            whisper = WhisperModel("medium", device=DEVICE, compute_type="float16" if "cuda" in DEVICE else "int8")
                            segments_raw, _ = whisper.transcribe(speech_audio, vad_filter=True)
                            segments = []
                            srt_lines = []
                            for idx_s, s in enumerate(segments_raw, 1):
                                if s.text.strip():
                                    s_start = round(s.start, 3)
                                    s_end = round(s.end, 3)
                                    srt_lines.append(f"{idx_s}\n{format_timestamp_srt(s_start)} --> {format_timestamp_srt(s_end)}\n{s.text.strip()}\n")
                                    segments.append({
                                        "id": idx_s,
                                        "start": s_start,
                                        "end": s_end,
                                        "text": s.text.strip(),
                                        "speaker": "Speaker_01",
                                        "assigned_voice": ""
                                    })
                            del whisper
                            clear_vram()
                            base_name = os.path.splitext(os.path.basename(v_path))[0]
                            auto_srt = os.path.join(dub_state["folders"]["subtitles"], f"{base_name}_whisper.srt")
                            with open(auto_srt, "w", encoding="utf-8") as f:
                                f.write("\n".join(srt_lines) + "\n")
                            sub_entry.delete(0, tk.END)
                            sub_entry.insert(0, auto_srt)
                            dub_state["data"]["sub_path"] = auto_srt
                            dub_state["data"]["subtitles"] = segments
                        else:
                            log(f"[Error] Subtitle file not found: {actual_srt}")
                            return False
                    else:
                        log(f"[Error] Subtitle file not found: {actual_srt}")
                        return False
                else:
                    log(f"[Subtitles] Loading subtitles: {os.path.basename(actual_srt)}...")
                    segments = parse_srt_file(actual_srt)
            else:
                log("[STT] Transcribing speech to text with Whisper...")
                from faster_whisper import WhisperModel
                whisper = WhisperModel("medium", device=DEVICE, compute_type="float16" if "cuda" in DEVICE else "int8")
                segments_raw, _ = whisper.transcribe(speech_audio, vad_filter=True)
                segments = []
                for s in segments_raw:
                    if s.text.strip():
                        segments.append({
                            "id": len(segments) + 1,
                            "start": round(s.start, 3),
                            "end": round(s.end, 3),
                            "text": s.text.strip(),
                            "speaker": "Speaker_01",
                            "assigned_voice": ""
                        })
                del whisper
                clear_vram()

            if dub_state["cancel_requested"]:
                return False

            has_diarization = dub_state["data"].get("steps_completed", {}).get("diarization", False)
            existing_speakers = dub_state["data"].get("speakers", {})
            force_re = force_reanalyze_var.get()

            if has_diarization and existing_speakers and not force_re:
                log(f"[Analysis] Using existing character analysis and voice assignments from project ({len(existing_speakers)} characters). Diarization skipped.")
                speakers = existing_speakers
                segments = sync_subtitles_with_disk(dub_state["data"].get("subtitles", []), segments, speakers)
                prog_var.config(value=80)
            else:
                max_spks = int(max_spk_spin.get())
                cur_prof = pitch_profile_var.get()
                segments, speakers = diarize_segments_direct(speech_audio, segments, max_speakers=max_spks, log_cb=log, profile=cur_prof)
                prog_var.config(value=80)

                if dub_state["cancel_requested"]:
                    return False

                log("[Preview] Preparing clean voice samples...")
                generate_speaker_previews(speech_audio, speakers, dub_state["folders"]["speakers"])

                # Assign each subtitle line its character's default voice (Supertonic as default)
                for seg in segments:
                    spk = seg.get("speaker", "Speaker_01")
                    spk_data = speakers.get(spk, {})
                    st_v = spk_data.get("supertonic_voice", "M1")
                    seg["assigned_voice"] = f"⚡ Supertonic [{st_v}]"

                dub_state["data"]["steps_completed"]["diarization"] = True

            dub_state["data"]["subtitles"] = segments
            dub_state["data"]["speakers"] = speakers

            save_project(dub_state["data"], dub_state["proj_file"])
            root.after(0, refresh_speaker_table)
            root.after(0, refresh_subtitle_table)
            prog_var.config(value=100)
            log(f"[Success] Analysis ready. Characters: {len(speakers)}, Subtitle lines: {len(segments)}.")
            return True

        except Exception as e:
            log(f"[ANALYSIS ERROR] {str(e)}")
            if not is_batch:
                messagebox.showerror("Error", f"Error during character analysis:\n{str(e)}")
            return False
        finally:
            if not is_batch:
                analyze_btn.config(state="normal")
                dub_btn.config(state="normal")
                batch_dub_btn.config(state="normal")
                cancel_btn.config(state="disabled")

    def process_dub(is_batch=False):
        try:
            if not dub_state["data"].get("speakers"):
                if not is_batch:
                    messagebox.showwarning("Notice", "Please run Step 1 (Analysis) first.")
                return False

            if not is_batch:
                analyze_btn.config(state="disabled")
                dub_btn.config(state="disabled")
                batch_dub_btn.config(state="disabled")
                cancel_btn.config(state="normal")
                dub_state["cancel_requested"] = False

            tgt_lang = lang_combo.get().strip()
            dub_state["data"]["target_lang"] = tgt_lang

            # Automatically check and sync with changes in the .srt file on disk
            actual_srt = sub_entry.get().strip() or dub_state["data"].get("sub_path")
            if actual_srt and os.path.exists(actual_srt):
                disk_segments = parse_srt_file(actual_srt)
                dub_state["data"]["subtitles"] = sync_subtitles_with_disk(
                    dub_state["data"].get("subtitles", []),
                    disk_segments,
                    dub_state["data"].get("speakers", {})
                )
                save_project(dub_state["data"], dub_state["proj_file"])
                root.after(0, refresh_subtitle_table)

            raw_subtitles = dub_state["data"].get("subtitles", [])
            spks_map = dub_state["data"].get("speakers", {})
            subtitles = merge_continuous_subtitles(raw_subtitles, max_gap=0.40, speakers=spks_map)

            # Clean legacy index-based audio chunks (hash-based chunks are preserved)
            gen_dir = dub_state["folders"]["generated"]
            if os.path.exists(gen_dir):
                _valid_keys = set()
                for _s in subtitles:
                    _valid_keys.add(_s.get("audio_chunk_key", ""))
                for f in os.listdir(gen_dir):
                    m = re.match(r'^(?:raw|sub)_(\d{5})\.(?:wav|txt)$', f)
                    if m:
                        sub_num = int(m.group(1))
                        if sub_num > len(subtitles):
                            try:
                                os.remove(os.path.join(gen_dir, f))
                            except Exception:
                                pass

            eng_choice = engine_var.get()
            dub_engine = "supertonic" if "Supertonic" in eng_choice else ("xtts" if "Klonovanie" in eng_choice else "edge")

            dub_state["tts_engine"] = DynamicDubbingEngine(target_lang=tgt_lang, engine_type=dub_engine, state=dub_state)

            total_subs = len(subtitles)
            generated_items = []
            cached_lines_count = 0
            new_lines_count = 0

            for idx, item in enumerate(subtitles):
                if dub_state["cancel_requested"]:
                    log("[Dubbing] Dubbing cancelled by user.")
                    return False

                sub_id = item["id"]
                spk = item["speaker"]
                text = " ".join(item["text"].replace('\r', ' ').replace('\n', ' ').split())
                
                allocated_time = max(0.6, item["end"] - item["start"])
                for nxt_idx in range(idx + 1, min(idx + 5, total_subs)):
                    nxt = subtitles[nxt_idx]
                    if nxt["speaker"] == spk:
                        if nxt["start"] > item["start"]:
                            allocated_time = min(allocated_time, max(0.5, nxt["start"] - item["start"] - 0.04))
                        break

                # Resolve TTS engine and voice for this specific subtitle line
                assigned_v_str = item.get("assigned_voice", "")
                act_engine, act_voice, act_ref_wav = resolve_voice_for_subtitle(assigned_v_str, spk, dub_state, lang_code=tgt_lang)

                # Stable content-hash chunk naming (prevents line deletions from causing cascade re-generation)
                import hashlib as _hashlib
                _voice_key = act_voice or os.path.basename(act_ref_wav or "")
                cache_sig = f"[{act_engine}:{_voice_key}]\n{text}".strip()
                _chunk_hash = _hashlib.md5(cache_sig.encode("utf-8")).hexdigest()[:12]
                # Backward compatibility: reuse legacy sub_{sub_id} audio chunk if identical
                _old_key = item.get("audio_chunk_key", f"sub_{sub_id:05d}")
                raw_wav = os.path.join(dub_state["folders"]["generated"], f"raw_{_chunk_hash}.wav")
                final_gen_wav = os.path.join(dub_state["folders"]["generated"], f"sub_{_chunk_hash}.wav")
                sub_txt = os.path.join(dub_state["folders"]["generated"], f"sub_{_chunk_hash}.txt")
                # Backward compatibility: check legacy sub_{sub_id} files
                _old_wav = os.path.join(dub_state["folders"]["generated"], f"{_old_key}.wav")
                _old_txt = os.path.join(dub_state["folders"]["generated"], f"{_old_key}.txt")
                if not os.path.exists(final_gen_wav) and os.path.exists(_old_wav):
                    try:
                        import shutil as _shutil
                        _shutil.copy2(_old_wav, final_gen_wav)
                        if os.path.exists(_old_txt):
                            _shutil.copy2(_old_txt, sub_txt)
                    except Exception:
                        pass
                cache_valid = False

                if os.path.exists(final_gen_wav) and os.path.getsize(final_gen_wav) > 100 and os.path.exists(sub_txt):
                    try:
                        with open(sub_txt, "r", encoding="utf-8-sig") as f:
                            if f.read().strip() == cache_sig:
                                cache_valid = True
                    except Exception:
                        cache_valid = False

                if cache_valid:
                    cached_lines_count += 1
                else:
                    new_lines_count += 1
                    log(f"[Synthesis ({act_engine})] ({idx+1}/{total_subs}) [{spk} | {act_voice or os.path.basename(act_ref_wav or '')}] '{text[:35]}...'")
                    
                    _, persona_eq = dub_state["tts_engine"].generate_speech(
                        text=text,
                        spk_id=spk,
                        ref_voice_wav=act_ref_wav if act_engine == "xtts" else None,
                        output_wav_path=raw_wav,
                        target_duration=allocated_time,
                        log_cb=log,
                        explicit_voice=act_voice,
                        explicit_engine=act_engine
                    )
                    fit_audio_strictly_to_slot(raw_wav, final_gen_wav, allocated_time, eq_persona=persona_eq)

                    with open(sub_txt, "w", encoding="utf-8") as f:
                        f.write(cache_sig + "\n")

                item["final_audio"] = final_gen_wav
                item["audio_chunk_key"] = f"sub_{_chunk_hash}"
                item["actual_start"] = item["start"]
                generated_items.append(item)
                prog_var.config(value=int((idx + 1) / max(1, total_subs) * 75))

            log(f"[Synthesis] Done: {cached_lines_count} lines loaded from cache, {new_lines_count} newly synthesized.")
            dub_state["tts_engine"].unload_model()

            if dub_state["cancel_requested"]:
                log("[Dubbing] Dubbing was cancelled by user.")
                return False

            v_path = dub_state["video_path"]
            total_video_dur = get_audio_duration(v_path)
            
            v_rms = 0.075
            vocals_file = dub_state["data"].get("vocals_audio")
            if vocals_file and os.path.exists(vocals_file):
                v_data, _ = sf.read(vocals_file)
                if len(v_data.shape) > 1:
                    v_data = np.mean(v_data, axis=1)
                active_v = v_data[np.abs(v_data) > 0.015]
                if len(active_v) > 0:
                    v_rms = float(np.clip(np.sqrt(np.mean(active_v**2)), 0.050, 0.085))

            log(f"[Mixer] Running Anti-Collision Manager (Matched RMS: {v_rms:.3f})...")
            dialogue_track = assemble_dialogue_track_anticollision(generated_items, total_video_dur, target_rms=v_rms)

            mixed_master_wav = os.path.join(dub_state["folders"]["output"], "final_master_audio.wav")
            orig_audio = dub_state["data"]["extracted_audio"]
            bgm_audio = dub_state["data"].get("bgm_audio", orig_audio)

            d_vol = float(dlg_vol_var.get())
            b_vol = float(bg_vol_var.get())
            duck_val = float(duck_var.get())

            log(f"[Mixer] Applying Master Pass-through (Dubbing: {int(d_vol*100)}%, Background: {int(b_vol*100)}%, Ducking: {int(duck_val*100)}%)...")
            create_adaptive_master_mix(orig_audio, bgm_audio, dialogue_track, generated_items, mixed_master_wav, dlg_vol=d_vol, bg_vol=b_vol, ducking=duck_val)

            ts_str = time.strftime("%Y%m%d_%H%M%S")
            base_vname = os.path.splitext(os.path.basename(v_path))[0]
            out_video = os.path.join(dub_state["folders"]["output"], f"{base_vname}_dubbed_{tgt_lang}_{ts_str}.mkv")

            mix_and_export_video(v_path, mixed_master_wav, out_video, target_lang=tgt_lang)

            prog_var.config(value=100)
            dub_state["data"]["steps_completed"]["final_export"] = True
            save_project(dub_state["data"], dub_state["proj_file"])

            log(f"[Done] Video created: {out_video}")
            if not is_batch:
                messagebox.showinfo("Done", f"Dubbing completed!\nOutput file:\n{out_video}")
            return True

        except Exception as e:
            log(f"[DUBBING ERROR] {str(e)}")
            if not is_batch:
                messagebox.showerror("Error", f"Error during dubbing:\n{str(e)}")
            return False
        finally:
            if not is_batch:
                analyze_btn.config(state="normal")
                dub_btn.config(state="normal")
                batch_dub_btn.config(state="normal")
                cancel_btn.config(state="disabled")

    def task_transcribe_whisper():
        v = video_entry.get().strip()
        if not (v and os.path.exists(v)):
            messagebox.showwarning("Notice", "Please select a valid video/audio file first.")
            return

        init_project_state(v)
        dub_state["cancel_requested"] = False
        analyze_btn.config(state="disabled")
        dub_btn.config(state="disabled")
        batch_dub_btn.config(state="disabled")
        if 'whisper_sub_btn' in locals() or 'whisper_sub_btn' in globals():
            whisper_sub_btn.config(state="disabled")
        cancel_btn.config(state="normal")

        try:
            log("[Whisper] Starting automated subtitle transcription...")
            prog_var.config(value=10)

            extracted_audio = os.path.join(dub_state["folders"]["audio"], "extracted_audio.wav")
            if not os.path.exists(extracted_audio):
                log("[Audio] Extracting audio track from video...")
                extract_audio_from_video(v, extracted_audio)
            dub_state["data"]["extracted_audio"] = extracted_audio

            if dub_state["cancel_requested"]:
                log("[Whisper] Cancelled by user.")
                return

            vocals_path = os.path.join(dub_state["folders"]["separation"], "vocals.wav")
            speech_audio = vocals_path if os.path.exists(vocals_path) else extracted_audio

            prog_var.config(value=30)
            log(f"[Whisper] Transcribing speech from '{os.path.basename(speech_audio)}' (Medium model)...")

            from faster_whisper import WhisperModel
            whisper = WhisperModel("medium", device=DEVICE, compute_type="float16" if "cuda" in DEVICE else "int8")
            segments_raw, info = whisper.transcribe(speech_audio, vad_filter=True)

            srt_lines = []
            parsed_segments = []
            idx = 1
            for s in segments_raw:
                if dub_state["cancel_requested"]:
                    break
                txt = s.text.strip()
                if txt:
                    s_start = round(s.start, 3)
                    s_end = round(s.end, 3)
                    srt_lines.append(f"{idx}\n{format_timestamp_srt(s_start)} --> {format_timestamp_srt(s_end)}\n{txt}\n")
                    parsed_segments.append({
                        "id": idx,
                        "start": s_start,
                        "end": s_end,
                        "text": txt,
                        "speaker": "Speaker_01",
                        "assigned_voice": ""
                    })
                    idx += 1

            del whisper
            clear_vram()

            if dub_state["cancel_requested"]:
                log("[Whisper] Cancelled by user.")
                return

            if not parsed_segments:
                log("[Whisper] No speech detected in the audio track.")
                messagebox.showinfo("Whisper", "No speech detected in the audio track.")
                return

            base_name = os.path.splitext(os.path.basename(v))[0]
            srt_out_path = os.path.join(dub_state["folders"]["subtitles"], f"{base_name}_whisper.srt")
            with open(srt_out_path, "w", encoding="utf-8") as f:
                f.write("\n".join(srt_lines) + "\n")

            sub_entry.delete(0, tk.END)
            sub_entry.insert(0, srt_out_path)
            mode_var.set("EXISTING_SRT")
            dub_state["data"]["sub_path"] = srt_out_path
            dub_state["data"]["subtitles"] = parsed_segments
            save_project(dub_state["data"], dub_state["proj_file"])

            root.after(0, refresh_subtitle_table)
            prog_var.config(value=100)
            log(f"[Whisper] Generated {len(parsed_segments)} subtitle lines -> {srt_out_path}")
            messagebox.showinfo("Whisper Subtitles", f"Successfully generated {len(parsed_segments)} subtitle lines!\n\nSaved to:\n{srt_out_path}")

        except Exception as e:
            log(f"[Whisper Error] {e}")
            messagebox.showerror("Whisper Error", f"Failed to transcribe subtitles:\n{e}")
        finally:
            analyze_btn.config(state="normal")
            dub_btn.config(state="normal")
            batch_dub_btn.config(state="normal")
            if 'whisper_sub_btn' in locals() or 'whisper_sub_btn' in globals():
                whisper_sub_btn.config(state="normal")
            cancel_btn.config(state="disabled")

    def task_analyze():
        v = video_entry.get().strip()
        s = sub_entry.get().strip()
        process_analyze(v, srt_p=s, is_batch=False)

    def task_dub():
        process_dub(is_batch=False)

    def task_batch_dub():
        try:
            v_entry_val = video_entry.get().strip()
            if v_entry_val and os.path.exists(v_entry_val):
                folder_to_scan = os.path.dirname(os.path.abspath(v_entry_val))
            else:
                folder_to_scan = SCRIPT_DIR

            log(f"[Batch] Scanning folder: {folder_to_scan}")
            tgt_lang = lang_combo.get().strip()
            pairs = scan_directory_for_video_pairs(folder_to_scan, target_lang=tgt_lang)

            if not pairs and folder_to_scan != SCRIPT_DIR:
                log(f"[Batch] No videos in {folder_to_scan}, checking script directory: {SCRIPT_DIR}")
                folder_to_scan = SCRIPT_DIR
                pairs = scan_directory_for_video_pairs(folder_to_scan, target_lang=tgt_lang)

            if not pairs:
                log(f"[Batch] No videos with matching subtitles (.srt/.ass) found in '{folder_to_scan}'.")
                messagebox.showinfo(
                    "Batch Dubbing",
                    f"In folder:\n{folder_to_scan}\n\nNo video files found with matching subtitle files (.srt/.ass with identical name or language suffix like video.en.srt)."
                )
                return

            analyze_btn.config(state="disabled")
            dub_btn.config(state="disabled")
            batch_dub_btn.config(state="disabled")
            cancel_btn.config(state="normal")
            dub_state["cancel_requested"] = False

            total_videos = len(pairs)
            log(f"[Batch] Found {total_videos} videos with subtitles queued for processing.")
            for v_f, s_f in pairs:
                log(f"   -> Video: {os.path.basename(v_f)} | Subtitles: {os.path.basename(s_f)}")

            completed_count = 0
            for b_idx, (vid_file, sub_file) in enumerate(pairs):
                if dub_state["cancel_requested"]:
                    log("[Batch] Batch dubbing was interrupted by user.")
                    break

                # Check: if video is already dubbed and .srt is unchanged, skip
                if not force_reanalyze_var.get() and is_batch_video_up_to_date(vid_file, sub_file, tgt_lang):
                    log(f"[Batch] ({b_idx + 1}/{total_videos}) Video {os.path.basename(vid_file)} already up to date (.srt unchanged). Skipping.")
                    completed_count += 1
                    continue

                log("[Batch] ============================================================")
                log(f"[Batch] ({b_idx + 1}/{total_videos}) STARTING: {os.path.basename(vid_file)}")
                log(f"[Batch] Subtitles: {os.path.basename(sub_file)}")
                log(f"[Dávka] ============================================================")
                
                # Aktualizácia polí v GUI
                root.after(0, lambda v=vid_file, s=sub_file: [
                    video_entry.delete(0, tk.END),
                    video_entry.insert(0, v),
                    sub_entry.delete(0, tk.END),
                    sub_entry.insert(0, s)
                ])

                # Step 1: Character Analysis & Audio Separation
                ok_analyze = process_analyze(vid_file, srt_p=sub_file, is_batch=True)
                if not ok_analyze or dub_state["cancel_requested"]:
                    log(f"[Batch] Analysis cancelled or failed for {os.path.basename(vid_file)}.")
                    break

                # Step 2: Speech Synthesis & Video Export
                ok_dub = process_dub(is_batch=True)
                if not ok_dub or dub_state["cancel_requested"]:
                    log(f"[Batch] Dubbing cancelled or failed for {os.path.basename(vid_file)}.")
                    break

                completed_count += 1
                log(f"[Batch] ({b_idx + 1}/{total_videos}) COMPLETED: {os.path.basename(vid_file)}")

            if dub_state["cancel_requested"]:
                log(f"[Batch] Interrupted on user request. Completed: {completed_count} of {total_videos} videos.")
                messagebox.showinfo("Cancelled", f"Batch processing was cancelled by user.\nCompleted {completed_count} of {total_videos} videos.")
            else:
                log(f"[Batch] Done! All {completed_count} videos successfully dubbed and exported.")
                messagebox.showinfo("Batch Dubbing Done", f"Successfully completed batch dubbing for all {completed_count} videos!")

        except Exception as e:
            log(f"[BATCH ERROR] {str(e)}")
            messagebox.showerror("Batch Error", f"Error during batch processing:\n{str(e)}")
        finally:
            analyze_btn.config(state="normal")
            dub_btn.config(state="normal")
            batch_dub_btn.config(state="normal")
            cancel_btn.config(state="disabled")

    def handle_export_work():
        if not dub_state["data"].get("subtitles"):
            messagebox.showinfo("Info", "Please run character analysis first.")
            return
        fn = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text", "*.txt")])
        if fn:
            export_working_file(dub_state["data"]["subtitles"], fn)
            log(f"[Mode B] Exported to: {fn}")
            messagebox.showinfo("Done", f"File saved successfully:\n{fn}")

    def handle_import_work():
        fn = filedialog.askopenfilename(filetypes=[("Text", "*.txt")])
        if fn:
            items = import_working_file(fn)
            if items:
                dub_state["data"]["subtitles"] = items
                save_project(dub_state["data"], dub_state["proj_file"])
                refresh_subtitle_table()
                log(f"[Mode B] Loaded {len(items)} subtitle lines.")
                messagebox.showinfo("Done", f"Loaded {len(items)} edited subtitle lines.")

    # ==========================================================================
    # LOGIKA PRE TAB 2 (TEXT NARRATION / MP3 S VIACHLASOVÝMI FRÁZAMI)
    # ==========================================================================
    def task_generate_narration():
        try:
            speed = float(speed_combo.get())
            lang_code = narr_lang_combo.get().split()[0]

            if not narr_phrases:
                parse_and_update_tab2_phrases(force_default=False)

            if not narr_phrases:
                messagebox.showerror("Error", "Input text is empty!")
                return

            narr_gen_btn.config(state="disabled")
            narr_cancel_btn.config(state="normal")
            narr_play_out_btn.config(state="disabled")
            narr_save_btn.config(state="disabled")
            narr_state["cancel_requested"] = False
            prog_var.config(value=5)

            log(f"[Narration] Starting multi-voice narration ({len(narr_phrases)} sentences, Language: {lang_code})...")

            temp_narr_dir = os.path.join(SCRIPT_DIR, "output_narration")
            os.makedirs(temp_narr_dir, exist_ok=True)

            tot = len(narr_phrases)
            audio_pieces = []
            OUTPUT_SR = 44100
            silence_sentence = np.zeros(int(OUTPUT_SR * 0.05), dtype=np.float32)

            # Inicializujeme generátor
            active_engine_obj = DynamicDubbingEngine(target_lang=lang_code, engine_type="supertonic")

            for idx, item in enumerate(narr_phrases):
                if narr_state["cancel_requested"]:
                    break

                chunk_text = item["text"].strip()
                v_info = item["voice_info"]
                v_type = v_info.get("type", "supertonic")
                v_id = v_info.get("voice_id")
                ref_wav = v_info.get("path") if v_type == "clone" else None

                # Referenčný prepis pre klonovanie
                if ref_wav and os.path.exists(ref_wav):
                    get_or_create_reference_transcription(ref_wav, log_cb=log)

                tmp_piece = os.path.join(temp_narr_dir, f"chunk_{idx:04d}.wav")
                tmp_txt = os.path.join(temp_narr_dir, f"chunk_{idx:04d}.txt")

                # Cache signature: voice type + voice ID/path + speed + phrase text
                voice_sig = f"[{v_type}:{v_id or os.path.basename(ref_wav or '')}:{speed}]"
                expected_cache = f"{voice_sig}\n{chunk_text}".strip()

                cache_valid = False
                if os.path.exists(tmp_piece) and os.path.getsize(tmp_piece) > 1000 and os.path.exists(tmp_txt):
                    try:
                        with open(tmp_txt, "r", encoding="utf-8-sig") as f:
                            if f.read().strip() == expected_cache:
                                cache_valid = True
                    except Exception:
                        cache_valid = False

                if cache_valid:
                    log(f"[Narration] ({idx+1}/{tot}) Using cached chunk ({item['voice_label'][:20]}): '{chunk_text[:35]}...'")
                else:
                    log(f"[Narration] ({idx+1}/{tot}) Synthesizing [{item['voice_label'][:25]}]: '{chunk_text[:35]}...'")
                    active_engine_obj.generate_speech(
                        text=chunk_text,
                        spk_id="Narrator",
                        ref_voice_wav=ref_wav,
                        output_wav_path=tmp_piece,
                        target_duration=None,
                        log_cb=log,
                        explicit_voice=v_id,
                        explicit_engine=v_type
                    )
                    with open(tmp_txt, "w", encoding="utf-8") as f:
                        f.write(expected_cache + "\n")

                if os.path.exists(tmp_piece) and os.path.getsize(tmp_piece) > 1000:
                    data, piece_sr = sf.read(tmp_piece)
                    if len(data.shape) > 1:
                        data = np.mean(data, axis=1)
                    data = trim_audio_silence(data, piece_sr, db_thresh=30)
                    
                    if piece_sr != OUTPUT_SR:
                        import scipy.signal
                        num_samples = int(round(len(data) * OUTPUT_SR / piece_sr))
                        data = scipy.signal.resample(data, num_samples)

                    audio_pieces.append(data.astype(np.float32))
                    audio_pieces.append(silence_sentence)

                prog_var.config(value=int(10 + (idx + 1) / max(1, tot) * 80))

            active_engine_obj.unload_model()

            if narr_state["cancel_requested"]:
                log("[Narration] Generation was cancelled.")
                return

            if not audio_pieces:
                raise RuntimeError("Nepodarilo sa vytvoriť zvukovú stopu.")

            combined = np.concatenate(audio_pieces)
            max_v = np.max(np.abs(combined))
            if max_v > 0:
                combined = (combined / max_v) * 0.92

            master_wav = os.path.join(temp_narr_dir, "master_narration.wav")
            sf.write(master_wav, combined, OUTPUT_SR)

            ts = time.strftime("%Y%m%d_%H%M%S")
            final_mp3 = os.path.join(temp_narr_dir, f"narration_{lang_code}_{ts}.mp3")
            convert_to_mp3_160k(master_wav, final_mp3)

            narr_state["last_mp3"] = final_mp3
            prog_var.config(value=100)
            narr_play_out_btn.config(state="normal")
            narr_save_btn.config(state="normal")

            dur = get_audio_duration(final_mp3)
            log(f"[Done] MP3 audio ready ({dur:.1f} s).")
            messagebox.showinfo("Done", f"Narration successfully generated!\n\nDuration: {dur:.1f} s\nFile: {final_mp3}")

        except Exception as e:
            log(f"[ERROR] {str(e)}")
            messagebox.showerror("Error", f"Error during speech generation:\n{str(e)}")
        finally:
            narr_gen_btn.config(state="normal")
            narr_cancel_btn.config(state="disabled")

    def play_narr_output():
        if narr_state["last_mp3"] and os.path.exists(narr_state["last_mp3"]):
            data, sr = sf.read(narr_state["last_mp3"])
            sd.stop()
            sd.play(data, sr)

    def save_narr_output():
        if not (narr_state["last_mp3"] and os.path.exists(narr_state["last_mp3"])):
            return
        dest = filedialog.asksaveasfilename(
            defaultextension=".mp3",
            filetypes=[("MP3 Audio (160 kbps)", "*.mp3")]
        )
        if dest:
            shutil.copy2(narr_state["last_mp3"], dest)
            messagebox.showinfo("Saved", f"MP3 file was saved:\n{dest}")

    # Priradenie tlačidiel
    def reset_project_memory():
        v = video_entry.get().strip()
        if not v:
            messagebox.showinfo("Info", "Zadajte najprv video súbor.")
            return
        proj_dir = os.path.splitext(v)[0] + "_ai_dub"
        proj_f = os.path.join(proj_dir, "project.json")
        if not os.path.exists(proj_f):
            messagebox.showinfo("Info", f"No saved project exists yet for video '{os.path.basename(v)}'.")
            return
        if messagebox.askyesno("Clear Project Cache", f"Are you sure you want to clear saved analysis and settings for video:\n{os.path.basename(v)}?\n\n(On next run, character analysis and voices will be re-analyzed from scratch)"):
            try:
                os.remove(proj_f)
                dub_state["data"] = load_or_init_project(v, proj_f)
                refresh_speaker_table()
                refresh_subtitle_table()
                log(f"[Project] Project cache {os.path.basename(proj_f)} was cleared.")
                messagebox.showinfo("Done", "Project cache was successfully cleared.")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to delete project file: {e}")

    reset_proj_btn.config(command=reset_project_memory)
    analyze_btn.config(command=lambda: threading.Thread(target=task_analyze, daemon=True).start())
    dub_btn.config(command=lambda: threading.Thread(target=task_dub, daemon=True).start())
    batch_dub_btn.config(command=lambda: threading.Thread(target=task_batch_dub, daemon=True).start())
    cancel_btn.config(command=lambda: dub_state.update({"cancel_requested": True}))
    export_work_btn.config(command=handle_export_work)
    import_work_btn.config(command=handle_import_work)

    narr_gen_btn.config(command=lambda: threading.Thread(target=task_generate_narration, daemon=True).start())
    narr_cancel_btn.config(command=lambda: narr_state.update({"cancel_requested": True}))
    narr_play_out_btn.config(command=play_narr_output)
    narr_save_btn.config(command=save_narr_output)

    # Inicializácia roletiek a počiatočného stavu
    update_tab1_voices_dropdown()
    update_spk_voices_dropdown()
    update_narr_voices_dropdown()
    parse_and_update_tab2_phrases(force_default=False)

    root.after(120, process_logs)
    return root

# ==============================================================================
# UKÁŽKA A PREHRÁVANIE SAMPLOV HLASOV
# ==============================================================================

def play_any_voice_sample(voice_label, target_lang="cs", log_cb=print):
    """
    Auditions any selected voice (Supertonic, Edge-TTS, or Custom Clone).
    Uses localized sample sentences and disk caching for instant 0ms playback.
    """
    cat, v_map = get_all_available_voices(target_lang)
    info = v_map.get(voice_label)

    # Fallback matching: handles format differences like "⚡ Supertonic [M1]" vs "⚡ [Supertonic] M1"
    if not info and voice_label:
        v_lbl_lower = voice_label.lower()
        # Substring match in either direction
        for k, v in v_map.items():
            k_lower = k.lower()
            if v_lbl_lower in k_lower or k_lower in v_lbl_lower:
                info = v
                break

    # Supertonic token match (M1..M5, F1..F5)
    if not info and voice_label:
        import re
        st_m = re.search(r'\b([MF][1-5])\b', voice_label, re.IGNORECASE)
        if st_m:
            target_vid = st_m.group(1).upper()
            for k, v in v_map.items():
                if v.get("type") == "supertonic" and v.get("voice_id") == target_vid:
                    info = v
                    break

    # Voice ID / Name match
    if not info and voice_label:
        for k, v in v_map.items():
            vid = str(v.get("voice_id", "")).lower()
            if vid and (vid in voice_label.lower() or voice_label.lower() in vid):
                info = v
                break

    if not info:
        return False, f"Voice '{voice_label}' not found."

    lang_code = target_lang.split()[0].lower() if target_lang else "cs"

    # Localized sample phrases for natural auditioning in project language
    localized_samples = {
        "cs": "Toto je ukázka syntézy hlasu pro dabing.",
        "sk": "Toto je ukážka syntézy hlasu pre dabing.",
        "ja": "これはローカル日本語吹き替えのサンプル音声です。",
        "de": "Dies ist ein Sprachbeispiel für die Synchronisation.",
        "pl": "To jest próbka głosu do dubbingu.",
        "en": "This is a voice sample for video dubbing."
    }
    sample_txt = localized_samples.get(lang_code, localized_samples["en"])

    if info["type"] == "clone":
        v_path = info.get("path")
        if v_path and os.path.exists(v_path):
            data, sr = sf.read(v_path)
            sd.stop()
            sd.play(data, sr)
            return True, f"Playing custom voice: {os.path.basename(v_path)}"
        return False, "Voice file does not exist."

    elif info["type"] == "supertonic":
        v_id = info["voice_id"]
        test_wav = os.path.join(VOICES_DIR, f"preview_st_{v_id}_{lang_code}.wav")
        if not (os.path.exists(test_wav) and os.path.getsize(test_wav) > 1000):
            st_lang = lang_code if lang_code in ["cs", "sk", "ja", "en", "de", "pl"] else "en"
            engine = DynamicDubbingEngine(target_lang=lang_code, engine_type="supertonic")
            engine.ensure_supertonic_loaded(log_cb=log_cb)
            style = engine.supertonic_model.get_voice_style(voice_name=v_id)
            wav, _ = engine.supertonic_model.synthesize(text=sample_txt, voice_style=style, lang=st_lang, total_steps=8, speed=1.0)
            engine.supertonic_model.save_audio(wav, test_wav)
        data, sr = sf.read(test_wav)
        sd.stop()
        sd.play(data, sr)
        return True, f"Playing Supertonic [{v_id}] preview ({lang_code.upper()})."

    elif info["type"] == "edge":
        v_id = info["voice_id"]
        clean_vid = v_id.replace(":", "_").replace("/", "_")
        test_wav = os.path.join(VOICES_DIR, f"preview_edge_{clean_vid}_{lang_code}.wav")
        if not (os.path.exists(test_wav) and os.path.getsize(test_wav) > 1000):
            tmp_mp3 = test_wav + ".mp3"
            comm = edge_tts.Communicate(sample_txt, v_id)
            asyncio.run(comm.save(tmp_mp3))
            cmd = ["ffmpeg", "-y", "-i", tmp_mp3, "-ar", str(SAMPLE_RATE), "-ac", "1", test_wav]
            run_cmd(cmd)
            if os.path.exists(tmp_mp3):
                try:
                    os.remove(tmp_mp3)
                except Exception:
                    pass
        data, sr = sf.read(test_wav)
        sd.stop()
        sd.play(data, sr)
        return True, f"Playing Edge-TTS [{v_id}] preview."

    return False, "Unknown voice type."

if __name__ == "__main__":
    app = create_app()
    app.mainloop()
