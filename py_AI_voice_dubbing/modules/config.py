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
SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VOICES_DIR = os.path.join(SCRIPT_DIR, "voices")
SPEAKER_PROFILES_DIR = os.path.join(SCRIPT_DIR, "speaker_profiles")
PROFILES_META_FILE = os.path.join(SPEAKER_PROFILES_DIR, "profiles.json")
for _d in (VOICES_DIR, SPEAKER_PROFILES_DIR):
    try:
        os.makedirs(_d, exist_ok=True)
    except PermissionError:
        pass

SUPERTONIC_MALE_VOICES = ["M1", "M2", "M3", "M4", "M5"]
SUPERTONIC_FEMALE_VOICES = ["F1", "F2", "F3", "F4", "F5"]

# Ranked voice assignment order by speaking duration (longest speaking character gets first voice)
SUPERTONIC_MALE_RANKED = ["M5", "M2", "M4", "M3", "M1"]
SUPERTONIC_FEMALE_RANKED = ["F4", "F3", "F2", "F5", "F1"]

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

