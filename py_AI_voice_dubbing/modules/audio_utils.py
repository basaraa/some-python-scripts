from modules.config import *

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
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception:
            pass

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
    cmd = [
        sys.executable, "-m", "demucs.separate",
        "-n", "htdemucs",
        "--two-stems", "vocals",
        "-d", DEVICE,
        "-j", "1",
        "-o", output_dir,
        audio_path
    ]
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

