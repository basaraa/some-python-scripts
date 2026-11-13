from modules.config import *
from modules.audio_utils import run_cmd, clear_vram, get_audio_duration, trim_audio_silence

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
    import math
    import scipy.signal

    def load_and_resample(path, target_sr):
        data, file_sr = sf.read(path)
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)
        if file_sr != target_sr:
            # Polyphase FIR resampler: O(1) memory, high-fidelity anti-aliasing without giant FFT bloat
            g = math.gcd(int(target_sr), int(file_sr))
            up = int(target_sr) // g
            down = int(file_sr) // g
            data = scipy.signal.resample_poly(data, up, down)
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
    del orig_data, bgm_data, dlg_data, active_dubbing_mask, orig_vocals, passthrough_vocals, processed_bgm, final_mix
    clear_vram()
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

