from modules.config import *
from modules.audio_utils import (
    run_cmd, trim_audio_silence, estimate_pitch_f0, clear_vram,
    scan_voices_directory, extract_audio_from_video, separate_sources
)

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
                import math
                import scipy.signal
                g = math.gcd(16000, int(sr))
                up = 16000 // g
                down = int(sr) // g
                wav_data = scipy.signal.resample_poly(wav_data, up, down)
            tensor = torch.tensor(wav_data, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                emb = self.encoder.encode_batch(tensor).squeeze().cpu().numpy()
            del tensor
            return emb / (np.linalg.norm(emb) + 1e-8)
        elif self.backend == "resemblyzer":
            from resemblyzer import preprocess_wav
            wav_pre = preprocess_wav(wav_data, source_sr=sr)
            emb = self.encoder.embed_utterance(wav_pre)
            return emb / (np.linalg.norm(emb) + 1e-8)
        else:
            return np.zeros(192, dtype=np.float32)

def _safe_profile_path(filename):
    """Resolves a filename strictly inside SPEAKER_PROFILES_DIR (path traversal guard)."""
    base_dir = os.path.abspath(SPEAKER_PROFILES_DIR)
    target = os.path.abspath(os.path.join(base_dir, os.path.basename(filename)))
    if not (target == base_dir or target.startswith(base_dir + os.sep)):
        raise ValueError(f"Unsafe profile path: {filename}")
    return target

def _infer_voice_from_filename(name):
    """Infers a default Supertonic voice label from a filename suffix like '_M4' or '_F2'."""
    m = re.search(r'(?:^|[_\-\s])([MF][1-5])(?:$|[_\-\s])', name, re.IGNORECASE)
    if m:
        vid = m.group(1).upper()
        return f"⚡ Supertonic [{vid}]"
    return "⚡ Supertonic [M1]"

def load_speaker_profiles_meta():
    """Loads speaker_profiles/profiles.json and auto-discovers any new audio files in speaker_profiles/."""
    try:
        os.makedirs(SPEAKER_PROFILES_DIR, exist_ok=True)
    except PermissionError:
        if not os.path.isdir(SPEAKER_PROFILES_DIR):
            return {}
    meta = {}
    if os.path.exists(PROFILES_META_FILE):
        try:
            with open(PROFILES_META_FILE, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                if isinstance(loaded, dict):
                    meta = loaded
        except Exception:
            meta = {}

    valid_exts = ('.wav', '.mp3', '.flac', '.ogg', '.m4a')
    found_files = {}
    for fname in sorted(os.listdir(SPEAKER_PROFILES_DIR)):
        if fname.lower().endswith(valid_exts):
            pname = os.path.splitext(fname)[0]
            found_files[pname] = fname

    changed = False
    for pname, fname in found_files.items():
        if pname not in meta:
            meta[pname] = {
                "file": fname,
                "voice": _infer_voice_from_filename(pname),
                "enabled": True
            }
            changed = True
        elif not meta[pname].get("file"):
            meta[pname]["file"] = fname
            changed = True

    # Remove entries whose audio file no longer exists on disk
    missing = [k for k, v in meta.items() if k not in found_files and not os.path.exists(_safe_profile_path(v.get("file", f"{k}.wav")))]
    for k in missing:
        del meta[k]
        changed = True

    if changed:
        save_speaker_profiles_meta(meta)
    return meta

def save_speaker_profiles_meta(meta):
    """Persists speaker_profiles/profiles.json."""
    os.makedirs(SPEAKER_PROFILES_DIR, exist_ok=True)
    with open(PROFILES_META_FILE, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

def _read_audio_mono_float(audio_path):
    """Reads wav/mp3/m4a/flac into mono float32 array and sample rate."""
    try:
        data, sr = sf.read(audio_path)
    except Exception:
        tmp_wav = _safe_profile_path("_tmp_decode.wav")
        run_cmd(["ffmpeg", "-y", "-i", audio_path, "-ac", "1", "-ar", "16000", tmp_wav])
        data, sr = sf.read(tmp_wav)
        if os.path.exists(tmp_wav):
            os.remove(tmp_wav)
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)
    return data.astype(np.float32), int(sr)

def get_or_compute_profile_embedding(profile_name, wav_path, encoder=None, log_cb=print):
    """Loads cached .npy speaker embedding or computes it in 10s chunks using OfflineSpeakerEncoder."""
    npy_path = _safe_profile_path(f"{profile_name}.npy")
    if os.path.exists(npy_path) and os.path.exists(wav_path):
        try:
            if os.path.getmtime(npy_path) >= os.path.getmtime(wav_path):
                emb = np.load(npy_path)
                norm = np.linalg.norm(emb)
                if norm > 1e-8:
                    return (emb / norm).astype(np.float32)
        except Exception:
            pass

    if encoder is None:
        return None

    data, sr = _read_audio_mono_float(wav_path)
    data = trim_audio_silence(data, sr, db_thresh=26)
    if len(data) < int(sr * 0.5):
        return None

    chunk_len = int(sr * 10.0)
    embs = []
    for start in range(0, len(data), chunk_len):
        chunk = data[start:start + chunk_len]
        if len(chunk) < int(sr * 0.5) and len(embs) > 0:
            continue
        embs.append(encoder.embed_utterance(chunk, sr=sr))

    if not embs:
        return None
    avg_emb = np.mean(np.stack(embs, axis=0), axis=0)
    avg_emb = (avg_emb / (np.linalg.norm(avg_emb) + 1e-8)).astype(np.float32)
    try:
        np.save(npy_path, avg_emb)
    except Exception as e:
        log_cb(f"[Profiles] Could not cache embedding for '{profile_name}': {e}")
    return avg_emb

def load_active_speaker_profiles(encoder, log_cb=print):
    """Returns list of active profile dicts: [{'name', 'voice', 'embedding'}]."""
    meta = load_speaker_profiles_meta()
    active = []
    for pname, info in sorted(meta.items()):
        if not info.get("enabled", True):
            continue
        fname = info.get("file", f"{pname}.wav")
        try:
            fpath = _safe_profile_path(fname)
        except ValueError:
            continue
        if not os.path.exists(fpath):
            continue
        emb = get_or_compute_profile_embedding(pname, fpath, encoder=encoder, log_cb=log_cb)
        if emb is not None:
            active.append({
                "name": pname,
                "voice": info.get("voice", "⚡ Supertonic [M1]"),
                "embedding": emb
            })
    return active

def _extract_short_voice_tag(voice_str):
    """Extracts a clean short tag like 'M4' or 'AntoninNeural' from a UI voice string."""
    s = (voice_str or "").strip()
    m = re.search(r'\b([MF][1-5])\b', s)
    if m:
        return m.group(1)
    m_edge = re.search(r'([a-z]{2}-[A-Z]{2}-(\w+?)Neural)', s)
    if m_edge:
        return m_edge.group(2)
    cleaned = re.sub(r'[^A-Za-z0-9]+', '_', s).strip('_')
    return cleaned[:20] if cleaned else "M1"

def create_speaker_profile_from_segments(video_path, selected_segments, target_voice_str, speech_audio_path=None, use_separation=True, log_cb=print):
    """
    Extracts clean speech from selected subtitle segments, concatenates up to 40s,
    saves <video>_<voice>.wav + .npy into speaker_profiles/, and registers it in profiles.json.
    """
    if not selected_segments:
        raise ValueError("No subtitle lines selected.")

    vid_base = re.sub(r'[^A-Za-z0-9_\-]+', '_', os.path.splitext(os.path.basename(video_path or "video"))[0]).strip("_") or "video"
    vid_base = vid_base[:40]
    voice_tag = _extract_short_voice_tag(target_voice_str)
    base_name = f"{vid_base}_{voice_tag}"

    meta = load_speaker_profiles_meta()
    profile_name = base_name
    suffix = 2
    while profile_name in meta or os.path.exists(_safe_profile_path(f"{profile_name}.wav")):
        profile_name = f"{base_name}_{suffix}"
        suffix += 1

    temp_files = []
    try:
        # If no pre-extracted audio exists yet, extract only the needed span via ffmpeg
        if not speech_audio_path or not os.path.exists(speech_audio_path):
            if not video_path or not os.path.exists(video_path):
                raise ValueError("No audio or video source available to extract voice sample.")
            tmp_raw = _safe_profile_path(f"_tmp_raw_{profile_name}.wav")
            temp_files.append(tmp_raw)
            log_cb("[Profiles] Extracting audio from video for voice profile...")
            extract_audio_from_video(video_path, tmp_raw)
            if use_separation:
                tmp_sep_dir = _safe_profile_path(f"_tmp_sep_{profile_name}")
                os.makedirs(tmp_sep_dir, exist_ok=True)
                vocals_p, _ = separate_sources(tmp_raw, tmp_sep_dir, log_cb=log_cb)
                speech_audio_path = vocals_p
            else:
                speech_audio_path = tmp_raw

        audio_data, sr = _read_audio_mono_float(speech_audio_path)
        chunks = []
        gap = np.zeros(int(sr * 0.12), dtype=np.float32)
        total_samples = 0
        max_samples = int(sr * 40.0)

        for seg in sorted(selected_segments, key=lambda x: float(x.get("start", 0))):
            s_idx = max(0, int(float(seg["start"]) * sr))
            e_idx = min(len(audio_data), int(float(seg["end"]) * sr))
            if e_idx <= s_idx:
                continue
            chunk = trim_audio_silence(audio_data[s_idx:e_idx], sr, db_thresh=26)
            if len(chunk) < int(sr * 0.20):
                continue
            if total_samples + len(chunk) > max_samples:
                remaining = max_samples - total_samples
                if remaining > int(sr * 0.5):
                    chunks.append(chunk[:remaining])
                    total_samples += remaining
                break
            chunks.append(chunk)
            chunks.append(gap)
            total_samples += len(chunk) + len(gap)

        if total_samples < int(sr * 1.5):
            raise ValueError("Selected lines contain less than 1.5s of clean speech. Please select more lines.")

        combined = np.concatenate(chunks).astype(np.float32)
        peak = float(np.max(np.abs(combined)))
        if peak > 1e-5:
            combined = combined * (0.90 / peak)

        out_wav = _safe_profile_path(f"{profile_name}.wav")
        sf.write(out_wav, combined, sr)

        encoder = OfflineSpeakerEncoder(log_cb=log_cb)
        emb = get_or_compute_profile_embedding(profile_name, out_wav, encoder=encoder, log_cb=log_cb)
        del encoder
        clear_vram()

        if emb is None:
            raise ValueError("Failed to compute speaker embedding for the selected audio.")

        meta[profile_name] = {
            "file": f"{profile_name}.wav",
            "voice": target_voice_str,
            "enabled": True
        }
        save_speaker_profiles_meta(meta)
        dur_sec = len(combined) / float(sr)
        log_cb(f"[Profiles] Created '{profile_name}' ({dur_sec:.1f}s) -> {target_voice_str}")
        return profile_name, dur_sec
    finally:
        for tf in temp_files:
            if os.path.exists(tf):
                try:
                    os.remove(tf)
                except Exception:
                    pass
        tmp_sep_dir = _safe_profile_path(f"_tmp_sep_{profile_name}")
        if os.path.isdir(tmp_sep_dir):
            shutil.rmtree(tmp_sep_dir, ignore_errors=True)

def assign_supertonic_voices_to_speakers(speakers, profile="asia"):
    """
    Assigns Supertonic voices (M1-M5 / F1-F5) strictly according to character gender and speaking duration:
      - Cutoffs:
          * Asia (China, Japan, Korea): Male < 165.0 Hz, Female >= 165.0 Hz
          * Western / Europe (US, EU): Male < 155.0 Hz, Female >= 155.0 Hz
      - Definitive spk_info["gender"] = "male" | "female" is stored for all UI views.
      - Male with longest speaking duration gets M1, second M2, etc.
      - Female with longest speaking duration gets F1, second F2, etc.
      - Skips speakers locked to a Known Voice Profile (is_profile=True).
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

        if spk_info.get("is_profile"):
            continue

        if is_fem:
            female_speakers.append((spk_id, spk_info, dur))
        else:
            male_speakers.append((spk_id, spk_info, dur))

    # Sort each gender independently by total speaking duration (longest speaking first!)
    male_speakers.sort(key=lambda x: x[2], reverse=True)
    female_speakers.sort(key=lambda x: x[2], reverse=True)

    # Assign Male voices M5, M2, M4, M3, M1 by duration
    for rank, (spk_id, spk_info, dur) in enumerate(male_speakers):
        v_idx = rank % len(SUPERTONIC_MALE_RANKED)
        chosen = SUPERTONIC_MALE_RANKED[v_idx]
        spk_info["supertonic_voice"] = chosen
        spk_info["assigned_voice"] = f"⚡ Supertonic [{chosen}]"
        for seg in spk_info.get("segments", []):
            seg["assigned_voice"] = f"⚡ Supertonic [{chosen}]"

    # Assign Female voices F4, F3, F2, F5, F1 by duration
    for rank, (spk_id, spk_info, dur) in enumerate(female_speakers):
        v_idx = rank % len(SUPERTONIC_FEMALE_RANKED)
        chosen = SUPERTONIC_FEMALE_RANKED[v_idx]
        spk_info["supertonic_voice"] = chosen
        spk_info["assigned_voice"] = f"⚡ Supertonic [{chosen}]"
        for seg in spk_info.get("segments", []):
            seg["assigned_voice"] = f"⚡ Supertonic [{chosen}]"


def diarize_segments_direct(audio_path, segments, max_speakers=6, log_cb=print, profile="asia", similarity_threshold=0.50, active_profiles=None):
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

    if active_profiles is None:
        active_profiles = load_active_speaker_profiles(encoder, log_cb=log_cb)

    if active_profiles:
        log_cb(f"[Profiles] Loaded {len(active_profiles)} active voice profile(s) (threshold={similarity_threshold:.2f}).")

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
            if len(features) % 50 == 0:
                clear_vram()
        except Exception:
            continue

    del encoder
    clear_vram()

    for seg in segments:
        seg["speaker"] = None
        seg["f0"] = 140.0

    # 1. Match against active Known Voice Profiles first
    matched_profile_map = {}
    unmatched_positions = []
    profile_hit_counts = {}

    for pos, val_idx in enumerate(valid_indices):
        feat = features[pos]
        feat_norm = feat / (np.linalg.norm(feat) + 1e-8)
        best_prof = None
        best_sim = -1.0

        if active_profiles:
            for prof in active_profiles:
                sim = float(np.dot(feat_norm, prof["embedding"]))
                if sim > best_sim:
                    best_sim = sim
                    best_prof = prof

        if best_prof is not None and best_sim >= similarity_threshold:
            pname = best_prof["name"]
            segments[val_idx]["speaker"] = pname
            segments[val_idx]["f0"] = pitches[pos]
            segments[val_idx]["assigned_voice"] = best_prof["voice"]
            matched_profile_map[pname] = best_prof
            profile_hit_counts[pname] = profile_hit_counts.get(pname, 0) + 1
        else:
            unmatched_positions.append(pos)

    if profile_hit_counts:
        summary = ", ".join(f"{k}: {v} lines" for k, v in sorted(profile_hit_counts.items()))
        log_cb(f"[Profiles] Matched lines -> {summary}")

    # 2. Cluster remaining unmatched segments
    from sklearn.cluster import AgglomerativeClustering
    if len(unmatched_positions) > 1:
        unm_features = np.array([features[p] for p in unmatched_positions])
        norms = np.linalg.norm(unm_features, axis=1, keepdims=True)
        unm_features = unm_features / np.maximum(norms, 1e-8)

        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=0.65,
            metric="cosine",
            linkage="average"
        )
        labels = clustering.fit_predict(unm_features)

        cluster_durations = {}
        for lab, pos in zip(labels, unmatched_positions):
            val_idx = valid_indices[pos]
            seg_dur = segments[val_idx]["end"] - segments[val_idx]["start"]
            cluster_durations[lab] = cluster_durations.get(lab, 0.0) + seg_dur

        sorted_by_duration = sorted(cluster_durations.keys(), key=lambda c: cluster_durations[c], reverse=True)
        top_clusters = sorted_by_duration[:max_speakers]

        centroids = {}
        for c_id in top_clusters:
            mask = (labels == c_id)
            c_vec = np.mean(unm_features[mask], axis=0)
            centroids[c_id] = c_vec / (np.linalg.norm(c_vec) + 1e-8)

        final_cluster_ids = []
        for i, lab in enumerate(labels):
            if lab in centroids:
                final_cluster_ids.append(lab)
            else:
                feat = unm_features[i]
                best_c = max(centroids.keys(), key=lambda c_key: np.dot(feat, centroids[c_key]))
                final_cluster_ids.append(best_c)

        rank_mapping = {c_id: f"Speaker_{rank+1:02d}" for rank, c_id in enumerate(top_clusters)}

        for i, pos in enumerate(unmatched_positions):
            val_idx = valid_indices[pos]
            spk_name = rank_mapping[final_cluster_ids[i]]
            segments[val_idx]["speaker"] = spk_name
            segments[val_idx]["f0"] = pitches[pos]
    elif len(unmatched_positions) == 1:
        val_idx = valid_indices[unmatched_positions[0]]
        segments[val_idx]["speaker"] = "Speaker_01"
        segments[val_idx]["f0"] = pitches[unmatched_positions[0]]

    for seg in segments:
        if seg["speaker"] is None:
            seg["speaker"] = "Speaker_01"
            seg["f0"] = 120.0
            seg["assigned_voice"] = "⚡ Supertonic [M1]"

    speakers = build_speakers_dict(segments)

    # Lock profile-matched speakers before assigning Supertonic defaults to remaining characters
    voices_dict = scan_voices_directory()
    for pname, prof in matched_profile_map.items():
        if pname in speakers:
            spk_info = speakers[pname]
            v_str = prof["voice"]
            spk_info["is_profile"] = True
            spk_info["assigned_voice"] = v_str
            spk_info["mode"] = "Voice Profile"
            m_st = re.search(r'\b([MF][1-5])\b', v_str)
            if m_st:
                spk_info["supertonic_voice"] = m_st.group(1)
            if "klon" in v_str.lower() or "clone" in v_str.lower():
                for vname, vpath in voices_dict.items():
                    if vname.lower() in v_str.lower():
                        spk_info["custom_voice_path"] = vpath
                        break
            for seg in spk_info.get("segments", []):
                seg["assigned_voice"] = v_str

    assign_supertonic_voices_to_speakers(speakers, profile=profile)

    for spk_id, info in sorted(speakers.items(), key=lambda x: x[1]["total_duration"], reverse=True):
        f0_val = info.get("avg_f0", 140.0)
        gender_str = "Female 👩" if info.get("is_female") else "Male 👨"
        conf = info.get("gender_confidence", "")
        tag = f"Profile ({info.get('assigned_voice')})" if info.get("is_profile") else f"Supertonic: {info.get('supertonic_voice')}"
        log_cb(f"[Character] {spk_id}: {gender_str} ({conf}) | F0={f0_val:.0f}Hz | Duration: {info['total_duration']:.1f}s | {tag}")

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
                "assigned_voice": old_data.get("assigned_voice", ""),
                "is_profile": old_data.get("is_profile", False),
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

