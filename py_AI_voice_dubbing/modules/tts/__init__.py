from modules.config import *
from modules.audio_utils import clear_vram, get_all_available_voices, run_cmd
from modules.tts.supertonic_engine import load_supertonic_model, synthesize_supertonic
from modules.tts.edge_engine import get_default_edge_voice, synth_edge_async
from modules.tts.xtts_engine import load_xtts_model, synthesize_xtts


class DynamicDubbingEngine:
    def __init__(self, target_lang="cs", engine_type="supertonic", state=None):
        self.target_lang = target_lang
        self.engine_type = engine_type  # "supertonic", "edge", "xtts"
        self.state = state or {}
        self.xtts_model = None
        self.supertonic_model = None
        self.supertonic_styles = {}

    def ensure_supertonic_loaded(self, log_cb=print):
        if self.supertonic_model is None:
            self.supertonic_model = load_supertonic_model(log_cb=log_cb)

    def ensure_xtts_loaded(self, log_cb=print):
        if self.xtts_model is None:
            self.xtts_model = load_xtts_model(log_cb=log_cb)

    def get_assigned_edge_voice(self, spk_id):
        speakers = self.state.get("data", {}).get("speakers", {})
        spk_data = speakers.get(spk_id, {})
        avg_f0 = spk_data.get("avg_f0", 140.0)
        return get_default_edge_voice(self.target_lang, is_female=(avg_f0 >= 175.0))

    def get_assigned_supertonic_voice(self, spk_id):
        speakers = self.state.get("data", {}).get("speakers", {})
        spk_data = speakers.get(spk_id, {})
        assigned = spk_data.get("supertonic_voice")
        if assigned:
            return assigned
        avg_f0 = spk_data.get("avg_f0", 140.0)
        is_female = avg_f0 >= 175.0
        return random.choice(SUPERTONIC_FEMALE_VOICES) if is_female else random.choice(SUPERTONIC_MALE_VOICES)

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

        if active_engine in ("xtts", "clone") or (ref_voice_wav and os.path.exists(ref_voice_wav) and active_engine == "clone"):
            log_cb(f"[Voice Cloner] Cloning voice for {spk_id} from reference: {os.path.basename(ref_voice_wav)}")
            self.ensure_xtts_loaded(log_cb=log_cb)
            synthesize_xtts(self.xtts_model, text, ref_voice_wav, self.target_lang, output_wav_path, speed=calc_speed, log_cb=log_cb)
            return output_wav_path, "neutral"

        elif active_engine == "supertonic":
            self.ensure_supertonic_loaded(log_cb=log_cb)
            st_voice = explicit_voice if explicit_voice else self.get_assigned_supertonic_voice(spk_id)
            log_cb(f"[Supertonic] Synthesizing voice {st_voice} ({self.target_lang}) for {spk_id}")
            synthesize_supertonic(
                self.supertonic_model, self.supertonic_styles, text, st_voice,
                self.target_lang, output_wav_path, speed=calc_speed, log_cb=log_cb
            )
            return output_wav_path, "neutral"

        else:
            voice_name = explicit_voice if explicit_voice else self.get_assigned_edge_voice(spk_id)
            log_cb(f"[Edge-TTS] Synthesizing voice {voice_name} for {spk_id}")
            asyncio.run(synth_edge_async(text, output_wav_path, voice_name, speed=calc_speed))
            return output_wav_path, "neutral"

    def unload_model(self):
        clear_vram()


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
