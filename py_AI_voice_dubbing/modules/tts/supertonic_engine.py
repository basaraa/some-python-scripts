from modules.config import *
from modules.audio_utils import run_cmd


def load_supertonic_model(log_cb=print):
    """Initializes and returns the local ONNX Supertonic TTS engine."""
    if SupertonicTTS is None:
        raise RuntimeError("Knižnica 'supertonic' nie je nainštalovaná! Spustite: pip install supertonic")
    log_cb("[Supertonic] Initializing local ONNX model...")
    model = SupertonicTTS(auto_download=True)
    log_cb("[Supertonic] Supertonic TTS engine is ready.")
    return model


def synthesize_supertonic(model, styles_cache, text, voice_id, target_lang, output_wav_path, speed=1.0, log_cb=print):
    """Synthesizes speech using Supertonic ONNX and resamples to SAMPLE_RATE."""
    if voice_id not in styles_cache:
        styles_cache[voice_id] = model.get_voice_style(voice_name=voice_id)
    style = styles_cache[voice_id]

    temp_super_wav = output_wav_path + ".st_temp.wav"
    st_lang = target_lang if target_lang in ["cs", "sk", "ja", "en", "de", "pl"] else "en"

    wav, _ = model.synthesize(
        text=text.strip(),
        voice_style=style,
        lang=st_lang,
        total_steps=8,
        speed=float(np.clip(speed, 0.7, 1.6))
    )
    model.save_audio(wav, temp_super_wav)

    cmd = [
        "ffmpeg", "-y", "-i", temp_super_wav,
        "-ar", str(SAMPLE_RATE), "-ac", "1",
        output_wav_path
    ]
    run_cmd(cmd)

    if os.path.exists(temp_super_wav):
        os.remove(temp_super_wav)
    return output_wav_path
