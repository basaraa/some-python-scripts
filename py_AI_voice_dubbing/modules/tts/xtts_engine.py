from modules.config import *
from modules.audio_utils import get_or_create_reference_transcription


def load_xtts_model(log_cb=print):
    """Loads Coqui XTTS-v2 multilingual voice cloning model into GPU VRAM."""
    try:
        import transformers.pytorch_utils
        if not hasattr(transformers.pytorch_utils, "isin_mps_friendly"):
            transformers.pytorch_utils.isin_mps_friendly = torch.isin
        from TTS.api import TTS
        log_cb("[Voice Cloner] Loading offline XTTS-v2 model into VRAM for voice cloning...")
        model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(DEVICE)
        log_cb("[Voice Cloner] XTTS-v2 model is ready.")
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load XTTS-v2 voice cloning model: {e}")


def synthesize_xtts(xtts_model, text, ref_voice_wav, target_lang, output_wav_path, speed=1.0, log_cb=print):
    """Clones voice from ref_voice_wav using Coqui XTTS-v2."""
    get_or_create_reference_transcription(ref_voice_wav, log_cb=log_cb)
    xtts_lang = "cs" if target_lang in ["cs", "sk"] else target_lang
    clean_text = text.strip()
    if not clean_text.endswith((".", "!", "?")):
        clean_text += "."
    clean_text += " "

    xtts_model.tts_to_file(
        text=clean_text,
        speaker_wav=ref_voice_wav,
        language=xtts_lang,
        file_path=output_wav_path,
        speed=speed,
        enable_text_splitting=False
    )
    return output_wav_path
