from modules.config import *
from modules.audio_utils import run_cmd


def get_default_edge_voice(target_lang, is_female=False):
    """Returns default Microsoft Edge Neural voice for the target language and gender."""
    if target_lang == "cs":
        return "cs-CZ-VlastaNeural" if is_female else "cs-CZ-AntoninNeural"
    elif target_lang == "sk":
        return "sk-SK-ViktoriaNeural" if is_female else "sk-SK-LukasNeural"
    elif target_lang == "ja":
        return "ja-JP-NanamiNeural" if is_female else "ja-JP-KeitaNeural"
    return "en-US-JennyNeural" if is_female else "en-US-GuyNeural"


async def synth_edge_async(text, output_wav, voice_name, speed=1.0):
    """Asynchronously synthesizes speech via Edge-TTS and converts to mono SAMPLE_RATE WAV."""
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
