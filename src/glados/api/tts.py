import io
from functools import lru_cache

import soundfile as sf

from glados.TTS import SpeechSynthesizerProtocol, get_speech_synthesizer
from glados.utils import spoken_text_converter


@lru_cache(maxsize=4)
def _get_tts_model(voice: str) -> SpeechSynthesizerProtocol:
    return get_speech_synthesizer(voice)


@lru_cache(maxsize=1)
def _get_converter() -> spoken_text_converter.SpokenTextConverter:
    return spoken_text_converter.SpokenTextConverter()


def write_speech_audio_file(f: str | io.BytesIO, text: str, *, voice: str = "af_heart", format: str) -> None:
    """Generate speech audio from text and write it to a file.

    Parameters:
    f: File path or BytesIO object to write the audio to
    text: Text to convert to speech
    voice: GLaDOS or Kokoro voice identifier
    format: Audio format (e.g., "mp3", "wav", "ogg")
    """
    tts_model = _get_tts_model(voice)
    converter = _get_converter()
    converted_text = converter.text_to_spoken(text)
    audio = tts_model.generate_speech_audio(converted_text)
    sf.write(
        f,
        audio,
        tts_model.sample_rate,
        format=format.upper(),
    )


def write_glados_audio_file(f: str | io.BytesIO, text: str, *, format: str) -> None:
    """Generate GLaDOS-style speech audio from text and write it to a file."""
    write_speech_audio_file(f, text, voice="glados", format=format)
