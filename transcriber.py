import whisper
import tempfile
import os
import subprocess
import numpy as np

os.environ["PATH"] += os.pathsep + r"C:\ffmpeg-8.1-essentials_build\bin"
FFMPEG  = r"C:\ffmpeg-8.1-essentials_build\bin\ffmpeg.exe"
FFPROBE = r"C:\ffmpeg-8.1-essentials_build\bin\ffprobe.exe"

# ── Model ─────────────────────────────────────────────────────────────────────
# "base"   → fastest on CPU, good Hinglish accuracy  ✅ recommended
# "small"  → slower (~34 s/chunk on CPU), slightly better accuracy
# "medium" → much slower, best multilingual accuracy (use with GPU only)
print("⏳ Loading Whisper model... (please wait, do NOT press Ctrl+C)")
model = whisper.load_model("base")
print("✅ Whisper model loaded and ready!")

# ── Config ────────────────────────────────────────────────────────────────────
# None = auto-detect language per chunk → handles Hindi, English, Hinglish ✅
# "hi" = force Hindi only
# "en" = English only
# "mr" = Marathi (uses Hindi script internally)
LANGUAGE = "hi"  # default; frontend can switch via set_lang message

MIN_BYTES   = 6_000  # ignore near-empty browser blobs
SILENCE_RMS = 0.008  # RMS below this → silence, skip

WHISPER_OPTIONS = dict(
    language=LANGUAGE,
    task="transcribe",
    condition_on_previous_text=False,
    no_speech_threshold=0.5,
    logprob_threshold=-1.2,
    word_timestamps=False,
    verbose=False,
)

# Common Whisper hallucinations on silence (language-agnostic)
HALLUCINATIONS = {
    "", ".", "..", "...", "you", "bye", "thank you", "thanks",
    "धन्यवाद", "शुक्रिया", "हाँ", "हां",
    "okay", "ok", "hmm", "um", "uh", "ah",
    "subtitles by", "subscribe", "like and subscribe",
    "www.", ".com",
}


def _convert_to_wav(webm_bytes: bytes) -> str | None:
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as f:
        f.write(webm_bytes)
        webm_path = f.name

    wav_path = webm_path.replace(".webm", ".wav")
    try:
        subprocess.run(
            [
                FFMPEG, "-y", "-loglevel", "error",
                "-i", webm_path,
                "-ar", "16000", "-ac", "1", "-sample_fmt", "s16",
                wav_path,
            ],
            capture_output=True, check=True,
        )
        return wav_path
    except subprocess.CalledProcessError as e:
        print(f"  ↳ ffmpeg error: {e.stderr.decode()}")
        return None
    finally:
        try:
            if os.path.exists(webm_path):
                os.unlink(webm_path)
        except OSError as e:
            print(f"  ↳ cleanup warning (webm): {e}")


def _is_silent(wav_path: str) -> bool:
    try:
        r = subprocess.run(
            [FFPROBE, "-v", "error", "-select_streams", "a:0",
             "-show_entries", "stream=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", wav_path],
            capture_output=True, text=True,
        )
        if float(r.stdout.strip() or "0") < 0.4:
            return True
    except Exception:
        pass

    try:
        audio = whisper.load_audio(wav_path)
        if float(np.sqrt(np.mean(audio ** 2))) < SILENCE_RMS:
            return True
    except Exception:
        pass

    return False


def transcribe_audio(audio_bytes: bytes) -> str:
    if len(audio_bytes) < MIN_BYTES:
        print(f"  ↳ skip: too small ({len(audio_bytes)} B)")
        return ""

    wav_path = _convert_to_wav(audio_bytes)
    if not wav_path:
        return ""

    try:
        if _is_silent(wav_path):
            print("  ↳ skip: silence")
            return ""

        result = model.transcribe(wav_path, **WHISPER_OPTIONS)
        text = result["text"].strip()

        if text.lower() in HALLUCINATIONS:
            print(f"  ↳ skip: hallucination → '{text}'")
            return ""

        print(f"  ↳ [{result.get('language', '?')}] {text}")
        return text

    except Exception as e:
        print(f"  ↳ error: {e}")
        return ""
    finally:
        try:
            if wav_path and os.path.exists(wav_path):
                os.unlink(wav_path)
        except OSError as e:
            print(f"  ↳ cleanup warning (wav): {e}")