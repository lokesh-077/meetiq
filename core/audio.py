import subprocess
import logging
import wave
import numpy as np
import os
from pathlib import Path
from core.config import settings

logger = logging.getLogger("meetiq.audio")

_whisper_model_instance = None

def get_whisper_model():
    """Singleton Whisper model loader to avoid repeated disk reads and GPU reloads"""
    global _whisper_model_instance
    if _whisper_model_instance is None:
        import whisper
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading Whisper model '{settings.whisper_model}' onto {device.upper()}...")
        _whisper_model_instance = whisper.load_model(settings.whisper_model, device=device)
        logger.info("Whisper model loaded successfully.")
    return _whisper_model_instance

def extract_audio(media_path: str, output_path: str) -> str:
    """Extract clean 16kHz mono PCM audio using FFmpeg without damaging bandpass filters"""
    cmd = [
        "ffmpeg", "-y", "-i", media_path,
        "-vn", "-ar", "16000", "-ac", "1",
        "-c:a", "pcm_s16le", output_path
    ]
    logger.info(f"Extracting audio from {media_path} -> {output_path}")
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        err = result.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(f"FFmpeg audio extraction failed: {err}")
    return output_path

def detect_speakers_energy(audio_path: str, segments: list) -> list:
    """Multi-speaker segmentation based on energy profiling and pause boundaries"""
    try:
        with wave.open(audio_path, 'rb') as wf:
            frames = wf.readframes(wf.getnframes())
            fr = wf.getframerate()
            sw = wf.getsampwidth()
        
        samples = (np.frombuffer(frames, dtype=np.int16).astype(np.float32)
                   if sw == 2 else np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128)
        max_val = np.max(np.abs(samples))
        if max_val > 0:
            samples /= max_val

        labeled = []
        cur_speaker = 1
        prev_end = 0.0
        energy_levels = []

        for seg in segments:
            s = seg.get("start", 0)
            e = seg.get("end", s + 2)
            txt = seg.get("text", "").strip()
            if not txt:
                continue

            chunk = samples[int(s * fr):int(e * fr)]
            energy = float(np.sqrt(np.mean(chunk ** 2))) if len(chunk) > 0 else 0

            # Speaker transition heuristic: pause > 2.0s or marked energy shift
            if (s - prev_end > 2.0) and energy_levels:
                avg_recent = sum(energy_levels[-3:]) / len(energy_levels[-3:])
                if abs(energy - avg_recent) > 0.12:
                    cur_speaker = (cur_speaker % 4) + 1 # Supports up to 4 speakers

            energy_levels.append(energy)
            labeled.append({
                "start": round(s, 2),
                "end": round(e, 2),
                "speaker": f"Speaker {cur_speaker}",
                "text": txt
            })
            prev_end = e

        return labeled
    except Exception as e:
        logger.warning(f"Energy speaker detection fallback: {e}")
        return [{
            "start": s.get("start", 0),
            "end": s.get("end", 0),
            "speaker": "Speaker 1",
            "text": s.get("text", "").strip()
        } for s in segments if s.get("text", "").strip()]

def format_speaker_transcript(labeled: list) -> str:
    """Format labeled segments into clean multi-speaker dialog blocks"""
    lines, prev_spk, buffer = [], None, []
    for item in labeled:
        spk = item["speaker"]
        txt = item["text"].strip()
        if not txt:
            continue
        if spk != prev_spk:
            if buffer:
                lines.append(f"{prev_spk}: {' '.join(buffer)}")
            buffer = []
            prev_spk = spk
        buffer.append(txt)
    if buffer and prev_spk:
        lines.append(f"{prev_spk}: {' '.join(buffer)}")
    return "\n".join(lines)

def transcribe_audio_file(audio_path: str, language: str = None) -> tuple:
    """Transcribe audio with Whisper, deduplicating repetitive hallucinations"""
    import torch
    model = get_whisper_model()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    kwargs = {
        "verbose": False,
        "fp16": (device == "cuda"),
        "task": "transcribe",
        "condition_on_previous_text": False,
        "temperature": (0.0, 0.2, 0.4),
        "compression_ratio_threshold": 2.4,
        "no_speech_threshold": 0.6
    }
    if language:
        kwargs["language"] = language

    logger.info("Transcribing audio file with Whisper...")
    result = model.transcribe(audio_path, **kwargs)
    detected_lang = result.get("language", "unknown").upper()

    raw_segments = result.get("segments", [])
    clean_segments = []
    prev_text = ""

    # Filter consecutive duplicate hallucination loops
    for seg in raw_segments:
        txt = seg["text"].strip()
        if not txt:
            continue
        # Drop exact repeat loops
        if txt == prev_text:
            continue
        words = txt.split()
        if len(words) > 4 and len(set(words)) / len(words) < 0.25:
            continue
        clean_segments.append(seg)
        prev_text = txt

    labeled = detect_speakers_energy(audio_path, clean_segments)
    speaker_tx = format_speaker_transcript(labeled)
    plain_tx = " ".join(s["text"] for s in clean_segments).strip() or result.get("text", "").strip()

    return plain_tx, speaker_tx, labeled, detected_lang
