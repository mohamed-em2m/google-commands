"""Audio recording and feature extraction utilities."""

import librosa
import numpy as np

from google_commands.config import DURATION_SECONDS, NUM_MFCC, SAMPLE_RATE, SILENCE_THRESHOLD


def load_and_resample(path: str, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    audio, sr = librosa.load(path, sr=target_sr)
    audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio


def pad_to_length(audio: np.ndarray, length: int = SAMPLE_RATE) -> np.ndarray:
    pad_width = max(length - len(audio), 0)
    return np.concatenate((np.zeros(pad_width), audio))


def is_silence(audio: np.ndarray, threshold: float = SILENCE_THRESHOLD) -> bool:
    return audio.max() < threshold


def extract_mfcc(audio: np.ndarray, n_mfcc: int = NUM_MFCC) -> np.ndarray:
    return librosa.feature.mfcc(y=audio[:SAMPLE_RATE], n_mfcc=n_mfcc).T


def record_audio(duration: float = DURATION_SECONDS, samplerate: int = SAMPLE_RATE) -> np.ndarray:
    import sounddevice as sd

    recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1)
    sd.wait()
    return recording


def save_wav(path: str, audio: np.ndarray, samplerate: int = SAMPLE_RATE) -> None:
    from scipy.io.wavfile import write

    write(path, samplerate, audio)
