"""Ensemble prediction pipeline."""

import logging

import numpy as np

from google_commands.audio import (
    extract_mfcc,
    is_silence,
    load_and_resample,
    pad_to_length,
    record_audio,
    save_wav,
)
from google_commands.config import LABELS, TEMP_AUDIO_FILE
from google_commands.models import ModelRegistry

logger = logging.getLogger(__name__)


class Predictor:
    def __init__(self, models_dir: str | None = None) -> None:
        self.registry = ModelRegistry(models_dir)

    def predict_from_file(self, audio_path: str) -> str:
        audio = load_and_resample(audio_path)
        return self._predict(audio)

    def predict_from_mic(self) -> str:
        logger.info("Recording...")
        audio = record_audio()
        save_wav(TEMP_AUDIO_FILE, audio)
        return self._predict(audio.flatten())

    def _predict(self, audio: np.ndarray) -> str:
        audio = pad_to_length(audio)

        if is_silence(audio):
            logger.info("Silence detected")
            return "silence"

        features = extract_mfcc(audio)
        preds = self.registry.predict_all(features)

        votes = np.zeros(len(LABELS), dtype=int)
        for pred in preds.values():
            votes[pred.item()] += 1

        if votes.max() < 2:
            return LABELS[preds["lstm"].item()]

        return LABELS[votes.argmax()]
