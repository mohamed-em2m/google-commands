"""Model loading and inference."""

import logging

import numpy as np
from tensorflow import keras

from google_commands.config import MODEL_FILES, MODELS_DIR

logger = logging.getLogger(__name__)


class ModelRegistry:
    def __init__(self, models_dir: str | None = None) -> None:
        self.models_dir = models_dir or str(MODELS_DIR)
        self._models: dict[str, keras.Model] = {}

    def load(self, name: str) -> keras.Model:
        if name not in self._models:
            filepath = f"{self.models_dir}/{MODEL_FILES[name]}"
            logger.info("Loading model %s from %s", name, filepath)
            self._models[name] = keras.saving.load_model(filepath)
        return self._models[name]

    def load_all(self) -> dict[str, keras.Model]:
        for name in MODEL_FILES:
            self.load(name)
        return self._models

    def predict_all(self, features: np.ndarray) -> dict[str, np.ndarray]:
        predictions: dict[str, np.ndarray] = {}
        for name, model in self.load_all().items():
            if name == "cnn1d":
                x = features.reshape((1, 44 * 13, 1))
            elif name == "cnn2d":
                x = features.reshape((1, 44, 13, 1))
            else:
                x = features.reshape((1, 44, 13))
            predictions[name] = np.argmax(model.predict(x, verbose=0), axis=1)
        return predictions
