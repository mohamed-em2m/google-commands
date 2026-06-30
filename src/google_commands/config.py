"""Configuration constants for the audio command recognition system."""

from pathlib import Path

# Audio parameters
SAMPLE_RATE: int = 22050
DURATION_SECONDS: float = 1.0
NUM_MFCC: int = 13
SILENCE_THRESHOLD: float = 0.1

# Paths
MODELS_DIR: Path = Path(__file__).resolve().parent.parent.parent / "models"
TEMP_AUDIO_FILE: str = "temp.wav"

# Command labels (alphabetically sorted to match LabelEncoder ordering)
LABELS: list[str] = [
    "backward", "bed", "bird", "cat", "dog", "down", "eight", "five",
    "follow", "forward", "four", "go", "happy", "house", "learn",
    "left", "marvin", "nine", "no", "off", "on", "one", "right",
    "seven", "sheila", "six", "stop", "three", "tree", "two", "up",
    "visual", "wow", "yes", "zero",
]

# Model filenames
MODEL_FILES: dict[str, str] = {
    "cnn1d": "cnn1d.h5",
    "cnn2d": "cnn2d.h5",
    "lstm": "lstm.h5",
}
