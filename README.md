# Audio Command Recognition

Real-time spoken command recognition using an ensemble of three deep learning models (1D CNN, 2D CNN, LSTM). The system records audio from the microphone, extracts MFCC features, and predicts the spoken command via majority voting.

## Quick Start

```bash
pip install -r requirements.txt
python -m google_commands
```

Say **"stop"** to exit.

## Features

- **Real-time inference** — records 1-second audio chunks and classifies them instantly
- **35-word vocabulary** — digits 0–9, directional commands, and common words
- **Ensemble voting** — combines 1D CNN, 2D CNN, and LSTM predictions (majority wins, LSTM breaks ties)
- **Silence detection** — ignores quiet recordings automatically
- **Installable package** — `pip install -e .` to use the `google-commands` CLI

## Installation

```bash
# Clone the repository
git clone https://github.com/mohamed-em2m/google-commands.git
cd google-commands

# Install dependencies
pip install -r requirements.txt

# (Optional) Install as a package
pip install -e .
```

## Usage

```bash
# Continuous listening mode
python -m google_commands

# Single prediction from microphone
python -m google_commands --once

# Predict a file
python -m google_commands --once --file path/to/audio.wav

# Adjust interval between recordings (default 1s)
python -m google_commands --interval 0.5

# Enable debug logging
python -m google_commands --verbose
```

If installed via `pip install -e .`:

```bash
google-commands
google-commands --once
```

## Project Structure

```
├── src/google_commands/       # Main package
│   ├── config.py              # Constants and paths
│   ├── audio.py               # Recording and MFCC extraction
│   ├── models.py              # Model loading and inference
│   ├── pipeline.py            # Ensemble prediction pipeline
│   └── cli.py                 # Command-line interface
├── models/                    # Pre-trained model files (.h5)
├── notebooks/                 # Jupyter notebooks for training
├── tests/                     # Test suite
├── scripts/                   # Utility scripts
├── pyproject.toml             # Project metadata and build config
├── requirements.txt           # Runtime dependencies
└── README.md
```

## Models

Three pre-trained models are included:

| Model | Input Shape | Architecture |
|-------|-------------|--------------|
| 1D CNN | `(44×13,)` | 1D convolution over MFCC features |
| 2D CNN | `(44, 13, 1)` | 2D convolution treating MFCC as image |
| LSTM | `(44, 13)` | Recurrent sequence model |

Each model was trained on the [TensorFlow Speech Commands dataset](https://www.tensorflow.org/datasets/catalog/speech_commands) (v0.02).

## License

MIT
