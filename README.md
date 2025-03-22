# Audio Command Recognition

This project implements a simple real-time audio command recognition system using three different pre-trained deep learning models. The system records audio from the microphone, processes the recording, and predicts a spoken command by combining the outputs from three different models (1D CNN, 2D CNN, and LSTM). The models are loaded from saved `.h5` files and the predictions are further processed with a label encoder stored using `joblib`.

## Features

- **Real-time audio recording:** Uses the microphone to capture a 1-second audio clip.
- **Audio processing:** Resamples audio and extracts MFCC features using `librosa`.
- **Ensemble prediction:** Combines predictions from three pre-trained models to determine the final command.
- **Stop command:** The system continuously listens until the predicted command is `"stop"`.

## Files

- **cnn1d.h5**: Pre-trained 1D CNN model.
- **cnn2d.h5**: Pre-trained 2D CNN model.
- **lstm.h5**: Pre-trained LSTM model.
- **encode**: Joblib file that stores the label encoder.
- **temp.wav**: Temporary file to store recorded audio (generated at runtime).
- **main.py**: Python script containing the source code.

## Requirements

The project uses several libraries for deep learning, audio processing, and recording. See the [`requirements.txt`](requirements.txt) file for the complete list.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/yourproject.git
   cd yourproject

    Install dependencies:

    pip install -r requirements.txt

    Ensure the following files are present in the project directory:

        cnn1d.h5

        cnn2d.h5

        lstm.h5

        encode (the joblib file containing the label encoder)

Usage

Run the main script:

python main.py

The system will start recording audio from your microphone, predict the spoken command, and print the prediction. When the command "stop" is predicted, the program will exit.
Notes

    Audio Input: Ensure that your microphone is set up and working properly.

    Models and Encoder: Make sure the pre-trained model files and the encoder file are correctly placed in the working directory.

    Dependencies: If you encounter any issues with audio recording or processing, verify that your system’s audio drivers and permissions are correctly configured.

License

This project is licensed under the MIT License.
