"""Command-line interface for the audio command recognition system."""

import argparse
import logging
import sys
from time import sleep

from google_commands.pipeline import Predictor


def run_realtime(interval: float = 1.0) -> None:
    predictor = Predictor()
    print("Listening for commands... Say 'stop' to exit.")
    while True:
        prediction = predictor.predict_from_mic()
        print(prediction)
        if prediction == "stop":
            break
        sleep(interval)


def run_once(audio_path: str | None = None) -> None:
    predictor = Predictor()
    if audio_path:
        result = predictor.predict_from_file(audio_path)
    else:
        result = predictor.predict_from_mic()
    print(result)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Real-time audio command recognition using ensemble deep learning."
    )
    parser.add_argument(
        "--once", "-o",
        action="store_true",
        help="Predict a single command and exit.",
    )
    parser.add_argument(
        "--file", "-f",
        type=str,
        default=None,
        help="Path to an audio file to predict (requires --once).",
    )
    parser.add_argument(
        "--interval", "-i",
        type=float,
        default=1.0,
        help="Seconds to wait between recording iterations (default: 1.0).",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging.",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(message)s")

    try:
        if args.once:
            run_once(args.file)
        else:
            run_realtime(args.interval)
    except KeyboardInterrupt:
        print("\nExiting.")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
