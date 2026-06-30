#!/usr/bin/env python3
"""Generate the LabelEncoder file from the canonical label list.

Run: python scripts/generate_encode.py
Output: models/encode
"""

import joblib
from sklearn.preprocessing import LabelEncoder

from google_commands.config import LABELS, MODELS_DIR


def main() -> None:
    encoder = LabelEncoder()
    encoder.fit(LABELS)
    output_path = MODELS_DIR / "encode"
    joblib.dump(encoder, str(output_path))
    print(f"Encoder saved to {output_path}")
    print(f"Classes ({len(encoder.classes_)}): {list(encoder.classes_)}")


if __name__ == "__main__":
    main()
