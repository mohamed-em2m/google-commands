"""Tests for the prediction pipeline."""

import numpy as np

from google_commands.audio import extract_mfcc, is_silence, pad_to_length
from google_commands.config import LABELS, NUM_MFCC, SAMPLE_RATE


def test_pad_to_length_short() -> None:
    audio = np.ones(1000)
    result = pad_to_length(audio, length=SAMPLE_RATE)
    assert len(result) == SAMPLE_RATE
    assert np.allclose(result[:SAMPLE_RATE - 1000], 0)
    assert np.allclose(result[SAMPLE_RATE - 1000:], 1)


def test_pad_to_length_exact() -> None:
    audio = np.ones(SAMPLE_RATE)
    result = pad_to_length(audio, length=SAMPLE_RATE)
    assert len(result) == SAMPLE_RATE
    assert np.allclose(result, 1)


def test_is_silence_loud() -> None:
    assert not is_silence(np.ones(100), threshold=0.1)


def test_is_silence_quiet() -> None:
    assert is_silence(np.zeros(100), threshold=0.1)


def test_extract_mfcc_shape() -> None:
    audio = np.random.randn(SAMPLE_RATE)
    mfcc = extract_mfcc(audio, n_mfcc=NUM_MFCC)
    assert mfcc.shape[1] == NUM_MFCC


def test_labels_count() -> None:
    assert len(LABELS) == 35


def test_labels_sorted() -> None:
    assert LABELS == sorted(LABELS)
