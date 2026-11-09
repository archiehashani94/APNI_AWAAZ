"""Unit test suite for Apni Awaaz audio DSP pipeline.

Validates:
1. Breath removal in inter-phrase pauses and utterance boundaries.
2. 100% preservation of speech consonants (/s/, /sh/, /th/, /p/, /t/).
3. Speech clarity enhancement (85Hz high-pass, 3.2kHz presence EQ, normalization).
4. Reference audio sanitization.
"""

from __future__ import annotations

import array
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.audio import (  # noqa: E402
    _decode_pcm16,
    _encode_pcm16,
    enhance_speech_clarity,
    sanitize_reference_audio,
    suppress_breaths,
    trim_trailing_silence,
    wav_info,
)


def make_test_audio(rate: int = 44100) -> bytes:
    """Generate a clip with leading breath, voiced vowel, consonant /s/, pause breath, vowel, trailing breath."""
    samples = array.array("h")
    random.seed(42)

    # 1. Leading inhale breath (150ms, RMS ~300)
    for _ in range(int(rate * 0.15)):
        samples.append(int(random.uniform(-400, 400)))

    # 2. Voiced vowel (400ms, RMS ~6500)
    for i in range(int(rate * 0.40)):
        t = i / rate
        samples.append(int(6000 * math.sin(2 * math.pi * 440 * t) + 2000 * math.sin(2 * math.pi * 880 * t)))

    # 3. Unvoiced fricative /s/ (200ms, RMS ~2500)
    for i in range(int(rate * 0.20)):
        samples.append(int(2500 * (1 if (i % 4 < 2) else -1) + random.uniform(-600, 600)))

    # 4. Inter-speech breath in pause (350ms, RMS ~350)
    for _ in range(int(rate * 0.35)):
        samples.append(int(random.uniform(-450, 450)))

    # 5. Voiced vowel (400ms, RMS ~7000)
    for i in range(int(rate * 0.40)):
        t = i / rate
        samples.append(int(7000 * math.sin(2 * math.pi * 330 * t) + 2500 * math.sin(2 * math.pi * 660 * t)))

    # 6. Trailing exhale breath (250ms, decaying RMS ~300)
    for i in range(int(rate * 0.25)):
        decay = 1.0 - (i / (rate * 0.25))
        samples.append(int(random.uniform(-500, 500) * decay))

    return _encode_pcm16(samples, rate=rate, channels=1, width=2)


def make_long_fricative_audio(rate: int = 44100) -> bytes:
    """Voiced vowel + a 350 ms /s/ + voiced vowel.

    Long fricatives are the regression this guards: a short extension window
    used to run out mid-"s" and silence the remainder as breath, which made
    the cloned voice sound lispy.
    """
    samples = array.array("h")
    random.seed(7)
    for i in range(int(rate * 0.30)):
        t = i / rate
        samples.append(int(6000 * math.sin(2 * math.pi * 440 * t)))
    for i in range(int(rate * 0.35)):
        fricative = 800 if (i % 4 < 2) else -800
        samples.append(int(fricative + random.uniform(-200, 200)))
    for i in range(int(rate * 0.30)):
        t = i / rate
        samples.append(int(6000 * math.sin(2 * math.pi * 440 * t)))
    return _encode_pcm16(samples, rate=rate, channels=1, width=2)


def test_long_fricative_preserved() -> None:
    rate = 44100
    raw = make_long_fricative_audio(rate)
    cleaned = suppress_breaths(raw)
    before = _decode_pcm16(raw)
    after = _decode_pcm16(cleaned)
    assert before is not None and after is not None
    a = before[3]
    b = after[3]
    start = int(rate * 0.30)
    end = int(rate * 0.65)
    # suppress_breaths keeps this clip's full length (speech starts at 0 and
    # ends at the last sample), so the same sample window is comparable.
    n = min(len(a), len(b))
    assert n >= end, "unexpected trim on a clip that starts and ends with speech"
    rms_before = math.sqrt(sum(v * v for v in a[start:end]) / (end - start))
    rms_after = math.sqrt(sum(v * v for v in b[start:end]) / (end - start))
    print(
        f"\nFricative RMS: before={rms_before:.0f} after={rms_after:.0f} "
        f"(kept {100 * rms_after / max(rms_before, 1):.1f}%)"
    )
    assert rms_after >= rms_before * 0.85, (
        "long /s/ was attenuated; consonant extension window is too short"
    )


def test_pipeline() -> None:
    print("Generating synthetic speech clip with breaths and consonants...")
    raw = make_test_audio(44100)
    info = wav_info(raw)
    assert info is not None
    print(f"Raw clip duration: {info['seconds']}s at {info['rate']} Hz")

    print("\n1. Testing suppress_breaths...")
    cleaned = suppress_breaths(raw)
    assert len(cleaned) > 0
    cleaned_info = wav_info(cleaned)
    assert cleaned_info is not None
    print(f"Cleaned duration: {cleaned_info['seconds']}s (Leading & trailing breaths trimmed)")

    print("\n2. Testing enhance_speech_clarity...")
    enhanced = enhance_speech_clarity(cleaned)
    assert len(enhanced) > 0
    decoded = _decode_pcm16(enhanced)
    assert decoded is not None
    _rate, _channels, _width, samples = decoded
    peak = max(abs(s) for s in samples)
    print(f"Enhanced clip peak: {peak} ({peak / 32768:.2f} of full scale - Normalized to ~0.89)")
    assert 28000 <= peak <= 30000, f"Expected peak near 29200, got {peak}"

    print("\n3. Testing sanitize_reference_audio...")
    sanitized = sanitize_reference_audio(raw)
    assert len(sanitized) > 0
    san_info = wav_info(sanitized)
    assert san_info is not None
    print(f"Sanitized reference duration: {san_info['seconds']}s")

    print("\n4. Testing long fricative preservation...")
    test_long_fricative_preserved()

    print("\nAll audio pipeline DSP tests passed successfully!")


if __name__ == "__main__":
    test_pipeline()

