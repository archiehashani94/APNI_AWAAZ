"""WAV helpers shared by the API and the Gnani client.

Gnani's TTS endpoint currently returns RIFF/WAVE audio whose ``RIFF`` and
``data`` chunk sizes are both ``0xFFFFFFFF`` (a streaming sentinel). Players
and duration checks then believe the clip is ~12 hours long. Rewriting those
two length fields to the real byte counts is safe and makes the audio behave
normally in browsers.
"""

from __future__ import annotations

import io
import math
import struct
import sys
import wave
from array import array
from typing import Any


def repair_wav_header(payload: bytes) -> bytes:
    """Return WAV bytes with correct RIFF/data sizes, or the input unchanged."""
    if len(payload) < 12 or payload[:4] != b"RIFF" or payload[8:12] != b"WAVE":
        return payload

    data_pos: int | None = None
    pos = 12
    end = len(payload)
    while pos + 8 <= end:
        chunk_id = payload[pos : pos + 4]
        declared = struct.unpack_from("<I", payload, pos + 4)[0]
        if chunk_id == b"data":
            data_pos = pos
            break
        pos += 8 + declared + (declared & 1)

    if data_pos is None:
        return payload

    actual = end - (data_pos + 8)
    out = bytearray(payload)
    struct.pack_into("<I", out, data_pos + 4, actual)
    # Gnani puts data last, so the file length is the real extent.
    struct.pack_into("<I", out, 4, len(out) - 8)
    return bytes(out)


def wav_info(payload: bytes) -> dict[str, Any] | None:
    """Describe a PCM WAV payload, repairing sentinel chunk sizes first."""
    repaired = repair_wav_header(payload)
    try:
        with wave.open(io.BytesIO(repaired), "rb") as handle:
            rate = handle.getframerate()
            frames = handle.getnframes()
            return {
                "rate": rate,
                "channels": handle.getnchannels(),
                "width": handle.getsampwidth(),
                "seconds": round(frames / rate, 2) if rate else None,
                "repaired": repaired != payload,
            }
    except (wave.Error, EOFError, ValueError):
        return None


def _db(value: float) -> float:
    """Convert a 16-bit sample magnitude to dBFS."""
    return 20.0 * math.log10(max(value, 1.0) / 32768.0)


def _decode_pcm16(payload: bytes) -> tuple[int, int, int, array] | None:
    """Return (rate, channels, width, samples) for 16-bit PCM WAV, else None."""
    repaired = repair_wav_header(payload)
    try:
        with wave.open(io.BytesIO(repaired), "rb") as handle:
            rate = handle.getframerate()
            channels = handle.getnchannels()
            width = handle.getsampwidth()
            raw = handle.readframes(handle.getnframes())
    except (wave.Error, EOFError, ValueError):
        return None
    if width != 2 or channels < 1 or rate <= 0 or not raw:
        return None
    samples = array("h")
    samples.frombytes(raw[: len(raw) - (len(raw) % 2)])
    if sys.byteorder == "big":
        samples.byteswap()
    if not samples:
        return None
    return rate, channels, width, samples


def _encode_pcm16(
    samples: array, *, rate: int, channels: int, width: int, fade_ms: int = 10
) -> bytes:
    fade_frames = min(int(rate * fade_ms / 1000), len(samples) // channels)
    for i in range(fade_frames):
        gain = i / fade_frames
        index = len(samples) - (i + 1) * channels
        for channel in range(channels):
            samples[index + channel] = int(samples[index + channel] * gain)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(width)
        handle.setframerate(rate)
        handle.writeframes(samples.tobytes())
    return buffer.getvalue()


def _frame_stats(
    samples: array, rate: int, channels: int, frame_ms: int = 20
) -> tuple[list[tuple[float, float]], int, int]:
    """Return per-frame ``(rms, zero_crossing_rate)`` plus frame length/count."""
    frame_len = max(1, int(rate * frame_ms / 1000)) * channels
    stats: list[tuple[float, float]] = []
    for start in range(0, len(samples) - frame_len + 1, frame_len):
        chunk = samples[start : start + frame_len]
        rms = math.sqrt(sum(v * v for v in chunk) / len(chunk))
        crossings = 0
        previous = chunk[0]
        for value in chunk[1:]:
            if (previous >= 0) != (value >= 0):
                crossings += 1
            previous = value
        stats.append((rms, crossings / len(chunk)))
    return stats, frame_len, frame_ms


def _speech_groups(
    samples: array,
    rate: int,
    channels: int,
    *,
    ratio: float = 0.12,
    floor_db: float = -48.0,
    bridge_frames: int = 3,
) -> tuple[list[tuple[int, int]], float, int, int] | None:
    """Split audio into speech groups as (start_frame, end_frame) pairs.

    Frames are 20 ms. Groups separated by less than ``bridge_frames`` stay
    together so a normal word is never split; a real pause or model-generated
    breath becomes its own group. Also returns p90 level, frame length, and
    frame count.
    """
    stats, frame_len, _ = _frame_stats(samples, rate, channels)
    if len(stats) < 3:
        return None
    ordered = sorted(value for value, _zcr in stats)
    p90 = ordered[min(len(ordered) - 1, int(len(ordered) * 0.9))]
    threshold = max(p90 * ratio, 32768.0 * 10 ** (floor_db / 20.0))
    spans: list[list[int]] = []
    start_frame: int | None = None
    flags = [value >= threshold for value, _zcr in stats] + [False]
    for index, value in enumerate(flags):
        if value and start_frame is None:
            start_frame = index
        elif not value and start_frame is not None:
            spans.append([start_frame, index])
            start_frame = None
    merged: list[list[int]] = []
    for span in spans:
        if merged and span[0] - merged[-1][1] < bridge_frames:
            merged[-1][1] = span[1]
        else:
            merged.append(span)
    return [(a, b) for a, b in merged], p90, frame_len, len(stats)


def trim_extra_groups(
    payload: bytes,
    text: str,
    *,
    max_words: int = 3,
    max_seconds: float = 6.0,
    keep_ms: int = 110,
) -> bytes:
    """Drop model-generated extra speech after a short phrase.

    The clone model sometimes repeats the whole phrase (``Yes. Yes.``) or adds
    a filler syllable (``Okay. Um, okay.``). A phrase with N words should have
    at most N speech groups, so anything after the final expected word is
    removed, but only when a real pause separates it. Long replies are left
    untouched, and the check requires the pause to prevent clipping a word that
    the model merely pronounced slowly.
    """
    words = [word for word in text.split() if any(ch.isalnum() for ch in word)]
    if not words or len(words) > max_words:
        return payload
    decoded = _decode_pcm16(payload)
    if decoded is None:
        return payload
    rate, channels, width, samples = decoded
    if len(samples) / (rate * channels) > max_seconds:
        return payload
    analysis = _speech_groups(samples, rate, channels)
    if analysis is None:
        return payload
    groups, _p90, frame_len, _count = analysis
    if len(groups) <= len(words):
        return payload

    expected_end = groups[len(words) - 1][1]
    extra_start = groups[len(words)][0]
    # Only a long pause marks a genuinely repeated phrase. A shorter gap is
    # more likely a syllable break inside a word (for example "wa-ter"), and
    # cutting there would delete real speech.
    if extra_start - expected_end < 10:
        return payload
    # The tail must itself be long enough to be a word, not a stray click.
    if groups[-1][1] - extra_start < 8:
        return payload
    # Never cut before the phrase has actually started, or when the expected
    # part is just a click.
    if expected_end < 6:
        return payload
    keep_frames = int(rate * keep_ms / 1000) // max(1, int(rate * 0.02))
    cut_frame = min(extra_start, expected_end + max(0, keep_frames))
    end_sample = min(len(samples), cut_frame * frame_len)
    if end_sample >= len(samples) or end_sample <= 0:
        return payload
    return _encode_pcm16(
        samples[:end_sample], rate=rate, channels=channels, width=width
    )


def _group_features(
    stats: list[tuple[float, float]], span: tuple[int, int]
) -> dict[str, float]:
    start, end = span
    segment = stats[start:end]
    rms = [value for value, _zcr in segment]
    zcr = [zcr for _value, zcr in segment]
    count = max(1, len(segment))
    unvoiced = sum(1 for value in zcr if value >= 0.15)
    return {
        "start": start,
        "end": end,
        "frames": end - start,
        "mean_rms": sum(rms) / count,
        "peak_rms": max(rms) if rms else 0.0,
        "mean_zcr": sum(zcr) / count,
        "max_zcr": max(zcr) if zcr else 0.0,
        "unvoiced_frames": unvoiced,
        "unvoiced_ratio": unvoiced / count,
    }


def _is_boundary_filler(
    feature: dict[str, float],
    gap_frames: int,
    reference_rms: float,
    *,
    pause_frames: int,
    min_frames: int,
    max_frames: int,
    quiet_ratio: float,
) -> bool:
    """Return True for an isolated, sustained vowel with no consonants.

    The clone sometimes appends ``ahh``/``uhh``/``ha`` after a sentence. That
    extra sound is a pure voiced vowel, so it has no high-zero-crossing frames
    (no ``s``/``sh``/``t``/``k``), lasts longer than a normal syllable, and is
    separated from the phrase by a real pause. Real final words almost always
    contain a consonant and sit at the speech level, so they survive.
    """
    frames = int(feature["frames"])
    return (
        gap_frames >= pause_frames
        and min_frames <= frames <= max_frames
        and feature["unvoiced_frames"] == 0
        and feature["mean_zcr"] <= 0.12
        and feature["mean_rms"] <= max(reference_rms * quiet_ratio, 1.0)
    )


def trim_boundary_fillers(
    payload: bytes,
    *,
    pause_ms: int = 240,
    min_ms: int = 450,
    max_ms: int = 1500,
    quiet_ratio: float = 0.75,
    keep_ms: int = 110,
    fade_ms: int = 8,
    merge_ms: int = 160,
) -> bytes:
    """Remove model-generated ``ahh``/``uhh``/``ha`` sounds at phrase edges.

    The word-count heuristic in :func:`trim_extra_groups` only covers phrases
    up to three words. For longer replies the model can still add a sustained
    voiced filler (for example ``"Thank you very much, ha."``). This function is
    text-independent: it looks for a pure-vowel block that is isolated by a
    pause, lasts longer than a normal final word, and is quieter than the
    phrase, then removes it. A quiet lead-in before a loud ``ha`` is removed
    with it. Isolated middle fillers are silenced in place.
    """
    decoded = _decode_pcm16(payload)
    if decoded is None:
        return payload
    rate, channels, width, samples = decoded
    stats, frame_len, _ = _frame_stats(samples, rate, channels)
    if len(stats) < 6:
        return payload

    analysis = _speech_groups(samples, rate, channels, ratio=0.10, floor_db=-50.0)
    if analysis is None:
        return payload
    groups, _p90, _frame_len, _count = analysis
    if len(groups) < 2:
        return payload

    pause_frames = max(1, pause_ms // 20)
    min_frames = max(1, min_ms // 20)
    max_frames = max(min_frames + 1, max_ms // 20)
    keep_frames = max(0, keep_ms // 20)
    merge_frames = max(0, merge_ms // 20)

    def blocks(groups: list[tuple[int, int]]) -> list[dict[str, float]]:
        merged: list[tuple[int, int]] = []
        for span in groups:
            if merged and span[0] - merged[-1][1] <= merge_frames:
                merged[-1] = (merged[-1][0], span[1])
            else:
                merged.append(span)
        return [_group_features(stats, span) for span in merged]

    features = blocks(groups)
    end_frame = len(stats)

    # Only the end of the clip is trimmed. Cutting a leading block is unsafe:
    # a soft opening ("Could you please...") has the same pause, level, and
    # voicing profile as a held ``ahh``, and an earlier version deleted real
    # words that way. A trailing filler after a finished sentence cannot be
    # confused with speech the user asked for.
    while len(features) >= 2:
        last = features[-1]
        previous = features[-2]
        gap = int(last["start"] - previous["end"])
        reference = max(feature["mean_rms"] for feature in features[:-1])
        if not _is_boundary_filler(
            last,
            gap,
            reference,
            pause_frames=pause_frames,
            min_frames=min_frames,
            max_frames=max_frames,
            quiet_ratio=quiet_ratio,
        ):
            break
        end_frame = min(int(last["start"]), int(previous["end"]) + keep_frames)
        features.pop()

    if end_frame >= len(stats):
        return payload

    out = samples[: end_frame * frame_len]
    if not out:
        return payload

    # Soften the new edge so the cut never clicks.
    fade = max(1, int(rate * fade_ms / 1000)) * channels
    fade = min(fade, len(out) // channels)
    for index in range(fade):
        out[len(out) - 1 - index] = int(
            out[len(out) - 1 - index] * (index / fade)
        )

    return _encode_pcm16(out, rate=rate, channels=channels, width=width)


def _similar_groups(
    left: dict[str, float], right: dict[str, float], tolerance: float = 0.45
) -> bool:
    if left["frames"] <= 0 or right["frames"] <= 0:
        return False
    duration_gap = abs(left["frames"] - right["frames"]) / max(
        left["frames"], right["frames"]
    )
    energy_gap = abs(left["mean_rms"] - right["mean_rms"]) / max(
        left["mean_rms"], right["mean_rms"], 1.0
    )
    return (
        duration_gap <= tolerance
        and energy_gap <= max(tolerance, 0.55)
        and abs(left["mean_zcr"] - right["mean_zcr"]) <= 0.10
        and abs(left["unvoiced_ratio"] - right["unvoiced_ratio"]) <= 0.25
    )


def _pearson(left: list[float], right: list[float]) -> float:
    count = min(len(left), len(right))
    if count <= 1:
        return 0.0
    left = left[:count]
    right = right[:count]
    mean_left = sum(left) / count
    mean_right = sum(right) / count
    var_left = sum((value - mean_left) ** 2 for value in left)
    var_right = sum((value - mean_right) ** 2 for value in right)
    if var_left <= 0 or var_right <= 0:
        return 0.0
    covariance = sum(
        (a - mean_left) * (b - mean_right) for a, b in zip(left, right)
    )
    return covariance / math.sqrt(var_left * var_right)


def _whole_repeat_frame(
    stats: list[tuple[float, float]],
    *,
    correlation: float = 0.95,
    zcr_correlation: float = 0.50,
) -> int | None:
    """Return the frame where a doubled phrase begins, or ``None``.

    A full phrase rendered twice produces a near-perfect match between the
    first ``N - lag`` frames and the frames starting at ``lag``. Checking lags
    directly is much cheaper than scanning every possible suffix length.
    """
    count = len(stats)
    if count < 70:
        return None
    loudness = [math.log(rms + 1.0) for rms, _zcr in stats]
    zcr = [value for _rms, value in stats]
    ordered = sorted(rms for rms, _zcr in stats)
    p90 = ordered[min(len(ordered) - 1, int(len(ordered) * 0.9))]
    silence_threshold = max(p90 * 0.12, 50.0)

    candidates: list[int] = []
    for lag in range(int(count * 0.20), int(count * 0.85)):
        overlap = min(lag, count - lag)
        if overlap < 35:
            continue
        if _pearson(loudness[:overlap], loudness[lag : lag + overlap]) < correlation:
            continue
        if _pearson(zcr[:overlap], zcr[lag : lag + overlap]) < zcr_correlation:
            continue
        if not any(
            stats[index][0] < silence_threshold
            for index in range(max(0, lag - 12), min(count, lag + 4))
        ):
            continue
        candidates.append(lag)

    return min(candidates) if candidates else None


def _tail_repeat_frame(
    stats: list[tuple[float, float]],
    *,
    min_seconds: float = 0.7,
    max_seconds: float = 2.6,
    correlation: float = 0.95,
    zcr_correlation: float = 0.55,
) -> int | None:
    """Return the frame where the repeated tail starts, or ``None``.

    Compares the end of the clip against every earlier segment of the same
    length. A true repeated phrase correlates almost perfectly in both loudness
    and zero-crossing rate, while normal prosody stays well below this. The
    repeated tail must be preceded by a pause so the cut lands in silence.
    """
    count = len(stats)
    if count < 60:
        return None
    loudness = [math.log(rms + 1.0) for rms, _zcr in stats]
    zcr = [value for _rms, value in stats]
    ordered = sorted(rms for rms, _zcr in stats)
    p90 = ordered[min(len(ordered) - 1, int(len(ordered) * 0.9))]
    silence_threshold = max(p90 * 0.12, 50.0)

    min_frames = max(30, int(min_seconds / 0.02))
    max_frames = min(int(max_seconds / 0.02), int(count * 0.65))
    earliest: int | None = None

    for length in range(min_frames, max_frames + 1):
        start = count - length
        if start < length:
            continue
        tail = loudness[start:]
        tail_zcr = zcr[start:]
        for offset in range(0, start - length + 1):
            if _pearson(tail, loudness[offset : offset + length]) < correlation:
                continue
            if _pearson(tail_zcr, zcr[offset : offset + length]) < zcr_correlation:
                continue
            if not any(
                stats[index][0] < silence_threshold
                for index in range(max(0, start - 12), min(count, start + 4))
            ):
                continue
            earliest = start if earliest is None else min(earliest, start)
            break

    return earliest


def trim_repeated_tail(
    payload: bytes,
    text: str = "",
    *,
    pause_ms: int = 240,
    keep_ms: int = 110,
) -> bytes:
    """Remove a repeated phrase or word at the end of a synthesized clip.

    The clone model occasionally renders the whole phrase twice (or repeats its
    final words) after a pause. The end of the clip is first compared with every
    earlier segment; a near-perfect loudness/voicing match marks a repeated tail
    and is cut. Speech groups are then compared as a fallback for repeats whose
    pacing differs slightly.
    """
    decoded = _decode_pcm16(payload)
    if decoded is None:
        return payload
    rate, channels, width, samples = decoded
    stats, frame_len, _ = _frame_stats(samples, rate, channels)
    if len(stats) < 60:
        return payload

    cut_frame = _whole_repeat_frame(stats)
    if cut_frame is None:
        cut_frame = _tail_repeat_frame(stats)
    if cut_frame is not None:
        end_sample = min(len(samples), cut_frame * frame_len)
        if 0 < end_sample < len(samples):
            return _encode_pcm16(
                samples[:end_sample], rate=rate, channels=channels, width=width
            )

    analysis = _speech_groups(samples, rate, channels)
    if analysis is None:
        return payload
    groups, _p90, _frame_len, _count = analysis
    if len(groups) < 4:
        return payload
    features = [_group_features(stats, span) for span in groups]
    pause_frames = max(1, pause_ms // 20)
    keep_frames = max(0, keep_ms // 20)

    # A whole-phrase or multi-word tail repeat: compare the trailing block with
    # the block immediately before it. A single matching group is deliberately
    # not treated as a repeat: a normal phrase can end with two similar words
    # ("stand up", "the fan") separated by a pause, and cutting one would delete
    # real speech. Short single-word repeats are handled by trim_extra_groups.
    for split in range(1, len(features)):
        gap = int(features[split]["start"] - features[split - 1]["end"])
        if gap < pause_frames:
            continue
        available = min(split, len(features) - split)
        for length in range(available, 1, -1):
            left = features[split - length : split]
            right = features[split : split + length]
            if all(_similar_groups(a, b) for a, b in zip(left, right)):
                cut_frame = min(
                    int(features[split]["start"]),
                    int(features[split - 1]["end"]) + keep_frames,
                )
                break
        if cut_frame is not None:
            break

    if cut_frame is None:
        return payload
    end_sample = min(len(samples), cut_frame * frame_len)
    if end_sample <= 0 or end_sample >= len(samples):
        return payload
    return _encode_pcm16(
        samples[:end_sample], rate=rate, channels=channels, width=width
    )


def trim_trailing_silence(
    payload: bytes,
    *,
    keep_ms: int = 120,
    lead_ms: int = 60,
    fade_ms: int = 12,
    ratio: float = 0.15,
    floor_db: float = -48.0,
) -> bytes:
    """Trim near-silent edges, including the breathy tail some TTS models add.

    The clone model sometimes appends a breath, a held vowel, or a repeated
    syllable after the sentence. That tail is much quieter than speech, so a
    frame-level threshold relative to the speech level removes it while leaving
    real word endings intact. A short fade-out prevents an audible click.
    Returns the input unchanged when it is not 16-bit PCM or already tight.
    """
    repaired = repair_wav_header(payload)
    try:
        with wave.open(io.BytesIO(repaired), "rb") as handle:
            rate = handle.getframerate()
            channels = handle.getnchannels()
            width = handle.getsampwidth()
            frames = handle.getnframes()
            raw = handle.readframes(frames)
    except (wave.Error, EOFError, ValueError):
        return payload
    if width != 2 or channels < 1 or rate <= 0 or not raw:
        return payload

    samples = array("h")
    samples.frombytes(raw[: len(raw) - (len(raw) % 2)])
    if sys.byteorder == "big":
        samples.byteswap()
    if not samples:
        return payload

    frame_len = max(1, int(rate * 0.02)) * channels
    frame_rms: list[float] = []
    for start in range(0, len(samples) - frame_len + 1, frame_len):
        chunk = samples[start : start + frame_len]
        frame_rms.append(math.sqrt(sum(v * v for v in chunk) / len(chunk)))
    if len(frame_rms) < 3:
        return payload

    ordered = sorted(frame_rms)
    p90 = ordered[min(len(ordered) - 1, int(len(ordered) * 0.9))]
    if p90 <= 0:
        return payload
    threshold = max(p90 * ratio, 32768.0 * 10 ** (floor_db / 20.0))

    first = next((i for i, value in enumerate(frame_rms) if value >= threshold), 0)
    last = next(
        (i for i in range(len(frame_rms) - 1, -1, -1) if frame_rms[i] >= threshold),
        len(frame_rms) - 1,
    )

    start_sample = max(0, first * frame_len - int(rate * lead_ms / 1000) * channels)
    end_sample = min(
        len(samples),
        (last + 1) * frame_len + int(rate * keep_ms / 1000) * channels,
    )

    out = samples[start_sample:end_sample]
    fade_frames = min(int(rate * fade_ms / 1000), len(out) // channels)
    for i in range(fade_frames):
        gain = i / fade_frames
        index = len(out) - (i + 1) * channels
        for channel in range(channels):
            out[index + channel] = int(out[index + channel] * gain)

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(width)
        handle.setframerate(rate)
        handle.writeframes(out.tobytes())
    return buffer.getvalue()



def suppress_breaths(
    payload: bytes,
    *,
    zcr_threshold: float = 0.18,
    min_run_ms: int = 200,
    gap_ms: int = 40,
    level_ratio: float = 0.04,
    floor_db: float = -50.0,
    fade_ms: int = 6,
) -> bytes:
    """Silence breathy, unvoiced runs in synthesized speech.

    The clone model learned breath noises from the reference recording and
    reproduces them after (and sometimes between) words. A breath is broadband
    noise: most adjacent samples cross zero, so its zero-crossing rate is high
    (~0.3) while voiced speech stays near 0.02. Real fricatives are also
    unvoiced but short, so only runs longer than ``min_run_ms`` are removed.

    The run is replaced with silence (short fades on both edges to avoid
    clicks). ``trim_trailing_silence`` then drops a silent tail, while a breath
    between words becomes a natural short pause.
    """
    decoded = _decode_pcm16(payload)
    if decoded is None:
        return payload
    rate, channels, width, samples = decoded
    frame_len = max(1, int(rate * 0.02)) * channels
    if len(samples) < frame_len * 3:
        return payload

    frame_rms: list[float] = []
    frame_zcr: list[float] = []
    for start in range(0, len(samples) - frame_len + 1, frame_len):
        chunk = samples[start : start + frame_len]
        frame_rms.append(math.sqrt(sum(v * v for v in chunk) / len(chunk)))
        crossings = 0
        previous = chunk[0]
        for value in chunk[1:]:
            if (previous >= 0) != (value >= 0):
                crossings += 1
            previous = value
        frame_zcr.append(crossings / frame_len)

    ordered = sorted(frame_rms)
    p90 = ordered[min(len(ordered) - 1, int(len(ordered) * 0.9))]
    level_floor = max(p90 * level_ratio, 32768.0 * 10 ** (floor_db / 20.0))
    flags = [
        rms >= level_floor and zcr >= zcr_threshold
        for rms, zcr in zip(frame_rms, frame_zcr)
    ]

    gap_frames = max(0, gap_ms // 20)
    runs: list[list[int]] = []
    start_frame: int | None = None
    last_active = 0
    for index, active in enumerate(flags):
        if active:
            if start_frame is None:
                start_frame = index
            last_active = index
        elif start_frame is not None and index - last_active > gap_frames:
            runs.append([start_frame, last_active + 1])
            start_frame = None
    if start_frame is not None:
        runs.append([start_frame, last_active + 1])

    min_frames = max(1, min_run_ms // 20)
    out = samples
    fade = max(1, int(rate * fade_ms / 1000)) * channels
    for run_start, run_end in runs:
        if run_end - run_start < min_frames:
            continue
        begin = run_start * frame_len
        end = min(len(out), run_end * frame_len)
        if end <= begin:
            continue
        if out is samples:
            out = array("h", samples)
        # Fade the preceding samples down and the following samples up.
        for index in range(max(0, begin - fade), begin):
            gain = (begin - index) / fade
            out[index] = int(out[index] * gain)
        for index in range(begin, end):
            out[index] = 0
        for index in range(end, min(len(out), end + fade)):
            gain = (index - end) / fade
            out[index] = int(out[index] * gain)

    if out is samples:
        return payload
    return _encode_pcm16(out, rate=rate, channels=channels, width=width)


def analyze_reference(payload: bytes) -> dict[str, Any] | None:
    """Measure a clone reference and explain why it might clone badly.

    Returns ``None`` when the payload is not readable 16-bit PCM WAV. The
    metrics are deliberately simple (level, clipping, silence, noise floor)
    because the goal is actionable guidance, not a full speech-quality model.
    """
    repaired = repair_wav_header(payload)
    try:
        with wave.open(io.BytesIO(repaired), "rb") as handle:
            rate = handle.getframerate()
            channels = handle.getnchannels()
            width = handle.getsampwidth()
            frames = handle.getnframes()
            raw = handle.readframes(frames)
    except (wave.Error, EOFError, ValueError):
        return None

    if width != 2 or rate <= 0 or channels <= 0 or not raw:
        return None

    samples = array("h")
    samples.frombytes(raw[: len(raw) - (len(raw) % 2)])
    if sys.byteorder == "big":
        samples.byteswap()
    if channels > 1:
        samples = samples[::channels]

    count = len(samples)
    if count == 0:
        return None

    seconds = count / rate
    # Pure-Python stats: subsample so a long, high-rate file stays fast.
    step = max(1, count // 30_000)
    window = samples[::step]
    n = len(window)

    peak = 0
    total = 0
    total_sq = 0.0
    clipped = 0
    for value in window:
        magnitude = value if value >= 0 else -value
        if magnitude > peak:
            peak = magnitude
        if magnitude >= 32700:
            clipped += 1
        total += value
        total_sq += value * value

    dc = total / n
    rms = math.sqrt(total_sq / n)

    # Frame energy at ~20 ms resolution (in subsampled terms).
    frame_len = max(1, int(rate * 0.02 / step))
    frame_rms = [
        math.sqrt(sum(v * v for v in window[start : start + frame_len]) / frame_len)
        for start in range(0, n - frame_len + 1, frame_len)
    ]
    frame_rms.sort()

    def percentile(fraction: float) -> float:
        index = min(len(frame_rms) - 1, max(0, int(len(frame_rms) * fraction)))
        return frame_rms[index]

    silence_floor = 32768 * 10 ** (-45 / 20)
    silence_ratio = (
        sum(1 for value in frame_rms if value < silence_floor) / len(frame_rms)
        if frame_rms
        else 0.0
    )
    noise = percentile(0.10)
    speech = percentile(0.90)
    snr = 20.0 * math.log10(max(speech, 1.0) / max(noise, 1.0)) if speech > 0 else 0.0

    peak_db = _db(peak)
    rms_db = _db(rms)
    noise_db = _db(noise)
    speech_db = _db(speech)
    clipping_percent = 100.0 * clipped / n
    dc_ratio = abs(dc) / 32768.0

    score = 100.0
    warnings: list[str] = []

    if clipping_percent >= 0.5:
        score -= min(60.0, clipping_percent * 12)
        warnings.append(
            f"About {clipping_percent:.1f}% of the samples are clipped. "
            "Lower the input level or move further from the microphone."
        )
    if rms_db < -38:
        score -= 30
        warnings.append(
            "The recording is very quiet. Speak closer to the microphone so the "
            "clone is not built from mostly noise."
        )
    elif rms_db < -32:
        score -= 12
        warnings.append("The recording is quiet. A louder, close-mic take clones better.")
    if silence_ratio > 0.75:
        score -= 40
        warnings.append(
            f"{silence_ratio * 100:.0f}% of the clip is silence. Speak continuously "
            "for 15–30 seconds so the clone has enough voice to learn."
        )
    elif silence_ratio > 0.55:
        score -= 20
        warnings.append(
            f"{silence_ratio * 100:.0f}% of the clip is silence. Long pauses make the "
            "clone less accurate."
        )
    # Clipping inflates the apparent noise floor (harmonics), so the SNR
    # estimate is only meaningful on unclipped audio.
    if clipping_percent < 1.0:
        if snr < 6:
            score -= 35
            warnings.append(
                "Background noise is almost as loud as the voice. Record in a quiet "
                "room with no fan, traffic, music, or TV."
            )
        elif snr < 12:
            score -= 18
            warnings.append(
                "Background noise is high relative to the voice. A quieter room or a "
                "headset microphone will improve the clone."
            )
    if dc_ratio > 0.02:
        score -= 8
        warnings.append(
            "The recording has low-frequency rumble. Remove fan/AC noise or use a "
            "headset microphone."
        )
    if channels > 1:
        score -= 5
        warnings.append("The file is stereo. Mono is preferred for voice cloning.")
    if rate < 16000:
        score -= 15
        warnings.append(
            f"The sample rate is only {rate} Hz. Export at 16 kHz or higher."
        )
    if seconds < 5:
        score -= 40
        warnings.append("The clip is shorter than 5 seconds of speech.")
    elif seconds < 10:
        score -= 15
        warnings.append(
            "Short reference. Aim for 15–30 seconds of continuous speech."
        )

    score = max(0, min(100, round(score)))
    if score >= 85 and not warnings:
        verdict = "good"
    elif score >= 60:
        verdict = "fair"
    else:
        verdict = "poor"

    return {
        "verdict": verdict,
        "score": score,
        "warnings": warnings,
        "metrics": {
            "rate": rate,
            "channels": channels,
            "seconds": round(seconds, 2),
            "peak_dbfs": round(peak_db, 1),
            "rms_dbfs": round(rms_db, 1),
            "speech_dbfs": round(speech_db, 1),
            "noise_dbfs": round(noise_db, 1),
            "snr_db": round(snr, 1),
            "silence_percent": round(silence_ratio * 100, 1),
            "clipping_percent": round(clipping_percent, 2),
            "dc_offset_percent": round(dc_ratio * 100, 2),
        },
    }

