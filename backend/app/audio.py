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
    keep_ms: int = 80,
) -> bytes:
    """Keep the clearest rendition of a short cloned phrase.

    The clone model renders short replies more than once: a quiet first copy,
    a pause, then a louder copy (``Yes. ... Yes.``). It can also prepend a soft
    breath or ``uh``. Counting speech groups as words (the previous approach)
    then treats the quiet pre-roll as the first word and cuts the real phrase,
    which is why a tap could play only a stray ``ahh``.

    Word counting cannot separate a pre-roll from a real word, so it is not
    used. Instead the clip is split into renditions at long pauses and the
    rendition with the most speech energy is kept. A real word is never cut
    away from its own phrase because the guard requires the best rendition to
    be long enough for the text; longer replies are left untouched.
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

    stats, frame_len, _ = _frame_stats(samples, rate, channels)
    if len(stats) < 2:
        return payload
    analysis = _speech_groups(samples, rate, channels, ratio=0.10, floor_db=-50.0)
    if analysis is None:
        return payload
    groups, _p90, _frame_len, _count = analysis
    if len(groups) < 2:
        return payload

    # Merge groups that belong to the same rendition. The clone puts a long
    # pause between copies (>=240 ms in practice); words inside one copy are
    # usually closer than 180 ms.
    rendition_gap = max(1, 180 // 20)
    renditions: list[list[int]] = []
    for start, end in groups:
        if renditions and start - renditions[-1][1] < rendition_gap:
            renditions[-1][1] = end
        else:
            renditions.append([start, end])
    if len(renditions) < 2:
        return payload

    features = [_group_features(stats, (start, end)) for start, end in renditions]
    best = max(features, key=lambda item: sum(
        stats[index][0] ** 2 for index in range(int(item["start"]), int(item["end"]))
    ))
    # A stray click or breath is not a phrase. Require roughly 80 ms of speech
    # per word (120 ms floor) before trusting the strongest rendition.
    min_frames = max(6, (80 * len(words)) // 20)
    if int(best["frames"]) < min_frames:
        return payload

    keep_frames = max(0, keep_ms // 20)
    start_sample = int(best["start"]) * frame_len
    end_sample = min(
        len(samples),
        (int(best["end"]) + keep_frames) * frame_len,
    )
    if start_sample <= 0 and end_sample >= len(samples):
        return payload
    if end_sample - start_sample <= 0:
        return payload

    out = array("h", samples[start_sample:end_sample])
    # A cut at a rendition start can click; ease in over 8 ms.
    fade = min(int(rate * 0.008) * channels, len(out) // channels)
    for index in range(fade):
        out[index] = int(out[index] * (index / fade))
    return _encode_pcm16(out, rate=rate, channels=channels, width=width)


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
    keep_ms: int = 80,
    lead_ms: int = 40,
    fade_ms: int = 12,
    ratio: float = 0.07,
    floor_db: float = -52.0,
) -> bytes:
    """Trim near-silent edges without clipping soft consonants or keeping trailing breaths.

    Uses a sensitive threshold relative to speech level so initial and final
    consonants (like 'th', 'f', 's', 't') survive, with a smooth fade-out.
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

    frame_len = max(1, int(rate * 0.015)) * channels
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
    lead_margin_ms: int = 25,
    tail_margin_ms: int = 40,
    min_pause_ms: int = 220,
    fade_ms: int = 10,
    attenuation: float = 0.05,
    **_kwargs: Any,
) -> bytes:
    """Silence breathy, unvoiced sounds in pauses and boundaries.

    Breaths occur before speech begins (leading inhale), in pauses between phrases
    (inter-speech breathing), and after the utterance ends (trailing exhale/sigh).
    Unlike speech consonants (such as /s/, /sh/, /th/) which have high energy or
    are tightly attached to voiced vowels within words, breath sounds sit isolated
    within pause gaps and boundaries and lack vocal periodicity.

    This function identifies true speech cores (voiced phonemes and primary syllables)
    and expands them to encompass all contiguous consonants and plosives (up to
    400 ms, so long fricatives survive). Breath in long inter-phrase pauses
    (>= 220 ms) is attenuated to about -26 dB with smooth crossfades, and
    leading/trailing breaths are cleanly trimmed with soft edge fades. Speech
    consonants remain 100% intact.
    """
    decoded = _decode_pcm16(payload)
    if decoded is None:
        return payload
    rate, channels, width, samples = decoded

    frame_ms = 10
    frame_len = max(1, int(rate * frame_ms / 1000)) * channels
    n_frames = len(samples) // frame_len
    if n_frames < 6:
        return payload

    rms_list: list[float] = []
    ac_list: list[float] = []

    min_lag = max(1, int(rate / 400)) * channels
    max_lag = max(min_lag + 1, int(rate / 70)) * channels

    for i in range(n_frames):
        chunk = samples[i * frame_len : (i + 1) * frame_len]
        mono = chunk[::channels] if channels > 1 else chunk
        m_len = len(mono)
        if m_len < 2:
            rms_list.append(0.0)
            ac_list.append(0.0)
            continue

        rms = math.sqrt(sum(s * s for s in mono) / m_len)
        m = sum(mono) / m_len
        centered = [x - m for x in mono]
        en = sum(x * x for x in centered)
        best_ac = 0.0
        if en > 100:
            limit = min(max_lag // channels, m_len // 2)
            start_lag = min_lag // channels
            for lag in range(start_lag, limit):
                c = sum(centered[j] * centered[j + lag] for j in range(m_len - lag))
                nc = c / en
                if nc > best_ac:
                    best_ac = nc

        rms_list.append(rms)
        ac_list.append(best_ac)

    sorted_rms = sorted(rms_list)
    p90 = sorted_rms[min(len(sorted_rms) - 1, int(len(sorted_rms) * 0.9))]
    noise_floor = sorted_rms[min(len(sorted_rms) - 1, int(len(sorted_rms) * 0.08))]
    if p90 <= 0:
        return payload

    # 1. Identify primary speech cores. A loud frame is not automatically
    #    speech: a breath or sigh can reach 20% of the speech level and used to
    #    be marked as a core, so the exact sound the clone appends after a
    #    phrase survived every later pass. Require periodicity for anything
    #    below full speech level; attached consonants are recovered by the
    #    expansion step below.
    is_core = [
        (r >= 0.09 * p90 and ac >= 0.28) or (r >= 0.50 * p90 and ac >= 0.18)
        for r, ac in zip(rms_list, ac_list)
    ]
    if not any(is_core):
        return payload

    # 2. Expand speech boundaries forward and backward to capture connected
    #    consonants. The window is generous but not unlimited: a 400 ms reach
    #    swallowed whole breaths that followed a word, so they were treated as
    #    speech and kept. 250 ms still covers normal /s/, /sh/ and /f/.
    is_speech = list(is_core)
    max_consonant_frames = max(1, int(250 / frame_ms))
    consonant_floor = max(p90 * 0.025, noise_floor * 1.2, 32768.0 * 10 ** (-52.0 / 20.0))

    for i in range(n_frames):
        if is_core[i]:
            j = i + 1
            ext = 0
            while j < n_frames and ext < max_consonant_frames:
                if is_speech[j]:
                    break
                if rms_list[j] >= consonant_floor:
                    is_speech[j] = True
                    ext += 1
                    j += 1
                else:
                    break

    for i in range(n_frames - 1, -1, -1):
        if is_core[i]:
            j = i - 1
            ext = 0
            while j >= 0 and ext < max_consonant_frames:
                if is_speech[j]:
                    break
                if rms_list[j] >= consonant_floor:
                    is_speech[j] = True
                    ext += 1
                    j -= 1
                else:
                    break

    # 3. Group speech spans with a 60 ms bridge across short stops
    bridge_frames = max(1, int(60 / frame_ms))
    spans: list[tuple[int, int]] = []
    start = None
    for i, sp in enumerate(is_speech):
        if sp and start is None:
            start = i
        elif not sp and start is not None:
            spans.append((start, i))
            start = None
    if start is not None:
        spans.append((start, n_frames))

    if not spans:
        return payload

    merged: list[tuple[int, int]] = [spans[0]]
    for s, e in spans[1:]:
        if s - merged[-1][1] <= bridge_frames:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))

    speech_spans = [span for span in merged if (span[1] - span[0]) * frame_ms >= 30]
    if not speech_spans:
        return payload

    # 4. Trim leading breath and trailing breath/sigh. Boundaries are anchored
    #    to the first and last voiced core, not to the expanded span: expansion
    #    deliberately reaches into consonants, and at the clip edges that reach
    #    is what preserved a model-generated inhale/exhale. A short consonant
    #    allowance is kept so a real initial or final /s/ is not clipped.
    core_indices = [index for index, core in enumerate(is_core) if core]
    first_core, last_core = core_indices[0], core_indices[-1]
    lead_allowed = max(1, int(120 / frame_ms))
    tail_allowed = max(1, int(200 / frame_ms))
    lead_start_frame = max(
        0,
        first_core
        - lead_allowed
        - max(1, int(lead_margin_ms / frame_ms)),
    )
    tail_end_frame = min(
        n_frames,
        last_core
        + tail_allowed
        + max(1, int(tail_margin_ms / frame_ms)),
    )

    fade_frames = max(1, int(rate * fade_ms / 1000)) * channels
    out = array("h", samples[lead_start_frame * frame_len : tail_end_frame * frame_len])
    offset_frames = lead_start_frame

    # Smooth leading edge fade-in
    fade_in_len = min(fade_frames, len(out) // channels)
    for i in range(fade_in_len):
        gain = i / fade_in_len
        for c in range(channels):
            out[i * channels + c] = int(out[i * channels + c] * gain)

    # Smooth trailing edge fade-out
    fade_out_len = min(fade_frames, len(out) // channels)
    for i in range(fade_out_len):
        gain = i / fade_out_len
        idx = len(out) - 1 - (i * channels)
        for c in range(channels):
            out[idx - c] = int(out[idx - c] * gain)

    # 5. Silence all breath sounds in inter-speech pauses
    pause_margin_frames = max(1, int(20 / frame_ms))
    for idx in range(len(speech_spans) - 1):
        prev_end = speech_spans[idx][1] - offset_frames
        next_start = speech_spans[idx + 1][0] - offset_frames

        gap_ms = (next_start - prev_end) * frame_ms
        if gap_ms >= min_pause_ms:
            g_start = prev_end + pause_margin_frames
            g_end = next_start - pause_margin_frames
            if g_end > g_start:
                s_start = g_start * frame_len
                s_end = g_end * frame_len
                for i in range(fade_frames):
                    pos = s_start - fade_frames + i
                    if 0 <= pos < len(out):
                        gain = 1.0 - (i / fade_frames)
                        out[pos] = int(out[pos] * gain)
                for pos in range(s_start, min(len(out), s_end)):
                    out[pos] = int(out[pos] * attenuation)
                for i in range(fade_frames):
                    pos = s_end + i
                    if 0 <= pos < len(out):
                        gain = i / fade_frames
                        out[pos] = int(out[pos] * gain)

    return _encode_pcm16(out, rate=rate, channels=channels, width=width)


def enhance_speech_clarity(
    payload: bytes,
    *,
    hp_cutoff_hz: float = 85.0,
    presence_freq_hz: float = 3200.0,
    presence_gain_db: float = 2.2,
    target_peak_ratio: float = 0.89,
) -> bytes:
    """Enhance voice clarity, remove sub-bass rumble, and normalize speech volume.

    1. Applies a 2nd-order Butterworth high-pass filter at 85 Hz to remove DC bias
       and low-end muddy synthesizer rumble.
    2. Applies a vocal presence peaking filter (+2.2 dB at 3200 Hz, Q=1.0) to bring
       formant intelligibility and consonant articulation forward.
    3. Peak-normalizes to -1.0 dBFS (~29,200 amplitude out of 32,768) so spoken
       speech is consistently loud, clear, and confident without clipping.
    """
    decoded = _decode_pcm16(payload)
    if decoded is None:
        return payload
    rate, channels, width, samples = decoded
    if len(samples) < 100:
        return payload

    # 1. 2nd-order High-Pass Filter (85 Hz)
    hp_f0 = max(20.0, min(hp_cutoff_hz, rate * 0.45))
    hp_q = 0.707
    w0 = 2.0 * math.pi * hp_f0 / rate
    alpha = math.sin(w0) / (2.0 * hp_q)
    b0 = (1.0 + math.cos(w0)) / 2.0
    b1 = -(1.0 + math.cos(w0))
    b2 = (1.0 + math.cos(w0)) / 2.0
    a0 = 1.0 + alpha
    a1 = -2.0 * math.cos(w0)
    a2 = 1.0 - alpha

    hp_b0, hp_b1, hp_b2 = b0 / a0, b1 / a0, b2 / a0
    hp_a1, hp_a2 = a1 / a0, a2 / a0

    # 2. Vocal presence peaking filter (3200 Hz, +2.2 dB, Q=1.0)
    f_p = max(200.0, min(presence_freq_hz, rate * 0.45))
    A = 10.0 ** (presence_gain_db / 40.0)
    w0_p = 2.0 * math.pi * f_p / rate
    alpha_p = math.sin(w0_p) / 2.0

    pb0 = 1.0 + alpha_p * A
    pb1 = -2.0 * math.cos(w0_p)
    pb2 = 1.0 - alpha_p * A
    pa0 = 1.0 + alpha_p / A
    pa1 = -2.0 * math.cos(w0_p)
    pa2 = 1.0 - alpha_p / A

    pk_b0, pk_b1, pk_b2 = pb0 / pa0, pb1 / pa0, pb2 / pa0
    pk_a1, pk_a2 = pa1 / pa0, pa2 / pa0

    out = array("h", [0] * len(samples))

    for ch in range(channels):
        hp_x1 = hp_x2 = hp_y1 = hp_y2 = 0.0
        pk_x1 = pk_x2 = pk_y1 = pk_y2 = 0.0

        for idx in range(ch, len(samples), channels):
            x = float(samples[idx])
            # High-pass filter
            y_hp = hp_b0 * x + hp_b1 * hp_x1 + hp_b2 * hp_x2 - hp_a1 * hp_y1 - hp_a2 * hp_y2
            hp_x2, hp_x1 = hp_x1, x
            hp_y2, hp_y1 = hp_y1, y_hp

            # Presence boost filter
            y_pk = pk_b0 * y_hp + pk_b1 * pk_x1 + pk_b2 * pk_x2 - pk_a1 * pk_y1 - pk_a2 * pk_y2
            pk_x2, pk_x1 = pk_x1, y_hp
            pk_y2, pk_y1 = pk_y1, y_pk

            val = int(y_pk)
            out[idx] = max(-32768, min(32767, val))

    # 3. Peak normalization to target_peak_ratio (-1.0 dBFS)
    peak = max(abs(s) for s in out) if out else 0
    if peak > 0:
        target_val = int(32768 * target_peak_ratio)
        gain = target_val / peak
        gain = min(gain, 4.0)
        for i in range(len(out)):
            val = int(out[i] * gain)
            out[i] = max(-32768, min(32767, val))

    return _encode_pcm16(out, rate=rate, channels=channels, width=width)


def sanitize_reference_audio(payload: bytes) -> bytes:
    """Sanitize reference recordings before generating voice clone embeddings.

    Removes DC bias, silences breath noises in pauses and edges, eliminates sub-bass
    rumble, and peak-normalizes. This prevents the cloning model from learning
    the speaker's breath patterns.
    """
    cleaned = suppress_breaths(
        payload,
        min_pause_ms=200,
        lead_margin_ms=20,
        tail_margin_ms=30,
        attenuation=0.02,
    )
    trimmed = trim_trailing_silence(cleaned, keep_ms=80, lead_ms=30, ratio=0.07)
    return enhance_speech_clarity(trimmed, hp_cutoff_hz=80.0, presence_gain_db=1.5, target_peak_ratio=0.92)


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

