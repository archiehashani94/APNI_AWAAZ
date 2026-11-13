from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

import httpx
import websockets

from app.audio import (
    enhance_speech_clarity,
    repair_wav_header,
    suppress_breaths,
    trim_boundary_fillers,
    trim_extra_groups,
    trim_repeated_tail,
    trim_trailing_silence,
    wav_info,
)
from app.config import settings

log = logging.getLogger("apni_awaaz.gnani")

CREDIT_HEADER_CANDIDATES = (
    "x-credits-used",
    "x-credit-used",
    "x-credits-consumed",
    "x-gnani-credits",
    "x-remaining-credits",
)


class BudgetExceeded(Exception):
    pass


class GnaniError(Exception):
    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


def upstream_error(response: httpx.Response) -> str:
    """Extract the actionable message from a Gnani error response.

    Gnani returns errors shaped like
    {"detail": {"error_code": "INVALID_API_KEY", "message": "Unauthorized"}}.
    Keeping that detail makes a 401 or a rejected-audio error diagnosable
    instead of surfacing a generic "cloning failed".
    """
    try:
        payload = response.json()
    except ValueError:
        return (response.text or "").strip()[:300]

    detail = payload.get("detail") if isinstance(payload, dict) else None
    if isinstance(detail, dict):
        code = detail.get("error_code") or detail.get("code")
        message = detail.get("message") or detail.get("detail")
        if code and message:
            return f"{code}: {message}"
        if code or message:
            return str(code or message)
    if isinstance(detail, str):
        return detail[:300]
    if isinstance(payload, dict):
        return str(payload.get("message") or payload.get("error") or payload)[:300]
    return str(payload)[:300]


@dataclass
class CreditEvent:
    ts: float
    service: str
    cached: bool
    estimated: float
    billed: float | None
    headers: dict[str, str]
    total_after: float


@dataclass
class CreditLedger:
    used: float = 0.0
    budget: float = 5000.0
    events: list[CreditEvent] = field(default_factory=list)

    def snapshot(self) -> dict[str, Any]:
        return {
            "used": round(self.used, 4),
            "budget": self.budget,
            "remaining": round(max(self.budget - self.used, 0.0), 4),
            "events": [
                {
                    "ts": e.ts,
                    "service": e.service,
                    "cached": e.cached,
                    "estimated": e.estimated,
                    "billed": e.billed,
                    "total_after": e.total_after,
                }
                for e in self.events[-50:]
            ],
        }


def _header_credits(headers: httpx.Headers) -> tuple[float | None, dict[str, str]]:
    found: dict[str, str] = {}
    billed: float | None = None
    for key, value in headers.items():
        lk = key.lower()
        if "credit" in lk:
            found[lk] = value
            if lk in CREDIT_HEADER_CANDIDATES:
                try:
                    billed = float(value)
                except ValueError:
                    pass
    return billed, found


def tts_cache_key(text: str, voice: str, emotion: str, model: str) -> str:
    raw = f"{text}\0{voice}\0{emotion}\0{model}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


# TODO: vachana-vc-v1 (docs.gnani.ai/api/VC/vc-inference) has no emotion field.
# Timbre v3.0 tags are inline in `text` and documented as timbre-v3.0 only
# (docs.gnani.ai/api/TTS/tts-audio-tags). Names confirmed in examples:
# <laugh> <giggle> <whispers> <shouting> <pauses>
# Catalog table HTML was stripped; do not invent tags. Clone path uses plain text.
TIMBRE_EMOTION_PREFIX = {
    "warm": "",
    "urgent": "<shouting> ",
    "playful": "<giggle> ",
    "calm": "<whispers> ",
}

# Timbre v2.5 keeps its own voice catalog: a v3.0 voice such as Shivani is
# rejected with 400, and each v2.5 voice has one preferred language. These are
# the documented pairings used when no cloned voice is active.
TIMBRE_V25_VOICES = {
    "hi-IN": "Nalini",
    "en-IN": "Kaveri",
    "hi-en": "Poorvi",
    "ta-IN": "Noopur",
    "te-IN": "Suhana",
    "kn-IN": "Saanvi",
    "ml-IN": "Reshma",
    "mr-IN": "Zahira",
    "bn-IN": "Kirra",
    "gu-IN": "Falak",
    "pa-IN": "Mehuli",
}


def has_devanagari(text: str) -> bool:
    return any("\u0900" <= ch <= "\u097f" for ch in text)


_TRAILING_PUNCT = ".!?;:।॥…"
_TRAILING_BRACKETS = "\"'”’»"


def natural_text(text: str) -> str:
    """Clean whitespace and give short replies sentence punctuation.

    TTS prosody is noticeably better when a reply ends in punctuation instead
    of a bare word. Keep existing punctuation and closing quotes intact.
    """
    clean = re.sub(r"\s+", " ", text).strip()
    if not clean:
        return clean
    closing = ""
    while clean and clean[-1] in _TRAILING_BRACKETS:
        closing = clean[-1] + closing
        clean = clean[:-1].rstrip()
    if clean and clean[-1] not in _TRAILING_PUNCT:
        clean += "।" if has_devanagari(clean) else "."
    return f"{clean}{closing}"


class GnaniClient:
    """Single entry point for every Gnani call. Logs credits, caches TTS, enforces budget."""

    def __init__(self) -> None:
        self.ledger = CreditLedger(budget=settings.credit_budget)
        self._tts_cache: dict[str, bytes] = {}
        self._http = httpx.Client(timeout=90.0)
        # Gnani allows only one TTS inference at a time. Serialising requests
        # here avoids a 429 (and the wrong-voice stock fallback) when two taps
        # overlap or a retry races the previous call.
        self._tts_lock = threading.Lock()

    def _auth(self) -> dict[str, str]:
        if not settings.gnani_api_key:
            log.error(
                "503: GNANI_API_KEY is missing — no upstream Gnani call. "
                "Set it in backend/.env and restart uvicorn."
            )
            raise GnaniError("GNANI_API_KEY is not set", 503)
        return {"X-API-Key-ID": settings.gnani_api_key}

    def _guard(self, estimated: float) -> None:
        if self.ledger.used + estimated > self.ledger.budget:
            raise BudgetExceeded(
                f"Credit budget {self.ledger.budget} would be exceeded "
                f"(used {self.ledger.used}, next {estimated})"
            )

    def _record(
        self,
        service: str,
        estimated: float,
        *,
        cached: bool = False,
        failed: bool = False,
        headers: httpx.Headers | None = None,
    ) -> None:
        billed, extra = _header_credits(headers) if headers is not None else (None, {})
        charge = 0.0 if (cached or failed) else (billed if billed is not None else estimated)
        self.ledger.used += charge
        event = CreditEvent(
            ts=time.time(),
            service=service,
            cached=cached,
            estimated=estimated,
            billed=billed,
            headers=extra,
            total_after=self.ledger.used,
        )
        self.ledger.events.append(event)
        log.info(
            "gnani %s cached=%s estimated=%.4f billed=%s total=%.4f headers=%s",
            service,
            cached,
            estimated,
            billed,
            self.ledger.used,
            extra,
        )

    def create_embeddings(self, audio: bytes, filename: str, content_type: str) -> dict[str, Any]:
        estimated = 50.0
        self._guard(estimated)
        files = {"audio_file": (filename, audio, content_type or "application/octet-stream")}
        url = f"{settings.gnani_base_url}/api/v1/tts/voice-clone/embeddings"
        log.info("Gnani POST %s (voice-clone embeddings)", url)
        response = self._http.post(url, headers=self._auth(), files=files)
        self._record("voice-clone-embeddings", estimated, headers=response.headers)
        if response.status_code >= 400:
            detail = upstream_error(response)
            log.error(
                "embeddings failed: POST %s status=%s detail=%s",
                url,
                response.status_code,
                detail,
            )
            raise GnaniError(
                f"Gnani embeddings failed ({response.status_code})"
                + (f": {detail}" if detail else ""),
                response.status_code,
            )
        payload = response.json()
        data = payload.get("data") or payload
        embedding = data.get("voice_clone_embedding") or data.get("speaker_embedding") or data
        if "embedding" not in embedding:
            raise GnaniError(f"Unexpected embeddings response: {payload}")
        return {
            "embedding": embedding["embedding"],
            "shape": embedding.get("shape", [1, 768]),
            "dtype": embedding.get("dtype", "torch.bfloat16"),
        }

    def synthesize(
        self,
        text: str,
        *,
        speaker_embedding: dict[str, Any] | None,
        emotion: str = "warm",
        language: str = "auto",
        allow_stock_fallback: bool = True,
    ) -> tuple[bytes, str, bool]:
        """Returns (wav_bytes, voice_label, used_stock_fallback). Never call from a render loop."""
        clean = natural_text(text)
        if not clean:
            raise GnaniError("Empty text", 400)

        if speaker_embedding:
            model = "vachana-vc-v1"
            voice_id = "clone:" + hashlib.sha256(
                str(speaker_embedding.get("embedding", "")).encode()
            ).hexdigest()[:16]
            # Plain text: VC docs list text + model + audio_config + speaker_embedding only.
            synth_text = clean
            sample_rate = settings.clone_tts_sample_rate
        else:
            model = settings.stock_tts_model
            if model == "timbre-v2.5":
                # v2.5 has one preferred language per voice. "auto" is resolved
                # from the script so the voice and language always agree.
                resolved = language
                if resolved not in TIMBRE_V25_VOICES:
                    resolved = "hi-IN" if has_devanagari(clean) else "en-IN"
                voice_id = TIMBRE_V25_VOICES[resolved]
                language = resolved
                prefix = ""
            else:
                voice_id = settings.stock_voice
                prefix = TIMBRE_EMOTION_PREFIX.get(emotion, "")
            synth_text = f"{prefix}{clean}" if prefix else clean
            sample_rate = settings.stock_tts_sample_rate

        key = tts_cache_key(synth_text, voice_id, emotion, model)
        cached = self._tts_cache.get(key)
        if cached:
            self._record("tts", 0.0, cached=True)
            return cached, voice_id, speaker_embedding is None

        estimated = max(len(synth_text), 1)
        self._guard(estimated)

        body: dict[str, Any] = {
            "text": synth_text,
            "model": model,
            "audio_config": {
                "sample_rate": sample_rate,
                "num_channels": 1,
                "sample_width": 2,
                "encoding": "linear_pcm",
                "container": "wav",
            },
        }
        if speaker_embedding:
            body["speaker_embedding"] = {
                "embedding": speaker_embedding["embedding"],
                "shape": speaker_embedding.get("shape", [1, 768]),
                "dtype": speaker_embedding.get("dtype", "torch.bfloat16"),
            }
        else:
            body["voice"] = voice_id
            body["language"] = language
            body["speed"] = 1.0

        url = f"{settings.gnani_base_url}/api/v1/tts/inference"
        log.info("Gnani POST %s (tts model=%s)", url, model)
        headers = {**self._auth(), "Content-Type": "application/json"}
        response: httpx.Response | None = None
        # Retry a 429 instead of instantly falling back to the stock voice. The
        # account allows one active inference, and a short cooldown is common
        # after a burst of taps.
        with self._tts_lock:
            for attempt in range(4):
                response = self._http.post(url, headers=headers, json=body)
                if response.status_code != 429 or attempt == 3:
                    break
                wait = 1.5 * (2**attempt)
                log.warning(
                    "TTS rate-limited (attempt %s/4); retrying in %.1fs",
                    attempt + 1,
                    wait,
                )
                time.sleep(wait)

        assert response is not None
        self._record(
            "tts",
            estimated,
            failed=response.status_code >= 400,
            headers=response.headers,
        )

        if (
            response.status_code >= 400
            and speaker_embedding
            and allow_stock_fallback
            and response.status_code != 429
        ):
            log.warning(
                "clone TTS failed status=%s detail=%s; falling back to stock Timbre",
                response.status_code,
                upstream_error(response),
            )
            return self.synthesize(
                clean,
                speaker_embedding=None,
                emotion=emotion,
                language=language,
                allow_stock_fallback=False,
            )
        if response.status_code >= 400:
            detail = upstream_error(response)
            log.error(
                "upstream TTS failed: POST %s status=%s model=%s detail=%s",
                url,
                response.status_code,
                model,
                detail,
            )
            raise GnaniError(
                f"Gnani TTS failed ({response.status_code})"
                + (f": {detail}" if detail else ""),
                503 if response.status_code >= 500 else response.status_code,
            )

        audio = response.content
        repaired = repair_wav_header(audio)
        if repaired != audio:
            log.warning(
                "Repaired sentinel WAV header from Gnani (%s bytes)",
                len(audio),
            )
            audio = repaired
        cleaned = suppress_breaths(audio)
        if cleaned != audio:
            log.info("Silenced breath noise in %s-byte TTS clip", len(audio))
            audio = cleaned
        trimmed = trim_extra_groups(audio, clean)
        if trimmed != audio:
            log.info("Removed extra speech tail from %s bytes", len(audio) - len(trimmed))
            audio = trimmed
        trimmed = trim_boundary_fillers(audio)
        if trimmed != audio:
            log.info("Removed ahh/uhh filler from %s bytes", len(audio) - len(trimmed))
            audio = trimmed
        trimmed = trim_repeated_tail(audio, clean)
        if trimmed != audio:
            log.info("Removed repeated phrase from %s bytes", len(audio) - len(trimmed))
            audio = trimmed
        trimmed = trim_trailing_silence(audio)
        if trimmed != audio:
            log.info("Trimmed TTS tail from %s to %s bytes", len(audio), len(trimmed))
            audio = trimmed
        enhanced = enhance_speech_clarity(audio)
        if enhanced != audio:
            log.info("Enhanced speech clarity and normalized audio (%s bytes)", len(enhanced))
            audio = enhanced
        info = wav_info(audio)
        if info:
            log.info(
                "Gnani audio: %.2fs rate=%s channels=%s width=%s model=%s",
                info["seconds"],
                info["rate"],
                info["channels"],
                info["width"],
                model,
            )
        self._tts_cache[key] = audio
        return audio, voice_id, speaker_embedding is None

    async def prisma_stream(
        self,
        pcm_iter: AsyncIterator[bytes | str],
        *,
        sample_rate: int = 16000,
        language: str = "en-IN",
        output_format: str = "verbatim",
    ) -> AsyncIterator[dict[str, Any]]:
        import asyncio

        estimated = 20.0
        self._guard(estimated)
        # Prisma v2.5 is the required model here. It takes one concrete locale
        # per session and only the verbatim/transcribe output styles.
        if output_format not in {"verbatim", "transcribe"}:
            output_format = "verbatim"
        language_code = (language.split(",")[0] or "en-IN").strip()
        headers = {
            "x-api-key-id": settings.gnani_api_key,
            "model": "prisma-v2.5",
            "language_code": language_code,
            "x-sample-rate": str(sample_rate),
            "x-idle-timeout-sec": str(settings.stt_idle_timeout_sec),
            "x-vad-threshold": str(settings.stt_vad_threshold),
            "x-min-speech-ms": str(settings.stt_min_speech_ms),
            "x-min-segment-ms": str(settings.stt_min_segment_ms),
            "x-min-silence-ms": str(settings.stt_min_silence_ms),
            "x-max-speech-s": str(settings.stt_max_speech_s),
            "format": output_format,
        }
        self._record("stt-stream-open", estimated)
        async with websockets.connect(
            settings.gnani_stt_ws,
            additional_headers=headers,
            max_size=8 * 1024 * 1024,
        ) as ws:
            async def sender() -> None:
                async for chunk in pcm_iter:
                    if chunk:
                        await ws.send(chunk)

            send_task = asyncio.create_task(sender())
            try:
                async for raw in ws:
                    if isinstance(raw, bytes):
                        continue
                    try:
                        frame = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    yield frame
            finally:
                send_task.cancel()


gnani = GnaniClient()
