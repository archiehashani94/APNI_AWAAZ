from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

import httpx
import websockets

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


class GnaniClient:
    """Single entry point for every Gnani call. Logs credits, caches TTS, enforces budget."""

    def __init__(self) -> None:
        self.ledger = CreditLedger(budget=settings.credit_budget)
        self._tts_cache: dict[str, bytes] = {}
        self._http = httpx.Client(timeout=90.0)

    def _auth(self) -> dict[str, str]:
        if not settings.gnani_api_key:
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
        headers: httpx.Headers | None = None,
    ) -> None:
        billed, extra = _header_credits(headers) if headers is not None else (None, {})
        charge = 0.0 if cached else (billed if billed is not None else estimated)
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
        response = self._http.post(url, headers=self._auth(), files=files)
        self._record("voice-clone-embeddings", estimated, headers=response.headers)
        if response.status_code >= 400:
            raise GnaniError(response.text[:500], response.status_code)
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
        clean = text.strip()
        if not clean:
            raise GnaniError("Empty text", 400)

        if speaker_embedding:
            model = "vachana-vc-v1"
            voice_id = "clone:" + hashlib.sha256(
                str(speaker_embedding.get("embedding", "")).encode()
            ).hexdigest()[:16]
            # Plain text: VC docs list text + model + audio_config + speaker_embedding only.
            synth_text = clean
        else:
            model = settings.stock_tts_model
            voice_id = settings.stock_voice
            prefix = TIMBRE_EMOTION_PREFIX.get(emotion, "")
            synth_text = f"{prefix}{clean}" if prefix else clean

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
                "sample_rate": 24000,
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
            body["voice"] = settings.stock_voice
            body["language"] = language
            body["speed"] = 1.0

        url = f"{settings.gnani_base_url}/api/v1/tts/inference"
        response = self._http.post(url, headers={**self._auth(), "Content-Type": "application/json"}, json=body)
        self._record("tts", estimated, headers=response.headers)

        if response.status_code >= 400 and speaker_embedding and allow_stock_fallback:
            log.warning("clone TTS failed (%s); falling back to stock Timbre", response.status_code)
            return self.synthesize(
                clean,
                speaker_embedding=None,
                emotion=emotion,
                language=language,
                allow_stock_fallback=False,
            )
        if response.status_code >= 400:
            raise GnaniError(response.text[:500], response.status_code)

        audio = response.content
        self._tts_cache[key] = audio
        return audio, voice_id, speaker_embedding is None

    async def prisma_stream(
        self, pcm_iter: AsyncIterator[bytes | str]
    ) -> AsyncIterator[dict[str, Any]]:
        import asyncio

        estimated = 20.0
        self._guard(estimated)
        headers = {
            "x-api-key-id": settings.gnani_api_key,
            "model": "prisma-v3.0",
            "language_code": "indic-auto",
            "x-sample-rate": "16000",
            "x-idle-timeout-sec": "300",
            "x-min-silence-ms": "700",
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
