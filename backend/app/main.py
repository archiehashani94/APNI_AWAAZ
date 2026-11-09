from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from contextlib import suppress
from typing import Annotated, Any, AsyncIterator

from fastapi import FastAPI, File, Header, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from app.audio import analyze_reference, sanitize_reference_audio, wav_info
from app.config import BACKEND_DIR, settings
from app.gnani.client import BudgetExceeded, GnaniError, gnani
from app.llm.provider import suggest_with_fallback
from app.recordings import (
    RecordingError,
    delete_recording,
    list_recordings,
    resolve_recording,
    save_recording,
)
from app.store import store
from app.voices import (
    clear_current_voice,
    current_voice_id,
    delete_voice,
    list_voices,
    load_voice,
    save_voice,
    set_current_voice,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("apni_awaaz")

FAVOURITES = [
    {"id": "yes", "text": "Yes", "emotion": "calm"},
    {"id": "no", "text": "No", "emotion": "calm"},
    {"id": "thanks", "text": "Thank you", "emotion": "warm"},
    {"id": "sorry", "text": "Sorry", "emotion": "warm"},
    {"id": "wait", "text": "Please wait", "emotion": "calm"},
    {"id": "repeat", "text": "Please repeat", "emotion": "calm"},
    {"id": "okay", "text": "Okay", "emotion": "calm"},
    {"id": "help", "text": "I need help", "emotion": "urgent"},
    {"id": "water", "text": "Water please", "emotion": "warm"},
    {"id": "love", "text": "I love you", "emotion": "warm"},
    {"id": "hello", "text": "Hello", "emotion": "warm"},
    {"id": "bye", "text": "Bye", "emotion": "warm"},
]

EMOTIONS = {"warm", "urgent", "playful", "calm"}
SPEAK_LANGUAGES = {
    "auto",
    "hi-IN",
    "en-IN",
    "hi-en",
    "ta-IN",
    "te-IN",
    "kn-IN",
    "ml-IN",
    "mr-IN",
    "pa-IN",
    "bn-IN",
    "gu-IN",
}
STT_LANGUAGES = {
    "indic-auto",
    "hi-IN",
    "en-IN",
    "hi-en",
    "ta-IN",
    "te-IN",
    "kn-IN",
    "ml-IN",
    "mr-IN",
    "pa-IN",
    "bn-IN",
    "gu-IN",
}
STT_FORMATS = {"verbatim", "transcribe", "codemix", "transliterate"}


def normalized_language(
    value: str | None,
    allowed: set[str],
    default: str,
    *,
    allow_list: bool = False,
) -> str:
    if not value:
        return default
    candidate = value.strip()
    if candidate in allowed:
        return candidate
    # Prisma v3.0 also accepts up to four comma-separated language codes.
    if allow_list:
        parts = [p.strip() for p in candidate.split(",")]
        if 1 <= len(parts) <= 4 and all(
            p in STT_LANGUAGES and p not in {"indic-auto", "hi-en"} for p in parts
        ):
            return ",".join(parts)
    return default


def wav_duration_seconds(payload: bytes) -> float | None:
    """Return duration for a WAV upload, or None for compressed formats."""
    info = wav_info(payload)
    return info["seconds"] if info else None


INVALID_KEY_HINT = (
    "Gnani rejected the API key (INVALID_API_KEY). Create a new key at "
    "app.gnani.ai/voice, put it in backend/.env as GNANI_API_KEY, then restart "
    "the backend."
)


def enrol_failure_message(exc: GnaniError) -> str:
    """Turn an upstream enrol error into a message the user can act on."""
    status = exc.status_code
    if status in (401, 403):
        return f"{INVALID_KEY_HINT} The stock Timbre voice will not work either until the key is fixed."
    if status in (400, 415, 422):
        return (
            f"Gnani rejected the reference audio ({exc}). Record a clean "
            "20–30 second WAV of one person speaking, then try again."
        )
    if status == 402:
        return "Gnani credits are exhausted. Using a stock Timbre voice until credits are added."
    if status and status >= 500:
        return "Gnani is unavailable right now. Using a stock Timbre voice; try again shortly."
    return f"Cloning failed ({exc}). Using a stock Timbre voice."


app = FastAPI(title="Apni Awaaz")
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list or ["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=[
        "X-AI-Generated",
        "X-Voice-Source",
        "X-Fallback",
        "X-Emotion",
        "X-Audio-Sample-Rate",
        "X-Audio-Seconds",
    ],
)


def session_from(session_id: str | None):
    if not session_id:
        raise HTTPException(400, "Missing X-Session-Id")
    return store.require(session_id)


@app.on_event("startup")
def startup() -> None:
    log.info(
        "config: env_file=%s GNANI_API_KEY=%s LLM_PROVIDER=%r LLM_API_KEY=%s",
        BACKEND_DIR / ".env",
        "set" if settings.gnani_api_key else "missing",
        settings.llm_provider,
        "set" if settings.llm_api_key else "missing",
    )


@app.get("/health")
def health() -> dict[str, Any]:
    key = settings.gnani_api_key
    return {
        "ok": True,
        "has_gnani_key": bool(key),
        # Safe fingerprint so you can confirm which key the process loaded
        # without exposing the secret.
        "gnani_key_tail": key[-4:] if key else None,
        "gnani_key_sha256_10": hashlib.sha256(key.encode()).hexdigest()[:10] if key else None,
        "llm_provider": settings.llm_provider or "canned",
        "has_llm_key": bool(settings.llm_api_key),
        "stt_language": settings.stt_language,
        "clone_tts_sample_rate": settings.clone_tts_sample_rate,
    }


@app.post("/api/session")
def create_session(
    x_client_id: Annotated[str | None, Header()] = None,
) -> dict[str, Any]:
    session = store.create()
    voices = list_voices(x_client_id)
    embedding = load_voice(x_client_id) if voices else None
    if embedding:
        session.speaker_embedding = embedding
        session.clone_ready = True
        session.consent = True
        session.using_stock_voice = False
        saved = current_voice_id(x_client_id)
        known = {item["id"] for item in voices}
        session.voice_id = saved if saved in known else voices[0]["id"]
    return {
        "session_id": session.id,
        "clone_ready": session.clone_ready,
        "consent": session.consent,
        "voice_id": session.voice_id,
    }


@app.get("/api/session")
def read_session(
    x_session_id: Annotated[str | None, Header()] = None,
    x_client_id: Annotated[str | None, Header()] = None,
) -> dict[str, Any]:
    """Report whether a stored session still exists after a backend restart.

    If the session was lost but this browser already enrolled a voice, the
    saved embedding is restored in place. That avoids charging for a new
    embedding (or asking the user to record again) after a restart.
    """
    session = store.get(x_session_id or "")
    if session is None:
        voices = list_voices(x_client_id)
        embedding = load_voice(x_client_id) if voices else None
        if embedding is None:
            raise HTTPException(404, "Session not found")
        session = store.require(x_session_id)
        session.speaker_embedding = embedding
        session.clone_ready = True
        session.consent = True
        session.using_stock_voice = False
        saved = current_voice_id(x_client_id)
        known = {item["id"] for item in voices}
        session.voice_id = saved if saved in known else voices[0]["id"]
    return {
        "session_id": session.id,
        "consent": session.consent,
        "clone_ready": session.clone_ready,
        "using_stock_voice": session.using_stock_voice,
        "voice_id": session.voice_id,
    }


class ConsentBody(BaseModel):
    agreed: bool


@app.post("/api/consent")
def consent(
    body: ConsentBody,
    x_session_id: Annotated[str | None, Header()] = None,
) -> dict[str, bool]:
    session = session_from(x_session_id)
    if not body.agreed:
        raise HTTPException(400, "Consent is required before cloning a voice")
    session.consent = True
    return {"consent": True}


@app.post("/api/enrol")
async def enrol(
    audio: UploadFile = File(...),
    x_session_id: Annotated[str | None, Header()] = None,
    x_client_id: Annotated[str | None, Header()] = None,
    x_recording_name: Annotated[str | None, Header()] = None,
) -> dict[str, Any]:
    session = session_from(x_session_id)
    if not session.consent:
        raise HTTPException(403, "Consent required")
    max_bytes = int(settings.max_upload_mb * 1024 * 1024)
    payload = await audio.read(max_bytes + 1)
    if len(payload) > max_bytes:
        raise HTTPException(413, f"Audio must be under {settings.max_upload_mb:.0f} MB")
    if len(payload) < 1000:
        raise HTTPException(400, "Audio too short — record about 20 seconds")

    duration = wav_duration_seconds(payload)
    if duration is not None:
        info = wav_info(payload) or {}
        rate = info.get("rate")
        channels = info.get("channels")
        log.info(
            "enrol reference: %.2fs rate=%s channels=%s width=%s",
            duration,
            rate,
            channels,
            info.get("width"),
        )
        if rate and rate > 44100:
            raise HTTPException(
                400,
                f"Reference audio is {rate} Hz. Export it at 16 kHz or 44.1 kHz mono, then try again.",
            )
        if channels and channels > 1:
            log.warning("reference has %s channels; Gnani expects mono", channels)
        if duration < settings.min_enrol_seconds:
            raise HTTPException(
                400,
                f"Only {duration:.1f}s of audio. Record at least "
                f"{settings.min_enrol_seconds:.0f} seconds of clear speech.",
            )
        if duration > settings.max_enrol_seconds:
            raise HTTPException(
                400,
                f"Audio is {duration:.1f}s. Use at most "
                f"{settings.max_enrol_seconds:.0f} seconds.",
            )

    quality = analyze_reference(payload)
    if quality:
        log.info(
            "enrol quality: verdict=%s score=%s metrics=%s",
            quality["verdict"],
            quality["score"],
            quality["metrics"],
        )
        for warning in quality["warnings"]:
            log.warning("enrol reference warning: %s", warning)

    # Sanitize reference audio: eliminate all breath sounds, DC rumble, and normalize peak
    clean_payload = sanitize_reference_audio(payload)
    if clean_payload != payload:
        log.info(
            "Sanitized enrolment reference: removed breaths and enhanced clarity (%s -> %s bytes)",
            len(payload),
            len(clean_payload),
        )
        payload = clean_payload
        duration = wav_duration_seconds(payload) or duration

    try:
        embedding = await run_in_threadpool(
            gnani.create_embeddings,
            payload,
            audio.filename or "voice.wav",
            audio.content_type or "audio/wav",
        )
        session.speaker_embedding = embedding
        session.clone_ready = True
        session.using_stock_voice = False
        recording_name = (x_recording_name or "").strip()
        if not recording_name or resolve_recording(recording_name) is None:
            recording_name = ""
        saved = save_voice(
            x_client_id,
            embedding,
            seconds=duration,
            recording=recording_name or None,
        )
        session.voice_id = saved["id"] if saved else None
        if session.voice_id:
            set_current_voice(x_client_id, session.voice_id)
        return {
            "ok": True,
            "clone_ready": True,
            "fallback": False,
            "voice_id": session.voice_id,
            "duration_seconds": round(duration, 2) if duration else None,
            "quality": quality,
        }
    except BudgetExceeded as exc:
        raise HTTPException(402, str(exc)) from exc
    except GnaniError as exc:
        log.warning("enrol clone failed status=%s err=%s; using stock voice", exc.status_code, exc)
        session.speaker_embedding = None
        session.clone_ready = False
        session.using_stock_voice = True
        return {
            "ok": True,
            "clone_ready": False,
            "fallback": True,
            "fallback_kind": "stock_voice",
            "upstream_status": exc.status_code,
            "message": enrol_failure_message(exc),
        }
    except Exception as exc:
        log.exception("unexpected enrol failure; using stock voice: %s", exc)
        session.speaker_embedding = None
        session.clone_ready = False
        session.using_stock_voice = True
        return {
            "ok": True,
            "clone_ready": False,
            "fallback": True,
            "fallback_kind": "stock_voice",
            "message": "Cloning failed because of an unexpected server error. Using a stock Timbre voice.",
        }


class SpeakBody(BaseModel):
    text: str = Field(min_length=1, max_length=500)
    emotion: str = "warm"
    language: str = "auto"


@app.post("/api/speak")
def speak(
    body: SpeakBody,
    x_session_id: Annotated[str | None, Header()] = None,
) -> Response:
    session = session_from(x_session_id)
    emotion = body.emotion if body.emotion in EMOTIONS else "warm"
    language = normalized_language(body.language, SPEAK_LANGUAGES, "auto")
    try:
        audio, _voice, used_stock = gnani.synthesize(
            body.text,
            speaker_embedding=session.speaker_embedding,
            emotion=emotion,
            language=language,
        )
    except BudgetExceeded as exc:
        raise HTTPException(402, str(exc)) from exc
    except GnaniError as exc:
        code = exc.status_code or 502
        log.error(
            "speak failed: returning %s because %s",
            code,
            exc,
        )
        detail = INVALID_KEY_HINT if code in (401, 403) else str(exc)
        raise HTTPException(code, detail) from exc

    if used_stock:
        session.using_stock_voice = True
    session.turns.append({"role": "user", "text": body.text})
    info = wav_info(audio)
    if info is None:
        log.error("Gnani returned non-WAV audio (%s bytes)", len(audio))
        raise HTTPException(502, "Gnani returned audio in an unexpected format.")
    expected_rate = (
        settings.stock_tts_sample_rate
        if used_stock or not session.clone_ready
        else settings.clone_tts_sample_rate
    )
    if info["rate"] != expected_rate:
        log.warning(
            "Gnani audio rate=%s Hz but requested %s Hz (%s)",
            info["rate"],
            expected_rate,
            "stock" if used_stock else "clone",
        )
    headers = {
        "X-AI-Generated": "true",
        "X-Voice-Source": "stock" if used_stock or not session.clone_ready else "clone",
        "X-Fallback": "true" if used_stock else "false",
        "X-Emotion": emotion,
        "X-Audio-Sample-Rate": str(info["rate"]),
        "X-Audio-Seconds": str(info["seconds"]),
        "Cache-Control": "no-store",
    }
    return Response(content=audio, media_type="audio/wav", headers=headers)


class SuggestBody(BaseModel):
    turns: list[dict[str, str]] | None = None
    user_gender: str | None = None
    language: str | None = None


def _clean_gender(value: str | None) -> str | None:
    if value in {"male", "female"}:
        return value
    return None


@app.post("/api/suggest")
def suggest(
    body: SuggestBody,
    x_session_id: Annotated[str | None, Header()] = None,
) -> dict[str, Any]:
    session = session_from(x_session_id)
    turns = body.turns if body.turns is not None else session.turns[-6:]
    language_hint = body.language if body.language in {"en", "hi", "hinglish"} else None
    replies, fallback = suggest_with_fallback(
        turns, _clean_gender(body.user_gender), language_hint
    )
    return {
        "replies": [r.as_dict() for r in replies],
        "fallback": fallback,
        "fallback_kind": "canned_replies" if fallback else None,
    }


@app.get("/api/favourites")
def favourites() -> dict[str, Any]:
    return {"favourites": FAVOURITES}


@app.post("/api/recordings")
async def save_voice_recording(
    audio: UploadFile = File(...),
    x_client_id: Annotated[str | None, Header()] = None,
    x_session_id: Annotated[str | None, Header()] = None,
) -> dict[str, Any]:
    """Keep a playable MP4 copy of every reference the user records."""
    session_from(x_session_id)
    max_bytes = int(settings.max_upload_mb * 1024 * 1024)
    payload = await audio.read(max_bytes + 1)
    if not payload:
        raise HTTPException(400, "Empty recording")
    if len(payload) > max_bytes:
        raise HTTPException(413, f"Audio must be under {settings.max_upload_mb:.0f} MB")
    info = wav_info(payload) or {}
    try:
        meta = await run_in_threadpool(
            save_recording,
            payload,
            client_id=x_client_id,
            duration=info.get("seconds"),
        )
    except RecordingError as exc:
        raise HTTPException(500, f"Could not save the MP4 copy: {exc}") from exc
    return {"ok": True, **meta}


@app.get("/api/recordings")
def recordings(x_client_id: Annotated[str | None, Header()] = None) -> dict[str, Any]:
    return {"recordings": list_recordings(x_client_id)}


@app.get("/api/recordings/{name}")
def recording_file(name: str, inline: bool = False) -> FileResponse:
    path = resolve_recording(name)
    if path is None:
        raise HTTPException(404, "Recording not found")
    return FileResponse(
        path,
        media_type="video/mp4",
        filename=path.name,
        content_disposition_type="inline" if inline else "attachment",
    )


@app.delete("/api/recordings/{name}")
def delete_voice_recording(
    name: str,
    x_client_id: Annotated[str | None, Header()] = None,
) -> dict[str, Any]:
    """Delete one saved MP4 take for this browser."""
    if not delete_recording(x_client_id, name):
        raise HTTPException(404, "Recording not found")
    return {"ok": True, "recordings": list_recordings(x_client_id)}


@app.get("/api/voices")
def voices(
    x_client_id: Annotated[str | None, Header()] = None,
    x_session_id: Annotated[str | None, Header()] = None,
) -> dict[str, Any]:
    """List every clone saved for this browser, newest first.

    Voices stored before the recording link existed are matched to the take
    that was saved just before the enrolment, so the picker does not offer to
    pay for a clone of audio that is already cloned.
    """
    items = list_voices(x_client_id)
    recordings = list_recordings(x_client_id)
    used = {item.get("recording") for item in items if item.get("recording")}
    for item in items:
        if item.get("recording"):
            continue
        candidates = [
            rec
            for rec in recordings
            if rec["name"] not in used
            and rec["created_at"] <= item["saved_at"] + 300_000
            and item["saved_at"] - rec["created_at"] <= 7_200_000
        ]
        if candidates:
            best = max(candidates, key=lambda rec: rec["created_at"])
            item["recording"] = best["name"]
            used.add(best["name"])
    session = store.get(x_session_id or "")
    for item in items:
        if item.get("seconds") is None and item.get("recording"):
            match = next(
                (rec for rec in recordings if rec["name"] == item["recording"]),
                None,
            )
            if match and match.get("seconds"):
                item["seconds"] = match["seconds"]
    return {
        "voices": items,
        "recordings": recordings,
        "active": session.voice_id if session else current_voice_id(x_client_id),
    }


@app.post("/api/voices/{voice_id}/use")
def use_saved_voice(
    voice_id: str,
    x_session_id: Annotated[str | None, Header()] = None,
    x_client_id: Annotated[str | None, Header()] = None,
) -> dict[str, Any]:
    """Switch the session to an already-created clone. No Gnani credits used."""
    session = session_from(x_session_id)
    embedding = load_voice(x_client_id, voice_id)
    if embedding is None:
        raise HTTPException(404, "Saved voice not found")
    session.speaker_embedding = embedding
    session.clone_ready = True
    session.using_stock_voice = False
    session.consent = True
    session.voice_id = voice_id
    set_current_voice(x_client_id, voice_id)
    return {"ok": True, "clone_ready": True, "voice_id": voice_id}


@app.delete("/api/voices/{voice_id}")
def delete_saved_voice(
    voice_id: str,
    x_session_id: Annotated[str | None, Header()] = None,
    x_client_id: Annotated[str | None, Header()] = None,
) -> dict[str, Any]:
    """Delete a saved clone. If it was in use, fall back to another take."""
    if not delete_voice(x_client_id, voice_id):
        raise HTTPException(404, "Saved voice not found")
    remaining = list_voices(x_client_id)
    next_id = remaining[0]["id"] if remaining else None
    session = store.get(x_session_id or "")
    if session is not None and session.voice_id == voice_id:
        if next_id:
            embedding = load_voice(x_client_id, next_id)
            session.speaker_embedding = embedding
            session.clone_ready = embedding is not None
            session.using_stock_voice = embedding is None
            session.voice_id = next_id
            set_current_voice(x_client_id, next_id)
        else:
            session.speaker_embedding = None
            session.clone_ready = False
            session.using_stock_voice = True
            session.voice_id = None
            clear_current_voice(x_client_id)
    return {
        "ok": True,
        "voices": remaining,
        "active": (
            session.voice_id if session is not None else current_voice_id(x_client_id)
        ),
    }


@app.get("/api/credits")
def credits() -> dict[str, Any]:
    return gnani.ledger.snapshot()


@app.post("/api/demo")
def demo(x_session_id: Annotated[str | None, Header()] = None) -> dict[str, Any]:
    session = session_from(x_session_id)
    session.consent = True
    session.speaker_embedding = None
    session.clone_ready = False
    session.using_stock_voice = True
    session.voice_id = None
    session.turns = [
        {"role": "partner", "text": "Aap theek ho? Water chahiye kya?"},
    ]
    replies, _ = suggest_with_fallback(session.turns)
    return {
        "ok": True,
        "demo": True,
        "fallback": True,
        "fallback_kind": "stock_voice",
        "partner": session.turns[0]["text"],
        "replies": [r.as_dict() for r in replies],
    }


@app.websocket("/ws/listen")
async def listen(websocket: WebSocket) -> None:
    await websocket.accept()
    session_id = websocket.query_params.get("session_id")
    if not session_id:
        await websocket.send_json({"type": "error", "message": "Missing session_id"})
        await websocket.close()
        return
    session = store.require(session_id)
    language = normalized_language(
        websocket.query_params.get("language"),
        STT_LANGUAGES,
        settings.stt_language,
        allow_list=True,
    )
    # Hinglish is Hindi speech written in Latin script. Prisma has no
    # dedicated locale for it, so listen in Hindi and transliterate the output.
    hinglish = language == "hi-en"
    if hinglish:
        language = "hi-IN"
    output_format = websocket.query_params.get("format") or (
        "transliterate" if hinglish else settings.stt_format
    )
    if output_format not in STT_FORMATS:
        output_format = settings.stt_format
    queue: asyncio.Queue[bytes | str | None] = asyncio.Queue()

    async def pcm_iter() -> AsyncIterator[bytes | str]:
        while True:
            chunk = await queue.get()
            if chunk is None:
                break
            yield chunk

    receiver: asyncio.Task | None = None
    fallback_sent = False

    async def send_fallback(message: str) -> None:
        nonlocal fallback_sent
        if fallback_sent:
            return
        fallback_sent = True
        with suppress(Exception):
            await websocket.send_json(
                {
                    "type": "fallback",
                    "fallback_kind": "browser_speech",
                    "message": message,
                }
            )

    async def pump_prisma() -> None:
        try:
            async for frame in gnani.prisma_stream(
                pcm_iter(),
                sample_rate=16000,
                language=language,
                output_format=output_format,
            ):
                kind = frame.get("type")
                if kind == "transcript":
                    text = (frame.get("text") or "").strip()
                    if text:
                        session.turns.append({"role": "partner", "text": text})
                    frame = {**frame, "source": "prisma"}
                elif kind == "error":
                    log.warning("Prisma error frame: %s", frame.get("message"))
                with suppress(Exception):
                    await websocket.send_json(frame)
            await send_fallback("Live listening ended. Using browser speech recognition.")
        except Exception as exc:
            log.warning("Prisma stream failed: %s", exc)
            await send_fallback("Live listening fell back to the browser.")

    if not settings.gnani_api_key:
        await send_fallback("No Gnani key. Using browser speech recognition.")
    else:
        receiver = asyncio.create_task(pump_prisma())
        await websocket.send_json(
            {
                "type": "ready",
                "engine": "prisma-v3.0",
                "language": language,
                "format": output_format,
            }
        )

    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break
            data = message.get("bytes")
            text = message.get("text")
            if data is not None:
                if receiver is not None and not receiver.done():
                    await queue.put(data)
            elif text:
                try:
                    payload = json.loads(text)
                except json.JSONDecodeError:
                    log.warning("Ignoring malformed browser control frame")
                    continue
                kind = payload.get("type")
                if kind == "commit":
                    await queue.put(json.dumps({"type": "commit"}))
                elif kind == "browser_transcript":
                    text_value = (payload.get("text") or "").strip()
                    if text_value:
                        session.turns.append({"role": "partner", "text": text_value})
                        with suppress(Exception):
                            await websocket.send_json(
                                {"type": "transcript", "text": text_value, "source": "browser"}
                            )
    except WebSocketDisconnect:
        pass
    finally:
        await queue.put(None)
        if receiver:
            receiver.cancel()
            with suppress(Exception):
                await receiver


@app.exception_handler(BudgetExceeded)
def budget_handler(_request, exc: BudgetExceeded) -> JSONResponse:
    return JSONResponse({"detail": str(exc)}, status_code=402)
