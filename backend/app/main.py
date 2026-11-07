from __future__ import annotations

import asyncio
import json
import logging
from contextlib import suppress
from typing import Annotated, Any, AsyncIterator

from fastapi import FastAPI, File, Header, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

from app.config import settings
from app.gnani.client import BudgetExceeded, GnaniError, gnani
from app.llm.provider import suggest_with_fallback
from app.store import store

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("apni_awaaz")

FAVOURITES = [
    {"id": "pain", "text": "I'm in pain", "emotion": "urgent"},
    {"id": "thanks", "text": "thank you", "emotion": "warm"},
    {"id": "yes", "text": "Yes", "emotion": "calm"},
    {"id": "no", "text": "No", "emotion": "calm"},
    {"id": "wait", "text": "Please wait", "emotion": "warm"},
]

app = FastAPI(title="Apni Awaaz")
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list or ["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-AI-Generated", "X-Voice-Source", "X-Fallback", "X-Emotion"],
)


def session_from(session_id: str | None):
    if not session_id:
        raise HTTPException(400, "Missing X-Session-Id")
    return store.require(session_id)


@app.get("/health")
def health() -> dict[str, Any]:
    return {"ok": True, "has_gnani_key": bool(settings.gnani_api_key)}


@app.post("/api/session")
def create_session() -> dict[str, str]:
    return {"session_id": store.create().id}


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
) -> dict[str, Any]:
    session = session_from(x_session_id)
    if not session.consent:
        raise HTTPException(403, "Consent required")
    payload = await audio.read()
    if len(payload) < 1000:
        raise HTTPException(400, "Audio too short — record about 20 seconds")
    try:
        embedding = gnani.create_embeddings(
            payload, audio.filename or "voice.wav", audio.content_type or "audio/wav"
        )
        session.speaker_embedding = embedding
        session.clone_ready = True
        session.using_stock_voice = False
        return {"ok": True, "clone_ready": True, "fallback": False}
    except BudgetExceeded as exc:
        raise HTTPException(402, str(exc)) from exc
    except (GnaniError, Exception) as exc:
        log.warning("enrol clone failed: %s", exc)
        session.speaker_embedding = None
        session.clone_ready = False
        session.using_stock_voice = True
        return {
            "ok": True,
            "clone_ready": False,
            "fallback": True,
            "fallback_kind": "stock_voice",
            "message": "Cloning failed. Using a stock Timbre voice.",
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
    emotion = body.emotion if body.emotion in {"warm", "urgent", "playful", "calm"} else "warm"
    try:
        audio, _voice, used_stock = gnani.synthesize(
            body.text,
            speaker_embedding=session.speaker_embedding,
            emotion=emotion,
            language=body.language,
        )
    except BudgetExceeded as exc:
        raise HTTPException(402, str(exc)) from exc
    except GnaniError as exc:
        raise HTTPException(exc.status_code or 502, str(exc)) from exc

    if used_stock:
        session.using_stock_voice = True
    session.turns.append({"role": "user", "text": body.text})
    headers = {
        "X-AI-Generated": "true",
        "X-Voice-Source": "stock" if used_stock or not session.clone_ready else "clone",
        "X-Fallback": "true" if used_stock else "false",
        "X-Emotion": emotion,
        "Cache-Control": "no-store",
    }
    return Response(content=audio, media_type="audio/wav", headers=headers)


class SuggestBody(BaseModel):
    turns: list[dict[str, str]] | None = None


@app.post("/api/suggest")
def suggest(
    body: SuggestBody,
    x_session_id: Annotated[str | None, Header()] = None,
) -> dict[str, Any]:
    session = session_from(x_session_id)
    turns = body.turns if body.turns is not None else session.turns[-6:]
    replies, fallback = suggest_with_fallback(turns)
    return {
        "replies": [r.as_dict() for r in replies],
        "fallback": fallback,
        "fallback_kind": "canned_replies" if fallback else None,
    }


@app.get("/api/favourites")
def favourites() -> dict[str, Any]:
    return {"favourites": FAVOURITES}


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
    queue: asyncio.Queue[bytes | str | None] = asyncio.Queue()

    async def pcm_iter() -> AsyncIterator[bytes | str]:
        while True:
            chunk = await queue.get()
            if chunk is None:
                break
            yield chunk

    prisma_ok = True
    receiver: asyncio.Task | None = None

    async def pump_prisma() -> None:
        nonlocal prisma_ok
        try:
            async for frame in gnani.prisma_stream(pcm_iter()):
                kind = frame.get("type")
                if kind == "transcript":
                    text = (frame.get("text") or "").strip()
                    if text:
                        session.turns.append({"role": "partner", "text": text})
                await websocket.send_json(frame)
        except Exception as exc:
            log.warning("Prisma stream failed: %s", exc)
            prisma_ok = False
            with suppress(Exception):
                await websocket.send_json(
                    {
                        "type": "fallback",
                        "fallback_kind": "browser_speech",
                        "message": "Live listening fell back to the browser.",
                    }
                )

    if not settings.gnani_api_key:
        await websocket.send_json(
            {
                "type": "fallback",
                "fallback_kind": "browser_speech",
                "message": "No Gnani key. Using browser speech recognition.",
            }
        )
    else:
        try:
            receiver = asyncio.create_task(pump_prisma())
            await websocket.send_json({"type": "ready", "engine": "prisma-v3.0"})
        except Exception as exc:
            log.warning("Prisma connect failed: %s", exc)
            await websocket.send_json(
                {
                    "type": "fallback",
                    "fallback_kind": "browser_speech",
                    "message": "Live listening fell back to the browser.",
                }
            )
            prisma_ok = False

    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break
            data = message.get("bytes")
            text = message.get("text")
            if data is not None:
                await queue.put(data)
            elif text:
                payload = json.loads(text)
                if payload.get("type") == "commit":
                    await queue.put(json.dumps({"type": "commit"}))
                elif payload.get("type") == "browser_transcript":
                    text_value = (payload.get("text") or "").strip()
                    if text_value:
                        session.turns.append({"role": "partner", "text": text_value})
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
