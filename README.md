# Apni Awaaz

Type or tap a reply. The app speaks it in a clone of your voice.

Browser talks only to this FastAPI server. Gnani keys never leave the backend (`X-API-Key-ID`).

## Run locally (Windows)

1. Copy `backend/.env.example` to `backend/.env` and set `GNANI_API_KEY`.
2. Backend:

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

3. Frontend (second terminal):

```powershell
cd frontend
copy .env.example .env.local
npm install
npm run dev
```

Open http://localhost:3000

## API tests (from `backend/`)

```powershell
python scripts/test_evon.py
python scripts/test_embeddings.py --audio path\to\voice.wav
python scripts/test_tts.py --text "Namaste"
python scripts/test_stt.py --audio path\to\speech.wav
```

## Deploy

- **Backend (Render):** New Web Service, root `backend`, start `uvicorn app.main:app --host 0.0.0.0 --port $PORT`. Set `GNANI_API_KEY`, `CORS_ORIGINS=*` (or your Vercel URL).
- **Frontend (Vercel):** Root `frontend`. Set `NEXT_PUBLIC_API_URL` to the Render HTTPS URL (no trailing slash).

## Notes from Gnani docs

- Clone embeddings: `POST /api/v1/tts/voice-clone/embeddings`
- Clone speak: `POST /api/v1/tts/inference` with `model: vachana-vc-v1` and `speaker_embedding` (`embedding`, `shape`, `dtype`)
- Stock fallback: Timbre `timbre-v3.0` voice `Shivani`
- **TODO:** `vachana-vc-v1` has no documented emotion field. Timbre v3 tags (`<whispers>`, `<shouting>`, `<giggle>`, `<pauses>`, `<laugh>`) are `timbre-v3.0` only. Clone synthesis uses plain text. UI emotions still tag replies and Timbre fallback.
- Prisma live STT: `wss://api.vachana.ai/stt/v3/stream`, headers `x-api-key-id`, `model: prisma-v3.0`, `language_code: indic-auto`, 16 kHz PCM16
- Evon hosted chat is not in docs.gnani.ai (closed beta). Swap the LLM in **one file:** `backend/app/llm/provider.py`. If `EVON_CHAT_URL` is empty, canned replies are used.
