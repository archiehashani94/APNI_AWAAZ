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
.\start.ps1
```

`start.ps1` frees port 8001 before starting. Running `uvicorn ... --port 8001` by
hand while a server is already up fails on Windows with
`[WinError 10013] An attempt was made to access a socket in a way forbidden by
its access permissions` — that means the port is in use, not that a firewall is
blocking it.

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
- **Frontend (Vercel):** Root `frontend`. Set `NEXT_PUBLIC_API_BASE` to the Render HTTPS URL (no trailing slash).

## Notes from Gnani docs

- Clone embeddings: `POST /api/v1/tts/voice-clone/embeddings`
- Clone speak: `POST /api/v1/tts/inference` with `model: vachana-vc-v1` and `speaker_embedding` (`embedding`, `shape`, `dtype`)
- Stock fallback: Timbre `timbre-v3.0` voice `Shivani`
- **TODO:** `vachana-vc-v1` has no documented emotion field. Timbre v3 tags (`<whispers>`, `<shouting>`, `<giggle>`, `<pauses>`, `<laugh>`) are `timbre-v3.0` only. Clone synthesis uses plain text. UI emotions still tag replies and Timbre fallback.

## Reply suggestions (LLM)

The three tappable replies come from an OpenAI-compatible Chat Completions endpoint. **`LLM_PROVIDER=canned` disables the LLM and always returns fixed lines**, so replies feel unrelated to the partner. Set any other name to enable it:

```dotenv
LLM_PROVIDER=gemini
LLM_CHAT_URL=https://generativelanguage.googleapis.com/v1beta/openai/chat/completions
LLM_MODEL=gemini-3.1-flash-lite
LLM_API_KEY=your-key
```

Use `gemini-3.1-flash-lite` rather than `gemini-2.5-flash`: the latter has a 20-requests-per-day free-tier cap that runs out quickly. Flash-Lite answers in about 2 seconds; `gemini-3.8-flash` was accurate but took ~60 seconds, which is too slow for live conversation.

- The prompt tells the model the user cannot speak, to answer the partner's **last** line, to include a clear yes/no when asked, to match the partner's language and script, and to avoid generic filler.
- Only the last 8 conversation turns are sent, including the user's previous replies, so follow-up answers stay in context.
- The Suggest card has a **Reply as** setting (Female / Male / Not set). It is saved in `localStorage` (`aa_user_gender`) and tells the model which first-person Hindi/Hinglish verb forms to use. When unset, the prompt tells the model not to guess the user's gender.
- Gemini 2.5 Flash spends most of `max_tokens` on hidden thinking and was truncating the JSON (`finish_reason=length`). The client sends `reasoning_effort=none` and a larger token budget, and retries without those optional fields if a provider rejects them.
- Responses are parsed as either `{"replies":[...]}` or a bare JSON array; duplicate contexts are cached for 90 s to protect free-tier rate limits.
- On any LLM failure or `429`, the app falls back to the intent-aware lines in `backend/app/llm/canned.py` and shows a `canned_replies` badge.

There is no fine-tuning step. Accuracy comes from enabling the real model, a task-specific prompt, and conversation context. If you collect a few example conversations, they can be added to the prompt as few-shot examples.

## Favourites

The default list is served by `GET /api/favourites`. Users can add their own phrases in the Favourites card; custom entries are stored in `localStorage` (`aa_custom_favourites`) and can be removed with the × button. Each phrase carries an emotion used by the stock-voice fallback.

## Voice quality notes

- Record **15–30 seconds** of continuous, natural speech in a quiet room. Use **one speaker only**: no music, no singing, no TV, no second voice, and speak the **same language** you will use for replies. The backend rejects clips shorter than 5 seconds (Gnani's documented minimum) or longer than 30 seconds.
- The browser removes DC offset, low-frequency rumble, and long breath runs, then trims leading/trailing silence, resamples to 16 kHz mono, peak-normalises the clip, and gives you a preview before uploading. Breath removal matters because the clone learns the reference's breath pattern and replays it after every reply. A noisy, musical, or multi-speaker reference is the most common reason a clone sounds unlike the user.
- After embedding, the backend scores the reference and returns `quality.warnings` (clipping, low level, excess silence, high noise, rumble, short clip). The UI shows these and offers "Record a better sample". A low score does not block cloning, but fixing it is the single biggest quality lever.
- `vachana-vc-v1` is the only Gnani model that accepts `speaker_embedding` (`timbre-*` returns `400`). Clone synthesis requests 44.1 kHz WAV and adds sentence punctuation to short replies for more natural prosody.
- Gnani's WAV responses declare `RIFF`/`data` sizes as `0xFFFFFFFF`; `backend/app/audio.py` rewrites them to the real sizes before caching or serving, so players and duration checks see the true length.
- The clone model can append a breath, an "um", or a repeated word after the sentence. `backend/app/audio.py` runs four text-independent passes on every clip: it silences long unvoiced breath runs (high zero-crossing rate, 200 ms or more), removes an isolated voiced filler at the **end** of the clip (pure vowel, no consonants, 450 ms or longer, separated by a 240 ms pause, quieter than the phrase), removes a repeated tail/whole phrase by matching loudness and voicing against the earlier audio, then trims the quiet tail. Only the end of the clip is ever cut: an opening such as "Could you please…" has the same pause and level profile as a held "ahh", so trimming the front can delete real words. This is why a favourite such as "Yes" no longer plays twice and why "Thank you very much" no longer ends with "ha".
- The frontend uses one `Audio` element per clip and an atomic `speakingRef` lock, so a rapid double tap cannot start a second copy of the same phrase.
- If embedding creation fails, the app clearly reports that it fell back to the stock Timbre voice and offers a "Record voice again" button.
- Every preview is also converted to **MP4/AAC** and stored in `backend/recordings/`. The enrolment card lists these files with an inline player and a "Download MP4" link, so a take is never lost. Conversion uses the static FFmpeg bundled with `imageio-ffmpeg` (no system install needed).
- Opening "Record voice again" does not discard the current clone. While a previous voice exists, the enrolment card shows **"Use previous voice"**, which returns to the talk screen without re-uploading anything.
- Backend sessions live in memory, but clone embeddings are persisted to `backend/data/voices/` (gitignored) keyed by the first 12 characters of the browser's client id — the same slug used in recording filenames. Each enrolment is stored as its own file instead of overwriting the last one, and the last selected voice is remembered. `POST /api/session` and `GET /api/session` restore that clone with `X-Client-Id`, so restarting the backend never costs another embedding credit or asks the user to record again.
- The Voice picker on the conversation screen lists every take the browser has recorded. `GET /api/voices` returns saved clones, and `POST /api/voices/{voice_id}/use` switches the live session to one for free. A saved MP4 that was never cloned shows **Create voice**, which re-clones that take through the normal enrolment path (about 50 Gnani credits, labelled in the UI).

## Troubleshooting: "Cloning failed. Using a stock Timbre voice."

First check the backend log or `GET /health`. The app now passes Gnani's own error text through, so the message is usually explicit:

| Symptom | Cause | Fix |
| --- | --- | --- |
| `401 INVALID_API_KEY: Unauthorized` | `GNANI_API_KEY` is missing, revoked, or from the wrong project | Generate a new key at [app.gnani.ai/voice](https://app.gnani.ai/voice), update `backend/.env`, restart uvicorn |
| `400` / `415` / `422` about the audio | Reference clip is silent, corrupt, too short, or an unsupported format | Record 20–30 s of clean speech; use WAV/MP3/M4A/OGG/FLAC/AAC |
| `402` or budget message | Gnani credits exhausted | Add credits or raise `CREDIT_BUDGET` |
| `503` | Gnani is temporarily unavailable | Retry in a minute |

A `401` also blocks stock Timbre playback and Prisma live STT — the whole Gnani integration is down until the key is fixed. The UI hides the "Record voice again" button when it detects this, because re-recording cannot help.

### Check which key the backend actually loaded

Settings are read once at startup, so editing `.env` does nothing until uvicorn restarts.

```powershell
cd backend
.\.venv\Scripts\python.exe scripts\check_gnani_key.py
```

The script prints only the key's length, last 4 characters, and a SHA-256 prefix — never the key itself. It sends a deliberately invalid TTS model, so Gnani authenticates the key but rejects the body before synthesizing anything. That makes the test free.

- `400 ... model ...` → key is valid
- `401 INVALID_API_KEY` → the key value is wrong, revoked, or from another account
- `403 SCOPE_NOT_ALLOWED` → key is valid but lacks TTS/Voice Cloning scope

`GET /health` also reports `gnani_key_tail` and `gnani_key_sha256_10` so you can confirm the running process picked up a new key.

**Creating a new key does not fix a stale `.env` or an unrestarted backend.** Replace the value, restart uvicorn, then re-run the checker. Delete the leaked key in the Gnani dashboard instead of leaving it unused.

## Live listening notes

- Prisma v3.0 requires raw mono PCM16 audio in **exactly 1,024-byte frames**, sent at roughly 32 ms cadence. The frontend downsamples the microphone to 16 kHz and paces frames from a timer rather than sending audio-callback bursts.
- The microphone stream is explicitly stopped when listening ends.
- If Prisma cannot connect or the stream drops, the app falls back to browser `SpeechRecognition` (Chrome/Edge). Browser transcripts are sent to the backend and included in suggestion turns even if the WebSocket is down.
- Streaming sessions are capped by Gnani at 15 minutes. The backend requests a 300-second idle timeout.
- VAD defaults can be tuned with `STT_VAD_THRESHOLD`, `STT_MIN_SPEECH_MS`, `STT_MIN_SEGMENT_MS`, `STT_MIN_SILENCE_MS`, and `STT_MAX_SPEECH_S`.

## Optional tuning environment variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `MIN_ENROL_SECONDS` | `5` | Reject shorter reference clips |
| `MAX_ENROL_SECONDS` | `30` | Reject longer reference clips |
| `MAX_UPLOAD_MB` | `10` | Upload size cap |
| `CLONE_TTS_SAMPLE_RATE` | `44100` | Cloned TTS output rate |
| `STOCK_TTS_SAMPLE_RATE` | `48000` | Stock Timbre output rate |
| `STT_LANGUAGE` | `indic-auto` | Default Prisma language |
| `STT_FORMAT` | `verbatim` | `verbatim`, `transcribe`, `codemix`, or `transliterate` |

- Prisma live STT: `wss://api.vachana.ai/stt/v3/stream`, headers `x-api-key-id`, `model: prisma-v3.0`, `language_code: indic-auto`, 16 kHz PCM16
- Swap the LLM in **one file:** `backend/app/llm/provider.py`. `LLM_PROVIDER=canned` (or empty) uses canned replies and never 503s.
