"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { API_URL, api, type Emotion, type Favourite, type Reply, wsUrl } from "@/lib/api";
import {
  analyzeSample,
  downsample,
  encodeWav,
  floatToPcm16,
  levelPercent,
  prepareVoiceSample,
} from "@/lib/wav";

type Stage = "consent" | "enrol" | "talk";
type ListenState = "idle" | "connecting" | "live" | "fallback";
type RecState = "idle" | "recording" | "review";

const STT_RATE = 16000;
const STT_FRAME_SAMPLES = 512; // Gnani requires exactly 1024 bytes per binary frame
const STT_FRAME_MS = 32;
const REFERENCE_RATE = 16000; // standard rate for speaker embeddings
const MAX_SOURCE_MB = 100;
const MAX_SOURCE_SECONDS = 600;

function closeAudio(ctx: AudioContext | null) {
  if (ctx && ctx.state !== "closed") void ctx.close();
}

/** Proper anti-aliased resample via OfflineAudioContext (not naive decimation). */
async function resampleMono(
  input: Float32Array,
  fromRate: number,
  toRate: number,
): Promise<Float32Array> {
  if (fromRate === toRate || input.length === 0) return input;
  const length = Math.max(1, Math.round((input.length * toRate) / fromRate));
  const offline = new OfflineAudioContext(1, length, toRate);
  const buffer = offline.createBuffer(1, input.length, fromRate);
  buffer.getChannelData(0).set(input);
  const source = offline.createBufferSource();
  source.buffer = buffer;
  source.connect(offline.destination);
  source.start();
  const rendered = await offline.startRendering();
  return rendered.getChannelData(0);
}

/** Read duration without decoding the whole file, so a huge video is rejected early. */
function probeMediaDuration(file: File): Promise<number> {
  return new Promise((resolve) => {
    const el = document.createElement("video");
    el.preload = "metadata";
    const url = URL.createObjectURL(file);
    const done = (value: number) => {
      URL.revokeObjectURL(url);
      resolve(value);
    };
    el.onloadedmetadata = () => done(Number.isFinite(el.duration) ? el.duration : 0);
    el.onerror = () => done(0);
    el.src = url;
  });
}

type BrowserRec = {
  continuous: boolean;
  interimResults: boolean;
  lang: string;
  onresult: ((event: unknown) => void) | null;
  onerror: ((event: { error?: string }) => void) | null;
  onend: (() => void) | null;
  start: () => void;
  stop: () => void;
};

type FavouriteItem = Favourite & { custom?: boolean };

const CUSTOM_FAVOURITES_KEY = "aa_custom_favourites";
const USER_GENDER_KEY = "aa_user_gender";
const CLIENT_ID_KEY = "aa_client";

type SavedRecording = {
  name: string;
  url: string;
  seconds: number | null;
  created_at: number;
  bytes: number;
};

function getClientId(): string {
  let id = localStorage.getItem(CLIENT_ID_KEY);
  if (!id) {
    id =
      typeof crypto !== "undefined" && "randomUUID" in crypto
        ? crypto.randomUUID()
        : `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
    localStorage.setItem(CLIENT_ID_KEY, id);
  }
  return id;
}

export default function Home() {
  const [sessionId, setSessionId] = useState("");
  const [stage, setStage] = useState<Stage>("consent");
  const [agreed, setAgreed] = useState(false);
  const [status, setStatus] = useState("");
  const [error, setError] = useState("");
  const [seconds, setSeconds] = useState(0);
  const [cloneReady, setCloneReady] = useState(false);
  const [voiceIssue, setVoiceIssue] = useState("");
  const [qualityWarnings, setQualityWarnings] = useState<string[]>([]);
  const [qualityScore, setQualityScore] = useState<number | null>(null);
  const [listening, setListening] = useState(false);
  const [liveText, setLiveText] = useState("");
  const [replies, setReplies] = useState<Reply[]>([]);
  const [draft, setDraft] = useState("");
  const [emotion, setEmotion] = useState<Emotion>("warm");
  const [defaultFavourites, setDefaultFavourites] = useState<Favourite[]>([]);
  const [customFavourites, setCustomFavourites] = useState<FavouriteItem[]>([]);
  const [favText, setFavText] = useState("");
  const [favEmotion, setFavEmotion] = useState<Emotion>("warm");
  const [speaking, setSpeaking] = useState(false);
  const [aiLabel, setAiLabel] = useState(false);
  const [credits, setCredits] = useState<string>("");
  const [fallbacks, setFallbacks] = useState<string[]>([]);
  const [listenState, setListenState] = useState<ListenState>("idle");
  const [listenLang, setListenLang] = useState("indic-auto");
  const [speakLang, setSpeakLang] = useState("auto");
  const [userGender, setUserGender] = useState<"neutral" | "female" | "male">("neutral");
  const [recState, setRecState] = useState<RecState>("idle");
  const [recLevel, setRecLevel] = useState(0);
  const [previewUrl, setPreviewUrl] = useState("");
  const [previewName, setPreviewName] = useState("voice.wav");
  const [previewBlob, setPreviewBlob] = useState<Blob | null>(null);
  const [previewWarnings, setPreviewWarnings] = useState<string[]>([]);
  const [previewScore, setPreviewScore] = useState<number | null>(null);
  const [fileBusy, setFileBusy] = useState(false);
  const [fileNote, setFileNote] = useState("");
  const [clientId, setClientId] = useState("");
  const [savedRecordings, setSavedRecordings] = useState<SavedRecording[]>([]);
  const [savingRecording, setSavingRecording] = useState(false);
  const [recordingNote, setRecordingNote] = useState("");

  const favourites: FavouriteItem[] = [
    ...defaultFavourites.map((item) => ({ ...item, custom: false })),
    ...customFavourites,
  ];

  const recChunks = useRef<Float32Array[]>([]);
  const recRate = useRef(16000);
  const recTimer = useRef<number | null>(null);
  const recCtx = useRef<AudioContext | null>(null);
  const recProc = useRef<ScriptProcessorNode | null>(null);
  const recStream = useRef<MediaStream | null>(null);
  const recLevelAt = useRef(0);
  const recClipped = useRef(false);
  const stoppingRec = useRef(false);
  const listenWs = useRef<WebSocket | null>(null);
  const listenCtx = useRef<AudioContext | null>(null);
  const listenStream = useRef<MediaStream | null>(null);
  const listenSource = useRef<MediaStreamAudioSourceNode | null>(null);
  const listenProc = useRef<ScriptProcessorNode | null>(null);
  const listenPacer = useRef<number | null>(null);
  const pcmQueue = useRef<Int16Array>(new Int16Array(0));
  const listeningRef = useRef(false);
  const closingListen = useRef(false);
  const browserRecActive = useRef(false);
  const browserRecRestart = useRef<number | null>(null);
  const lastTranscript = useRef<{ text: string; at: number }>({ text: "", at: 0 });
  const partnerTurns = useRef<{ role: string; text: string }[]>([]);
  const suggestSeq = useRef(0);
  const suggestTimer = useRef<number | null>(null);
  const userGenderRef = useRef<"neutral" | "female" | "male">("neutral");
  const voiceVersion = useRef(0);
  const audioUrl = useRef("");
  const previewUrlRef = useRef("");
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const speakingRef = useRef(false);
  const fileInput = useRef<HTMLInputElement | null>(null);
  const favCache = useRef<Map<string, Blob>>(new Map());
  const speechRec = useRef<BrowserRec | null>(null);

  const addFallback = useCallback((kind: string) => {
    setFallbacks((prev) => (prev.includes(kind) ? prev : [...prev, kind]));
  }, []);

  useEffect(() => {
    const client = getClientId();
    setClientId(client);
    let sid = localStorage.getItem("aa_session");
    (async () => {
      if (sid) {
        // The backend keeps sessions in memory, so an id saved before a
        // restart points at nothing. Validate it before reusing it; the
        // backend restores a saved clone from the client id when possible.
        try {
          const check = await fetch(`${API_URL}/api/session`, {
            headers: { "X-Session-Id": sid, "X-Client-Id": client },
          });
          if (!check.ok) {
            localStorage.removeItem("aa_session");
            sid = null;
          } else {
            const info: {
              consent?: boolean;
              clone_ready?: boolean;
            } = await check.json();
            if (info.consent) setAgreed(true);
            if (info.clone_ready) {
              setCloneReady(true);
              setStage("talk");
            }
          }
        } catch {
          // Backend may still be waking up; keep the id and retry on use.
        }
      }
      if (!sid) {
        const res = await fetch(`${API_URL}/api/session`, {
          method: "POST",
          headers: { "X-Client-Id": client },
        });
        const data = await res.json();
        sid = data.session_id as string;
        localStorage.setItem("aa_session", sid);
        if (data.consent) setAgreed(true);
        if (data.clone_ready) {
          setCloneReady(true);
          setStage("talk");
        }
      }
      setSessionId(sid!);
      const fav = await fetch(`${API_URL}/api/favourites`);
      const favData: { favourites?: Favourite[] } = await fav.json();
      setDefaultFavourites(favData.favourites || []);
    })().catch((err) => setError(String(err)));
  }, []);

  useEffect(() => {
    if (!clientId) return;
    let cancelled = false;
    (async () => {
      try {
        const res = await fetch(`${API_URL}/api/recordings`, {
          headers: { "X-Client-Id": clientId },
        });
        if (!res.ok) return;
        const data: { recordings?: SavedRecording[] } = await res.json();
        if (!cancelled) setSavedRecordings(data.recordings || []);
      } catch {
        /* saved recordings are optional */
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [clientId]);

  useEffect(() => {
    try {
      const raw = localStorage.getItem(CUSTOM_FAVOURITES_KEY);
      if (!raw) return;
      const parsed: unknown = JSON.parse(raw);
      if (!Array.isArray(parsed)) return;
      const valid = parsed.filter(
        (item): item is FavouriteItem =>
          typeof item === "object" &&
          item !== null &&
          typeof (item as FavouriteItem).id === "string" &&
          typeof (item as FavouriteItem).text === "string" &&
          ["warm", "urgent", "playful", "calm"].includes(
            (item as FavouriteItem).emotion,
          ),
      );
      setCustomFavourites(valid.map((item) => ({ ...item, custom: true })));
    } catch {
      /* corrupted local storage should not break the app */
    }
  }, []);

  useEffect(() => {
    const saved = localStorage.getItem(USER_GENDER_KEY);
    if (saved === "female" || saved === "male") {
      setUserGender(saved);
      userGenderRef.current = saved;
    }
  }, []);

  useEffect(() => {
    return () => {
      closingListen.current = true;
      listeningRef.current = false;
      stopCapture();
      stopBrowserSpeech();
      if (recTimer.current) window.clearInterval(recTimer.current);
      if (suggestTimer.current) window.clearTimeout(suggestTimer.current);
      recProc.current?.disconnect();
      recStream.current?.getTracks().forEach((t) => t.stop());
      closeAudio(recCtx.current);
      audioRef.current?.pause();
      if (audioUrl.current) URL.revokeObjectURL(audioUrl.current);
      if (previewUrlRef.current) URL.revokeObjectURL(previewUrlRef.current);
    };
  }, []);

  async function refreshCredits() {
    if (!sessionId) return;
    try {
      const res = await api("/api/credits", sessionId);
      const data = await res.json();
      setCredits(`${data.used} / ${data.budget} credits`);
    } catch {
      /* credit display is non-critical */
    }
  }

  async function submitConsent() {
    setError("");
    const res = await api("/api/consent", sessionId, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ agreed: true }),
    });
    if (!res.ok) {
      setError(await res.text());
      return;
    }
    setStage("enrol");
  }

  async function startDemo() {
    setError("");
    try {
      await api("/api/consent", sessionId, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ agreed: true }),
      });
      const res = await api("/api/demo", sessionId, { method: "POST" });
      const data = await res.json();
      setCloneReady(false);
      addFallback("stock_voice");
      if (data.fallback) addFallback(data.fallback_kind || "demo");
      setLiveText(data.partner || "");
      partnerTurns.current = data.partner ? [{ role: "partner", text: data.partner }] : [];
      voiceVersion.current = 0;
      favCache.current.clear();
      setReplies(data.replies || []);
      setStage("talk");
      setAiLabel(true);
      refreshCredits();
    } catch (err) {
      setError(String(err));
    }
  }

  function clearPreview() {
    if (previewUrlRef.current) {
      URL.revokeObjectURL(previewUrlRef.current);
      previewUrlRef.current = "";
    }
    setPreviewUrl("");
    setPreviewBlob(null);
    setPreviewName("voice.wav");
    setPreviewWarnings([]);
    setPreviewScore(null);
    setFileNote("");
    setRecState("idle");
  }

  function showPreview(blob: Blob, name: string) {
    if (previewUrlRef.current) URL.revokeObjectURL(previewUrlRef.current);
    const url = URL.createObjectURL(blob);
    previewUrlRef.current = url;
    setPreviewUrl(url);
    setPreviewBlob(blob);
    setPreviewName(name);
    setRecState("review");
    setStatus("");
  }

  async function persistRecording(blob: Blob, seconds: number) {
    if (!clientId || !sessionId) return;
    setSavingRecording(true);
    setRecordingNote("");
    try {
      const form = new FormData();
      form.append("audio", blob, "voice.wav");
      const res = await fetch(`${API_URL}/api/recordings`, {
        method: "POST",
        headers: { "X-Session-Id": sessionId, "X-Client-Id": clientId },
        body: form,
      });
      if (!res.ok) {
        const detail = await res.text();
        throw new Error(detail || `Save failed (${res.status})`);
      }
      const saved: SavedRecording = await res.json();
      setSavedRecordings((prev) => [
        saved,
        ...prev.filter((item) => item.name !== saved.name),
      ]);
      setRecordingNote("MP4 copy saved.");
    } catch (err) {
      setRecordingNote(
        `Could not save the MP4 copy: ${err instanceof Error ? err.message : String(err)}`,
      );
    } finally {
      setSavingRecording(false);
    }
  }

  async function startRecord() {
    if (recState === "recording") return;
    setError("");
    setQualityWarnings([]);
    setQualityScore(null);
    clearPreview();
    recChunks.current = [];
    setRecLevel(0);
    setSeconds(0);
    recClipped.current = false;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: false,
        },
      });
      recStream.current = stream;
      const ctx = new AudioContext();
      recCtx.current = ctx;
      recRate.current = ctx.sampleRate;
      const source = ctx.createMediaStreamSource(stream);
      const proc = ctx.createScriptProcessor(4096, 1, 1);
      recProc.current = proc;
      proc.onaudioprocess = (ev) => {
        const samples = new Float32Array(ev.inputBuffer.getChannelData(0));
        recChunks.current.push(samples);
        let peak = 0;
        for (let i = 0; i < samples.length; i++) {
          const value = Math.abs(samples[i]);
          if (value > peak) peak = value;
        }
        if (peak >= 0.995) recClipped.current = true;
        const now = performance.now();
        if (now - recLevelAt.current > 100) {
          recLevelAt.current = now;
          setRecLevel(levelPercent(samples));
        }
      };
      const mute = ctx.createGain();
      mute.gain.value = 0;
      source.connect(proc);
      proc.connect(mute);
      mute.connect(ctx.destination);
      stoppingRec.current = false;
      setRecState("recording");
      setStatus("");
      recTimer.current = window.setInterval(() => {
        setSeconds((s) => {
          const next = s + 1;
          if (next >= 30) {
            window.setTimeout(() => void stopRecord(), 0);
            return 30;
          }
          return next;
        });
      }, 1000);
    } catch {
      setRecState("idle");
      setError("Microphone unavailable. Allow microphone access and try again.");
    }
  }

  async function stopRecord() {
    if (stoppingRec.current) return;
    stoppingRec.current = true;
    if (recTimer.current) {
      window.clearInterval(recTimer.current);
      recTimer.current = null;
    }
    recProc.current?.disconnect();
    recProc.current = null;
    closeAudio(recCtx.current);
    recCtx.current = null;
    recStream.current?.getTracks().forEach((t) => t.stop());
    recStream.current = null;
    setRecLevel(0);
    setRecState("idle");

    const total = recChunks.current.reduce((n, c) => n + c.length, 0);
    const merged = new Float32Array(total);
    let o = 0;
    for (const c of recChunks.current) {
      merged.set(c, o);
      o += c.length;
    }
    const rawDuration = merged.length / recRate.current;
    if (rawDuration < 5) {
      setRecState("idle");
      setError(
        `Only ${rawDuration.toFixed(1)}s recorded. Record at least 5 seconds, ideally 20–30 seconds.`,
      );
      return;
    }
    if (levelPercent(merged) < 5) {
      setRecState("idle");
      setError("The recording is too quiet. Move closer to the microphone and record again.");
      return;
    }
    if (recClipped.current) {
      setError(
        "The recording clipped (input was too loud). If the preview sounds distorted, move back and record again.",
      );
    }

    // Trimming silence and normalising the level makes the clone far more
    // consistent than sending a raw, quiet laptop-mic recording upstream.
    // Speaker embeddings are trained around 16 kHz, so resample properly
    // instead of handing Gnani a 44.1/48 kHz reference.
    let prepared: Float32Array;
    let reference: Float32Array;
    try {
      reference = await resampleMono(merged, recRate.current, REFERENCE_RATE);
      prepared = prepareVoiceSample(reference, REFERENCE_RATE);
    } catch {
      setError("Could not process the recording. Stop listening and try again.");
      return;
    }
    const preparedDuration = prepared.length / REFERENCE_RATE;
    if (preparedDuration < 5) {
      setRecState("idle");
      setError(
        `After removing silence only ${preparedDuration.toFixed(1)}s remains. Record at least 5 seconds of speech.`,
      );
      return;
    }

    // Score the raw take, before trimming/normalising hides clipping or noise.
    const quality = analyzeSample(reference, REFERENCE_RATE);
    setPreviewWarnings(quality.warnings);
    setPreviewScore(quality.score);

    const wav = encodeWav(prepared, REFERENCE_RATE);
    showPreview(wav, "voice.wav");
    void persistRecording(wav, preparedDuration);
    setStatus(
      quality.warnings.length
        ? `Recorded ${preparedDuration.toFixed(1)}s, but the reference has problems — read the notes below before using it.`
        : `Recorded ${preparedDuration.toFixed(1)}s of speech. Listen to the preview, then tap "Use this voice".`,
    );
  }

  async function confirmEnrol() {
    if (!previewBlob || !sessionId || fileBusy) return;
    await uploadVoice(previewBlob, previewName);
  }

  async function uploadVoice(file: Blob, name: string) {
    setStatus("Creating voice embedding…");
    setError("");
    try {
      const form = new FormData();
      form.append("audio", file, name);
      const res = await api("/api/enrol", sessionId, {
        method: "POST",
        headers: clientId ? { "X-Client-Id": clientId } : undefined,
        body: form,
      });
      const data: {
        clone_ready?: boolean;
        fallback?: boolean;
        message?: string;
        detail?: string;
        duration_seconds?: number;
        upstream_status?: number;
        quality?: { verdict?: string; score?: number; warnings?: string[] };
      } = await res.json().catch(() => ({}));
      if (!res.ok) {
        setError(data.detail || `Enrol failed (${res.status})`);
        setStatus("");
        return;
      }
      setCloneReady(Boolean(data.clone_ready));
      setAiLabel(true);
      const warnings = data.quality?.warnings ?? [];
      setQualityWarnings(warnings);
      setQualityScore(typeof data.quality?.score === "number" ? data.quality.score : null);
      if (data.fallback) {
        addFallback("stock_voice");
        const blocked = data.upstream_status === 401 || data.upstream_status === 403;
        setVoiceIssue(blocked ? data.message || "Gnani rejected the API key." : "");
        setError(data.message || "Cloning failed. Using the stock voice; record again to retry.");
      } else {
        voiceVersion.current += 1;
        favCache.current.clear();
        setVoiceIssue("");
        clearPreview();
        setStatus(
          warnings.length
            ? "Clone ready, but the reference audio has problems — see the notes above the speak box."
            : "Clone ready. Tap “Test this voice” to check that it sounds like you.",
        );
      }
      setStage("talk");
      refreshCredits();
    } catch {
      setStatus("");
      setError("Could not reach the voice server. Check that the backend is running and try again.");
    }
  }

  async function onFile(ev: React.ChangeEvent<HTMLInputElement>) {
    const file = ev.target.files?.[0];
    ev.target.value = "";
    if (!file) return;
    if (recState === "recording") {
      setError("Stop the current recording before uploading a file.");
      return;
    }
    if (file.size === 0) {
      setError("That file is empty. Choose another audio file.");
      return;
    }
    if (file.size > MAX_SOURCE_MB * 1024 * 1024) {
      setError(
        `That file is over ${MAX_SOURCE_MB} MB. Trim the clip or export just the audio track.`,
      );
      return;
    }
    const looksAudio =
      file.type.startsWith("audio/") ||
      file.type === "video/mp4" ||
      file.type === "video/quicktime" ||
      file.type === "video/3gpp" ||
      /\.(wav|mp3|m4a|mp4|mov|3gp|3gpp|ogg|oga|opus|flac|aac|webm|amr)$/i.test(file.name);
    if (!looksAudio) {
      setError("Choose an audio file (WAV, MP3, M4A, MP4, OGG, FLAC or AAC).");
      return;
    }

    setError("");
    setFileNote("");
    setQualityWarnings([]);
    setQualityScore(null);
    setFileBusy(true);
    setStatus(`Reading ${file.name}…`);
    let ctx: AudioContext | null = null;
    try {
      const mediaSeconds = await probeMediaDuration(file);
      if (mediaSeconds > MAX_SOURCE_SECONDS) {
        setError(
          `That file is ${Math.round(mediaSeconds / 60)} minutes long. Use at most 10 minutes; only the first 30 seconds of speech are used.`,
        );
        setStatus("");
        return;
      }
      const bytes = await file.arrayBuffer();
      ctx = new AudioContext();
      const decoded = await new Promise<AudioBuffer>((resolve, reject) => {
        // Callback form keeps this working on older Safari as well as Chrome/Edge.
        ctx!.decodeAudioData(bytes.slice(0), resolve, reject);
      });
      const channels = decoded.numberOfChannels;
      const mono = new Float32Array(decoded.length);
      for (let c = 0; c < channels; c++) {
        const data = decoded.getChannelData(c);
        for (let i = 0; i < mono.length; i++) mono[i] += data[i] / channels;
      }
      // Speaker embeddings expect a low-rate mono reference. Use the browser's
      // filtered resampler rather than picking every Nth sample.
      const reference = await resampleMono(mono, decoded.sampleRate, REFERENCE_RATE);

      // Same cleaning as a studio take: remove DC/rumble, trim silence,
      // normalise, then send a plain 16 kHz WAV.
      let prepared = prepareVoiceSample(reference, REFERENCE_RATE);
      let duration = prepared.length / REFERENCE_RATE;
      if (duration < 5) {
        setError(
          `This file has only ${duration.toFixed(1)}s of audible speech. Use at least 5 seconds, ideally 20–30.`,
        );
        setStatus("");
        return;
      }
      let note = "";
      if (duration > 30) {
        prepared = prepared.slice(0, Math.floor(REFERENCE_RATE * 30));
        duration = prepared.length / REFERENCE_RATE;
        note = "The file was longer than 30s, so the first 30 seconds will be used.";
      }
      const wav = encodeWav(prepared, REFERENCE_RATE);
      const quality = analyzeSample(reference.slice(0, REFERENCE_RATE * 30), REFERENCE_RATE);
      setPreviewWarnings(quality.warnings);
      setPreviewScore(quality.score);
      showPreview(wav, "voice.wav");
      void persistRecording(wav, duration);
      setFileNote(note);
      setStatus(
        quality.warnings.length
          ? `Loaded ${duration.toFixed(1)}s, but the reference has problems — read the notes below before using it.`
          : `Loaded ${duration.toFixed(1)}s of speech from ${file.name}. Preview it, then tap "Use this voice".`,
      );
    } catch {
      setStatus("");
      setError(
        "This browser could not read the audio in that file. If it is an MP4/MOV video, convert it to MP3 or WAV, or record directly instead.",
      );
    } finally {
      setFileBusy(false);
      closeAudio(ctx);
    }
  }

  function rememberPartner(text: string) {
    const clean = text.trim();
    if (!clean) return false;
    const now = Date.now();
    if (lastTranscript.current.text === clean && now - lastTranscript.current.at < 1500) return false;
    lastTranscript.current = { text: clean, at: now };
    partnerTurns.current = [...partnerTurns.current, { role: "partner", text: clean }].slice(-12);
    return true;
  }

  function rememberUser(text: string) {
    const clean = text.trim();
    if (!clean) return;
    const turns = partnerTurns.current;
    const last = turns[turns.length - 1];
    if (last && last.role === "user" && last.text === clean) return;
    partnerTurns.current = [...turns, { role: "user", text: clean }].slice(-12);
  }

  function persistCustomFavourites(items: FavouriteItem[]) {
    try {
      localStorage.setItem(CUSTOM_FAVOURITES_KEY, JSON.stringify(items));
    } catch {
      /* private mode / quota: keep the in-memory list */
    }
  }

  function addCustomFavourite() {
    const text = favText.trim().replace(/\s+/g, " ").slice(0, 80);
    if (!text) return;
    if (favourites.some((item) => item.text.toLowerCase() === text.toLowerCase())) {
      setError("That phrase is already in favourites.");
      return;
    }
    const item: FavouriteItem = {
      id: `custom-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 6)}`,
      text,
      emotion: favEmotion,
      custom: true,
    };
    const next = [...customFavourites, item];
    setCustomFavourites(next);
    persistCustomFavourites(next);
    setFavText("");
    setError("");
  }

  function removeCustomFavourite(id: string) {
    const next = customFavourites.filter((item) => item.id !== id);
    setCustomFavourites(next);
    persistCustomFavourites(next);
  }

  function sendPcmFrame(socket: WebSocket): boolean {
    if (socket.readyState !== WebSocket.OPEN || pcmQueue.current.length < STT_FRAME_SAMPLES) {
      return false;
    }
    const frame = pcmQueue.current.slice(0, STT_FRAME_SAMPLES);
    pcmQueue.current = pcmQueue.current.slice(STT_FRAME_SAMPLES);
    try {
      socket.send(frame.buffer);
      return true;
    } catch {
      return false;
    }
  }

  function stopPacer() {
    if (listenPacer.current) {
      window.clearInterval(listenPacer.current);
      listenPacer.current = null;
    }
  }

  function startPacer(socket: WebSocket) {
    stopPacer();
    // Gnani wants one 1,024-byte frame every 32 ms. Sending from a timer
    // avoids the bursts produced by a ScriptProcessor callback.
    listenPacer.current = window.setInterval(() => {
      if (socket.readyState !== WebSocket.OPEN) return;
      const frames = pcmQueue.current.length >= STT_FRAME_SAMPLES * 3 ? 2 : 1;
      for (let i = 0; i < frames; i++) {
        if (!sendPcmFrame(socket)) break;
      }
    }, STT_FRAME_MS);
  }

  function stopCapture() {
    stopPacer();
    listenProc.current?.disconnect();
    listenProc.current = null;
    listenSource.current?.disconnect();
    listenSource.current = null;
    listenStream.current?.getTracks().forEach((t) => t.stop());
    listenStream.current = null;
    closeAudio(listenCtx.current);
    listenCtx.current = null;
    pcmQueue.current = new Int16Array(0);
  }

  function startBrowserSpeech() {
    if (browserRecActive.current || closingListen.current) return;
    const win = window as Window & {
      SpeechRecognition?: new () => BrowserRec;
      webkitSpeechRecognition?: new () => BrowserRec;
    };
    const SR = win.SpeechRecognition || win.webkitSpeechRecognition;
    if (!SR) {
      addFallback("browser_speech_unavailable");
      setError("This browser has no speech recognition. Use Chrome or Edge for live listening.");
      return;
    }
    addFallback("browser_speech");
    const rec = new SR();
    rec.continuous = true;
    rec.interimResults = true;
    rec.lang = listenLang === "indic-auto" ? "en-IN" : listenLang;
    rec.onresult = (raw) => {
      const event = raw as {
        resultIndex: number;
        results: { length: number; [i: number]: { isFinal: boolean; 0: { transcript: string } } };
      };
      let finalText = "";
      let interim = "";
      for (let i = event.resultIndex; i < event.results.length; i++) {
        const t = event.results[i][0].transcript;
        if (event.results[i].isFinal) finalText += t;
        else interim += t;
      }
      if (interim) setLiveText(interim);
      if (finalText) {
        const text = finalText.trim();
        if (!text) return;
        setLiveText(text);
        const added = rememberPartner(text);
        const socket = listenWs.current;
        if (socket?.readyState === WebSocket.OPEN) {
          try {
            socket.send(JSON.stringify({ type: "browser_transcript", text }));
          } catch {
            /* socket closed between checks */
          }
        }
        if (added) scheduleSuggestions();
      }
    };
    rec.onerror = (event) => {
      const code = event?.error;
      if (code === "not-allowed" || code === "service-not-allowed") {
        setError("Speech recognition permission was denied.");
      } else if (code === "audio-capture") {
        setError("No microphone found for speech recognition.");
      }
    };
    rec.onend = () => {
      browserRecActive.current = false;
      if (!closingListen.current && listeningRef.current && !browserRecRestart.current) {
        browserRecRestart.current = window.setTimeout(() => {
          browserRecRestart.current = null;
          startBrowserSpeech();
        }, 500);
      }
    };
    try {
      rec.start();
      browserRecActive.current = true;
      speechRec.current = rec;
      setListenState("fallback");
    } catch {
      browserRecActive.current = false;
      setError("Could not start browser speech recognition. Stop listening and try again.");
    }
  }

  function stopBrowserSpeech() {
    if (browserRecRestart.current) {
      window.clearTimeout(browserRecRestart.current);
      browserRecRestart.current = null;
    }
    browserRecActive.current = false;
    const rec = speechRec.current;
    speechRec.current = null;
    if (rec) {
      try {
        rec.stop();
      } catch {
        /* already stopped */
      }
    }
  }

  async function startListen() {
    if (!sessionId || listeningRef.current) return;
    closingListen.current = false;
    listeningRef.current = true;
    setListening(true);
    setListenState("connecting");
    setError("");
    setLiveText("");
    setStatus("");
    stopBrowserSpeech();

    let stream: MediaStream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });
    } catch {
      listeningRef.current = false;
      setListening(false);
      setListenState("idle");
      setError("Microphone unavailable. Allow microphone access and try again.");
      return;
    }

    if (!listeningRef.current) {
      stream.getTracks().forEach((t) => t.stop());
      return;
    }
    listenStream.current = stream;
    const socket = new WebSocket(
      wsUrl(
        `/ws/listen?session_id=${encodeURIComponent(sessionId)}&language=${encodeURIComponent(listenLang)}`,
      ),
    );
    listenWs.current = socket;
    socket.onmessage = (ev) => {
      let frame: {
        type?: string;
        text?: string;
        message?: string;
        fallback_kind?: string;
      };
      try {
        frame = JSON.parse(String(ev.data));
      } catch {
        return;
      }
      if (frame.type === "ready") {
        setListenState("connecting");
        setStatus("Connecting to Prisma…");
      } else if (frame.type === "connected") {
        setListenState("live");
        setStatus("");
      } else if (frame.type === "speech_start") {
        setStatus("Hearing speech…");
      } else if (frame.type === "processing") {
        setStatus("Transcribing…");
      } else if (frame.type === "transcript") {
        const text = (frame.text || "").trim();
        if (text) {
          setLiveText(text);
          const added = rememberPartner(text);
          setStatus("");
          if (added) scheduleSuggestions();
        }
      } else if (frame.type === "fallback") {
        addFallback(frame.fallback_kind || "browser_speech");
        setListenState("fallback");
        stopCapture();
        startBrowserSpeech();
      } else if (frame.type === "error") {
        setError(frame.message || "Live transcription error.");
      }
    };
    socket.onerror = () => {
      if (!closingListen.current) {
        addFallback("browser_speech");
        stopCapture();
        startBrowserSpeech();
      }
    };
    socket.onclose = () => {
      if (!closingListen.current && listeningRef.current) {
        addFallback("browser_speech");
        setListenState("fallback");
        stopCapture();
        startBrowserSpeech();
      }
    };

    if (!listeningRef.current || listenStream.current !== stream) {
      if (!listeningRef.current) {
        stopCapture();
        if (socket.readyState < WebSocket.CLOSING) socket.close();
      }
      return;
    }

    try {
      const ctx = new AudioContext();
      listenCtx.current = ctx;
      const source = ctx.createMediaStreamSource(stream);
      listenSource.current = source;
      const proc = ctx.createScriptProcessor(4096, 1, 1);
      listenProc.current = proc;
      proc.onaudioprocess = (ev) => {
        const raw = ev.inputBuffer.getChannelData(0);
        const ds = downsample(raw, ctx.sampleRate, STT_RATE);
        const pcm = floatToPcm16(ds);
        const combined = new Int16Array(pcmQueue.current.length + pcm.length);
        combined.set(pcmQueue.current);
        combined.set(pcm, pcmQueue.current.length);
        // Keep at most four seconds so a stalled socket cannot grow memory.
        const max = STT_RATE * 4;
        pcmQueue.current =
          combined.length > max ? combined.slice(combined.length - max) : combined;
      };
      const mute = ctx.createGain();
      mute.gain.value = 0;
      source.connect(proc);
      proc.connect(mute);
      mute.connect(ctx.destination);
      startPacer(socket);
    } catch {
      stopCapture();
      if (socket.readyState < WebSocket.CLOSING) socket.close();
      listeningRef.current = false;
      setListening(false);
      setListenState("idle");
      setError("Could not start audio capture. Try Chrome or Edge, then retry.");
    }
  }

  function stopListen() {
    if (!listeningRef.current && !listenWs.current && !listenStream.current) return;
    closingListen.current = true;
    listeningRef.current = false;
    setListening(false);
    setListenState("idle");
    setStatus("");

    const socket = listenWs.current;
    listenWs.current = null;
    if (socket) {
      if (socket.readyState === WebSocket.OPEN) {
        // Flush anything still queued, ask Prisma to finalise the utterance,
        // then give it a moment to return the last transcript.
        while (pcmQueue.current.length >= STT_FRAME_SAMPLES) sendPcmFrame(socket);
        try {
          socket.send(JSON.stringify({ type: "commit" }));
        } catch {
          /* ignore */
        }
        window.setTimeout(() => {
          if (socket.readyState < WebSocket.CLOSING) socket.close();
        }, 1000);
      } else if (socket.readyState < WebSocket.CLOSING) {
        socket.close();
      }
    }
    stopCapture();
    stopBrowserSpeech();
  }

  function scheduleSuggestions() {
    // Coalesce a burst of final transcripts into one LLM request.
    if (suggestTimer.current) window.clearTimeout(suggestTimer.current);
    suggestTimer.current = window.setTimeout(() => {
      suggestTimer.current = null;
      void loadSuggestions();
    }, 250);
  }

  async function loadSuggestions() {
    if (!sessionId) return;
    // Ignore a slow older response if a newer transcript already arrived.
    const seq = ++suggestSeq.current;
    const turns = partnerTurns.current.slice(-8);
    try {
      const res = await api("/api/suggest", sessionId, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(
          turns.length
            ? {
                turns,
                ...(userGenderRef.current === "neutral"
                  ? {}
                  : { user_gender: userGenderRef.current }),
              }
            : {},
        ),
      });
      if (!res.ok || seq !== suggestSeq.current) return;
      const data = await res.json();
      if (seq !== suggestSeq.current) return;
      if (data.fallback) addFallback("canned_replies");
      setReplies(data.replies || []);
    } catch {
      /* suggestions are non-critical; keep the previous options */
    }
  }

  function playBlob(blob: Blob): Promise<void> {
    return new Promise((resolve, reject) => {
      const url = URL.createObjectURL(blob);
      if (audioUrl.current) URL.revokeObjectURL(audioUrl.current);
      audioUrl.current = url;
      // Use a new element for each clip. Reusing one element meant the pause
      // event from the previous source could fire on the new handlers and
      // release the speaking lock while the new clip was still playing, which
      // made a single tap sound doubled.
      const previous = audioRef.current;
      if (previous) {
        previous.onended = null;
        previous.onerror = null;
        previous.onpause = null;
        previous.pause();
        previous.removeAttribute("src");
        previous.load();
      }
      const audio = new Audio();
      audioRef.current = audio;
      let settled = false;
      const finish = (action: () => void) => {
        if (settled) return;
        settled = true;
        URL.revokeObjectURL(url);
        if (audioUrl.current === url) audioUrl.current = "";
        if (audioRef.current === audio) audioRef.current = null;
        action();
      };
      audio.onended = () => finish(resolve);
      audio.onerror = () =>
        finish(() => reject(new Error("Could not play the generated audio.")));
      // A voice change or unmount pauses the element; resolve so speakText
      // does not stay stuck with its "speaking" lock held.
      audio.onpause = () => {
        if (audio.ended || !audio.paused) return;
        finish(resolve);
      };
      audio.src = url;
      audio.play().catch((err) => finish(() => reject(err)));
    });
  }

  async function speakText(text: string, emo: Emotion, recordTurn = true) {
    // State updates are async, so a fast double tap could pass a `speaking`
    // check twice. A ref guard is atomic and prevents overlapping playback.
    if (speakingRef.current) return;
    const clean = text.trim();
    if (!clean) return;
    speakingRef.current = true;
    setSpeaking(true);
    setError("");
    const key = `${clean}|${emo}|v${voiceVersion.current}`;
    try {
      let blob = favCache.current.get(key);
      if (!blob) {
        const res = await api("/api/speak", sessionId, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text: clean, emotion: emo, language: speakLang }),
        });
        if (!res.ok) {
          const detail: { detail?: string } = await res.json().catch(() => ({}));
          throw new Error(detail.detail || `Speak failed (${res.status})`);
        }
        if (res.headers.get("X-Fallback") === "true") addFallback("stock_voice");
        setAiLabel(res.headers.get("X-AI-Generated") === "true");
        blob = await res.blob();
        favCache.current.set(key, blob);
      }
      await playBlob(blob);
      if (recordTurn) rememberUser(clean);
      refreshCredits();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      speakingRef.current = false;
      setSpeaking(false);
    }
  }

  async function playFavourite(fav: Favourite) {
    await speakText(fav.text, fav.emotion);
  }

  const uploading = status.startsWith("Creating voice embedding");
  return (
    <main className="wrap">
      <div className="brand">APNI AWAAZ</div>
      <h1>Give your voice back</h1>
      <p className="lede">
        Type or tap a reply. The app speaks it in a clone of your voice.
      </p>
      <div className="row" style={{ marginBottom: "1rem" }}>
        {aiLabel && <span className="badge">AI-generated audio</span>}
        {fallbacks.map((f) => (
          <span className="badge warn" key={f}>
            Fallback: {f.replaceAll("_", " ")}
          </span>
        ))}
        {credits && <span className="badge">{credits}</span>}
      </div>
      {error && <p className="error">{error}</p>}

      {stage === "consent" && (
        <section className="card">
          <h2>Before we clone a voice</h2>
          <label className="consent">
            <input type="checkbox" checked={agreed} onChange={(e) => setAgreed(e.target.checked)} />
            <span>This is my own voice, or I have the speaker&apos;s permission.</span>
          </label>
          <div className="row" style={{ marginTop: "1rem" }}>
            <button disabled={!agreed || !sessionId} onClick={submitConsent}>
              Continue
            </button>
            <button className="secondary" disabled={!agreed || !sessionId} onClick={startDemo}>
              One-click demo
            </button>
          </div>
        </section>
      )}

      {stage === "enrol" && (
        <section className="card">
          <h2>Voice enrolment</h2>
          <p className="lede">
            Record 15–30 seconds of natural speech in a quiet room. Use one person only —
            no music, no singing, no TV, no second voice. Speak the same language you will
            use for replies. A clean, close-mic take makes the biggest difference to how
            closely the clone matches you.
          </p>
          <p className="meter">
            Good reference: continuous talking, normal volume, about a hand-width from the
            mic. Bad reference: whispering, shouting, singing, background music, or a video
            with several speakers.
          </p>
          <div className="row">
            {recState !== "recording" ? (
              <button onClick={startRecord} disabled={uploading || fileBusy}>
                {previewUrl ? "Record again" : "Record"}
              </button>
            ) : (
              <button onClick={stopRecord}>Stop ({seconds}s)</button>
            )}
            <button
              type="button"
              className="secondary"
              onClick={() => fileInput.current?.click()}
              disabled={fileBusy || uploading || recState === "recording"}
            >
              {fileBusy ? "Reading file…" : "Upload audio file"}
            </button>
            <input
              ref={fileInput}
              type="file"
              accept="audio/*,video/mp4,video/quicktime,video/3gpp,.m4a,.mp3,.wav,.ogg,.oga,.opus,.flac,.aac,.webm,.amr,.mp4,.mov,.3gp,.3gpp"
              hidden
              onChange={onFile}
            />
            {cloneReady && (
              <button
                type="button"
                className="secondary"
                disabled={uploading || recState === "recording"}
                onClick={() => {
                  clearPreview();
                  setStage("talk");
                }}
              >
                Use previous voice
              </button>
            )}
          </div>
          {fileNote && <p className="meter">{fileNote}</p>}
          {savingRecording && <p className="meter">Saving MP4 copy…</p>}
          {!savingRecording && recordingNote && (
            <p className="meter">{recordingNote}</p>
          )}
          {savedRecordings.length > 0 && (
            <div className="recordings">
              <p className="meter">
                <strong>Saved recordings</strong> — every take is kept as an MP4
                file on the server.
              </p>
              <ul>
                {savedRecordings.map((recording) => (
                  <li key={recording.name}>
                    <div className="recording-line">
                      <span>{new Date(recording.created_at).toLocaleString()}</span>
                      {recording.seconds ? (
                        <span>{recording.seconds.toFixed(1)}s</span>
                      ) : null}
                      <a href={`${API_URL}${recording.url}`} download={recording.name}>
                        Download MP4
                      </a>
                    </div>
                    <audio
                      controls
                      preload="none"
                      src={`${API_URL}${recording.url}?inline=1`}
                    />
                  </li>
                ))}
              </ul>
            </div>
          )}
          <div className="level" aria-hidden="true">
            <div style={{ width: `${recLevel}%` }} />
          </div>
          <p className="meter">
            {recState === "recording"
              ? `Recording… ${seconds}s / 30s. Keep the mic about a hand-width away.`
              : status}
          </p>
          {previewUrl && (
            <div className="preview">
              <p className="meter">
                Preview this recording before creating the clone. It must be one person
                speaking, with no music, singing, or second voice. If it is noisy, echoey,
                or quiet, record again.
              </p>
              {previewWarnings.length > 0 && (
                <div className="quality">
                  <p>
                    <strong>
                      This reference may clone poorly
                      {previewScore !== null ? ` · ${previewScore}/100` : ""}
                    </strong>
                  </p>
                  <ul>
                    {previewWarnings.map((warning) => (
                      <li key={warning}>{warning}</li>
                    ))}
                  </ul>
                  <p className="meter">
                    Fix the issue and record or upload again — the clone can only be as
                    good as this clip.
                  </p>
                </div>
              )}
              <audio controls src={previewUrl} />
              <div className="row" style={{ marginTop: "0.7rem" }}>
                <button onClick={confirmEnrol} disabled={uploading}>
                  {uploading ? "Creating voice…" : "Use this voice"}
                </button>
                <button className="secondary" onClick={startRecord} disabled={uploading}>
                  Record again
                </button>
              </div>
            </div>
          )}
        </section>
      )}

      {stage === "talk" && (
        <>
          <section className="card">
            <h2>Listen</h2>
            <p className="lede">
              {cloneReady
                ? "Your cloned voice is ready."
                : "Cloning has not succeeded, so replies currently use a stock Timbre voice."}
            </p>
            {qualityWarnings.length > 0 && (
              <div className="quality">
                <p>
                  <strong>
                    Reference audio check{qualityScore !== null ? ` · ${qualityScore}/100` : ""}
                  </strong>
                  {" — "}the clone may not match you closely because:
                </p>
                <ul>
                  {qualityWarnings.map((warning) => (
                    <li key={warning}>{warning}</li>
                  ))}
                </ul>
                <button
                  className="secondary"
                  onClick={() => {
                    clearPreview();
                    setStage("enrol");
                  }}
                >
                  Record a better sample
                </button>
              </div>
            )}
            <div className="row" style={{ marginBottom: "0.8rem" }}>
              {cloneReady && (
                <button
                  className="secondary"
                  disabled={speaking}
                  onClick={() =>
                    speakText(
                      "Hello, this is a test of my cloned voice. Does it sound like the person you recorded?",
                      emotion,
                      false,
                    )
                  }
                >
                  {speaking ? "Testing…" : "Test this voice"}
                </button>
              )}
              <button
                className="secondary"
                onClick={() => {
                  clearPreview();
                  setStage("enrol");
                }}
              >
                {voiceIssue && !cloneReady ? "Open voice setup" : "Record voice again"}
              </button>
            </div>
            {cloneReady && (
              <p className="meter">
                Recording a new sample replaces this cloned voice for the session.
              </p>
            )}
            {!cloneReady && voiceIssue && (
              <p className="meter">
                Voice cloning is blocked by the backend API key. Fix{" "}
                <code>GNANI_API_KEY</code> in <code>backend/.env</code> and restart the
                backend — recording again will fail until then.
              </p>
            )}
            <div className="row">
              <label className="field">
                Listener language
                <select value={listenLang} onChange={(e) => setListenLang(e.target.value)}>
                  <option value="indic-auto">Auto (Indian languages)</option>
                  <option value="hi-IN">Hindi</option>
                  <option value="en-IN">English</option>
                  <option value="hi-IN,en-IN">Hindi + English</option>
                  <option value="bn-IN">Bengali</option>
                  <option value="ta-IN">Tamil</option>
                  <option value="te-IN">Telugu</option>
                  <option value="mr-IN">Marathi</option>
                  <option value="gu-IN">Gujarati</option>
                  <option value="kn-IN">Kannada</option>
                  <option value="ml-IN">Malayalam</option>
                  <option value="pa-IN">Punjabi</option>
                </select>
              </label>
              {!listening ? (
                <button onClick={startListen} disabled={listenState === "connecting"}>
                  {listenState === "connecting" ? "Connecting…" : "Start listening"}
                </button>
              ) : (
                <button className="secondary" onClick={stopListen}>
                  Stop listening
                </button>
              )}
            </div>
            <p className="meter">
              {listenState === "connecting"
                ? "Connecting to Prisma…"
                : listenState === "live"
                  ? "Live · Prisma v3.0"
                  : listenState === "fallback"
                    ? "Browser speech fallback is active"
                    : status}
            </p>
            <p className="transcript" aria-live="polite">
              {liveText || "Partner speech will appear here."}
            </p>
          </section>

          <section className="card">
            <div className="row" style={{ justifyContent: "space-between", alignItems: "center" }}>
              <h2>Suggest</h2>
              <button className="secondary" onClick={() => void loadSuggestions()}>
                Refresh
              </button>
            </div>
            <div className="row">
              {replies.map((r) => (
                <button
                  className="option"
                  key={r.text}
                  disabled={speaking}
                  onClick={() => speakText(r.text, r.emotion)}
                >
                  <small>{r.emotion}</small>
                  {r.text}
                </button>
              ))}
            </div>
            <div className="row" style={{ marginTop: "0.8rem", alignItems: "flex-end" }}>
              <label className="field">
                Reply as
                <select
                  value={userGender}
                  onChange={(e) => {
                    const value = e.target.value as "neutral" | "female" | "male";
                    setUserGender(value);
                    userGenderRef.current = value;
                    try {
                      localStorage.setItem(USER_GENDER_KEY, value);
                    } catch {
                      /* private mode */
                    }
                  }}
                >
                  <option value="neutral">Not set</option>
                  <option value="female">Female</option>
                  <option value="male">Male</option>
                </select>
              </label>
              <span className="meter">
                Keeps Hindi/Hinglish verb forms right. Saved in this browser.
              </span>
            </div>
          </section>

          <section className="card">
            <h2>Speak</h2>
            <textarea
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              placeholder="Type your own reply"
            />
            <div className="row" style={{ marginTop: "0.7rem" }}>
              <label className="field">
                Language
                <select value={speakLang} onChange={(e) => setSpeakLang(e.target.value)}>
                  <option value="auto">Auto</option>
                  <option value="en-IN">English</option>
                  <option value="hi-IN">Hindi</option>
                  <option value="hi-en">Hinglish</option>
                  <option value="bn-IN">Bengali</option>
                  <option value="ta-IN">Tamil</option>
                  <option value="te-IN">Telugu</option>
                  <option value="mr-IN">Marathi</option>
                  <option value="gu-IN">Gujarati</option>
                  <option value="kn-IN">Kannada</option>
                  <option value="ml-IN">Malayalam</option>
                  <option value="pa-IN">Punjabi</option>
                </select>
              </label>
              {(["warm", "urgent", "playful", "calm"] as Emotion[]).map((e) => (
                <button
                  key={e}
                  className="secondary"
                  onClick={() => setEmotion(e)}
                  style={{ outline: emotion === e ? "2px solid var(--accent-2)" : undefined }}
                >
                  {e}
                </button>
              ))}
              <button disabled={speaking || !draft.trim()} onClick={() => speakText(draft, emotion)}>
                {speaking ? "Speaking…" : "Speak"}
              </button>
            </div>
            <p className="ai">Cloned audio is AI-generated.</p>
          </section>

          <section className="card">
            <div className="row" style={{ justifyContent: "space-between", alignItems: "center" }}>
              <h2>Favourites</h2>
              <span className="meter">Tap to speak</span>
            </div>
            <div className="fav-grid">
              {favourites.map((f) => (
                <div className="fav-item" key={f.id}>
                  <button
                    className="fav"
                    disabled={speaking}
                    onClick={() => playFavourite(f)}
                  >
                    {f.text}
                  </button>
                  {f.custom && (
                    <button
                      type="button"
                      className="fav-remove"
                      aria-label={`Remove ${f.text} from favourites`}
                      title="Remove"
                      onClick={() => removeCustomFavourite(f.id)}
                    >
                      ×
                    </button>
                  )}
                </div>
              ))}
            </div>
            <div className="fav-add">
              <input
                value={favText}
                onChange={(e) => setFavText(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") {
                    e.preventDefault();
                    addCustomFavourite();
                  }
                }}
                placeholder="Add your own words or phrase"
                maxLength={80}
                aria-label="New favourite phrase"
              />
              <select
                value={favEmotion}
                onChange={(e) => setFavEmotion(e.target.value as Emotion)}
                aria-label="Emotion for the new phrase"
              >
                <option value="warm">warm</option>
                <option value="calm">calm</option>
                <option value="urgent">urgent</option>
                <option value="playful">playful</option>
              </select>
              <button
                className="secondary"
                onClick={addCustomFavourite}
                disabled={!favText.trim()}
              >
                Add
              </button>
            </div>
            <p className="meter">
              Add anything you say often. Custom phrases are saved in this browser.
            </p>
          </section>
        </>
      )}

    </main>
  );
}
