"use client";

import { useCallback, useEffect, useRef, useState, type ReactNode } from "react";
import AnomalyOrb from "./AnomalyOrb";
import { API_URL, api, type Emotion, type Favourite, type Reply, wsUrl } from "@/lib/api";
import { attachAudio } from "@/lib/audioMonitor";
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

type BuddyMood = "idle" | "speaking" | "listening";

type Turn = { role: "partner" | "user"; text: string };

function MicIcon() {
  return (
    <svg
      viewBox="0 0 24 24"
      width="20"
      height="20"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.7}
      strokeLinecap="round"
      aria-hidden="true"
    >
      <rect x="9" y="3" width="6" height="10.5" rx="3" />
      <path d="M5 11a7 7 0 0 0 14 0" />
      <path d="M12 18v3" />
    </svg>
  );
}

function PlayIcon() {
  return (
    <svg viewBox="0 0 24 24" width="13" height="13" fill="currentColor" aria-hidden="true">
      <path d="M8 5.4v13.2L19 12z" />
    </svg>
  );
}

function RefreshIcon() {
  return (
    <svg
      viewBox="0 0 24 24"
      width="18"
      height="18"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.8}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d="M19.5 12a7.5 7.5 0 1 1-2.2-5.3" />
      <path d="M19.5 4.5V9H15" />
    </svg>
  );
}

function SendIcon() {
  return (
    <svg
      viewBox="0 0 24 24"
      width="18"
      height="18"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.8}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d="M5 12h13" />
      <path d="M12 5l7 7-7 7" />
    </svg>
  );
}

function PlusIcon() {
  return (
    <svg
      viewBox="0 0 24 24"
      width="18"
      height="18"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.8}
      strokeLinecap="round"
      aria-hidden="true"
    >
      <path d="M12 5v14M5 12h14" />
    </svg>
  );
}

/**
 * The 3D voice core sits at the centre of the page. It idles, perks up while
 * it listens, and pulses with the cloned audio while it speaks.
 */
function VoiceCore({
  mood,
  action,
}: {
  mood: BuddyMood;
  action?: ReactNode;
}) {
  const line =
    mood === "speaking"
      ? "Speaking in your voice"
      : mood === "listening"
        ? "Listening to you"
        : "Hi, I'm your voice";
  return (
    <div className="voice-core">
      <div className="orb-stage">
        <div className="orb-rings" aria-hidden="true">
          <span />
          <span />
          <span />
        </div>
        <AnomalyOrb mood={mood} />
      </div>
      <div className={`equalizer ${mood}`} aria-hidden="true">
        <span />
        <span />
        <span />
        <span />
        <span />
      </div>
      <p className="orb-caption" role="status" aria-live="polite">
        <span className={`status-dot ${mood}`} aria-hidden="true" />
        {line}
      </p>
      {action}
    </div>
  );
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
  const [thread, setThread] = useState<Turn[]>([]);
  const [heroInView, setHeroInView] = useState(true);
  const [favOpen, setFavOpen] = useState(false);

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
  const threadRef = useRef<HTMLElement | null>(null);
  const favCache = useRef<Map<string, Blob>>(new Map());
  const speechRec = useRef<BrowserRec | null>(null);
  const clockRef = useRef<HTMLSpanElement | null>(null);

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
    const tick = () => {
      if (!clockRef.current) return;
      const now = new Date();
      clockRef.current.textContent = `TIME ${now.toLocaleTimeString("en-GB", { hour12: false })}`;
    };
    tick();
    const id = window.setInterval(tick, 1000);
    return () => window.clearInterval(id);
  }, []);

  // The dock only appears once the hero has scrolled away. The hero markup is
  // observed from the outside so its component stays untouched.
  useEffect(() => {
    const hero = document.querySelector(".hero");
    if (!hero || typeof IntersectionObserver === "undefined") return;
    const observer = new IntersectionObserver(
      ([entry]) => setHeroInView(entry.isIntersecting),
      { threshold: 0 },
    );
    observer.observe(hero);
    return () => observer.disconnect();
  }, []);

  function scrollToThread() {
    threadRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
  }

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
    // Mirror the turn into render state for the chat thread. The suggestion
    // context above still reads the ref, so its behaviour is unchanged.
    // Progressive transcripts replace the previous partner bubble instead of
    // stacking near-duplicates.
    setThread((turns) => {
      const previous = turns[turns.length - 1];
      if (
        previous &&
        previous.role === "partner" &&
        (clean.startsWith(previous.text) || previous.text.startsWith(clean))
      ) {
        return [...turns.slice(0, -1), { role: "partner" as const, text: clean }];
      }
      return [...turns, { role: "partner" as const, text: clean }].slice(-40);
    });
    return true;
  }

  function rememberUser(text: string) {
    const clean = text.trim();
    if (!clean) return;
    const turns = partnerTurns.current;
    const last = turns[turns.length - 1];
    if (last && last.role === "user" && last.text === clean) return;
    partnerTurns.current = [...turns, { role: "user", text: clean }].slice(-12);
    setThread((turns) => [...turns, { role: "user" as const, text: clean }].slice(-40));
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
      // Route the clip through the shared analyser so the 3D voice core
      // reacts to the real audio. If it is unavailable, play normally.
      void attachAudio(audio)
        .catch(() => false)
        .then(() => {
          if (settled) return;
          audio.play().catch((err) => finish(() => reject(err)));
        });
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
  const mood: BuddyMood = speaking ? "speaking" : listening ? "listening" : "idle";
  const dockVisible = stage === "talk" && !heroInView;

  // The record action lives under the voice core, so the centre of the page is
  // also the place you start and stop a take.
  const heroAction =
    stage === "enrol" ? (
      <div className={`orb-action ${recState === "recording" ? "live" : ""}`}>
        {recState === "recording" ? (
          <button
            type="button"
            className="record-dot live"
            onClick={stopRecord}
            aria-label="Stop recording"
          >
            <span className="record-stop" aria-hidden="true" />
          </button>
        ) : (
          <button
            type="button"
            className="record-dot"
            onClick={startRecord}
            disabled={uploading || fileBusy}
            aria-label={previewUrl ? "Record again" : "Start recording"}
          >
            <span className="record-glyph" aria-hidden="true" />
          </button>
        )}
        <span className="orb-action-label">
          {recState === "recording"
            ? `Stop · ${seconds}s`
            : previewUrl
              ? "Record again"
              : "Record your voice"}
        </span>
      </div>
    ) : stage === "talk" ? (
      <div className={`orb-action ${listening ? "live" : ""}`}>
        {listening ? (
          <button
            type="button"
            className="record-dot live"
            onClick={stopListen}
            aria-label="Stop listening"
          >
            <span className="record-stop" aria-hidden="true" />
          </button>
        ) : (
          <button
            type="button"
            className="record-dot"
            onClick={() => {
              void startListen();
              scrollToThread();
            }}
            disabled={listenState === "connecting"}
            aria-label="Start listening"
          >
            <MicIcon />
          </button>
        )}
        <span className="orb-action-label">
          {listening
            ? "Listening · tap to stop"
            : listenState === "connecting"
              ? "Connecting…"
              : "Start listening"}
        </span>
      </div>
    ) : null;

  return (
    <main className="site">
      <div className="space-bg" aria-hidden="true" />
      <div className="grid-overlay" aria-hidden="true" />
      <div className="wrap">
        <div className="topbar">
          <span className="logo">
            <span className="logo-mark" aria-hidden="true" />
            APNI AWAAZ
          </span>
          <div className="topbar-right">
            <span className={`topbar-pill ${cloneReady ? "on" : ""}`}>
              {cloneReady ? "CLONE ONLINE" : "CLONE PENDING"}
            </span>
            <span ref={clockRef} className="topbar-time">
              TIME --:--:--
            </span>
          </div>
        </div>
        <header className="hero">
          <div className="hero-copy">
            <span className="eyebrow">
              APNI AWAAZ <i>//</i> VOICE SYNTHESIS
            </span>
            <h1>
              Give your
              <br />
              voice back
            </h1>
            <p className="lede">
              Type or tap a reply. The core speaks it in a clone of your voice —
              one take, kept across restarts, no re-recording.
            </p>
            <div className="hero-meta">
              <span className={`meta ${cloneReady ? "good" : ""}`}>
                {cloneReady ? "CLONE READY" : "CLONE PENDING"}
              </span>
              <span className="meta">
                {listenState === "live" ? "INPUT LIVE" : "INPUT IDLE"}
              </span>
            </div>
          </div>
          <VoiceCore mood={mood} action={heroAction} />
          <aside className="hero-panel" aria-label="Voice core status">
            <div className="hero-panel-title">VOICE CORE</div>
            <div className="status-item">
              <span>Clone</span>
              <strong>{cloneReady ? "Ready" : "Pending"}</strong>
            </div>
            <div className="status-item">
              <span>Input</span>
              <strong>{listenState === "live" ? "Live" : "Idle"}</strong>
            </div>
            <div className="status-item">
              <span>Engine</span>
              <strong>Gnani Timbre</strong>
            </div>
            <div className="status-item">
              <span>Language</span>
              <strong>{speakLang === "auto" ? "Auto" : speakLang}</strong>
            </div>
            <div className="status-item">
              <span>Spend</span>
              <strong>{credits || "—"}</strong>
            </div>
          </aside>
        </header>
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
        <div className="surface">
          <section className="thread" ref={threadRef} aria-label="Conversation">
            {thread.length === 0 && !liveText.trim() ? (
              <p className="thread-empty">Partner speech will appear here.</p>
            ) : (
              <ol className="bubbles">
                {thread.map((turn, index) =>
                  turn.role === "partner" ? (
                    <li className="bubble partner" key={`${turn.role}-${index}`}>
                      {turn.text}
                    </li>
                  ) : (
                    <li className="bubble mine" key={`${turn.role}-${index}`}>
                      <button
                        type="button"
                        className="bubble-replay"
                        disabled={speaking}
                        onClick={() => speakText(turn.text, emotion, false)}
                        aria-label={`Replay in your voice: ${turn.text}`}
                      >
                        <span>{turn.text}</span>
                        <PlayIcon />
                      </button>
                    </li>
                  ),
                )}
                {liveText.trim() &&
                thread[thread.length - 1]?.text !== liveText.trim() ? (
                  <li className="bubble partner live">{liveText.trim()}</li>
                ) : null}
              </ol>
            )}
            <p className="thread-status" role="status" aria-live="polite">
              {listenState === "connecting"
                ? "Connecting to Prisma…"
                : listenState === "fallback"
                  ? "Browser speech fallback is active"
                  : status || (listenState === "live" ? "Listening · Prisma v3.0" : "")}
            </p>
          </section>

          <div className="surface-divider" aria-hidden="true" />

          <section className="surface-block" aria-label="Suggestions">
            <div className="block-head">
              <span className="surface-label">Suggestions</span>
              <button
                type="button"
                className="icon-btn"
                onClick={() => void loadSuggestions()}
                aria-label="Refresh suggestions"
                title="Refresh suggestions"
              >
                <RefreshIcon />
              </button>
            </div>
            <div className="chip-scroll">
              {replies.length === 0 ? (
                <p className="chip-empty">
                  No suggestions yet — start listening, then tap refresh.
                </p>
              ) : (
                replies.map((r) => (
                  <button
                    type="button"
                    className="chip"
                    key={r.text}
                    disabled={speaking}
                    onClick={() => speakText(r.text, r.emotion)}
                  >
                    <span className="chip-tone">{r.emotion}</span>
                    {r.text}
                  </button>
                ))
              )}
            </div>
          </section>

          <div className="surface-divider" aria-hidden="true" />

          <section className="surface-block" aria-label="Favourites">
            <div className="block-head">
              <span className="surface-label">Favourites</span>
              <button
                type="button"
                className="icon-btn"
                onClick={() => setFavOpen((open) => !open)}
                aria-label={favOpen ? "Close the favourite editor" : "Add or remove favourites"}
                aria-expanded={favOpen}
                title={favOpen ? "Close the favourite editor" : "Add or remove favourites"}
              >
                <PlusIcon />
              </button>
            </div>
            <div className="chip-scroll">
              {favourites.map((f) => (
                <span className="chip-wrap" key={f.id}>
                  <button
                    type="button"
                    className="chip chip-plain"
                    disabled={speaking}
                    onClick={() => playFavourite(f)}
                  >
                    {f.text}
                  </button>
                  {f.custom && favOpen ? (
                    <button
                      type="button"
                      className="chip-remove"
                      aria-label={`Remove ${f.text} from favourites`}
                      title="Remove"
                      onClick={() => removeCustomFavourite(f.id)}
                    >
                      ×
                    </button>
                  ) : null}
                </span>
              ))}
            </div>
            {favOpen && (
              <div className="fav-edit">
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
                  aria-label="Tone for the new phrase"
                >
                  <option value="warm">warm</option>
                  <option value="calm">calm</option>
                  <option value="urgent">urgent</option>
                  <option value="playful">playful</option>
                </select>
                <button
                  type="button"
                  className="chip-add"
                  onClick={addCustomFavourite}
                  disabled={!favText.trim()}
                >
                  Add
                </button>
              </div>
            )}
          </section>

          {dockVisible && (
            <div className="dock" role="region" aria-label="Reply dock">
              <div className="dock-inner">
                <button
                  type="button"
                  className={`dock-mic ${listening ? "live" : ""}`}
                  onClick={listening ? stopListen : startListen}
                  disabled={!listening && listenState === "connecting"}
                  aria-label={listening ? "Stop listening" : "Start listening"}
                  title={listening ? "Stop listening" : "Start listening"}
                >
                  {listening ? (
                    <span className="dock-stop" aria-hidden="true" />
                  ) : (
                    <MicIcon />
                  )}
                </button>
                <input
                  className="dock-input"
                  value={draft}
                  onChange={(e) => setDraft(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && draft.trim() && !speaking) {
                      e.preventDefault();
                      void speakText(draft, emotion);
                    }
                  }}
                  placeholder="Type a reply"
                  aria-label="Your reply"
                />
                <span className="dock-tone">
                  <select
                    value={emotion}
                    onChange={(e) => setEmotion(e.target.value as Emotion)}
                    aria-label="Tone"
                  >
                    <option value="warm">warm</option>
                    <option value="urgent">urgent</option>
                    <option value="playful">playful</option>
                    <option value="calm">calm</option>
                  </select>
                </span>
                <button
                  type="button"
                  className="dock-send"
                  disabled={speaking || !draft.trim()}
                  onClick={() => void speakText(draft, emotion)}
                  aria-label="Speak this reply"
                  title="Speak this reply"
                >
                  <SendIcon />
                </button>
              </div>
            </div>
          )}
        </div>
      )}
      </div>
    </main>
  );
}
