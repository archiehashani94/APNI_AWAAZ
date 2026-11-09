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
type UiLang = "en" | "hi";

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
const LISTEN_LANG_KEY = "aa_listen_lang";
const UI_LANG_KEY = "aa_ui_lang";

const EMOTION_LABELS: Record<Emotion, { en: string; hi: string }> = {
  warm: { en: "warm", hi: "स्नेही" },
  calm: { en: "calm", hi: "शांत" },
  urgent: { en: "urgent", hi: "ज़रूरी" },
  playful: { en: "playful", hi: "चंचल" },
};

const FALLBACK_LABELS: Record<string, { en: string; hi: string }> = {
  stock_voice: { en: "stock voice", hi: "स्टॉक आवाज़" },
  canned_replies: { en: "canned replies", hi: "तैयार जवाब" },
  browser_speech: { en: "browser speech", hi: "ब्राउज़र वाक्" },
  browser_speech_unavailable: {
    en: "browser speech unavailable",
    hi: "ब्राउज़र वाक् उपलब्ध नहीं",
  },
  demo: { en: "demo", hi: "डेमो" },
};

type ContentLang = "en" | "hi" | "hinglish";

// Default phrases are localised by id; custom phrases are never rewritten.
const FAVOURITE_TEXT: Record<string, Record<ContentLang, string>> = {
  yes: { en: "Yes", hi: "हाँ", hinglish: "Haan" },
  no: { en: "No", hi: "नहीं", hinglish: "Nahi" },
  thanks: { en: "Thank you", hi: "धन्यवाद", hinglish: "Dhanyavaad" },
  sorry: { en: "Sorry", hi: "माफ़ कीजिए", hinglish: "Maaf kijiye" },
  wait: { en: "Please wait", hi: "कृपया रुकिए", hinglish: "Please rukiye" },
  repeat: {
    en: "Please repeat",
    hi: "कृपया दोबारा कहिए",
    hinglish: "Please dobara kahiye",
  },
  okay: { en: "Okay", hi: "ठीक है", hinglish: "Theek hai" },
  help: {
    en: "I need help",
    hi: "मुझे मदद चाहिए",
    hinglish: "Mujhe madad chahiye",
  },
  water: {
    en: "Water please",
    hi: "पानी दीजिए",
    hinglish: "Paani dijiye",
  },
  love: {
    en: "I love you",
    hi: "मैं आपसे प्यार करता/करती हूँ",
    hinglish: "Main aapse pyaar karta/karti hoon",
  },
  hello: { en: "Hello", hi: "नमस्ते", hinglish: "Namaste" },
  bye: { en: "Bye", hi: "अलविदा", hinglish: "Alvida" },
};

const SUPPORTED_LANGUAGES = [
  { code: "en-IN", label: "English", short: "EN", native: "English" },
  { code: "hi-IN", label: "Hindi", short: "HI", native: "हिन्दी" },
  {
    code: "hi-en",
    label: "Hinglish",
    short: "HI-EN",
    native: "हिन्दी + English",
  },
  { code: "ta-IN", label: "Tamil", short: "TA", native: "தமிழ்" },
  { code: "te-IN", label: "Telugu", short: "TE", native: "తెలుగు" },
  { code: "kn-IN", label: "Kannada", short: "KN", native: "ಕನ್ನಡ" },
  { code: "ml-IN", label: "Malayalam", short: "ML", native: "മലയാളം" },
  { code: "mr-IN", label: "Marathi", short: "MR", native: "मराठी" },
  { code: "bn-IN", label: "Bengali", short: "BN", native: "বাংলা" },
  { code: "gu-IN", label: "Gujarati", short: "GU", native: "ગુજરાતી" },
  { code: "pa-IN", label: "Punjabi", short: "PA", native: "ਪੰਜਾਬੀ" },
];

type SavedRecording = {
  name: string;
  url: string;
  seconds: number | null;
  created_at: number;
  bytes: number;
};

type SavedVoice = {
  id: string;
  saved_at: number;
  seconds: number | null;
  recording: string | null;
  label: string | null;
};

function takeLabel(seconds: number | null, at: number, lang: UiLang = "en"): string {
  const when = new Date(at).toLocaleString(lang === "hi" ? "hi-IN" : "en-GB", {
    day: "numeric",
    month: "short",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  });
  if (seconds) {
    return lang === "hi"
      ? `${Math.round(seconds)} सेकंड का टेक, ${when}`
      : `${Math.round(seconds)}s take, ${when}`;
  }
  return lang === "hi" ? `सहेजी गई आवाज़, ${when}` : `Saved voice, ${when}`;
}

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

function ChevronIcon() {
  return (
    <svg
      viewBox="0 0 24 24"
      width="14"
      height="14"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.9}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d="M6 9.5l6 6 6-6" />
    </svg>
  );
}

function CheckIcon() {
  return (
    <svg
      viewBox="0 0 24 24"
      width="14"
      height="14"
      fill="none"
      stroke="currentColor"
      strokeWidth={2}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d="M4.5 12.5l5 5 10-11" />
    </svg>
  );
}

function TrashIcon() {
  return (
    <svg
      viewBox="0 0 24 24"
      width="16"
      height="16"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.7}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d="M4 7h16" />
      <path d="M9 7V5h6v2" />
      <path d="M6.5 7l.8 12h9.4l.8-12" />
      <path d="M10 11v4.5M14 11v4.5" />
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
  lang = "en",
}: {
  mood: BuddyMood;
  action?: ReactNode;
  lang?: UiLang;
}) {
  const line =
    mood === "speaking"
      ? lang === "hi"
        ? "आपकी आवाज़ में बोल रहा हूँ"
        : "Speaking in your voice"
      : mood === "listening"
        ? lang === "hi"
          ? "आपको सुन रहा हूँ"
          : "Listening to you"
        : lang === "hi"
          ? "नमस्ते, मैं आपकी आवाज़ हूँ"
          : "Hi, I'm your voice";
  return (
    <div className="voice-core">
      <div className="orb-stage">
        <AnomalyOrb mood={mood} />
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
  const [listenLang, setListenLang] = useState("en-IN");
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
  const [savedVoices, setSavedVoices] = useState<SavedVoice[]>([]);
  const [activeVoiceId, setActiveVoiceId] = useState<string | null>(null);
  const [voiceMenuOpen, setVoiceMenuOpen] = useState(false);
  const [voiceBusy, setVoiceBusy] = useState<string | null>(null);
  const [deleteBusy, setDeleteBusy] = useState<string | null>(null);
  const [voiceError, setVoiceError] = useState("");
  const [savingRecording, setSavingRecording] = useState(false);
  const [recordingNote, setRecordingNote] = useState("");
  const [thread, setThread] = useState<Turn[]>([]);
  const [favOpen, setFavOpen] = useState(false);
  const [uiLang, setUiLang] = useState<UiLang>("en");
  const [uploading, setUploading] = useState(false);
  const [consentBusy, setConsentBusy] = useState<"" | "consent" | "demo">("");

  const t = (en: string, hi: string) => (uiLang === "hi" ? hi : en);
  const emotionLabel = (emo: Emotion) =>
    uiLang === "hi" ? EMOTION_LABELS[emo].hi : EMOTION_LABELS[emo].en;

  function changeUiLang(lang: UiLang) {
    setUiLang(lang);
    try {
      localStorage.setItem(UI_LANG_KEY, lang);
    } catch {
      /* private mode: keep the in-memory choice */
    }
    // Keep the partner's listening language in step with the interface
    // language, unless the user has deliberately picked another language.
    if (listenLang === "en-IN" || listenLang === "hi-IN") {
      changeListenLang(lang === "hi" ? "hi-IN" : "en-IN");
    }
  }

  const contentLang: ContentLang =
    listenLang === "hi-en" ? "hinglish" : uiLang === "hi" ? "hi" : "en";

  const favourites: FavouriteItem[] = [
    ...defaultFavourites.map((item) => ({
      ...item,
      text: FAVOURITE_TEXT[item.id]?.[contentLang] ?? item.text,
      custom: false,
    })),
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
  const previewRecording = useRef("");
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

  const loadVoices = useCallback(async () => {
    if (!clientId) return;
    try {
      const res = await fetch(`${API_URL}/api/voices`, {
        headers: {
          "X-Client-Id": clientId,
          ...(sessionId ? { "X-Session-Id": sessionId } : {}),
        },
      });
      if (!res.ok) return;
      const data: { voices?: SavedVoice[]; active?: string | null } = await res.json();
      setSavedVoices(data.voices || []);
      if (data.active) setActiveVoiceId(data.active);
    } catch {
      /* the picker is optional; the current clone keeps working */
    }
  }, [clientId, sessionId]);

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
              voice_id?: string | null;
            } = await check.json();
            if (info.consent) setAgreed(true);
            if (info.clone_ready) {
              setCloneReady(true);
              setStage("talk");
            }
            if (info.voice_id) setActiveVoiceId(info.voice_id);
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
        if (data.voice_id) setActiveVoiceId(data.voice_id);
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
    void loadVoices();
  }, [loadVoices]);

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
    const savedLang = localStorage.getItem(LISTEN_LANG_KEY);
    // Prisma and browser SpeechRecognition each accept exactly one locale.
    // A comma-separated value (from an older build) must never be replayed.
    if (savedLang && !savedLang.includes(",")) {
      setListenLang(savedLang);
    } else if (savedLang) {
      localStorage.removeItem(LISTEN_LANG_KEY);
    }
  }, []);

  useEffect(() => {
    const saved = localStorage.getItem(UI_LANG_KEY);
    if (saved === "en" || saved === "hi") {
      setUiLang(saved);
      // A Hindi interface should listen in Hindi by default.
      if (saved === "hi" && !localStorage.getItem(LISTEN_LANG_KEY)) {
        setListenLang("hi-IN");
        localStorage.setItem(LISTEN_LANG_KEY, "hi-IN");
      }
    }
  }, []);

  useEffect(() => {
    document.documentElement.lang = uiLang === "hi" ? "hi" : "en";
  }, [uiLang]);

  useEffect(() => {
    const tick = () => {
      if (!clockRef.current) return;
      const now = new Date();
      const prefix = uiLang === "hi" ? "समय" : "Time";
      clockRef.current.textContent = `${prefix} ${now.toLocaleTimeString("en-GB", { hour12: false })}`;
    };
    tick();
    const id = window.setInterval(tick, 1000);
    return () => window.clearInterval(id);
  }, [uiLang]);

  function scrollToThread() {
    threadRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  // Recording is the primary action, so it must stay reachable after a clone
  // exists. This reuses the original enrolment screen (and its record control
  // under the orb) instead of introducing a second recording path.
  function startNewRecording() {
    if (listeningRef.current) stopListen();
    clearPreview();
    setStage("enrol");
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  async function selectVoice(voice: SavedVoice, goToTalk = false) {
    if (voiceBusy) return;
    if (!sessionId) {
      setVoiceError(
        t(
          "The session is still starting. Wait a moment, then tap the voice again.",
          "सेशन अभी शुरू हो रहा है। थोड़ा रुकें, फिर आवाज़ पर दोबारा टैप करें।",
        ),
      );
      return;
    }
    setVoiceBusy(voice.id);
    setError("");
    setVoiceError("");
    try {
      const res = await api(`/api/voices/${voice.id}/use`, sessionId, {
        method: "POST",
        headers: clientId ? { "X-Client-Id": clientId } : undefined,
      });
      if (!res.ok) {
        let detail = "";
        try {
          const data: { detail?: string } = await res.json();
          detail = data.detail || "";
        } catch {
          detail = "";
        }
        setVoiceError(detail || t("Could not switch to that voice.", "उस आवाज़ पर स्विच नहीं हो सका।"));
        return;
      }
      voiceVersion.current += 1;
      favCache.current.clear();
      setCloneReady(true);
      setVoiceIssue("");
      setActiveVoiceId(voice.id);
      setVoiceMenuOpen(false);
      setStatus(
        t(
          `Voice switched to the ${takeLabel(voice.seconds, voice.saved_at, uiLang)} clone.`,
          `${takeLabel(voice.seconds, voice.saved_at, uiLang)} क्लोन पर आवाज़ स्विच हो गई।`,
        ),
      );
      if (goToTalk) setStage("talk");
    } catch {
      setVoiceError(
        t(
          "Could not reach the voice server. Check that the backend is running.",
          "वॉइस सर्वर से संपर्क नहीं हो सका। जाँचें कि बैकएंड चल रहा है।",
        ),
      );
    } finally {
      setVoiceBusy(null);
    }
  }

  async function deleteSavedVoice(voice: SavedVoice) {
    if (deleteBusy || voiceBusy) return;
    const label = takeLabel(voice.seconds, voice.saved_at, uiLang);
    if (
      !window.confirm(
        t(
          `Delete ${label}? The saved MP4 recording stays, so you can clone it again.`,
          `${label} हटाएँ? सहेजी गई MP4 रिकॉर्डिंग रहेगी, आप उसे दोबारा क्लोन कर सकते हैं।`,
        ),
      )
    ) {
      return;
    }
    setDeleteBusy(voice.id);
    setVoiceError("");
    try {
      const res = await api(`/api/voices/${voice.id}`, sessionId, {
        method: "DELETE",
        headers: clientId ? { "X-Client-Id": clientId } : undefined,
      });
      if (!res.ok) {
        setVoiceError(t("Could not delete that voice.", "वह आवाज़ हटाई नहीं जा सकी।"));
        return;
      }
      const data: { voices?: SavedVoice[]; active?: string | null } = await res.json();
      setSavedVoices(data.voices || []);
      setActiveVoiceId(data.active ?? null);
      if (voice.id === activeVoiceId) {
        voiceVersion.current += 1;
        favCache.current.clear();
        if (data.active) {
          setCloneReady(true);
          setVoiceIssue("");
        } else {
          setCloneReady(false);
          setVoiceIssue(
            t(
              "No saved voice left. Record a new one to speak in your own voice.",
              "कोई सहेजी गई आवाज़ नहीं बची। अपनी आवाज़ में बोलने के लिए नई आवाज़ रिकॉर्ड करें।",
            ),
          );
        }
      }
      setStatus(t("Voice deleted.", "आवाज़ हटा दी गई।"));
    } catch {
      setVoiceError(
        t("Could not reach the voice server.", "वॉइस सर्वर से संपर्क नहीं हो सका।"),
      );
    } finally {
      setDeleteBusy(null);
    }
  }

  async function deleteRecording(rec: SavedRecording) {
    if (deleteBusy || voiceBusy) return;
    if (
      !window.confirm(
        t(
          "Delete this saved MP4 recording? This cannot be undone.",
          "यह सहेजी गई MP4 रिकॉर्डिंग हटाएँ? इसे वापस नहीं लाया जा सकता।",
        ),
      )
    ) {
      return;
    }
    setDeleteBusy(rec.name);
    setVoiceError("");
    try {
      const res = await api(`/api/recordings/${encodeURIComponent(rec.name)}`, sessionId, {
        method: "DELETE",
        headers: clientId ? { "X-Client-Id": clientId } : undefined,
      });
      if (!res.ok) {
        setVoiceError(
          t("Could not delete that recording.", "वह रिकॉर्डिंग हटाई नहीं जा सकी।"),
        );
        return;
      }
      const data: { recordings?: SavedRecording[] } = await res.json();
      setSavedRecordings(data.recordings || []);
      setRecordingNote(t("MP4 recording deleted.", "MP4 रिकॉर्डिंग हटा दी गई।"));
    } catch {
      setVoiceError(
        t("Could not reach the voice server.", "वॉइस सर्वर से संपर्क नहीं हो सका।"),
      );
    } finally {
      setDeleteBusy(null);
    }
  }

  // Older takes are kept as MP4s. If a take was never cloned, clone it now,
  // reusing the same cleaning and enrolment pipeline as a fresh recording.
  async function cloneFromRecording(rec: SavedRecording) {
    if (voiceBusy || fileBusy) return;
    setVoiceBusy(rec.name);
    setError("");
    setVoiceError("");
    setStatus(t("Reading the saved take…", "सहेजा गया टेक पढ़ रहे हैं…"));
    let ctx: AudioContext | null = null;
    try {
      const res = await fetch(`${API_URL}${rec.url}?inline=1`);
      if (!res.ok) throw new Error("Could not load the saved take");
      const bytes = await res.arrayBuffer();
      ctx = new AudioContext();
      const decoded = await new Promise<AudioBuffer>((resolve, reject) => {
        ctx!.decodeAudioData(bytes.slice(0), resolve, reject);
      });
      const channels = decoded.numberOfChannels;
      const mono = new Float32Array(decoded.length);
      for (let c = 0; c < channels; c++) {
        const data = decoded.getChannelData(c);
        for (let i = 0; i < mono.length; i++) mono[i] += data[i] / channels;
      }
      const reference = await resampleMono(mono, decoded.sampleRate, REFERENCE_RATE);
      let prepared = prepareVoiceSample(reference, REFERENCE_RATE);
      if (prepared.length / REFERENCE_RATE > 30) {
        prepared = prepared.slice(0, REFERENCE_RATE * 30);
      }
      if (prepared.length / REFERENCE_RATE < 5) {
        setStatus("");
        setVoiceError(
          t(
            "That take is too short to clone. Record a new voice instead.",
            "वह टेक क्लोन करने के लिए बहुत छोटा है। इसके बजाय नई आवाज़ रिकॉर्ड करें।",
          ),
        );
        return;
      }
      previewRecording.current = rec.name;
      await uploadVoice(encodeWav(prepared, REFERENCE_RATE), "voice.wav");
    } catch {
      setStatus("");
      setVoiceError(
        t(
          "Could not read that saved take. Record a new voice instead.",
          "वह सहेजा गया टेक पढ़ा नहीं जा सका। इसके बजाय नई आवाज़ रिकॉर्ड करें।",
        ),
      );
    } finally {
      closeAudio(ctx);
      setVoiceBusy(null);
    }
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
      setCredits(
        t(
          `${data.used} / ${data.budget} credits`,
          `${data.used} / ${data.budget} क्रेडिट`,
        ),
      );
    } catch {
      /* credit display is non-critical */
    }
  }

  async function submitConsent() {
    if (consentBusy) return;
    setConsentBusy("consent");
    setError("");
    try {
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
    } finally {
      setConsentBusy("");
    }
  }

  async function startDemo() {
    if (consentBusy) return;
    setConsentBusy("demo");
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
    } finally {
      setConsentBusy("");
    }
  }

  function clearPreview() {
    if (previewUrlRef.current) {
      URL.revokeObjectURL(previewUrlRef.current);
      previewUrlRef.current = "";
    }
    previewRecording.current = "";
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
        throw new Error(detail || t(`Save failed (${res.status})`, `सहेजना विफल (${res.status})`));
      }
      const saved: SavedRecording = await res.json();
      setSavedRecordings((prev) => [
        saved,
        ...prev.filter((item) => item.name !== saved.name),
      ]);
      previewRecording.current = saved.name;
      setRecordingNote(t("MP4 copy saved.", "MP4 कॉपी सहेजी गई।"));
    } catch (err) {
      setRecordingNote(
        t(
          `Could not save the MP4 copy: ${err instanceof Error ? err.message : String(err)}`,
          `MP4 कॉपी सहेजी नहीं जा सकी: ${err instanceof Error ? err.message : String(err)}`,
        ),
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
          echoCancellation: false,
          noiseSuppression: false,
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
      setError(
        t(
          "Microphone unavailable. Allow microphone access and try again.",
          "माइक्रोफ़ोन उपलब्ध नहीं है। माइक्रोफ़ोन की अनुमति दें और दोबारा कोशिश करें।",
        ),
      );
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
        t(
          `Only ${rawDuration.toFixed(1)}s recorded. Record at least 5 seconds, ideally 20 to 30 seconds.`,
          `केवल ${rawDuration.toFixed(1)} सेकंड रिकॉर्ड हुए। कम से कम 5 सेकंड, बेहतर हो 20 से 30 सेकंड रिकॉर्ड करें।`,
        ),
      );
      return;
    }
    if (levelPercent(merged) < 5) {
      setRecState("idle");
      setError(
        t(
          "The recording is too quiet. Move closer to the microphone and record again.",
          "रिकॉर्डिंग बहुत धीमी है। माइक्रोफ़ोन के पास आएँ और दोबारा रिकॉर्ड करें।",
        ),
      );
      return;
    }
    if (recClipped.current) {
      setError(
        t(
          "The recording clipped (input was too loud). If the preview sounds distorted, move back and record again.",
          "रिकॉर्डिंग क्लिप हो गई (आवाज़ बहुत तेज़ थी)। अगर प्रीव्यू विकृत लगे, थोड़ा दूर जाकर दोबारा रिकॉर्ड करें।",
        ),
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
      setError(
        t(
          "Could not process the recording. Stop listening and try again.",
          "रिकॉर्डिंग प्रोसेस नहीं हो सकी। सुनना बंद करें और दोबारा कोशिश करें।",
        ),
      );
      return;
    }
    const preparedDuration = prepared.length / REFERENCE_RATE;
    if (preparedDuration < 5) {
      setRecState("idle");
      setError(
        t(
          `After removing silence only ${preparedDuration.toFixed(1)}s remains. Record at least 5 seconds of speech.`,
          `सन्नाटा हटाने के बाद केवल ${preparedDuration.toFixed(1)} सेकंड बचे। कम से कम 5 सेकंड की बोली रिकॉर्ड करें।`,
        ),
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
        ? t(
            `Recorded ${preparedDuration.toFixed(1)}s, but the reference has problems. Read the notes below before using it.`,
            `${preparedDuration.toFixed(1)} सेकंड रिकॉर्ड हुए, लेकिन रेफ़रेंस में समस्याएँ हैं। इस्तेमाल से पहले नीचे दिए नोट्स पढ़ें।`,
          )
        : t(
            `Recorded ${preparedDuration.toFixed(1)}s of speech. Listen to the preview, then tap "Use this voice".`,
            `${preparedDuration.toFixed(1)} सेकंड की बोली रिकॉर्ड हुई। प्रीव्यू सुनें, फिर "इस आवाज़ का उपयोग करें" पर टैप करें।`,
          ),
    );
  }

  async function confirmEnrol() {
    if (!previewBlob || !sessionId || fileBusy) return;
    await uploadVoice(previewBlob, previewName);
  }

  async function uploadVoice(file: Blob, name: string) {
    setStatus(t("Creating voice embedding…", "आवाज़ एम्बेडिंग बना रहे हैं…"));
    setUploading(true);
    setError("");
    const recording = previewRecording.current;
    try {
      const form = new FormData();
      form.append("audio", file, name);
      const headers: Record<string, string> = {};
      if (clientId) headers["X-Client-Id"] = clientId;
      if (recording) headers["X-Recording-Name"] = recording;
      const res = await api("/api/enrol", sessionId, {
        method: "POST",
        headers: Object.keys(headers).length ? headers : undefined,
        body: form,
      });
      const data: {
        clone_ready?: boolean;
        fallback?: boolean;
        message?: string;
        detail?: string;
        voice_id?: string | null;
        duration_seconds?: number;
        upstream_status?: number;
        quality?: { verdict?: string; score?: number; warnings?: string[] };
      } = await res.json().catch(() => ({}));
      if (!res.ok) {
        setError(data.detail || t(`Enrol failed (${res.status})`, `एनरोल विफल (${res.status})`));
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
        setVoiceIssue(
          blocked
            ? data.message || t("Gnani rejected the API key.", "Gnani ने API कुंजी अस्वीकार की।")
            : "",
        );
        setError(
          data.message ||
            t(
              "Cloning failed. Using the stock voice; record again to retry.",
              "क्लोनिंग विफल। स्टॉक आवाज़ इस्तेमाल हो रही है; दोबारा कोशिश के लिए फिर रिकॉर्ड करें।",
            ),
        );
      } else {
        voiceVersion.current += 1;
        favCache.current.clear();
        setVoiceIssue("");
        setActiveVoiceId(data.voice_id || null);
        setVoiceMenuOpen(false);
        clearPreview();
        void loadVoices();
        setStatus(
          warnings.length
            ? t(
                "Clone ready, but the reference audio has problems. See the notes above the speak box.",
                "क्लोन तैयार है, लेकिन रेफ़रेंस ऑडियो में समस्याएँ हैं। बोलने के बॉक्स के ऊपर दिए नोट्स देखें।",
              )
            : t(
                "Clone ready. Tap “Test this voice” to check that it sounds like you.",
                "क्लोन तैयार है। जाँचने के लिए “इस आवाज़ को आज़माएँ” पर टैप करें कि यह आप जैसी लगती है।",
              ),
        );
      }
      setStage("talk");
      refreshCredits();
    } catch {
      setStatus("");
      setError(
        t(
          "Could not reach the voice server. Check that the backend is running and try again.",
          "वॉइस सर्वर से संपर्क नहीं हो सका। जाँचें कि बैकएंड चल रहा है, फिर दोबारा कोशिश करें।",
        ),
      );
    } finally {
      setUploading(false);
    }
  }

  async function onFile(ev: React.ChangeEvent<HTMLInputElement>) {
    const file = ev.target.files?.[0];
    ev.target.value = "";
    if (!file) return;
    if (recState === "recording") {
      setError(
        t(
          "Stop the current recording before uploading a file.",
          "फ़ाइल अपलोड करने से पहले मौजूदा रिकॉर्डिंग रोकें।",
        ),
      );
      return;
    }
    if (file.size === 0) {
      setError(t("That file is empty. Choose another audio file.", "वह फ़ाइल खाली है। दूसरी ऑडियो फ़ाइल चुनें।"));
      return;
    }
    if (file.size > MAX_SOURCE_MB * 1024 * 1024) {
      setError(
        t(
          `That file is over ${MAX_SOURCE_MB} MB. Trim the clip or export just the audio track.`,
          `वह फ़ाइल ${MAX_SOURCE_MB} MB से बड़ी है। क्लिप छोटी करें या सिर्फ़ ऑडियो ट्रैक निर्यात करें।`,
        ),
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
      setError(
        t(
          "Choose an audio file (WAV, MP3, M4A, MP4, OGG, FLAC or AAC).",
          "ऑडियो फ़ाइल चुनें (WAV, MP3, M4A, MP4, OGG, FLAC या AAC)।",
        ),
      );
      return;
    }

    setError("");
    setFileNote("");
    setQualityWarnings([]);
    setQualityScore(null);
    setFileBusy(true);
    setStatus(t(`Reading ${file.name}…`, `${file.name} पढ़ रहे हैं…`));
    let ctx: AudioContext | null = null;
    try {
      const mediaSeconds = await probeMediaDuration(file);
      if (mediaSeconds > MAX_SOURCE_SECONDS) {
        setError(
          t(
            `That file is ${Math.round(mediaSeconds / 60)} minutes long. Use at most 10 minutes; only the first 30 seconds of speech are used.`,
            `वह फ़ाइल ${Math.round(mediaSeconds / 60)} मिनट लंबी है। अधिकतम 10 मिनट इस्तेमाल करें; केवल पहले 30 सेकंड की बोली ली जाती है।`,
          ),
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
          t(
            `This file has only ${duration.toFixed(1)}s of audible speech. Use at least 5 seconds, ideally 20 to 30.`,
            `इस फ़ाइल में केवल ${duration.toFixed(1)} सेकंड की सुनाई देने वाली बोली है। कम से कम 5 सेकंड, बेहतर हो 20 से 30 सेकंड इस्तेमाल करें।`,
          ),
        );
        setStatus("");
        return;
      }
      let note = "";
      if (duration > 30) {
        prepared = prepared.slice(0, Math.floor(REFERENCE_RATE * 30));
        duration = prepared.length / REFERENCE_RATE;
        note = t(
          "The file was longer than 30s, so the first 30 seconds will be used.",
          "फ़ाइल 30 सेकंड से लंबी थी, इसलिए पहले 30 सेकंड इस्तेमाल होंगे।",
        );
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
          ? t(
              `Loaded ${duration.toFixed(1)}s, but the reference has problems. Read the notes below before using it.`,
              `${duration.toFixed(1)} सेकंड लोड हुए, लेकिन रेफ़रेंस में समस्याएँ हैं। इस्तेमाल से पहले नीचे दिए नोट्स पढ़ें।`,
            )
          : t(
              `Loaded ${duration.toFixed(1)}s of speech from ${file.name}. Preview it, then tap "Use this voice".`,
              `${file.name} से ${duration.toFixed(1)} सेकंड की बोली लोड हुई। प्रीव्यू सुनें, फिर "इस आवाज़ का उपयोग करें" पर टैप करें।`,
            ),
      );
    } catch {
      setStatus("");
      setError(
        t(
          "This browser could not read the audio in that file. If it is an MP4/MOV video, convert it to MP3 or WAV, or record directly instead.",
          "यह ब्राउज़र उस फ़ाइल का ऑडियो नहीं पढ़ सका। अगर यह MP4/MOV वीडियो है, तो इसे MP3 या WAV में बदलें, या सीधे रिकॉर्ड करें।",
        ),
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
      setError(t("That phrase is already in favourites.", "वह वाक्यांश पहले से पसंदीदा में है।"));
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

  function startBrowserSpeech(langOverride?: string) {
    if (browserRecActive.current || closingListen.current) return;
    const win = window as Window & {
      SpeechRecognition?: new () => BrowserRec;
      webkitSpeechRecognition?: new () => BrowserRec;
    };
    const SR = win.SpeechRecognition || win.webkitSpeechRecognition;
    if (!SR) {
      addFallback("browser_speech_unavailable");
      setError(
        t(
          "This browser has no speech recognition. Use Chrome or Edge for live listening.",
          "इस ब्राउज़र में वाक् पहचान नहीं है। लाइव सुनने के लिए Chrome या Edge इस्तेमाल करें।",
        ),
      );
      return;
    }
    addFallback("browser_speech");
    const rec = new SR();
    rec.continuous = true;
    rec.interimResults = true;
    const speechLang = langOverride || listenLang;
    // Browser recognition has no Hinglish locale. English (India) keeps the
    // transcript in Latin script, which is the closest match for Hinglish.
    rec.lang =
      speechLang === "indic-auto" ||
      speechLang === "hi-en" ||
      speechLang.includes(",")
        ? "en-IN"
        : speechLang;
    rec.onresult = (raw) => {
      const event = raw as {
        resultIndex: number;
        results: { length: number; [i: number]: { isFinal: boolean; 0: { transcript: string } } };
      };
      let finalText = "";
      let interim = "";
      for (let i = event.resultIndex; i < event.results.length; i++) {
        const part = event.results[i][0].transcript;
        if (event.results[i].isFinal) finalText += part;
        else interim += part;
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
        setError(t("Speech recognition permission was denied.", "वाक् पहचान की अनुमति अस्वीकार हुई।"));
      } else if (code === "audio-capture") {
        setError(t("No microphone found for speech recognition.", "वाक् पहचान के लिए कोई माइक्रोफ़ोन नहीं मिला।"));
      }
    };
    rec.onend = () => {
      browserRecActive.current = false;
      if (!closingListen.current && listeningRef.current && !browserRecRestart.current) {
        browserRecRestart.current = window.setTimeout(() => {
          browserRecRestart.current = null;
          startBrowserSpeech(langOverride);
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
      setError(
        t(
          "Could not start browser speech recognition. Stop listening and try again.",
          "ब्राउज़र वाक् पहचान शुरू नहीं हो सकी। सुनना बंद करें और दोबारा कोशिश करें।",
        ),
      );
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

  function changeListenLang(lang: string) {
    setListenLang(lang);
    localStorage.setItem(LISTEN_LANG_KEY, lang);
    // Existing chips are in the previous language; fetch new ones for this one.
    if (partnerTurns.current.length) void loadSuggestions(lang);
    if (listeningRef.current) {
      stopListen();
      window.setTimeout(() => void startListen(lang), 120);
    }
  }

  async function startListen(forcedLang?: string) {
    if (!sessionId || listeningRef.current) return;
    const langToUse = forcedLang || listenLang;
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
      setError(
        t(
          "Microphone unavailable. Allow microphone access and try again.",
          "माइक्रोफ़ोन उपलब्ध नहीं है। माइक्रोफ़ोन की अनुमति दें और दोबारा कोशिश करें।",
        ),
      );
      return;
    }

    if (!listeningRef.current) {
      stream.getTracks().forEach((t) => t.stop());
      return;
    }
    listenStream.current = stream;
    const socket = new WebSocket(
      wsUrl(
        `/ws/listen?session_id=${encodeURIComponent(sessionId)}&language=${encodeURIComponent(langToUse)}`,
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
        setStatus(t("Connecting to Prisma…", "Prisma से कनेक्ट हो रहे हैं…"));
      } else if (frame.type === "connected") {
        setListenState("live");
        setStatus("");
      } else if (frame.type === "speech_start") {
        setStatus(t("Hearing speech…", "बोली सुनाई दे रही है…"));
      } else if (frame.type === "processing") {
        setStatus(t("Transcribing…", "ट्रांसक्राइब हो रहा है…"));
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
        startBrowserSpeech(langToUse);
      } else if (frame.type === "error") {
        setError(frame.message || t("Live transcription error.", "लाइव ट्रांसक्रिप्शन में त्रुटि।"));
      }
    };
    socket.onerror = () => {
      if (!closingListen.current) {
        addFallback("browser_speech");
        stopCapture();
        startBrowserSpeech(langToUse);
      }
    };
    socket.onclose = () => {
      if (!closingListen.current && listeningRef.current) {
        addFallback("browser_speech");
        setListenState("fallback");
        stopCapture();
        startBrowserSpeech(langToUse);
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
      setError(
        t(
          "Could not start audio capture. Try Chrome or Edge, then retry.",
          "ऑडियो कैप्चर शुरू नहीं हो सका। Chrome या Edge आज़माएँ, फिर दोबारा कोशिश करें।",
        ),
      );
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

  async function loadSuggestions(languageOverride?: string) {
    if (!sessionId) return;
    // Ignore a slow older response if a newer transcript already arrived.
    const seq = ++suggestSeq.current;
    const turns = partnerTurns.current.slice(-8);
    const listenLanguage = languageOverride || listenLang;
    const payload: Record<string, unknown> = {};
    if (turns.length) payload.turns = turns;
    // A Hinglish session should get Roman-script Hindi suggestions even when
    // the partner's transcript is short or mixed.
    if (listenLanguage === "hi-en") payload.language = "hinglish";
    if (userGenderRef.current !== "neutral") {
      payload.user_gender = userGenderRef.current;
    }
    try {
      const res = await api("/api/suggest", sessionId, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
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
        finish(() =>
          reject(
            new Error(
              t("Could not play the generated audio.", "बनाई गई ऑडियो चलाई नहीं जा सकी।"),
            ),
          ),
        );
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
          throw new Error(detail.detail || t(`Speak failed (${res.status})`, `बोलना विफल (${res.status})`));
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

  const mood: BuddyMood = speaking ? "speaking" : listening ? "listening" : "idle";
  const dockVisible = stage === "talk";
  const activeVoice = savedVoices.find((voice) => voice.id === activeVoiceId) ?? null;
  const unclonedRecordings = savedRecordings.filter(
    (rec) => !savedVoices.some((voice) => voice.recording === rec.name),
  );

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
            aria-label={t("Stop recording", "रिकॉर्डिंग रोकें")}
          >
            <span className="record-stop" aria-hidden="true" />
          </button>
        ) : (
          <button
            type="button"
            className="record-dot"
            onClick={startRecord}
            disabled={uploading || fileBusy}
            aria-label={
              previewUrl ? t("Record again", "दोबारा रिकॉर्ड करें") : t("Start recording", "रिकॉर्डिंग शुरू करें")
            }
          >
            <span className="record-glyph" aria-hidden="true" />
          </button>
        )}
        <span className="orb-action-label">
          {recState === "recording"
            ? t(`Stop ${seconds}s`, `रोकें ${seconds}से`)
            : previewUrl
              ? t("Record again", "दोबारा रिकॉर्ड करें")
              : t("Record your voice", "अपनी आवाज़ रिकॉर्ड करें")}
        </span>
      </div>
    ) : stage === "talk" ? (
      <div className={`orb-action ${listening ? "live" : ""}`}>
        {listening ? (
          <button
            type="button"
            className="record-dot live"
            onClick={stopListen}
            aria-label={t("Stop listening", "सुनना बंद करें")}
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
            aria-label={t("Start listening", "सुनना शुरू करें")}
          >
            <MicIcon />
          </button>
        )}
        <span className="orb-action-label">
          {listening
            ? t("Listening. Tap to stop.", "सुन रहे हैं। रोकने के लिए टैप करें।")
            : listenState === "connecting"
              ? t("Connecting…", "कनेक्ट हो रहे हैं…")
              : t("Start listening", "सुनना शुरू करें")}
        </span>
      </div>
    ) : null;

  return (
    <main className="site">
      <div className="ambient" aria-hidden="true" />
      <div className="wrap">
        <div className="topbar">
          <span className="logo">Apni Awaaz</span>
          <div className="topbar-right">
            <div
              className="ui-lang"
              role="group"
              aria-label={t("App language", "ऐप की भाषा")}
            >
              <button
                type="button"
                className={uiLang === "en" ? "active" : ""}
                onClick={() => changeUiLang("en")}
                aria-pressed={uiLang === "en"}
              >
                English
              </button>
              <button
                type="button"
                className={uiLang === "hi" ? "active" : ""}
                onClick={() => changeUiLang("hi")}
                aria-pressed={uiLang === "hi"}
              >
                हिन्दी
              </button>
            </div>
            <span ref={clockRef} className="topbar-time">
              Time --:--:--
            </span>
          </div>
        </div>
        <div className="workspace">
          <header className="hero">
            <div className="hero-copy">
              <h1>
                {t("Give your", "अपनी आवाज़")}
                <br />
                {t("voice back", "वापस पाएँ")}
              </h1>
              <p className="lede">
                {t(
                  "Type or tap a reply. The core speaks it in a clone of your voice: one take, kept across restarts, no re-recording.",
                  "जवाब टाइप करें या टैप करें। यह केंद्र आपकी आवाज़ के क्लोन में उसे बोलता है: एक ही टेक, रीस्टार्ट के बाद भी सुरक्षित, दोबारा रिकॉर्डिंग नहीं।",
                )}
              </p>
            </div>
            <VoiceCore mood={mood} action={heroAction} lang={uiLang} />
          </header>
          <div className="stage">
            <div className="tag-row">
              {aiLabel && <span className="badge">{t("AI-generated audio", "AI से बनाई ऑडियो")}</span>}
              {fallbacks.map((f) => (
                <span className="badge warn" key={f}>
                  {t("Fallback: ", "फ़ॉलबैक: ")}
                  {FALLBACK_LABELS[f]
                    ? t(FALLBACK_LABELS[f].en, FALLBACK_LABELS[f].hi)
                    : f.replaceAll("_", " ")}
                </span>
              ))}
              {credits && <span className="badge">{credits}</span>}
            </div>
      {error && <p className="error">{error}</p>}

      {stage === "consent" && (
        <section className="card">
          <h2>{t("Before we clone a voice", "आवाज़ क्लोन करने से पहले")}</h2>
          <label className="consent">
            <input type="checkbox" checked={agreed} onChange={(e) => setAgreed(e.target.checked)} />
            <span>
              {t(
                "This is my own voice, or I have the speaker's permission.",
                "यह मेरी अपनी आवाज़ है, या मेरे पास वक्ता की अनुमति है।",
              )}
            </span>
          </label>
          <div className="row" style={{ marginTop: "1rem" }}>
            <button
              type="button"
              disabled={!agreed || !sessionId || consentBusy !== ""}
              onClick={submitConsent}
            >
              {consentBusy === "consent"
                ? t("Starting…", "शुरू हो रहा है…")
                : t("Continue", "जारी रखें")}
            </button>
            <button
              type="button"
              className="secondary"
              disabled={!agreed || !sessionId || consentBusy !== ""}
              onClick={startDemo}
            >
              {consentBusy === "demo"
                ? t("Starting demo…", "डेमो शुरू हो रहा है…")
                : t("One-click demo", "वन-क्लिक डेमो")}
            </button>
          </div>
        </section>
      )}

      {stage === "enrol" && (
        <section className="card">
          <h2>{t("Voice enrolment", "आवाज़ एनरोलमेंट")}</h2>
          <p className="lede">
            {t(
              "Record 15 to 30 seconds of natural speech in a quiet room. Use one person only: no music, no singing, no TV, no second voice. Speak the same language you will use for replies. A clean, close-mic take makes the biggest difference to how closely the clone matches you.",
              "शांत कमरे में 15 से 30 सेकंड की सहज बोली रिकॉर्ड करें। केवल एक व्यक्ति: कोई संगीत, गाना, टीवी या दूसरी आवाज़ नहीं। वही भाषा बोलें जिसका इस्तेमाल आप जवाबों के लिए करेंगे। साफ़, माइक के पास की रिकॉर्डिंग क्लोन को आपसे सबसे ज़्यादा मिलती-जुलती बनाती है।",
            )}
          </p>
          <p className="meter">
            {t(
              "Good reference: continuous talking, normal volume, about a hand-width from the mic. Bad reference: whispering, shouting, singing, background music, or a video with several speakers.",
              "अच्छा रेफ़रेंस: लगातार बोलना, सामान्य आवाज़, माइक से लगभग एक हथेली की दूरी। खराब रेफ़रेंस: फुसफुसाना, चिल्लाना, गाना, पृष्ठभूमि संगीत, या कई वक्ताओं वाला वीडियो।",
            )}
          </p>
          <div className="row">
            <button
              type="button"
              className="secondary"
              onClick={() => fileInput.current?.click()}
              disabled={fileBusy || uploading || recState === "recording"}
            >
              {fileBusy
                ? t("Reading file…", "फ़ाइल पढ़ रहे हैं…")
                : t("Upload audio file", "ऑडियो फ़ाइल अपलोड करें")}
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
                {t("Use previous voice", "पिछली आवाज़ इस्तेमाल करें")}
              </button>
            )}
          </div>
          {fileNote && <p className="meter">{fileNote}</p>}
          {savingRecording && (
            <p className="meter">{t("Saving MP4 copy…", "MP4 कॉपी सहेजी जा रही है…")}</p>
          )}
          {!savingRecording && recordingNote && (
            <p className="meter">{recordingNote}</p>
          )}
          {savedRecordings.length > 0 && (
            <div className="recordings">
              <p className="meter">
                <strong>{t("Saved recordings", "सहेजी गई रिकॉर्डिंग")}</strong>
                {t(
                  ". Every take is kept as an MP4 file on the server. Tap “Use this voice” to switch back to an earlier take without recording again.",
                  ". हर टेक सर्वर पर MP4 फ़ाइल के रूप में रखा जाता है। बिना दोबारा रिकॉर्ड किए पुराने टेक पर लौटने के लिए “इस आवाज़ का उपयोग करें” पर टैप करें।",
                )}
              </p>
              {voiceError && (
                <p className="voice-alert" role="alert">
                  {voiceError}
                </p>
              )}
              <ul>
                {savedRecordings.map((recording) => {
                  const linked =
                    savedVoices.find((voice) => voice.recording === recording.name) ??
                    null;
                  const inUse = linked !== null && linked.id === activeVoiceId;
                  return (
                    <li key={recording.name}>
                      <div className="recording-line">
                        <span>
                          {new Date(recording.created_at).toLocaleString(
                            uiLang === "hi" ? "hi-IN" : "en-GB",
                          )}
                        </span>
                        {recording.seconds ? (
                          <span>
                            {recording.seconds.toFixed(1)}
                            {uiLang === "hi" ? "से" : "s"}
                          </span>
                        ) : null}
                        <a href={`${API_URL}${recording.url}`} download={recording.name}>
                          {t("Download MP4", "MP4 डाउनलोड करें")}
                        </a>
                        <button
                          type="button"
                          className="recording-delete"
                          onClick={() => void deleteRecording(recording)}
                          disabled={deleteBusy !== null || voiceBusy !== null}
                          aria-label={t(
                            "Delete this recording",
                            "यह रिकॉर्डिंग हटाएँ",
                          )}
                          title={t("Delete this recording", "यह रिकॉर्डिंग हटाएँ")}
                        >
                          {deleteBusy === recording.name ? (
                            <span className="voice-delete-busy" aria-hidden="true" />
                          ) : (
                            <TrashIcon />
                          )}
                        </button>
                      </div>
                      <audio
                        controls
                        preload="none"
                        src={`${API_URL}${recording.url}?inline=1`}
                      />
                      {linked ? (
                        <button
                          type="button"
                          className="secondary recording-use"
                          disabled={voiceBusy !== null || uploading || inUse}
                          onClick={() => void selectVoice(linked, true)}
                        >
                          {inUse
                            ? t("In use", "उपयोग में")
                            : voiceBusy === linked.id
                              ? t("Switching…", "स्विच हो रहा है…")
                              : t("Use this voice", "इस आवाज़ का उपयोग करें")}
                        </button>
                      ) : (
                        <button
                          type="button"
                          className="secondary recording-use"
                          disabled={voiceBusy !== null || uploading || fileBusy}
                          onClick={() => void cloneFromRecording(recording)}
                        >
                          {voiceBusy === recording.name
                            ? t("Creating voice…", "आवाज़ बना रहे हैं…")
                            : t("Clone this recording", "इस रिकॉर्डिंग को क्लोन करें")}
                        </button>
                      )}
                      {!linked && (
                        <p className="meter recording-note">
                          {t(
                            "This take was never cloned. Creating a voice from it uses about 50 Gnani credits.",
                            "यह टेक कभी क्लोन नहीं हुआ। इससे आवाज़ बनाने में लगभग 50 Gnani क्रेडिट लगते हैं।",
                          )}
                        </p>
                      )}
                    </li>
                  );
                })}
              </ul>
            </div>
          )}
          <div className="level" aria-hidden="true">
            <div style={{ transform: `scaleX(${recLevel / 100})` }} />
          </div>
          <p className="meter">
            {recState === "recording"
              ? t(
                  `Recording… ${seconds}s / 30s. Keep the mic about a hand-width away.`,
                  `रिकॉर्ड हो रहा है… ${seconds}से / 30से। माइक को लगभग एक हथेली की दूरी पर रखें।`,
                )
              : status}
          </p>
          {previewUrl && (
            <div className="preview">
              <p className="meter">
                {t(
                  "Preview this recording before creating the clone. It must be one person speaking, with no music, singing, or second voice. If it is noisy, echoey, or quiet, record again.",
                  "क्लोन बनाने से पहले यह रिकॉर्डिंग सुनें। इसमें एक ही व्यक्ति बोल रहा हो, कोई संगीत, गाना या दूसरी आवाज़ न हो। अगर शोर, गूँज या धीमी आवाज़ है, तो दोबारा रिकॉर्ड करें।",
                )}
              </p>
              {previewWarnings.length > 0 && (
                <div className="quality">
                  <p>
                    <strong>
                      {t("This reference may clone poorly", "यह रेफ़रेंस खराब क्लोन बन सकता है")}
                      {previewScore !== null ? ` (${previewScore}/100)` : ""}
                    </strong>
                  </p>
                  <ul>
                    {previewWarnings.map((warning) => (
                      <li key={warning}>{warning}</li>
                    ))}
                  </ul>
                  <p className="meter">
                    {t(
                      "Fix the issue and record or upload again. The clone can only be as good as this clip.",
                      "समस्या ठीक करें और दोबारा रिकॉर्ड या अपलोड करें। क्लोन इस क्लिप जितना ही अच्छा हो सकता है।",
                    )}
                  </p>
                </div>
              )}
              <audio controls src={previewUrl} />
              <div className="row" style={{ marginTop: "0.7rem" }}>
                <button onClick={confirmEnrol} disabled={uploading}>
                  {uploading
                    ? t("Creating voice…", "आवाज़ बना रहे हैं…")
                    : t("Use this voice", "इस आवाज़ का उपयोग करें")}
                </button>
                <button className="secondary" onClick={startRecord} disabled={uploading}>
                  {t("Record again", "दोबारा रिकॉर्ड करें")}
                </button>
              </div>
            </div>
          )}
        </section>
      )}

      {stage === "talk" && (
        <div className="surface">
          <div className="surface-top">
            <button
              type="button"
              className={`voice-current ${voiceMenuOpen ? "open" : ""}`}
              onClick={() => setVoiceMenuOpen((open) => !open)}
              aria-expanded={voiceMenuOpen}
              aria-controls="voice-menu"
            >
              <span className="voice-key">{t("Voice", "आवाज़")}</span>
              <span className="voice-value">
                {activeVoice
                  ? takeLabel(activeVoice.seconds, activeVoice.saved_at, uiLang)
                  : cloneReady
                    ? t("Saved voice", "सहेजी गई आवाज़")
                    : t("Stock voice", "स्टॉक आवाज़")}
              </span>
              <ChevronIcon />
            </button>
            <div className="surface-actions">
              {cloneReady && (
                <button
                  type="button"
                  className="secondary test-voice"
                  onClick={() =>
                    void speakText(
                      t(
                        "Your voice is now cloned. This is how you sound.",
                        "आपकी आवाज़ अब क्लोन हो गई है। यह सुनें कि आप कैसे सुनाई देते हैं।",
                      ),
                      emotion,
                      false,
                    )
                  }
                  disabled={speaking || uploading}
                >
                  {speaking
                    ? t("Testing…", "आज़मा रहे हैं…")
                    : t("Test this voice", "इस आवाज़ को आज़माएँ")}
                </button>
              )}
              <button
                type="button"
                className="record-again"
                onClick={startNewRecording}
                disabled={uploading}
              >
                {t("Record a new voice", "नई आवाज़ रिकॉर्ड करें")}
              </button>
            </div>
          </div>
          {voiceMenuOpen && (
            <div className="voice-menu" id="voice-menu">
              {voiceError && (
                <p className="voice-alert" role="alert">
                  {voiceError}
                </p>
              )}
              <p className="voice-hint">
                {savedVoices.length + unclonedRecordings.length > 1
                  ? t(
                      "Every take you have recorded is here. Switching between saved clones uses no credits.",
                      "आपके रिकॉर्ड किए हुए सभी टेक यहाँ हैं। सहेजे गए क्लोन के बीच स्विच करने में कोई क्रेडिट नहीं लगता।",
                    )
                  : t(
                      "Voices are saved on this device and survive restarts.",
                      "आवाज़ें इस डिवाइस पर सहेजी जाती हैं और रीस्टार्ट के बाद भी रहती हैं।",
                    )}
              </p>
              {savedVoices.map((voice) => {
                const active = voice.id === activeVoiceId;
                return (
                  <div
                    key={voice.id}
                    className={`voice-row-wrap ${active ? "active" : ""}`}
                  >
                    <button
                      type="button"
                      className={`voice-row ${active ? "active" : ""}`}
                      onClick={() => void selectVoice(voice)}
                      disabled={voiceBusy !== null || deleteBusy !== null}
                      aria-pressed={active}
                    >
                      <span className="voice-row-main">
                        <span className="voice-row-title">
                          {takeLabel(voice.seconds, voice.saved_at, uiLang)}
                        </span>
                        <span className="voice-row-meta">
                          {voice.recording
                            ? t("From a saved recording", "सहेजी गई रिकॉर्डिंग से")
                            : t("Saved clone", "सहेजा गया क्लोन")}
                        </span>
                      </span>
                      <span className="voice-row-act">
                        {active ? (
                          <>
                            <CheckIcon /> {t("In use", "उपयोग में")}
                          </>
                        ) : voiceBusy === voice.id ? (
                          t("Switching…", "स्विच हो रहा है…")
                        ) : (
                          t("Use", "इस्तेमाल करें")
                        )}
                      </span>
                    </button>
                    <button
                      type="button"
                      className="voice-delete"
                      onClick={() => void deleteSavedVoice(voice)}
                      disabled={deleteBusy !== null || voiceBusy !== null}
                      aria-label={t(
                        `Delete ${takeLabel(voice.seconds, voice.saved_at, uiLang)}`,
                        `${takeLabel(voice.seconds, voice.saved_at, uiLang)} हटाएँ`,
                      )}
                      title={t("Delete this voice", "यह आवाज़ हटाएँ")}
                    >
                      {deleteBusy === voice.id ? (
                        <span className="voice-delete-busy" aria-hidden="true" />
                      ) : (
                        <TrashIcon />
                      )}
                    </button>
                  </div>
                );
              })}
              {unclonedRecordings.map((rec) => (
                <button
                  type="button"
                  key={rec.name}
                  className="voice-row"
                  onClick={() => void cloneFromRecording(rec)}
                  disabled={voiceBusy !== null}
                >
                  <span className="voice-row-main">
                    <span className="voice-row-title">
                      {takeLabel(rec.seconds, rec.created_at, uiLang)}
                    </span>
                    <span className="voice-row-meta">
                      {t("Recording, not cloned yet", "रिकॉर्डिंग, अभी क्लोन नहीं हुई")}
                    </span>
                  </span>
                  <span className="voice-row-act">
                    {voiceBusy === rec.name
                      ? t("Creating…", "बना रहे हैं…")
                      : t("Create voice", "आवाज़ बनाएँ")}
                  </span>
                </button>
              ))}
              {savedVoices.length === 0 && unclonedRecordings.length === 0 && (
                <p className="chip-empty">
                  {t(
                    "No saved voices yet. Record one to get started.",
                    "अभी कोई सहेजी गई आवाज़ नहीं है। शुरू करने के लिए एक रिकॉर्ड करें।",
                  )}
                </p>
              )}
              {unclonedRecordings.length > 0 && (
                <p className="voice-hint">
                  {t(
                    "Creating a voice from a saved take uses about 50 Gnani credits.",
                    "सहेजे गए टेक से आवाज़ बनाने में लगभग 50 Gnani क्रेडिट लगते हैं।",
                  )}
                </p>
              )}
            </div>
          )}
          <div className="surface-divider" aria-hidden="true" />
          <div
            className="listen-lang-bar"
            role="region"
            aria-label={t("Partner speech language", "साथी की बोली की भाषा")}
          >
            <div className="listen-lang-label">
              <MicIcon />
              <span>{t("Hear speech in:", "बोली इस भाषा में सुनें:")}</span>
            </div>
            <div className="listen-lang-pills">
              <button
                type="button"
                className={`lang-pill ${listenLang === "en-IN" ? "active" : ""}`}
                onClick={() => changeListenLang("en-IN")}
                title={t("Transcribe partner in English", "साथी की बोली अंग्रेज़ी में ट्रांसक्राइब करें")}
              >
                {t("English", "अंग्रेज़ी")}
              </button>
              <button
                type="button"
                className={`lang-pill ${listenLang === "hi-IN" ? "active" : ""}`}
                onClick={() => changeListenLang("hi-IN")}
                title={t("Transcribe partner in Hindi", "साथी की बोली हिन्दी में ट्रांसक्राइब करें")}
              >
                हिन्दी (Hindi)
              </button>
              <button
                type="button"
                className={`lang-pill ${listenLang === "hi-en" ? "active" : ""}`}
                onClick={() => changeListenLang("hi-en")}
                title={t(
                  "Transcribe partner in Hinglish (Roman Hindi)",
                  "साथी की बोली Hinglish (Roman Hindi) में ट्रांसक्राइब करें",
                )}
              >
                Hinglish
              </button>
              <div className="lang-select-wrap">
                <select
                  value={listenLang}
                  onChange={(e) => changeListenLang(e.target.value)}
                  aria-label={t("All supported languages", "सभी समर्थित भाषाएँ")}
                  title={t("Select from 10+ languages", "10+ भाषाओं में से चुनें")}
                >
                  {SUPPORTED_LANGUAGES.map((l) => (
                    <option key={l.code} value={l.code}>
                      {l.label} ({l.native})
                    </option>
                  ))}
                </select>
              </div>
            </div>
          </div>
          <section className="thread" ref={threadRef} aria-label={t("Conversation", "बातचीत")}>
            {thread.length === 0 && !liveText.trim() ? (
              <p className="thread-empty">
                {t("Partner speech will appear here.", "साथी की बोली यहाँ दिखाई देगी।")}
              </p>
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
                        aria-label={t(
                          `Replay in your voice: ${turn.text}`,
                          `अपनी आवाज़ में फिर सुनें: ${turn.text}`,
                        )}
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
                ? t("Connecting to Prisma…", "Prisma से कनेक्ट हो रहे हैं…")
                : listenState === "fallback"
                  ? t("Browser speech fallback is active", "ब्राउज़र वाक् फ़ॉलबैक सक्रिय है")
                  : status ||
                    (listenState === "live"
                      ? t(
                          `Listening in ${SUPPORTED_LANGUAGES.find((l) => l.code === listenLang)?.label || listenLang} on Prisma v3.0`,
                          `${listenLang === "hi-en" ? "Hinglish" : SUPPORTED_LANGUAGES.find((l) => l.code === listenLang)?.native || listenLang} में सुन रहे हैं, Prisma v3.0`,
                        )
                      : "")}
            </p>
          </section>

          <div className="surface-divider" aria-hidden="true" />

          <section className="surface-block" aria-label={t("Suggestions", "सुझाव")}>
            <div className="block-head">
              <span className="surface-label">{t("Suggestions", "सुझाव")}</span>
              <button
                type="button"
                className="icon-btn"
                onClick={() => void loadSuggestions()}
                aria-label={t("Refresh suggestions", "सुझाव रीफ़्रेश करें")}
                title={t("Refresh suggestions", "सुझाव रीफ़्रेश करें")}
              >
                <RefreshIcon />
              </button>
            </div>
            <div className="chip-scroll">
              {replies.length === 0 ? (
                <p className="chip-empty">
                  {t(
                    "No suggestions yet. Start listening, then tap refresh.",
                    "अभी कोई सुझाव नहीं। सुनना शुरू करें, फिर रीफ़्रेश पर टैप करें।",
                  )}
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
                    <span className="chip-tone">{emotionLabel(r.emotion)}</span>
                    {r.text}
                  </button>
                ))
              )}
            </div>
          </section>

          <div className="surface-divider" aria-hidden="true" />

          <section className="surface-block" aria-label={t("Favourites", "पसंदीदा")}>
            <div className="block-head">
              <span className="surface-label">{t("Favourites", "पसंदीदा")}</span>
              <button
                type="button"
                className="icon-btn"
                onClick={() => setFavOpen((open) => !open)}
                aria-label={
                  favOpen
                    ? t("Close the favourite editor", "पसंदीदा संपादक बंद करें")
                    : t("Add or remove favourites", "पसंदीदा जोड़ें या हटाएँ")
                }
                aria-expanded={favOpen}
                title={
                  favOpen
                    ? t("Close the favourite editor", "पसंदीदा संपादक बंद करें")
                    : t("Add or remove favourites", "पसंदीदा जोड़ें या हटाएँ")
                }
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
                      aria-label={t(
                        `Remove ${f.text} from favourites`,
                        `${f.text} को पसंदीदा से हटाएँ`,
                      )}
                      title={t("Remove", "हटाएँ")}
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
                  placeholder={t("Add your own words or phrase", "अपने शब्द या वाक्यांश जोड़ें")}
                  maxLength={80}
                  aria-label={t("New favourite phrase", "नया पसंदीदा वाक्यांश")}
                />
                <select
                  value={favEmotion}
                  onChange={(e) => setFavEmotion(e.target.value as Emotion)}
                  aria-label={t("Tone for the new phrase", "नए वाक्यांश का भाव")}
                >
                  <option value="warm">{emotionLabel("warm")}</option>
                  <option value="calm">{emotionLabel("calm")}</option>
                  <option value="urgent">{emotionLabel("urgent")}</option>
                  <option value="playful">{emotionLabel("playful")}</option>
                </select>
                <button
                  type="button"
                  className="chip-add"
                  onClick={addCustomFavourite}
                  disabled={!favText.trim()}
                >
                  {t("Add", "जोड़ें")}
                </button>
              </div>
            )}
          </section>

          {dockVisible && (
            <div className="dock" role="region" aria-label={t("Reply dock", "जवाब डॉक")}>
              <div className="dock-shell">
                <div className="dock-inner">
                <button
                  type="button"
                  className={`dock-mic ${listening ? "live" : ""}`}
                  onClick={listening ? stopListen : () => void startListen()}
                  disabled={!listening && listenState === "connecting"}
                  aria-label={
                    listening ? t("Stop listening", "सुनना बंद करें") : t("Start listening", "सुनना शुरू करें")
                  }
                  title={
                    listening ? t("Stop listening", "सुनना बंद करें") : t("Start listening", "सुनना शुरू करें")
                  }
                >
                  {listening ? (
                    <span className="dock-stop" aria-hidden="true" />
                  ) : (
                    <MicIcon />
                  )}
                </button>
                <span className="dock-lang" title={t("Partner speech language (STT)", "साथी की बोली की भाषा (STT)")}>
                  <select
                    value={listenLang}
                    onChange={(e) => changeListenLang(e.target.value)}
                    aria-label={t("Partner speech language", "साथी की बोली की भाषा")}
                  >
                    {SUPPORTED_LANGUAGES.map((l) => (
                      <option key={l.code} value={l.code}>
                        {l.short} {l.label}
                      </option>
                    ))}
                  </select>
                </span>
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
                  placeholder={t("Type a reply", "जवाब टाइप करें")}
                  aria-label={t("Your reply", "आपका जवाब")}
                />
                <span className="dock-tone">
                  <select
                    value={emotion}
                    onChange={(e) => setEmotion(e.target.value as Emotion)}
                    aria-label={t("Tone", "भाव")}
                  >
                    <option value="warm">{emotionLabel("warm")}</option>
                    <option value="urgent">{emotionLabel("urgent")}</option>
                    <option value="playful">{emotionLabel("playful")}</option>
                    <option value="calm">{emotionLabel("calm")}</option>
                  </select>
                </span>
                <button
                  type="button"
                  className="dock-send"
                  disabled={speaking || !draft.trim()}
                  onClick={() => void speakText(draft, emotion)}
                  aria-label={t("Speak this reply", "यह जवाब बोलें")}
                  title={t("Speak this reply", "यह जवाब बोलें")}
                >
                  <SendIcon />
                </button>
                </div>
              </div>
            </div>
          )}
        </div>
      )}
          </div>
        </div>
      </div>
    </main>
  );
}
