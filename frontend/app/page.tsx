"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { API_URL, api, type Emotion, type Favourite, type Reply, wsUrl } from "@/lib/api";
import { downsample, encodeWav, floatToPcm16 } from "@/lib/wav";

type Stage = "consent" | "enrol" | "talk";

type BrowserRec = {
  continuous: boolean;
  interimResults: boolean;
  lang: string;
  onresult: ((event: unknown) => void) | null;
  start: () => void;
  stop: () => void;
};

export default function Home() {
  const [sessionId, setSessionId] = useState("");
  const [stage, setStage] = useState<Stage>("consent");
  const [agreed, setAgreed] = useState(false);
  const [status, setStatus] = useState("");
  const [error, setError] = useState("");
  const [recording, setRecording] = useState(false);
  const [seconds, setSeconds] = useState(0);
  const [cloneReady, setCloneReady] = useState(false);
  const [listening, setListening] = useState(false);
  const [liveText, setLiveText] = useState("");
  const [replies, setReplies] = useState<Reply[]>([]);
  const [draft, setDraft] = useState("");
  const [emotion, setEmotion] = useState<Emotion>("warm");
  const [favourites, setFavourites] = useState<Favourite[]>([]);
  const [speaking, setSpeaking] = useState(false);
  const [aiLabel, setAiLabel] = useState(false);
  const [credits, setCredits] = useState<string>("");
  const [fallbacks, setFallbacks] = useState<string[]>([]);

  const recChunks = useRef<Float32Array[]>([]);
  const recRate = useRef(16000);
  const recTimer = useRef<number | null>(null);
  const recCtx = useRef<AudioContext | null>(null);
  const recProc = useRef<ScriptProcessorNode | null>(null);
  const recStream = useRef<MediaStream | null>(null);
  const listenWs = useRef<WebSocket | null>(null);
  const listenCtx = useRef<AudioContext | null>(null);
  const pcmBuf = useRef<Int16Array>(new Int16Array(0));
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const favCache = useRef<Map<string, Blob>>(new Map());
  const speechRec = useRef<BrowserRec | null>(null);

  const addFallback = useCallback((kind: string) => {
    setFallbacks((prev) => (prev.includes(kind) ? prev : [...prev, kind]));
  }, []);

  useEffect(() => {
    let sid = localStorage.getItem("aa_session");
    (async () => {
      if (!sid) {
        const res = await fetch(`${API_URL}/api/session`, { method: "POST" });
        const data = await res.json();
        sid = data.session_id as string;
        localStorage.setItem("aa_session", sid);
      }
      setSessionId(sid!);
      const fav = await fetch(`${API_URL}/api/favourites`);
      setFavourites((await fav.json()).favourites);
    })().catch((err) => setError(String(err)));
  }, []);

  async function refreshCredits() {
    if (!sessionId) return;
    const res = await api("/api/credits", sessionId);
    const data = await res.json();
    setCredits(`${data.used} / ${data.budget} credits`);
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
      setReplies(data.replies || []);
      setStage("talk");
      setAiLabel(true);
      refreshCredits();
    } catch (err) {
      setError(String(err));
    }
  }

  async function startRecord() {
    setError("");
    recChunks.current = [];
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    recStream.current = stream;
    const ctx = new AudioContext();
    recCtx.current = ctx;
    recRate.current = ctx.sampleRate;
    const source = ctx.createMediaStreamSource(stream);
    const proc = ctx.createScriptProcessor(4096, 1, 1);
    recProc.current = proc;
    proc.onaudioprocess = (ev) => {
      recChunks.current.push(new Float32Array(ev.inputBuffer.getChannelData(0)));
    };
    const mute = ctx.createGain();
    mute.gain.value = 0;
    source.connect(proc);
    proc.connect(mute);
    mute.connect(ctx.destination);
    setRecording(true);
    setSeconds(0);
    recTimer.current = window.setInterval(() => {
      setSeconds((s) => {
        if (s >= 29) {
          void stopRecord();
          return 30;
        }
        return s + 1;
      });
    }, 1000);
  }

  async function stopRecord() {
    if (recTimer.current) window.clearInterval(recTimer.current);
    recProc.current?.disconnect();
    recCtx.current?.close();
    recStream.current?.getTracks().forEach((t) => t.stop());
    setRecording(false);
    const total = recChunks.current.reduce((n, c) => n + c.length, 0);
    const merged = new Float32Array(total);
    let o = 0;
    for (const c of recChunks.current) {
      merged.set(c, o);
      o += c.length;
    }
    const wav = encodeWav(merged, recRate.current);
    await uploadVoice(wav, "enrol.wav");
  }

  async function uploadVoice(file: Blob, name: string) {
    setStatus("Creating voice embedding…");
    const form = new FormData();
    form.append("audio", file, name);
    const res = await api("/api/enrol", sessionId, { method: "POST", body: form });
    const data = await res.json();
    if (!res.ok) {
      setError(data.detail || "Enrol failed");
      setStatus("");
      return;
    }
    setCloneReady(Boolean(data.clone_ready));
    if (data.fallback) addFallback("stock_voice");
    setStage("talk");
    setStatus("");
    setAiLabel(true);
    refreshCredits();
  }

  async function onFile(ev: React.ChangeEvent<HTMLInputElement>) {
    const file = ev.target.files?.[0];
    if (file) await uploadVoice(file, file.name);
  }

  function startBrowserSpeech() {
    const SR =
      (window as Window & { SpeechRecognition?: new () => BrowserRec; webkitSpeechRecognition?: new () => BrowserRec })
        .SpeechRecognition ||
      (window as Window & { webkitSpeechRecognition?: new () => BrowserRec }).webkitSpeechRecognition;
    if (!SR) {
      addFallback("browser_speech_unavailable");
      return;
    }
    addFallback("browser_speech");
    const rec = new SR();
    rec.continuous = true;
    rec.interimResults = true;
    rec.lang = "en-IN";
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
        setLiveText(finalText);
        listenWs.current?.send(JSON.stringify({ type: "browser_transcript", text: finalText }));
        void loadSuggestions();
      }
    };
    rec.start();
    speechRec.current = rec;
  }

  async function startListen() {
    if (!sessionId) return;
    setListening(true);
    setError("");
    const socket = new WebSocket(wsUrl(`/ws/listen?session_id=${sessionId}`));
    listenWs.current = socket;
    socket.onmessage = (ev) => {
      const frame = JSON.parse(ev.data);
      if (frame.type === "fallback") {
        addFallback(frame.fallback_kind || "browser_speech");
        startBrowserSpeech();
      }
      if (frame.type === "transcript" && frame.text) {
        setLiveText(frame.text);
        void loadSuggestions();
      }
    };
    socket.onerror = () => {
      addFallback("browser_speech");
      startBrowserSpeech();
    };
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const ctx = new AudioContext({ sampleRate: 16000 });
    listenCtx.current = ctx;
    const source = ctx.createMediaStreamSource(stream);
    const proc = ctx.createScriptProcessor(4096, 1, 1);
    proc.onaudioprocess = (ev) => {
      if (socket.readyState !== WebSocket.OPEN) return;
      const raw = ev.inputBuffer.getChannelData(0);
      const ds = downsample(raw, ctx.sampleRate, 16000);
      const pcm = floatToPcm16(ds);
      const merged = new Int16Array(pcmBuf.current.length + pcm.length);
      merged.set(pcmBuf.current);
      merged.set(pcm, pcmBuf.current.length);
      let offset = 0;
      while (offset + 512 <= merged.length) {
        socket.send(merged.subarray(offset, offset + 512).buffer);
        offset += 512;
      }
      pcmBuf.current = merged.subarray(offset);
    };
    const mute = ctx.createGain();
    mute.gain.value = 0;
    source.connect(proc);
    proc.connect(mute);
    mute.connect(ctx.destination);
  }

  function stopListen() {
    setListening(false);
    listenWs.current?.close();
    listenCtx.current?.close();
    speechRec.current?.stop();
  }

  async function loadSuggestions() {
    const res = await api("/api/suggest", sessionId, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({}),
    });
    const data = await res.json();
    if (data.fallback) addFallback("canned_replies");
    setReplies(data.replies || []);
  }

  async function speakText(text: string, emo: Emotion) {
    if (speaking) return;
    setSpeaking(true);
    setError("");
    const cached = favCache.current.get(`${text}|${emo}`);
    try {
      const blob =
        cached ||
        (await (async () => {
          const res = await api("/api/speak", sessionId, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text, emotion: emo }),
          });
          if (!res.ok) throw new Error((await res.json()).detail || "Speak failed");
          if (res.headers.get("X-Fallback") === "true") addFallback("stock_voice");
          setAiLabel(res.headers.get("X-AI-Generated") === "true");
          return res.blob();
        })());
      const url = URL.createObjectURL(blob);
      const audio = audioRef.current || new Audio();
      audioRef.current = audio;
      audio.src = url;
      await audio.play();
      refreshCredits();
    } catch (err) {
      setError(String(err));
    } finally {
      setSpeaking(false);
    }
  }

  async function playFavourite(fav: Favourite) {
    const key = `${fav.text}|${fav.emotion}`;
    if (!favCache.current.has(key) && !speaking) {
      const res = await api("/api/speak", sessionId, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: fav.text, emotion: fav.emotion }),
      });
      if (res.ok) favCache.current.set(key, await res.blob());
    }
    await speakText(fav.text, fav.emotion);
  }

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
          <p className="lede">Upload or record about 20 seconds of clear speech (5–30 s).</p>
          <div className="row">
            {!recording ? (
              <button onClick={startRecord}>Record</button>
            ) : (
              <button onClick={stopRecord}>Stop ({seconds}s)</button>
            )}
            <label className="badge">
              Upload file
              <input type="file" accept="audio/*" hidden onChange={onFile} />
            </label>
          </div>
          <p className="meter">{recording ? `Recording… ${seconds}s / 20s suggested` : status}</p>
        </section>
      )}

      {stage === "talk" && (
        <>
          <section className="card">
            <h2>Listen</h2>
            <p className="lede">
              {cloneReady ? "Cloned voice is ready." : "Using a stock Timbre voice until cloning succeeds."}
            </p>
            <div className="row">
              {!listening ? (
                <button onClick={startListen}>Start listening</button>
              ) : (
                <button className="secondary" onClick={stopListen}>
                  Stop listening
                </button>
              )}
            </div>
            <p className="transcript">{liveText || "Partner speech will appear here."}</p>
          </section>

          <section className="card">
            <h2>Suggest</h2>
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
          </section>

          <section className="card">
            <h2>Speak</h2>
            <textarea
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              placeholder="Type your own reply"
            />
            <div className="row" style={{ marginTop: "0.7rem" }}>
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
            <h2>Favourites</h2>
            <div className="row">
              {favourites.map((f) => (
                <button key={f.id} className="fav" disabled={speaking} onClick={() => playFavourite(f)}>
                  {f.text}
                </button>
              ))}
            </div>
          </section>
        </>
      )}

    </main>
  );
}
