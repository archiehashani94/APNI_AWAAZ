/**
 * A single analyser that every spoken clip is routed through, so the 3D voice
 * core can pulse with the actual cloned audio instead of a fake animation.
 *
 * The analyser is optional: if the browser will not give us a running
 * AudioContext, playback falls back to the normal media element path.
 */

type AudioCtor = typeof AudioContext;

let ctx: AudioContext | null = null;
let analyser: AnalyserNode | null = null;
let bins: Uint8Array<ArrayBuffer> | null = null;

function audioCtor(): AudioCtor | null {
  if (typeof window === "undefined") return null;
  const win = window as unknown as {
    AudioContext?: AudioCtor;
    webkitAudioContext?: AudioCtor;
  };
  return win.AudioContext ?? win.webkitAudioContext ?? null;
}

/** Route one media element through the shared analyser. Returns false on failure. */
export async function attachAudio(el: HTMLAudioElement): Promise<boolean> {
  const Ctor = audioCtor();
  if (!Ctor) return false;
  try {
    if (!ctx) {
      ctx = new Ctor();
      analyser = ctx.createAnalyser();
      analyser.fftSize = 512;
      analyser.smoothingTimeConstant = 0.72;
      analyser.connect(ctx.destination);
      bins = new Uint8Array(analyser.frequencyBinCount);
    }
    if (ctx.state !== "running") await ctx.resume();
    if (ctx.state !== "running" || !analyser) return false;
    // Each clip uses a fresh <audio> element, so one source per element is safe.
    const source = ctx.createMediaElementSource(el);
    source.connect(analyser);
    return true;
  } catch {
    return false;
  }
}

/** Smoothed broadband level, 0–1. */
export function audioLevel(): number {
  if (!analyser || !bins) return 0;
  analyser.getByteFrequencyData(bins);
  let sum = 0;
  for (let i = 0; i < bins.length; i++) sum += bins[i];
  return Math.min(1, (sum / bins.length / 255) * 2.6);
}
