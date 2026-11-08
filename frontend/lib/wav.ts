export function encodeWav(samples: Float32Array, sampleRate: number): Blob {
  const pcm = new Int16Array(samples.length);
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    pcm[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  const buffer = new ArrayBuffer(44 + pcm.length * 2);
  const view = new DataView(buffer);
  writeStr(view, 0, "RIFF");
  view.setUint32(4, 36 + pcm.length * 2, true);
  writeStr(view, 8, "WAVE");
  writeStr(view, 12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeStr(view, 36, "data");
  view.setUint32(40, pcm.length * 2, true);
  let offset = 44;
  for (let i = 0; i < pcm.length; i++, offset += 2) {
    view.setInt16(offset, pcm[i], true);
  }
  return new Blob([buffer], { type: "audio/wav" });
}

function writeStr(view: DataView, offset: number, text: string) {
  for (let i = 0; i < text.length; i++) view.setUint8(offset + i, text.charCodeAt(i));
}

export function downsample(input: Float32Array, fromRate: number, toRate: number): Float32Array {
  if (fromRate === toRate) return input;
  const ratio = fromRate / toRate;
  const out = new Float32Array(Math.floor(input.length / ratio));
  for (let i = 0; i < out.length; i++) out[i] = input[Math.floor(i * ratio)];
  return out;
}

export function floatToPcm16(input: Float32Array): Int16Array {
  const pcm = new Int16Array(input.length);
  for (let i = 0; i < input.length; i++) {
    const s = Math.max(-1, Math.min(1, input[i]));
    pcm[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  return pcm;
}

/** Remove DC offset, trim leading/trailing near-silence, and peak-normalise. */
export function prepareVoiceSample(
  input: Float32Array,
  sampleRate: number,
): Float32Array {
  if (input.length === 0) return input;

  // Remove any DC bias from the microphone/interface.
  let mean = 0;
  for (let i = 0; i < input.length; i++) mean += input[i];
  mean /= input.length;
  const centered = new Float32Array(input.length);
  for (let i = 0; i < input.length; i++) centered[i] = input[i] - mean;

  // A laptop mic often carries low-frequency rumble (fan, desk, AC). Speaker
  // embeddings latch onto that and the clone can sound deep or droney.
  const filtered = highPass(centered, sampleRate, 80);
  // Breaths in the reference are learned by the clone and replayed after every
  // reply, so remove them before the embedding is created.
  const cleaned = removeBreaths(filtered, sampleRate);
  const trimmed = trimQuietEdges(cleaned, sampleRate);
  const normalized = normalizePeak(trimmed, 0.92);
  return fadeEdges(normalized, sampleRate, 8);
}

/**
 * Silence long breath runs in a reference recording.
 *
 * A breath is broadband noise, so almost every pair of adjacent samples
 * crosses zero (zero-crossing rate ~0.3) while voiced speech sits near 0.02.
 * Fricatives are also unvoiced but brief, so only runs longer than 200 ms are
 * treated as breaths. The run is replaced with silence; if it sits between
 * words it becomes a natural pause, and at the edges `trimQuietEdges` removes
 * it entirely.
 */
function removeBreaths(
  input: Float32Array,
  sampleRate: number,
  zcrThreshold = 0.18,
  minRunMs = 200,
  levelRatio = 0.04,
): Float32Array {
  if (input.length === 0) return input;
  const frame = Math.max(1, Math.floor(sampleRate * 0.02));
  const count = Math.floor(input.length / frame);
  if (count < 3) return input;

  const rms = new Float32Array(count);
  const zcr = new Float32Array(count);
  for (let f = 0; f < count; f++) {
    const start = f * frame;
    let sum = 0;
    let crossings = 0;
    let previous = input[start];
    for (let i = start; i < start + frame; i++) {
      const value = input[i];
      sum += value * value;
      if ((previous >= 0) !== (value >= 0)) crossings++;
      previous = value;
    }
    rms[f] = Math.sqrt(sum / frame);
    zcr[f] = crossings / frame;
  }

  const ordered = Array.from(rms).sort((a, b) => a - b);
  const p90 = ordered[Math.min(ordered.length - 1, Math.floor(ordered.length * 0.9))];
  if (p90 <= 0) return input;
  const floor = Math.max(p90 * levelRatio, 0.002);

  const runs: Array<[number, number]> = [];
  let start = -1;
  let lastActive = -1;
  for (let f = 0; f < count; f++) {
    const active = rms[f] >= floor && zcr[f] >= zcrThreshold;
    if (active) {
      if (start < 0) start = f;
      lastActive = f;
    } else if (start >= 0 && f - lastActive > 2) {
      runs.push([start, lastActive + 1]);
      start = -1;
    }
  }
  if (start >= 0) runs.push([start, lastActive + 1]);

  const minFrames = Math.max(1, Math.floor(minRunMs / 20));
  const fade = Math.max(1, Math.floor((sampleRate * 6) / 1000));
  let out: Float32Array | null = null;
  for (const [runStart, runEnd] of runs) {
    if (runEnd - runStart < minFrames) continue;
    const begin = runStart * frame;
    const end = Math.min(input.length, runEnd * frame);
    if (end <= begin) continue;
    if (!out) out = input.slice();
    for (let i = Math.max(0, begin - fade); i < begin; i++) out[i] *= (begin - i) / fade;
    for (let i = begin; i < end; i++) out[i] = 0;
    for (let i = end; i < Math.min(out.length, end + fade); i++) out[i] *= (i - end) / fade;
  }
  return out ?? input;
}

/** First-order high-pass to remove DC/rumble below the cutoff. */
function highPass(input: Float32Array, sampleRate: number, cutoffHz: number): Float32Array {
  const rc = 1 / (2 * Math.PI * cutoffHz);
  const dt = 1 / sampleRate;
  const alpha = rc / (rc + dt);
  const out = new Float32Array(input.length);
  let prevIn = input[0] ?? 0;
  let prevOut = 0;
  for (let i = 0; i < input.length; i++) {
    const x = input[i];
    prevOut = alpha * (prevOut + x - prevIn);
    prevIn = x;
    out[i] = prevOut;
  }
  return out;
}

/**
 * Trim leading/trailing quiet audio relative to the clip's own speech level.
 * A fixed amplitude threshold keeps breaths and room tone in the reference,
 * and the clone then learns to add that breathy tail to every reply. Comparing
 * each 20 ms frame with the speech level removes the tail without clipping
 * quiet words.
 */
function trimQuietEdges(
  input: Float32Array,
  sampleRate: number,
  ratio = 0.12,
  keepMs = 100,
): Float32Array {
  if (input.length === 0) return input;
  const frame = Math.max(1, Math.floor(sampleRate * 0.02));
  const rms: number[] = [];
  for (let start = 0; start + frame <= input.length; start += frame) {
    let sum = 0;
    for (let i = start; i < start + frame; i++) sum += input[i] * input[i];
    rms.push(Math.sqrt(sum / frame));
  }
  if (rms.length < 3) return input;

  const ordered = [...rms].sort((a, b) => a - b);
  const p90 = ordered[Math.min(ordered.length - 1, Math.floor(ordered.length * 0.9))];
  if (p90 <= 0) return input;
  const threshold = Math.max(p90 * ratio, 0.0015);

  let first = 0;
  while (first < rms.length && rms[first] < threshold) first++;
  if (first >= rms.length) return input; // all silence: let the server reject it
  let last = rms.length - 1;
  while (last > first && rms[last] < threshold) last--;

  const keep = Math.floor((sampleRate * keepMs) / 1000);
  const start = Math.max(0, first * frame - keep);
  const end = Math.min(input.length, (last + 1) * frame + keep);
  return input.slice(start, end);
}

function normalizePeak(input: Float32Array, target: number): Float32Array {
  let peak = 0;
  for (let i = 0; i < input.length; i++) {
    const value = Math.abs(input[i]);
    if (value > peak) peak = value;
  }
  if (peak === 0 || Math.abs(peak - target) < 0.001) return input;
  const gain = target / peak;
  const out = new Float32Array(input.length);
  for (let i = 0; i < input.length; i++) {
    out[i] = Math.max(-1, Math.min(1, input[i] * gain));
  }
  return out;
}

function fadeEdges(input: Float32Array, sampleRate: number, ms: number): Float32Array {
  const fade = Math.min(Math.floor((sampleRate * ms) / 1000), Math.floor(input.length / 2));
  if (fade <= 0) return input;
  const out = input.slice();
  for (let i = 0; i < fade; i++) {
    const gain = i / fade;
    out[i] *= gain;
    out[out.length - 1 - i] *= gain;
  }
  return out;
}

/** Approximate RMS level in the 0-100 range for a simple UI meter. */
export function levelPercent(input: Float32Array): number {
  if (input.length === 0) return 0;
  let sum = 0;
  for (let i = 0; i < input.length; i++) sum += input[i] * input[i];
  const rms = Math.sqrt(sum / input.length);
  const db = 20 * Math.log10(rms + 1e-8);
  return Math.max(0, Math.min(100, ((db + 60) / 60) * 100));
}

export type SampleQuality = {
  score: number;
  verdict: "good" | "fair" | "poor";
  warnings: string[];
};

const dbfs = (value: number) => 20 * Math.log10(Math.max(value, 1e-6));

/**
 * Score a reference clip before it is trimmed and normalised. This is the last
 * chance to warn about clipping, noise, or long silences without spending an
 * embedding credit upstream.
 */
export function analyzeSample(input: Float32Array, sampleRate: number): SampleQuality {
  if (input.length === 0 || sampleRate <= 0) {
    return { score: 0, verdict: "poor", warnings: ["The clip is empty."] };
  }

  const step = Math.max(1, Math.floor(input.length / 30_000));
  const values: number[] = [];
  for (let i = 0; i < input.length; i += step) values.push(input[i]);
  const n = values.length;

  let peak = 0;
  let total = 0;
  let totalSq = 0;
  let clipped = 0;
  for (const value of values) {
    const magnitude = Math.abs(value);
    if (magnitude > peak) peak = magnitude;
    if (magnitude >= 0.995) clipped++;
    total += value;
    totalSq += value * value;
  }
  const dc = Math.abs(total / n);
  const rms = Math.sqrt(totalSq / n);

  const frameLen = Math.max(1, Math.floor((sampleRate * 0.02) / step));
  const frameRms: number[] = [];
  for (let start = 0; start + frameLen <= n; start += frameLen) {
    let sum = 0;
    for (let i = start; i < start + frameLen; i++) sum += values[i] * values[i];
    frameRms.push(Math.sqrt(sum / frameLen));
  }
  frameRms.sort((a, b) => a - b);
  const percentile = (fraction: number) =>
    frameRms[Math.min(frameRms.length - 1, Math.max(0, Math.floor(frameRms.length * fraction)))] ??
    0;

  const silenceRatio = frameRms.length
    ? frameRms.filter((value) => value < 0.004).length / frameRms.length
    : 0;
  const noise = percentile(0.1);
  const speech = percentile(0.9);
  const snr = speech > 0 ? 20 * Math.log10(Math.max(speech, 1e-6) / Math.max(noise, 1e-6)) : 0;
  const clippingPercent = (100 * clipped) / n;
  const rmsDb = dbfs(rms);

  let score = 100;
  const warnings: string[] = [];

  if (clippingPercent >= 0.5) {
    score -= Math.min(60, clippingPercent * 12);
    warnings.push(
      `About ${clippingPercent.toFixed(1)}% of the sample is clipped (distorted). Lower the input level or move back from the microphone.`,
    );
  }
  if (rmsDb < -38) {
    score -= 30;
    warnings.push("The clip is very quiet. Speak closer to the microphone.");
  } else if (rmsDb < -32) {
    score -= 12;
    warnings.push("The clip is quiet. A louder, close-mic take clones better.");
  }
  if (silenceRatio > 0.75) {
    score -= 40;
    warnings.push(
      `${Math.round(silenceRatio * 100)}% of the clip is silence. Speak continuously for 15–30 seconds.`,
    );
  } else if (silenceRatio > 0.55) {
    score -= 20;
    warnings.push(
      `${Math.round(silenceRatio * 100)}% of the clip is silence. Long pauses make the clone less accurate.`,
    );
  }
  if (clippingPercent < 1) {
    if (snr < 6) {
      score -= 35;
      warnings.push(
        "Background noise is almost as loud as the voice. Record in a quiet room with no fan, traffic, music, or TV.",
      );
    } else if (snr < 12) {
      score -= 18;
      warnings.push("Background noise is high. Use a quieter room or a headset microphone.");
    }
  }
  if (dc > 0.02) {
    score -= 8;
    warnings.push("There is low-frequency rumble. Remove fan/AC noise or use a headset microphone.");
  }

  score = Math.max(0, Math.min(100, Math.round(score)));
  const verdict = score >= 85 ? "good" : score >= 60 ? "fair" : "poor";
  return { score, verdict, warnings };
}
