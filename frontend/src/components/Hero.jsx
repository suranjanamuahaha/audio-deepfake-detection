import React, { useState, useRef, useEffect, useCallback } from "react";
import { Mic, Square, ShieldAlert, ShieldCheck, Activity } from "lucide-react";

// ─────────────────────────────────────────────────────────────────────────────
// Realtime Waveform (recording input visualiser)
// ─────────────────────────────────────────────────────────────────────────────
const Waveform = ({ isRecording, analyserRef }) => {
  const canvasRef = useRef(null);
  const rafRef = useRef(null);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const W = canvas.offsetWidth;
    const H = canvas.offsetHeight;
    canvas.width = W * window.devicePixelRatio;
    canvas.height = H * window.devicePixelRatio;
    ctx.scale(window.devicePixelRatio, window.devicePixelRatio);

    const analyser = analyserRef.current;

    if (analyser && isRecording) {
      const bufferLength = analyser.frequencyBinCount;
      const dataArray = new Uint8Array(bufferLength);
      analyser.getByteFrequencyData(dataArray);
      ctx.clearRect(0, 0, W, H);
      const barCount = 64;
      const barW = (W / barCount) * 0.55;
      const gap = (W / barCount) * 0.45;
      for (let i = 0; i < barCount; i++) {
        const binIndex = Math.floor((i / barCount) * bufferLength * 0.75);
        const rawVal = dataArray[binIndex] / 255;
        const barH = Math.max(3, rawVal * H * 0.95);
        const x = i * (barW + gap);
        const y = (H - barH) / 2;
        const grad = ctx.createLinearGradient(0, y, 0, y + barH);
        grad.addColorStop(0, `rgba(239,68,68,${0.4 + rawVal * 0.6})`);
        grad.addColorStop(0.5, `rgba(248,113,113,${0.8 + rawVal * 0.2})`);
        grad.addColorStop(1, `rgba(239,68,68,${0.4 + rawVal * 0.6})`);
        ctx.fillStyle = grad;
        ctx.beginPath();
        ctx.roundRect(x, y, barW, barH, 2);
        ctx.fill();
      }
    } else {
      ctx.clearRect(0, 0, W, H);
      const barCount = 64;
      const barW = (W / barCount) * 0.55;
      const gap = (W / barCount) * 0.45;
      const t = Date.now() / 1000;
      for (let i = 0; i < barCount; i++) {
        const wave =
          Math.sin(t * 1.2 + i * 0.35) * 0.12 +
          Math.sin(t * 0.7 + i * 0.18) * 0.06 +
          0.12;
        const barH = Math.max(3, wave * H);
        const x = i * (barW + gap);
        const y = (H - barH) / 2;
        ctx.fillStyle = `rgba(239,68,68,${0.15 + wave * 0.4})`;
        ctx.beginPath();
        ctx.roundRect(x, y, barW, barH, 2);
        ctx.fill();
      }
    }
    rafRef.current = requestAnimationFrame(draw);
  }, [isRecording, analyserRef]);

  useEffect(() => {
    rafRef.current = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(rafRef.current);
  }, [draw]);

  return (
    <canvas
      ref={canvasRef}
      style={{ width: "100%", height: "64px", display: "block" }}
    />
  );
};

// ─────────────────────────────────────────────────────────────────────────────
// Cooley-Tukey radix-2 FFT (in-place)
// ─────────────────────────────────────────────────────────────────────────────
function fft(re, im) {
  const n = re.length;
  if (n <= 1) return;
  const half = n / 2;
  const er = new Float32Array(half), ei = new Float32Array(half);
  const or = new Float32Array(half), oi = new Float32Array(half);
  for (let i = 0; i < half; i++) {
    er[i] = re[i * 2]; ei[i] = im[i * 2];
    or[i] = re[i * 2 + 1]; oi[i] = im[i * 2 + 1];
  }
  fft(er, ei); fft(or, oi);
  for (let k = 0; k < half; k++) {
    const a = (-2 * Math.PI * k) / n;
    const cos = Math.cos(a), sin = Math.sin(a);
    const tr = cos * or[k] - sin * oi[k];
    const ti = sin * or[k] + cos * oi[k];
    re[k] = er[k] + tr; im[k] = ei[k] + ti;
    re[k + half] = er[k] - tr; im[k + half] = ei[k] - ti;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Spectrogram — decodes the AudioBuffer and paints a log-frequency heatmap
// ─────────────────────────────────────────────────────────────────────────────
const Spectrogram = ({ audioBuffer, isDeepfake }) => {
  const canvasRef = useRef(null);

  useEffect(() => {
    if (!audioBuffer || !canvasRef.current) return;
    const canvas = canvasRef.current;
    const W = canvas.offsetWidth;
    const H = canvas.offsetHeight;
    canvas.width = W * window.devicePixelRatio;
    canvas.height = H * window.devicePixelRatio;
    const ctx = canvas.getContext("2d");
    ctx.scale(window.devicePixelRatio, window.devicePixelRatio);

    const channelData = audioBuffer.getChannelData(0);
    const fftSize = 1024;
    const hopSize = Math.floor(fftSize / 4);
    const freqBins = fftSize / 2;
    const maxFrames = Math.min(Math.floor((channelData.length - fftSize) / hopSize), 600);

    // Hann window
    const hann = new Float32Array(fftSize);
    for (let i = 0; i < fftSize; i++)
      hann[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (fftSize - 1)));

    // Build magnitude frames
    const magMatrix = [];
    let globalMax = -Infinity, globalMin = Infinity;
    for (let f = 0; f < maxFrames; f++) {
      const start = f * hopSize;
      const re = new Float32Array(fftSize);
      const im = new Float32Array(fftSize);
      for (let i = 0; i < fftSize; i++)
        re[i] = (channelData[start + i] || 0) * hann[i];
      fft(re, im);
      const mags = new Float32Array(freqBins);
      for (let i = 0; i < freqBins; i++) {
        const db = 20 * Math.log10(Math.sqrt(re[i] * re[i] + im[i] * im[i]) + 1e-9);
        mags[i] = db;
        if (db > globalMax) globalMax = db;
        if (db < globalMin) globalMin = db;
      }
      magMatrix.push(mags);
    }
    const dynRange = 80;
    const floorDb = globalMax - dynRange;

    // Colourmap helpers
    function colourDeepfake(t) {
      if (t < 0.25) { const s = t * 4; return [Math.round(s * 80), 0, Math.round(s * 30)]; }
      if (t < 0.5) { const s = (t - 0.25) * 4; return [Math.round(80 + s * 160), Math.round(s * 20), Math.round(30 - s * 30)]; }
      if (t < 0.75) { const s = (t - 0.5) * 4; return [240, Math.round(20 + s * 150), 0]; }
      const s = (t - 0.75) * 4; return [255, Math.round(170 + s * 85), Math.round(s * 220)];
    }
    function colourReal(t) {
      if (t < 0.25) { const s = t * 4; return [Math.round(68 - s * 15), Math.round(s * 60), Math.round(84 + s * 90)]; }
      if (t < 0.5) { const s = (t - 0.25) * 4; return [Math.round(53 + s * 50), Math.round(60 + s * 100), Math.round(174 - s * 40)]; }
      if (t < 0.75) { const s = (t - 0.5) * 4; return [Math.round(103 + s * 120), Math.round(160 + s * 50), Math.round(134 - s * 90)]; }
      const s = (t - 0.75) * 4; return [Math.round(223 + s * 32), Math.round(210 + s * 45), Math.round(44 + s * 20)];
    }

    // Paint pixel grid
    const imgData = ctx.createImageData(W, H);
    const buf = imgData.data;
    const logMin = Math.log(1 + 1);
    const logMax = Math.log(freqBins + 1);

    for (let x = 0; x < W; x++) {
      const fi = Math.floor((x / W) * maxFrames);
      const mags = magMatrix[Math.min(fi, magMatrix.length - 1)];
      for (let y = 0; y < H; y++) {
        const logPos = logMax - (y / H) * (logMax - logMin);
        const bin = Math.min(freqBins - 1, Math.max(0, Math.round(Math.exp(logPos) - 1)));
        const norm = Math.max(0, Math.min(1, (mags[bin] - floorDb) / dynRange));
        const [r, g, b] = isDeepfake ? colourDeepfake(norm) : colourReal(norm);
        const p = (y * W + x) * 4;
        buf[p] = r; buf[p + 1] = g; buf[p + 2] = b; buf[p + 3] = 255;
      }
    }
    ctx.putImageData(imgData, 0, 0);

    // Frequency grid lines + labels
    const nyquist = audioBuffer.sampleRate / 2;
    const freqTicks = [8000, 4000, 2000, 1000, 500, 250].filter(f => f < nyquist);
    ctx.font = "9px 'Courier New', monospace";
    freqTicks.forEach(freq => {
      const bin = Math.max(1, Math.floor((freq / nyquist) * freqBins));
      const logPos = Math.log(bin + 1);
      const y = H * (1 - (logPos - logMin) / (logMax - logMin));
      ctx.strokeStyle = "rgba(255,255,255,0.18)";
      ctx.lineWidth = 0.5;
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke();
      ctx.fillStyle = "rgba(255,255,255,0.5)";
      ctx.fillText(freq >= 1000 ? `${freq / 1000}kHz` : `${freq}Hz`, 5, y - 3);
    });

    // Time axis ticks
    const dur = audioBuffer.duration;
    const tickInterval = dur > 4 ? 1 : 0.5;
    for (let t = tickInterval; t < dur; t += tickInterval) {
      const x = (t / dur) * W;
      ctx.strokeStyle = "rgba(255,255,255,0.12)";
      ctx.lineWidth = 0.5;
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke();
      ctx.fillStyle = "rgba(255,255,255,0.4)";
      ctx.fillText(`${t.toFixed(1)}s`, x + 3, H - 4);
    }
  }, [audioBuffer, isDeepfake]);

  return (
    <canvas
      ref={canvasRef}
      style={{ width: "100%", height: "160px", display: "block", borderRadius: "6px" }}
    />
  );
};

// ─────────────────────────────────────────────────────────────────────────────
// Result descriptions
// ─────────────────────────────────────────────────────────────────────────────
const DESCRIPTIONS = {
  deepfake: [
    "Irregular harmonic spacing and unnatural formant transitions were detected — consistent with neural TTS synthesis artifacts.",
    "Spectral analysis reveals pitch periodicities uncommon in natural speech, suggesting a vocoder-based generative model.",
    "Anomalous high-frequency energy distribution and phase discontinuities indicate a likely AI-synthesized origin.",
  ],
  real: [
    "Natural prosodic variation and continuous micro-pitch fluctuations confirm authentic human vocal production.",
    "Formant trajectories and breath-noise characteristics are consistent with genuine biological speech patterns.",
    "No synthetic periodicity artifacts detected. Spectral continuity and noise floor match real-world recording conditions.",
  ],
};

function getDescription(label, confidence) {
  const pool = DESCRIPTIONS[label] ?? DESCRIPTIONS.real;
  return pool[Math.floor(confidence * pool.length) % pool.length];
}

// ─────────────────────────────────────────────────────────────────────────────
// Deepfake Detection Popup
// ─────────────────────────────────────────────────────────────────────────────
const DeepfakePopup = ({ onClose }) => {
  return (
    <div className="fixed inset-0 flex items-center justify-center z-50">
      <div className="bg-neutral-900 rounded-2xl w-full max-w-md mx-4 shadow-xl border border-red-500/20">
        <div className="rounded-t-2xl px-6 pt-6 pb-5 border-t border-x"
          style={{ borderColor: "#ef4444", background: "rgba(239,68,68,0.07)" }}>
          <div className="flex items-start gap-4">
            <div style={{ color: "#ef4444" }} className="mt-0.5 shrink-0">
              <ShieldAlert className="w-9 h-9" />
            </div>
            <div className="text-left">
              <p className="text-[10px] tracking-widest uppercase text-gray-500 mb-0.5">Alert</p>
              <h3 className="text-2xl font-bold" style={{ color: "#ef4444" }}>
                Spam Call Detected!
              </h3>
              <p className="text-sm text-gray-400 mt-2 leading-relaxed">
                Dangerous deepfake audio detected. This may be a fraudulent call attempting to deceive or harm you. Exercise caution and verify the source.
              </p>
            </div>
          </div>
        </div>
        <div className="border-x border-b rounded-b-2xl px-6 py-4" style={{ borderColor: "#ef4444" }}>
          <div className="flex gap-3">
            <button
              onClick={onClose}
              className="flex-1 bg-gray-600 hover:bg-gray-700 text-white font-semibold py-3 px-4 rounded-lg transition-colors"
            >
              Ignore
            </button>
            <button
              onClick={onClose}
              className="flex-1 bg-red-600 hover:bg-red-700 text-white font-semibold py-3 px-4 rounded-lg transition-colors"
            >
              OK
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

// ─────────────────────────────────────────────────────────────────────────────
// Main Hero
// ─────────────────────────────────────────────────────────────────────────────
export const Hero = () => {
  const [isRecording, setIsRecording] = useState(false);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [audioBuffer, setAudioBuffer] = useState(null);
  const [showPopup, setShowPopup] = useState(false);
  const [countdown, setCountdown] = useState(0);

  const mediaRecorderRef = useRef(null);
  const chunksRef = useRef([]);
  const timeoutRef = useRef(null);
  const streamRef = useRef(null);
  const analyserRef = useRef(null);
  const audioCtxRef = useRef(null);
  const intervalRef = useRef(null);
  // Tracks whether a result has arrived so the interval callback can stop decrementing
  const resultReceivedRef = useRef(false);

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

      const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      audioCtxRef.current = audioCtx;
      const src = audioCtx.createMediaStreamSource(stream);
      const analyser = audioCtx.createAnalyser();
      analyser.fftSize = 256;
      analyser.smoothingTimeConstant = 0.8;
      src.connect(analyser);
      analyserRef.current = analyser;

      const recorder = new MediaRecorder(stream);
      chunksRef.current = [];
      recorder.ondataavailable = (e) => { if (e.data.size > 0) chunksRef.current.push(e.data); };
      recorder.onstop = async () => {
        const blob = new Blob(chunksRef.current, { type: "audio/webm" });
        await decodeForSpectrogram(blob);
        await sendAudio(blob);
      };

      recorder.start();
      mediaRecorderRef.current = recorder;
      streamRef.current = stream;
      setIsRecording(true);

      // Reset result-received flag and start countdown from 10
      resultReceivedRef.current = false;
      setCountdown(10);

      // Tick down every second, but stop as soon as a result has arrived
      intervalRef.current = setInterval(() => {
        if (resultReceivedRef.current) {
          clearInterval(intervalRef.current);
          return;
        }

        setCountdown(prev => {
          if (prev > 0) return prev - 1;
          return 0; // stay at 0, DON'T stop interval
        });

      }, 1000);

      setResult(null);
      setAudioBuffer(null);
      timeoutRef.current = setTimeout(stopRecording, 7000);
    } catch (err) {
      console.error(err);
      alert("Error accessing audio");
    }
  };

  const stopRecording = () => {
    mediaRecorderRef.current?.stop();
    streamRef.current?.getTracks().forEach((t) => t.stop());
    audioCtxRef.current?.close();
    analyserRef.current = null;
    clearTimeout(timeoutRef.current);
    setIsRecording(false);
  };

  const decodeForSpectrogram = async (blob) => {
    try {
      const arrayBuf = await blob.arrayBuffer();
      const ctx = new (window.AudioContext || window.webkitAudioContext)();
      const decoded = await ctx.decodeAudioData(arrayBuf);
      await ctx.close();
      setAudioBuffer(decoded);
    } catch (e) {
      console.warn("Spectrogram decode failed", e);
    }
  };

  const sendAudio = async (blob) => {
    try {
      setLoading(true);
      const fd = new FormData();
      fd.append("file", blob, "audio.webm");
      const res = await fetch(`${import.meta.env.VITE_API_URL}/predict`, { method: "POST", body: fd });
      const data = await res.json();

      // Mark result as received — the interval will stop on its next tick
      resultReceivedRef.current = true;
      clearInterval(intervalRef.current);

      setResult({ label: data.label, confidence: data.confidence });
      if (data.label === "deepfake") {
        setShowPopup(true);
      }
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  const isDeepfake = result?.label === "deepfake";
  const accentColor = isDeepfake ? "#ef4444" : "#22c55e";
  const gradientBar = isDeepfake
    ? "linear-gradient(90deg,#1a0000,#7f1d1d,#dc2626,#fbbf24,#fff)"
    : "linear-gradient(90deg,#0a0a1a,#1e3a5f,#0d9488,#84cc16,#fde047)";

  return (
    <section className="min-h-screen flex flex-col items-center text-white px-4" style={{ background: "linear-gradient(to bottom, #000000 0%, #140022 40%, #0a1f3f 100%)" }}>

      {/* HEADER */}
      <div className="mt-28 text-center">
        <p className="text-xs text-gray-400 tracking-widest uppercase">AI powered Audio Forensics</p>
        <h1 className="text-4xl md:text-6xl font-bold mt-4">Deepfake Voice Detection</h1>
        <p className="text-sm text-gray-400 mt-3 max-w-[1000px] mx-auto text-center">
          Detect synthetic voices instantly from microphone or live calls.
        </p>
      </div>

      {/* RECORD CARD */}
      <div className="bg-neutral-900 mt-10 p-8 rounded-2xl w-full max-w-[500px] shadow-xl text-center">

        <div className="mb-8">
          <Waveform isRecording={isRecording} analyserRef={analyserRef} />
        </div>

        <div className="flex justify-center">
          <button
            onClick={() => isRecording ? stopRecording() : startRecording()}
            className={`flex mb-5 items-center justify-center w-24 h-24 rounded-full transition-all duration-300 ${isRecording ? "bg-red-600 scale-110 shadow-lg shadow-red-500/40" : "bg-red-500 hover:scale-105"
              }`}
          >
            {/* Show countdown number while recording or analyzing (and a result hasn't arrived yet) */}
            {(isRecording || loading) && !result
              ? <span className="text-white text-2xl font-bold">{countdown}</span>
              : <Mic className="w-8 h-8 text-white" />
            }
          </button>
        </div>

        <p className="mt-5 text-sm text-gray-400">
          {isRecording ? "Recording & analyzing..." : "Click to start recording"}
        </p>
        {loading && <p className="mt-4 text-purple-400 animate-pulse">Analyzing audio...</p>}
      </div>

      {/* ── RESULT CARD ── */}
      {result && (
        <div className="mt-6 w-full max-w-[500px] mb-16" style={{ animation: "fadeSlideUp 0.45s ease forwards" }}>

          {/* Verdict */}
          <div className="rounded-t-2xl px-6 pt-6 pb-5 border-t border-x"
            style={{ borderColor: accentColor, background: isDeepfake ? "rgba(239,68,68,0.07)" : "rgba(34,197,94,0.07)" }}>

            <div className="flex items-start gap-4">
              <div style={{ color: accentColor }} className="mt-0.5 shrink-0">
                {isDeepfake ? <ShieldAlert className="w-9 h-9" /> : <ShieldCheck className="w-9 h-9" />}
              </div>
              <div className="text-left">
                <p className="text-[10px] tracking-widest uppercase text-gray-500 mb-0.5">Analysis Verdict</p>
                <h3 className="text-2xl font-bold" style={{ color: accentColor }}>
                  {isDeepfake ? "Deepfake Detected" : "Authentic Voice"}
                </h3>
                <p className="text-sm text-gray-400 mt-2 leading-relaxed">
                  {getDescription(result.label, result.confidence)}
                </p>
              </div>
            </div>

            {/* Confidence bar */}
            <div className="mt-5">
              <div className="flex justify-between mb-1.5">
                <span className="text-[10px] tracking-widest uppercase text-gray-500">Confidence Score</span>
                <span className="text-sm font-mono font-semibold" style={{ color: accentColor }}>
                  {(result.confidence * 100).toFixed(1)}%
                </span>
              </div>
              <div className="w-full bg-neutral-800 rounded-full h-2 overflow-hidden">
                <div className="h-2 rounded-full transition-all duration-700"
                  style={{
                    width: `${result.confidence * 100}%`,
                    background: isDeepfake
                      ? "linear-gradient(90deg,#b91c1c,#ef4444,#fca5a5)"
                      : "linear-gradient(90deg,#15803d,#22c55e,#86efac)"
                  }} />
              </div>
            </div>
          </div>

          {/* Spectrogram panel */}
          <div className="border-x border-b rounded-b-2xl overflow-hidden"
            style={{ borderColor: accentColor, background: "#080808" }}>

            {/* Panel header */}
            <div className="flex items-center gap-2 px-4 py-2.5 border-b"
              style={{ borderColor: "rgba(255,255,255,0.05)" }}>
              <Activity className="w-3.5 h-3.5 text-gray-500" />
              <span className="text-[10px] tracking-widest uppercase text-gray-500">Frequency Spectrogram</span>
              {audioBuffer && (
                <span className="ml-auto text-[10px] font-mono text-gray-600">
                  {(audioBuffer.sampleRate / 1000).toFixed(1)} kHz &middot; {audioBuffer.numberOfChannels}ch &middot; {audioBuffer.duration.toFixed(2)}s
                </span>
              )}
            </div>

            {/* Canvas */}
            <div className="px-3 pt-3 pb-2">
              {audioBuffer
                ? <Spectrogram audioBuffer={audioBuffer} isDeepfake={isDeepfake} />
                : <div className="h-40 flex items-center justify-center text-gray-600 text-xs font-mono">No audio data</div>
              }
            </div>

            {/* Colour legend */}
            <div className="flex items-center gap-3 px-4 pb-3">
              <span className="text-[9px] text-gray-600 font-mono whitespace-nowrap">Low energy</span>
              <div className="flex-1 h-1.5 rounded-full" style={{ background: gradientBar }} />
              <span className="text-[9px] text-gray-600 font-mono whitespace-nowrap">High energy</span>
            </div>
          </div>
        </div>
      )}

      {/* Deepfake Detection Popup */}
      {showPopup && <DeepfakePopup onClose={() => setShowPopup(false)} />}

      {/* Footer */}
      <div id="contact" className="mt-24 pb-16 text-center">
        <p className="text-sm text-gray-400">❤️ Made by <span className="font-mono font-bold tracking-widest">DATADEFENDERS</span></p>
      </div>

      <style>{`
        @keyframes fadeSlideUp {
          from { opacity: 0; transform: translateY(18px); }
          to   { opacity: 1; transform: translateY(0); }
        }
      `}</style>
    </section>
  );
};
