import { useEffect, useMemo, useRef, useState } from 'react';
import {
  Code2,
  Sparkles,
  Target,
  Activity,
  Layers,
  Info,
  User,
  Camera,
  CheckCircle2,
  XCircle,
  Fingerprint,
} from 'lucide-react';
import {
  generateOlivettiFaces,
  GaussianNB,
  accuracy,
  confusionMatrix,
} from './data/olivettiFaces';

const CODE_SNIPPET = `import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

d = fetch_olivetti_faces()
X1, X2, y1, y2 = train_test_split(
    d.data, d.target, test_size=0.2, random_state=42
)

m = GaussianNB().fit(X1, y1)
p = m.predict(X2)
print("Accuracy:", accuracy_score(y2, p))

fig, ax = plt.subplots(2, 5, figsize=(10, 5))
for i, a in enumerate(ax.flat[:10]):
    a.imshow(X2[i].reshape(64, 64), cmap='gray')
    a.set_title(p[i])
    a.axis('off')
plt.show()`;

// Face Image Component using Canvas
function FaceImage({
  pixels,
  size = 128,
  label,
  predicted,
  isCorrect,
  showLabel = true,
  onClick,
}: {
  pixels: number[];
  size?: number;
  label?: number;
  predicted?: number;
  isCorrect?: boolean;
  showLabel?: boolean;
  onClick?: () => void;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const gridSize = 64;
    const imageData = ctx.createImageData(gridSize, gridSize);

    for (let i = 0; i < pixels.length; i++) {
      const val = Math.floor(pixels[i] * 255);
      imageData.data[i * 4] = val;
      imageData.data[i * 4 + 1] = val;
      imageData.data[i * 4 + 2] = val;
      imageData.data[i * 4 + 3] = 255;
    }

    // Draw to offscreen canvas then scale
    const offscreen = document.createElement('canvas');
    offscreen.width = gridSize;
    offscreen.height = gridSize;
    const offCtx = offscreen.getContext('2d');
    if (offCtx) {
      offCtx.putImageData(imageData, 0, 0);
      ctx.imageSmoothingEnabled = true;
      ctx.drawImage(offscreen, 0, 0, size, size);
    }
  }, [pixels, size]);

  return (
    <div
      className={`group relative overflow-hidden rounded-lg border transition ${
        isCorrect === undefined
          ? 'border-white/10'
          : isCorrect
          ? 'border-emerald-500/40'
          : 'border-rose-500/40'
      } ${onClick ? 'cursor-pointer hover:scale-105' : ''}`}
      onClick={onClick}
      style={{ width: size, height: size }}
    >
      <canvas
        ref={canvasRef}
        width={size}
        height={size}
        className="block"
      />
      {showLabel && (label !== undefined || predicted !== undefined) && (
        <div className="absolute inset-x-0 bottom-0 bg-gradient-to-t from-black/80 via-black/40 to-transparent p-1.5">
          <div className="flex items-center justify-between text-[10px] font-mono">
            {label !== undefined && (
              <span className="rounded bg-amber-500/20 px-1 text-amber-300">
                {label}
              </span>
            )}
            {predicted !== undefined && (
              <span
                className={`rounded px-1 ${
                  isCorrect
                    ? 'bg-emerald-500/20 text-emerald-300'
                    : 'bg-rose-500/20 text-rose-300'
                }`}
              >
                →{predicted}
              </span>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

// Probability bar component
function ProbabilityBar({
  proba,
  actualClass,
}: {
  proba: number[];
  actualClass: number;
}) {
  const topK = proba
    .map((p, i) => ({ class: i, prob: p }))
    .sort((a, b) => b.prob - a.prob)
    .slice(0, 5);
  const maxProb = Math.max(...proba);

  return (
    <div className="space-y-1.5">
      {topK.map((item) => (
        <div key={item.class} className="flex items-center gap-2 text-xs">
          <span
            className={`w-8 text-right font-mono ${
              item.class === actualClass ? 'text-emerald-400' : 'text-white/60'
            }`}
          >
            {item.class}
          </span>
          <div className="flex-1">
            <div className="h-1.5 overflow-hidden rounded-full bg-white/5">
              <div
                className={`h-full rounded-full transition-all ${
                  item.class === actualClass
                    ? 'bg-gradient-to-r from-emerald-500 to-teal-400'
                    : item.prob === maxProb
                    ? 'bg-gradient-to-r from-amber-500 to-orange-400'
                    : 'bg-white/30'
                }`}
                style={{ width: `${item.prob * 100}%` }}
              />
            </div>
          </div>
          <span className="w-12 text-right font-mono text-white/50">
            {(item.prob * 100).toFixed(1)}%
          </span>
        </div>
      ))}
    </div>
  );
}

export default function App() {
  const [showCode, setShowCode] = useState(false);
  const [selectedIdx, setSelectedIdx] = useState(0);
  const numSubjects = 10;

  // Generate faces
  const allFaces = useMemo(() => generateOlivettiFaces(numSubjects, 10), [numSubjects]);

  // Train/test split (80/20)
  const { train, test } = useMemo(() => {
    const split = Math.floor(allFaces.length * 0.8);
    return {
      train: allFaces.slice(0, split),
      test: allFaces.slice(split),
    };
  }, [allFaces]);

  // Train model
  const model = useMemo(() => {
    const m = new GaussianNB();
    m.fit(
      train.map((f) => f.pixels),
      train.map((f) => f.label)
    );
    return m;
  }, [train]);

  // Predict
  const predictions = useMemo(
    () => model.predict(test.map((f) => f.pixels)),
    [model, test]
  );

  const probabilities = useMemo(
    () => model.predictProba(test.map((f) => f.pixels)),
    [model, test]
  );

  // Stats
  const acc = useMemo(
    () => accuracy(test.map((f) => f.label), predictions),
    [test, predictions]
  );

  const confMatrix = useMemo(
    () => confusionMatrix(test.map((f) => f.label), predictions, numSubjects),
    [test, predictions, numSubjects]
  );

  // Compute per-class accuracy
  const classAccuracies = useMemo(() => {
    return Array.from({ length: numSubjects }, (_, c) => {
      const classIndices = test
        .map((f, i) => ({ label: f.label, idx: i }))
        .filter((item) => item.label === c);
      if (classIndices.length === 0) return 0;
      const correct = classIndices.filter(
        (item) => predictions[item.idx] === c
      ).length;
      return correct / classIndices.length;
    });
  }, [test, predictions, numSubjects]);

  // First 10 test samples (like the matplotlib plot)
  const first10Test = test.slice(0, 10);
  const first10Predictions = predictions.slice(0, 10);

  // Current selected sample
  const currentTest = test[selectedIdx];
  const currentPrediction = predictions[selectedIdx];
  const currentProba = probabilities[selectedIdx];
  const isCorrect = currentTest.label === currentPrediction;

  // Mean faces per class (averaged pixel values)
  const meanFaces = useMemo(() => {
    const means: number[][] = [];
    const size = 64 * 64;
    for (let c = 0; c < numSubjects; c++) {
      const classFaces = train.filter((f) => f.label === c);
      const meanPixels = Array(size).fill(0);
      classFaces.forEach((f) => {
        for (let i = 0; i < size; i++) meanPixels[i] += f.pixels[i];
      });
      for (let i = 0; i < size; i++) meanPixels[i] /= classFaces.length;
      means.push(meanPixels);
    }
    return means;
  }, [train, numSubjects]);

  return (
    <div className="relative min-h-screen overflow-hidden bg-[#1a0f08] text-white">
      {/* Animated gradient background orbs */}
      <div className="pointer-events-none fixed inset-0 overflow-hidden">
        <div className="absolute -top-40 -left-40 h-96 w-96 animate-pulse rounded-full bg-amber-600/15 blur-3xl" />
        <div
          className="absolute top-1/3 -right-40 h-96 w-96 animate-pulse rounded-full bg-orange-600/15 blur-3xl"
          style={{ animationDelay: '1s' }}
        />
        <div
          className="absolute -bottom-40 left-1/3 h-96 w-96 animate-pulse rounded-full bg-rose-600/15 blur-3xl"
          style={{ animationDelay: '2s' }}
        />
      </div>

      {/* Grid pattern */}
      <div
        className="pointer-events-none fixed inset-0 opacity-[0.03]"
        style={{
          backgroundImage:
            'linear-gradient(to right, white 1px, transparent 1px), linear-gradient(to bottom, white 1px, transparent 1px)',
          backgroundSize: '60px 60px',
        }}
      />

      <div className="relative">
        {/* Header */}
        <header className="border-b border-white/5 bg-black/40">
          <div className="mx-auto flex max-w-7xl items-center gap-4 px-6 py-4">
            <div className="relative flex h-11 w-11 items-center justify-center rounded-xl bg-gradient-to-br from-amber-500 via-orange-500 to-rose-500">
              <Fingerprint
                className="h-5 w-5 text-white"
                strokeWidth={2.5}
              />
            </div>
            <div>
              <h1 className="text-lg font-bold tracking-tight">
                Face Recognition
              </h1>
              <p className="text-xs text-white/50">
                Gaussian Naive Bayes · Olivetti Faces
              </p>
            </div>
            <button
              onClick={() => setShowCode((s) => !s)}
              className="ml-auto flex items-center gap-2 rounded-lg border border-white/10 bg-white/5 px-3 py-1.5 text-xs font-medium text-white/80 transition hover:border-amber-500/50 hover:bg-amber-500/10 hover:text-white"
            >
              <Code2 className="h-3.5 w-3.5" />
              {showCode ? 'Hide Code' : 'View Code'}
            </button>
          </div>
        </header>

        <main className="mx-auto max-w-7xl px-6 py-8">
          {/* Hero */}
          <div className="group relative mb-8 overflow-hidden rounded-2xl border border-white/10 bg-gradient-to-br from-amber-900/30 via-orange-900/20 to-rose-900/30 p-8">
            <div className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-amber-400 to-transparent" />
            <div className="absolute inset-x-0 bottom-0 h-px bg-gradient-to-r from-transparent via-rose-400 to-transparent" />

            <div className="flex flex-col gap-6 lg:flex-row lg:items-center lg:justify-between">
              <div className="max-w-2xl">
                <div className="mb-3 inline-flex items-center gap-1.5 rounded-full border border-amber-400/30 bg-amber-500/10 px-3 py-1 text-[11px] font-medium text-amber-300">
                  <Camera className="h-3 w-3" /> Image Classification
                </div>
                <h2 className="text-3xl font-bold leading-tight md:text-4xl">
                  Recognizing faces with{' '}
                  <span className="bg-gradient-to-r from-amber-400 via-orange-400 to-rose-400 bg-clip-text text-transparent">
                    Gaussian Naive Bayes
                  </span>
                </h2>
                <p className="mt-3 text-sm leading-relaxed text-white/70">
                  Each face is represented as a{' '}
                  <span className="font-mono text-amber-300">64×64</span>{' '}
                  grayscale image. The model assumes pixel values follow a
                  Gaussian distribution per class and uses Bayes' theorem to
                  classify.
                </p>
              </div>
              <div className="flex gap-3 lg:flex-col">
                <div className="flex-1 rounded-xl border border-white/10 bg-black/40 px-5 py-4 lg:min-w-[140px]">
                  <p className="bg-gradient-to-r from-amber-400 to-orange-400 bg-clip-text text-3xl font-bold text-transparent">
                    {allFaces.length}
                  </p>
                  <p className="text-xs text-white/50">Faces</p>
                </div>
                <div className="flex-1 rounded-xl border border-white/10 bg-black/40 px-5 py-4 lg:min-w-[140px]">
                  <p className="bg-gradient-to-r from-orange-400 to-rose-400 bg-clip-text text-3xl font-bold text-transparent">
                    {numSubjects}
                  </p>
                  <p className="text-xs text-white/50">Subjects</p>
                </div>
                <div className="flex-1 rounded-xl border border-white/10 bg-black/40 px-5 py-4 lg:min-w-[140px]">
                  <p className="bg-gradient-to-r from-rose-400 to-pink-400 bg-clip-text text-3xl font-bold text-transparent">
                    {(acc * 100).toFixed(0)}%
                  </p>
                  <p className="text-xs text-white/50">Accuracy</p>
                </div>
              </div>
            </div>
          </div>

          {/* Sample Gallery (2×5 grid like matplotlib) */}
          <div className="rounded-2xl border border-white/10 bg-black/40 p-6">
            <div className="mb-4 flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-gradient-to-br from-amber-500/20 to-rose-500/20 ring-1 ring-amber-400/30">
                  <User className="h-4 w-4 text-amber-300" />
                </div>
                <div>
                  <h3 className="text-sm font-semibold">Test Sample Gallery</h3>
                  <p className="text-xs text-white/50">
                    First 10 test images with predictions (click to inspect)
                  </p>
                </div>
              </div>
              <div className="flex items-center gap-3 text-xs">
                <div className="flex items-center gap-1.5">
                  <div className="h-2 w-2 rounded-full bg-emerald-500" />
                  <span className="text-white/70">Correct</span>
                </div>
                <div className="flex items-center gap-1.5">
                  <div className="h-2 w-2 rounded-full bg-rose-500" />
                  <span className="text-white/70">Wrong</span>
                </div>
              </div>
            </div>

            <div className="grid grid-cols-5 gap-2 sm:grid-cols-5 md:gap-3">
              {first10Test.map((face, i) => (
                <FaceImage
                  key={i}
                  pixels={face.pixels}
                  size={128}
                  label={face.label}
                  predicted={first10Predictions[i]}
                  isCorrect={face.label === first10Predictions[i]}
                />
              ))}
            </div>

            <div className="mt-4 grid grid-cols-10 gap-1 text-center text-[9px] font-mono">
              {first10Predictions.map((p, i) => (
                <div
                  key={i}
                  className={`rounded px-1 py-0.5 ${
                    first10Test[i].label === p
                      ? 'bg-emerald-500/10 text-emerald-300'
                      : 'bg-rose-500/10 text-rose-300'
                  }`}
                >
                  {p}
                </div>
              ))}
            </div>
          </div>

          {/* Main Grid: Explorer + Stats */}
          <div className="mt-6 grid grid-cols-1 gap-6 lg:grid-cols-3">
            {/* Sample Explorer */}
            <div className="lg:col-span-2">
              <div className="rounded-2xl border border-white/10 bg-black/40 p-6">
                <div className="mb-4 flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-gradient-to-br from-orange-500/20 to-pink-500/20 ring-1 ring-orange-400/30">
                      <Target className="h-4 w-4 text-orange-300" />
                    </div>
                    <div>
                      <h3 className="text-sm font-semibold">Sample Explorer</h3>
                      <p className="text-xs text-white/50">
                        Inspect predictions with probabilities
                      </p>
                    </div>
                  </div>
                  <div className="flex items-center gap-3">
                    <button
                      onClick={() => setSelectedIdx((s) => Math.max(0, s - 1))}
                      disabled={selectedIdx === 0}
                      className="rounded-md border border-white/10 bg-white/5 px-3 py-1 text-xs text-white/70 transition hover:bg-white/10 disabled:opacity-30"
                    >
                      ← Prev
                    </button>
                    <span className="font-mono text-xs text-white/50">
                      {selectedIdx + 1} / {test.length}
                    </span>
                    <button
                      onClick={() =>
                        setSelectedIdx((s) => Math.min(test.length - 1, s + 1))
                      }
                      disabled={selectedIdx === test.length - 1}
                      className="rounded-md border border-white/10 bg-white/5 px-3 py-1 text-xs text-white/70 transition hover:bg-white/10 disabled:opacity-30"
                    >
                      Next →
                    </button>
                  </div>
                </div>

                <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
                  {/* Face Image */}
                  <div className="flex flex-col items-center gap-2">
                    <FaceImage
                      pixels={currentTest.pixels}
                      size={192}
                      showLabel={false}
                    />
                    <div className="text-center text-xs">
                      <p className="text-white/40">Test Sample #{selectedIdx}</p>
                    </div>
                  </div>

                  {/* Prediction Card */}
                  <div className="space-y-3">
                    <div
                      className="rounded-xl border p-4"
                      style={{
                        borderColor: isCorrect
                          ? 'rgba(16, 185, 129, 0.5)'
                          : 'rgba(244, 63, 94, 0.5)',
                        backgroundColor: isCorrect
                          ? 'rgba(16, 185, 129, 0.1)'
                          : 'rgba(244, 63, 94, 0.1)',
                      }}
                    >
                      <p className="text-[10px] uppercase tracking-wider text-white/40">
                        Result
                      </p>
                      <div className="mt-2 flex items-center gap-2">
                        {isCorrect ? (
                          <>
                            <CheckCircle2 className="h-5 w-5 text-emerald-400" />
                            <span className="text-lg font-bold text-emerald-300">
                              Correct ✓
                            </span>
                          </>
                        ) : (
                          <>
                            <XCircle className="h-5 w-5 text-rose-400" />
                            <span className="text-lg font-bold text-rose-300">
                              Incorrect ✗
                            </span>
                          </>
                        )}
                      </div>
                    </div>

                    <div className="rounded-xl border border-amber-500/30 bg-amber-500/5 p-4">
                      <p className="text-[10px] uppercase tracking-wider text-amber-400/70">
                        Actual Subject
                      </p>
                      <p className="mt-1 font-mono text-2xl font-bold text-amber-300">
                        {currentTest.label}
                      </p>
                    </div>

                    <div
                      className="rounded-xl border p-4"
                      style={{
                        borderColor: `${
                          isCorrect ? '#10b981' : '#f43f5e'
                        }50`,
                        backgroundColor: `${
                          isCorrect ? '#10b981' : '#f43f5e'
                        }10`,
                      }}
                    >
                      <p className="text-[10px] uppercase tracking-wider text-white/40">
                        Predicted Subject
                      </p>
                      <p
                        className="mt-1 font-mono text-2xl font-bold"
                        style={{
                          color: isCorrect ? '#10b981' : '#f43f5e',
                        }}
                      >
                        {currentPrediction}
                      </p>
                    </div>
                  </div>

                  {/* Probabilities */}
                  <div className="rounded-xl border border-white/10 bg-white/[0.02] p-4">
                    <p className="mb-3 text-[10px] font-semibold uppercase tracking-wider text-white/60">
                      Class Probabilities (Top 5)
                    </p>
                    <ProbabilityBar
                      proba={currentProba}
                      actualClass={currentTest.label}
                    />
                    <div className="mt-3 border-t border-white/5 pt-3">
                      <p className="text-[10px] uppercase tracking-wider text-white/40">
                        Confidence
                      </p>
                      <p className="mt-1 font-mono text-lg font-bold text-amber-300">
                        {(Math.max(...currentProba) * 100).toFixed(1)}%
                      </p>
                    </div>
                  </div>
                </div>

                <div className="mt-4">
                  <input
                    type="range"
                    min={0}
                    max={test.length - 1}
                    value={selectedIdx}
                    onChange={(e) =>
                      setSelectedIdx(parseInt(e.target.value))
                    }
                    className="h-2 w-full cursor-pointer appearance-none rounded-full bg-white/10 accent-amber-500"
                  />
                </div>
              </div>
            </div>

            {/* Stats */}
            <div className="space-y-6">
              {/* Accuracy */}
              <div className="rounded-2xl border border-white/10 bg-black/40 p-5">
                <div className="mb-3 flex items-center gap-2">
                  <Activity className="h-4 w-4 text-amber-400" />
                  <h3 className="text-[11px] font-semibold uppercase tracking-wider text-white/60">
                    Model Performance
                  </h3>
                </div>
                <div className="text-center">
                  <p className="bg-gradient-to-r from-amber-400 via-orange-400 to-rose-400 bg-clip-text font-mono text-5xl font-bold text-transparent">
                    {(acc * 100).toFixed(1)}%
                  </p>
                  <p className="mt-1 text-xs text-white/50">Overall Accuracy</p>
                </div>
                <div className="mt-3 grid grid-cols-2 gap-2 text-xs">
                  <div className="rounded-lg border border-emerald-500/20 bg-emerald-500/5 p-2 text-center">
                    <p className="font-mono font-bold text-emerald-300">
                      {predictions.filter(
                        (p, i) => p === test[i].label
                      ).length}
                    </p>
                    <p className="text-[10px] text-white/50">Correct</p>
                  </div>
                  <div className="rounded-lg border border-rose-500/20 bg-rose-500/5 p-2 text-center">
                    <p className="font-mono font-bold text-rose-300">
                      {predictions.filter(
                        (p, i) => p !== test[i].label
                      ).length}
                    </p>
                    <p className="text-[10px] text-white/50">Wrong</p>
                  </div>
                </div>
              </div>

              {/* Per-class accuracy */}
              <div className="rounded-2xl border border-white/10 bg-black/40 p-5">
                <div className="mb-3 flex items-center gap-2">
                  <Layers className="h-4 w-4 text-orange-400" />
                  <h3 className="text-[11px] font-semibold uppercase tracking-wider text-white/60">
                    Per-Class Accuracy
                  </h3>
                </div>
                <div className="space-y-1.5">
                  {classAccuracies.map((acc, c) => (
                    <div key={c} className="flex items-center gap-2 text-xs">
                      <span className="w-5 font-mono text-white/60">{c}</span>
                      <div className="flex-1">
                        <div className="h-1.5 overflow-hidden rounded-full bg-white/5">
                          <div
                            className="h-full rounded-full bg-gradient-to-r from-amber-500 via-orange-500 to-rose-500"
                            style={{ width: `${acc * 100}%` }}
                          />
                        </div>
                      </div>
                      <span className="w-10 text-right font-mono text-white/50">
                        {(acc * 100).toFixed(0)}%
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* Mean Faces */}
          <div className="mt-6 rounded-2xl border border-white/10 bg-black/40 p-6">
            <div className="mb-4 flex items-center gap-3">
              <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-gradient-to-br from-rose-500/20 to-pink-500/20 ring-1 ring-rose-400/30">
                <Sparkles className="h-4 w-4 text-rose-300" />
              </div>
              <div>
                <h3 className="text-sm font-semibold">
                  Mean Faces per Subject
                </h3>
                <p className="text-xs text-white/50">
                  Average appearance learned by the model
                </p>
              </div>
            </div>

            <div
              className="grid gap-2"
              style={{ gridTemplateColumns: `repeat(${numSubjects}, 1fr)` }}
            >
              {meanFaces.map((pixels, c) => (
                <div key={c} className="flex flex-col items-center gap-1">
                  <FaceImage pixels={pixels} size={80} showLabel={false} />
                  <span className="font-mono text-[10px] text-amber-300">
                    {c}
                  </span>
                </div>
              ))}
            </div>
          </div>

          {/* Confusion Matrix */}
          <div className="mt-6 rounded-2xl border border-white/10 bg-black/40 p-6">
            <div className="mb-4 flex items-center gap-3">
              <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-gradient-to-br from-purple-500/20 to-pink-500/20 ring-1 ring-purple-400/30">
                <Target className="h-4 w-4 text-purple-300" />
              </div>
              <div>
                <h3 className="text-sm font-semibold">Confusion Matrix</h3>
                <p className="text-xs text-white/50">
                  Actual vs Predicted ({numSubjects}×{numSubjects})
                </p>
              </div>
            </div>

            <div className="overflow-x-auto">
              <div
                className="inline-grid gap-0.5"
                style={{
                  gridTemplateColumns: `auto repeat(${numSubjects}, 1fr)`,
                }}
              >
                {/* Header */}
                <div />
                {Array.from({ length: numSubjects }, (_, i) => (
                  <div
                    key={`h-${i}`}
                    className="flex h-8 items-center justify-center text-[10px] font-mono text-white/50"
                  >
                    {i}
                  </div>
                ))}

                {/* Rows */}
                {confMatrix.map((row, r) => (
                  <>
                    <div
                      key={`r-${r}`}
                      className="flex h-8 items-center pr-2 text-[10px] font-mono text-white/50"
                    >
                      {r}
                    </div>
                    {row.map((val, c) => {
                      const max = Math.max(...row, 1);
                      const isDiag = r === c;
                      const intensity = val / max;
                      return (
                        <div
                          key={`${r}-${c}`}
                          className={`flex h-8 w-8 items-center justify-center text-[10px] font-mono ${
                            val === 0
                              ? 'text-white/10'
                              : isDiag
                              ? 'text-emerald-300'
                              : 'text-rose-300'
                          }`}
                          style={{
                            backgroundColor: isDiag
                              ? `rgba(16, 185, 129, ${intensity * 0.4})`
                              : `rgba(244, 63, 94, ${intensity * 0.4})`,
                          }}
                          title={`Actual ${r} → Predicted ${c}: ${val}`}
                        >
                          {val > 0 ? val : ''}
                        </div>
                      );
                    })}
                  </>
                ))}
              </div>
            </div>

            <div className="mt-4 flex items-center gap-4 text-xs">
              <div className="flex items-center gap-2">
                <div className="h-3 w-3 rounded bg-emerald-500/40" />
                <span className="text-white/60">Diagonal = Correct</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="h-3 w-3 rounded bg-rose-500/40" />
                <span className="text-white/60">Off-diagonal = Errors</span>
              </div>
            </div>
          </div>

          {/* Code Preview */}
          {showCode && (
            <div className="mt-6 rounded-2xl border border-white/10 bg-black/60">
              <div className="flex items-center gap-3 border-b border-white/5 px-6 py-4">
                <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-gradient-to-br from-amber-500/20 to-orange-500/20 ring-1 ring-amber-400/30">
                  <Code2 className="h-4 w-4 text-amber-300" />
                </div>
                <div>
                  <h3 className="text-sm font-semibold">Python Source</h3>
                  <p className="text-xs text-white/50">
                    sklearn GaussianNB on Olivetti faces
                  </p>
                </div>
              </div>
              <div className="overflow-x-auto p-4">
                <pre className="font-mono text-sm leading-relaxed">
                  <code>
                    {CODE_SNIPPET.split('\n').map((line, i) => (
                      <div key={i} className="flex">
                        <span className="mr-4 inline-block w-6 select-none text-right text-white/20">
                          {i + 1}
                        </span>
                        <span className="text-white/80">{line}</span>
                      </div>
                    ))}
                  </code>
                </pre>
              </div>
            </div>
          )}

          {/* Bottom info cards */}
          <div className="mt-6 grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <div className="group relative overflow-hidden rounded-xl border border-white/10 bg-black/40 p-5 transition hover:border-amber-500/30">
              <div className="absolute -right-4 -top-4 h-16 w-16 rounded-full bg-amber-500/10 transition group-hover:bg-amber-500/20" />
              <div className="relative">
                <div className="mb-2 flex h-8 w-8 items-center justify-center rounded-lg bg-amber-500/20 ring-1 ring-amber-400/30">
                  <Sparkles className="h-4 w-4 text-amber-300" />
                </div>
                <h4 className="text-[10px] font-semibold uppercase tracking-wider text-white/40">
                  Algorithm
                </h4>
                <p className="mt-1 font-semibold text-white">
                  Gaussian Naive Bayes
                </p>
                <p className="mt-1 text-xs text-white/50">
                  Generative probabilistic model
                </p>
              </div>
            </div>

            <div className="group relative overflow-hidden rounded-xl border border-white/10 bg-black/40 p-5 transition hover:border-orange-500/30">
              <div className="absolute -right-4 -top-4 h-16 w-16 rounded-full bg-orange-500/10 transition group-hover:bg-orange-500/20" />
              <div className="relative">
                <div className="mb-2 flex h-8 w-8 items-center justify-center rounded-lg bg-orange-500/20 ring-1 ring-orange-400/30">
                  <Activity className="h-4 w-4 text-orange-300" />
                </div>
                <h4 className="text-[10px] font-semibold uppercase tracking-wider text-white/40">
                  Assumption
                </h4>
                <p className="mt-1 font-semibold text-white">
                  Feature Independence
                </p>
                <p className="mt-1 text-xs text-white/50">
                  Pixels are conditionally independent
                </p>
              </div>
            </div>

            <div className="group relative overflow-hidden rounded-xl border border-white/10 bg-black/40 p-5 transition hover:border-rose-500/30">
              <div className="absolute -right-4 -top-4 h-16 w-16 rounded-full bg-rose-500/10 transition group-hover:bg-rose-500/20" />
              <div className="relative">
                <div className="mb-2 flex h-8 w-8 items-center justify-center rounded-lg bg-rose-500/20 ring-1 ring-rose-400/30">
                  <Layers className="h-4 w-4 text-rose-300" />
                </div>
                <h4 className="text-[10px] font-semibold uppercase tracking-wider text-white/40">
                  Data
                </h4>
                <p className="mt-1 font-semibold text-white">
                  Olivetti Faces
                </p>
                <p className="mt-1 text-xs text-white/50">64×64 grayscale</p>
              </div>
            </div>

            <div className="group relative overflow-hidden rounded-xl border border-white/10 bg-black/40 p-5 transition hover:border-purple-500/30">
              <div className="absolute -right-4 -top-4 h-16 w-16 rounded-full bg-purple-500/10 transition group-hover:bg-purple-500/20" />
              <div className="relative">
                <div className="mb-2 flex h-8 w-8 items-center justify-center rounded-lg bg-purple-500/20 ring-1 ring-purple-400/30">
                  <Info className="h-4 w-4 text-purple-300" />
                </div>
                <h4 className="text-[10px] font-semibold uppercase tracking-wider text-white/40">
                  Features
                </h4>
                <p className="mt-1 font-mono font-semibold text-white">
                  4096
                </p>
                <p className="mt-1 text-xs text-white/50">
                  Pixel intensities
                </p>
              </div>
            </div>
          </div>

          <footer className="mt-8 pb-8 text-center text-xs text-white/40">
            Built with React · Tailwind CSS · TypeScript
          </footer>
        </main>
      </div>
    </div>
  );
}
