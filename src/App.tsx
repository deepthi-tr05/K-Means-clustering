import { useMemo, useState } from 'react';
import {
  Code2,
  Sparkles,
  Target,
  Activity,
  Info,
  Hexagon,
  Aperture,
} from 'lucide-react';
import {
  generateBreastCancerData,
  standardScaler,
  kMeans,
  silhouetteScore,
  clusterStats,
} from './data/kmeansData';

const CODE_SNIPPET = `import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

d = load_breast_cancer()
X = pd.DataFrame(
    d.data, columns=d.feature_names
)[['mean radius', 'mean texture']]

X = StandardScaler().fit_transform(X)
k = KMeans(n_clusters=4, random_state=0)
y = k.fit_predict(X)

for i in range(4):
    plt.scatter(
        X[y == i, 0], X[y == i, 1],
        label=f'C{i}'
    )
plt.legend()
plt.show()`;

// Cluster colors
const CLUSTER_COLORS = [
  '#a855f7', // violet
  '#ec4899', // pink
  '#06b6d4', // cyan
  '#f59e0b', // amber
  '#10b981', // emerald
  '#f43f5e', // rose
  '#3b82f6', // blue
  '#f97316', // orange
  '#14b8a6', // teal
  '#84cc16', // lime
];

export default function App() {
  const [k, setK] = useState(4);
  const [showCode, setShowCode] = useState(false);

  // Generate and scale data
  const data = useMemo(() => generateBreastCancerData(), []);
  const { scaled } = useMemo(
    () => standardScaler(data),
    [data]
  );

  // Run K-Means for current k
  const result = useMemo(() => kMeans(scaled, k, 100, 0), [scaled, k]);

  // Elbow method: compute inertia for k=1 to 10
  const elbowData = useMemo(() => {
    return Array.from({ length: 10 }, (_, i) => {
      const currentK = i + 1;
      const res = kMeans(scaled, currentK, 100, 0);
      return { k: currentK, inertia: res.inertia };
    });
  }, [scaled]);

  // Silhouette scores for k=2 to 10
  const silhouetteData = useMemo(() => {
    return Array.from({ length: 9 }, (_, i) => {
      const currentK = i + 2;
      const res = kMeans(scaled, currentK, 100, 0);
      const score = silhouetteScore(scaled, res.labels, currentK);
      return { k: currentK, score };
    });
  }, [scaled]);

  // Cluster stats
  const stats = useMemo(
    () => clusterStats(scaled, result.labels, k),
    [scaled, result.labels, k]
  );

  // Current silhouette score
  const currentSilhouette = useMemo(
    () => silhouetteScore(scaled, result.labels, k),
    [scaled, result.labels, k]
  );

  // Plot dimensions
  const plotWidth = 700;
  const plotHeight = 400;
  const padding = { top: 20, right: 20, bottom: 40, left: 50 };
  const innerWidth = plotWidth - padding.left - padding.right;
  const innerHeight = plotHeight - padding.top - padding.bottom;

  // Compute scales based on data range
  const xMin = Math.min(...scaled.map((d) => d.x)) - 0.5;
  const xMax = Math.max(...scaled.map((d) => d.x)) + 0.5;
  const yMin = Math.min(...scaled.map((d) => d.y)) - 0.5;
  const yMax = Math.max(...scaled.map((d) => d.y)) + 0.5;

  const xScale = (x: number) =>
    padding.left + ((x - xMin) / (xMax - xMin)) * innerWidth;
  const yScale = (y: number) =>
    padding.top + innerHeight - ((y - yMin) / (yMax - yMin)) * innerHeight;

  // Elbow plot dimensions
  const elbowWidth = 280;
  const elbowHeight = 120;
  const elbowPadding = { top: 10, right: 10, bottom: 25, left: 40 };
  const elbowInnerWidth = elbowWidth - elbowPadding.left - elbowPadding.right;
  const elbowInnerHeight =
    elbowHeight - elbowPadding.top - elbowPadding.bottom;

  const elbowXScale = (k: number) =>
    elbowPadding.left + ((k - 1) / 9) * elbowInnerWidth;
  const maxInertia = Math.max(...elbowData.map((d) => d.inertia));
  const elbowYScale = (inertia: number) =>
    elbowPadding.top +
    elbowInnerHeight -
    (inertia / maxInertia) * elbowInnerHeight;

  return (
    <div className="relative min-h-screen overflow-hidden bg-[#0f051a] text-white">
      {/* Animated gradient background orbs */}
      <div className="pointer-events-none fixed inset-0 overflow-hidden">
        <div className="absolute -top-40 -left-40 h-96 w-96 animate-pulse rounded-full bg-violet-600/15 blur-3xl" />
        <div
          className="absolute top-1/3 -right-40 h-96 w-96 animate-pulse rounded-full bg-pink-600/15 blur-3xl"
          style={{ animationDelay: '1s' }}
        />
        <div
          className="absolute -bottom-40 left-1/3 h-96 w-96 animate-pulse rounded-full bg-cyan-600/15 blur-3xl"
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
            <div className="relative flex h-11 w-11 items-center justify-center rounded-xl bg-gradient-to-br from-violet-500 via-pink-500 to-cyan-500">
              <Aperture className="h-5 w-5 text-white" strokeWidth={2.5} />
            </div>
            <div>
              <h1 className="text-lg font-bold tracking-tight">
                K-Means Clustering
              </h1>
              <p className="text-xs text-white/50">
                Unsupervised Learning · Breast Cancer Dataset
              </p>
            </div>
            <button
              onClick={() => setShowCode((s) => !s)}
              className="ml-auto flex items-center gap-2 rounded-lg border border-white/10 bg-white/5 px-3 py-1.5 text-xs font-medium text-white/80 transition hover:border-violet-500/50 hover:bg-violet-500/10 hover:text-white"
            >
              <Code2 className="h-3.5 w-3.5" />
              {showCode ? 'Hide Code' : 'View Code'}
            </button>
          </div>
        </header>

        <main className="mx-auto max-w-7xl px-6 py-8">
          {/* Hero */}
          <div className="group relative mb-8 overflow-hidden rounded-2xl border border-white/10 bg-gradient-to-br from-violet-900/30 via-pink-900/20 to-cyan-900/30 p-8">
            <div className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-violet-400 to-transparent" />
            <div className="absolute inset-x-0 bottom-0 h-px bg-gradient-to-r from-transparent via-cyan-400 to-transparent" />

            <div className="flex flex-col gap-6 lg:flex-row lg:items-center lg:justify-between">
              <div className="max-w-2xl">
                <div className="mb-3 inline-flex items-center gap-1.5 rounded-full border border-violet-400/30 bg-violet-500/10 px-3 py-1 text-[11px] font-medium text-violet-300">
                  <Sparkles className="h-3 w-3" /> Unsupervised
                </div>
                <h2 className="text-3xl font-bold leading-tight md:text-4xl">
                  Discovering structure with{' '}
                  <span className="bg-gradient-to-r from-violet-400 via-pink-400 to-cyan-400 bg-clip-text text-transparent">
                    K-Means
                  </span>
                </h2>
                <p className="mt-3 text-sm leading-relaxed text-white/70">
                  K-Means partitions the data into{' '}
                  <span className="font-semibold text-violet-300">k</span>{' '}
                  clusters by iteratively assigning points to the nearest
                  centroid and updating centroids to cluster means.
                </p>
              </div>
              <div className="flex gap-3 lg:flex-col">
                <div className="flex-1 rounded-xl border border-white/10 bg-black/40 px-5 py-4 lg:min-w-[140px]">
                  <p className="bg-gradient-to-r from-violet-400 to-pink-400 bg-clip-text text-3xl font-bold text-transparent">
                    569
                  </p>
                  <p className="text-xs text-white/50">Samples</p>
                </div>
                <div className="flex-1 rounded-xl border border-white/10 bg-black/40 px-5 py-4 lg:min-w-[140px]">
                  <p className="bg-gradient-to-r from-pink-400 to-cyan-400 bg-clip-text text-3xl font-bold text-transparent">
                    2
                  </p>
                  <p className="text-xs text-white/50">Features</p>
                </div>
                <div className="flex-1 rounded-xl border border-white/10 bg-black/40 px-5 py-4 lg:min-w-[140px]">
                  <p className="bg-gradient-to-r from-cyan-400 to-amber-400 bg-clip-text text-3xl font-bold text-transparent">
                    {k}
                  </p>
                  <p className="text-xs text-white/50">Clusters</p>
                </div>
              </div>
            </div>
          </div>

          {/* Main Visualization */}
          <div className="grid grid-cols-1 gap-6 lg:grid-cols-5">
            {/* Scatter Plot */}
            <div className="lg:col-span-3">
              <div className="rounded-2xl border border-white/10 bg-black/40 p-6">
                <div className="mb-4 flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-gradient-to-br from-violet-500/20 to-pink-500/20 ring-1 ring-violet-400/30">
                      <Hexagon className="h-4 w-4 text-violet-300" />
                    </div>
                    <div>
                      <h3 className="text-sm font-semibold">
                        Cluster Visualization
                      </h3>
                      <p className="text-xs text-white/50">
                        Mean Radius × Mean Texture (standardized)
                      </p>
                    </div>
                  </div>
                </div>

                <div className="overflow-hidden rounded-xl border border-white/5 bg-black/40">
                  <svg
                    width={plotWidth}
                    height={plotHeight}
                    className="w-full h-auto"
                  >
                    {/* Background */}
                    <rect
                      x={padding.left}
                      y={padding.top}
                      width={innerWidth}
                      height={innerHeight}
                      fill="rgba(255,255,255,0.02)"
                      stroke="rgba(255,255,255,0.05)"
                    />

                    {/* Grid lines */}
                    {Array.from({ length: 5 }, (_, i) => {
                      const xVal = xMin + (i / 4) * (xMax - xMin);
                      return (
                        <line
                          key={`xgrid-${i}`}
                          x1={xScale(xVal)}
                          y1={padding.top}
                          x2={xScale(xVal)}
                          y2={padding.top + innerHeight}
                          stroke="rgba(255,255,255,0.03)"
                        />
                      );
                    })}
                    {Array.from({ length: 5 }, (_, i) => {
                      const yVal = yMin + (i / 4) * (yMax - yMin);
                      return (
                        <line
                          key={`ygrid-${i}`}
                          x1={padding.left}
                          y1={yScale(yVal)}
                          x2={padding.left + innerWidth}
                          y2={yScale(yVal)}
                          stroke="rgba(255,255,255,0.03)"
                        />
                      );
                    })}

                    {/* Data points */}
                    {scaled.map((p, i) => (
                      <circle
                        key={i}
                        cx={xScale(p.x)}
                        cy={yScale(p.y)}
                        r={3.5}
                        fill={CLUSTER_COLORS[result.labels[i]]}
                        opacity={0.75}
                        stroke="rgba(255,255,255,0.2)"
                        strokeWidth={0.5}
                      />
                    ))}

                    {/* Centroids */}
                    {result.centroids.map((c, i) => (
                      <g key={`cent-${i}`}>
                        {/* Glow */}
                        <circle
                          cx={xScale(c[0])}
                          cy={yScale(c[1])}
                          r={16}
                          fill={CLUSTER_COLORS[i]}
                          opacity={0.2}
                        />
                        {/* Outer ring */}
                        <circle
                          cx={xScale(c[0])}
                          cy={yScale(c[1])}
                          r={10}
                          fill="none"
                          stroke="white"
                          strokeWidth={2}
                        />
                        {/* Inner dot */}
                        <circle
                          cx={xScale(c[0])}
                          cy={yScale(c[1])}
                          r={6}
                          fill={CLUSTER_COLORS[i]}
                        />
                        {/* X marker */}
                        <line
                          x1={xScale(c[0]) - 4}
                          y1={yScale(c[1]) - 4}
                          x2={xScale(c[0]) + 4}
                          y2={yScale(c[1]) + 4}
                          stroke="white"
                          strokeWidth={1.5}
                        />
                        <line
                          x1={xScale(c[0]) - 4}
                          y1={yScale(c[1]) + 4}
                          x2={xScale(c[0]) + 4}
                          y2={yScale(c[1]) - 4}
                          stroke="white"
                          strokeWidth={1.5}
                        />
                        {/* Label */}
                        <text
                          x={xScale(c[0]) + 14}
                          y={yScale(c[1]) + 4}
                          fill="white"
                          fontSize="11"
                          fontWeight="600"
                        >
                          C{i}
                        </text>
                      </g>
                    ))}

                    {/* Axes */}
                    <line
                      x1={padding.left}
                      y1={padding.top + innerHeight}
                      x2={padding.left + innerWidth}
                      y2={padding.top + innerHeight}
                      stroke="rgba(255,255,255,0.2)"
                    />
                    <line
                      x1={padding.left}
                      y1={padding.top}
                      x2={padding.left}
                      y2={padding.top + innerHeight}
                      stroke="rgba(255,255,255,0.2)"
                    />

                    {/* X ticks */}
                    {Array.from({ length: 5 }, (_, i) => {
                      const v = xMin + (i / 4) * (xMax - xMin);
                      return (
                        <g key={`xtick-${i}`}>
                          <line
                            x1={xScale(v)}
                            y1={padding.top + innerHeight}
                            x2={xScale(v)}
                            y2={padding.top + innerHeight + 4}
                            stroke="rgba(255,255,255,0.2)"
                          />
                          <text
                            x={xScale(v)}
                            y={padding.top + innerHeight + 18}
                            textAnchor="middle"
                            fill="rgba(255,255,255,0.5)"
                            fontSize="10"
                            fontFamily="monospace"
                          >
                            {v.toFixed(1)}
                          </text>
                        </g>
                      );
                    })}

                    {/* Y ticks */}
                    {Array.from({ length: 5 }, (_, i) => {
                      const v = yMin + (i / 4) * (yMax - yMin);
                      return (
                        <g key={`ytick-${i}`}>
                          <text
                            x={padding.left - 8}
                            y={yScale(v) + 3}
                            textAnchor="end"
                            fill="rgba(255,255,255,0.5)"
                            fontSize="10"
                            fontFamily="monospace"
                          >
                            {v.toFixed(1)}
                          </text>
                        </g>
                      );
                    })}

                    {/* Labels */}
                    <text
                      x={padding.left + innerWidth / 2}
                      y={padding.top + innerHeight + 34}
                      textAnchor="middle"
                      fill="rgba(255,255,255,0.7)"
                      fontSize="11"
                      fontWeight="600"
                    >
                      Mean Radius (standardized)
                    </text>
                    <text
                      x={padding.left - 35}
                      y={padding.top + innerHeight / 2}
                      textAnchor="middle"
                      fill="rgba(255,255,255,0.7)"
                      fontSize="11"
                      fontWeight="600"
                      transform={`rotate(-90, ${padding.left - 35}, ${
                        padding.top + innerHeight / 2
                      })`}
                    >
                      Mean Texture (standardized)
                    </text>
                  </svg>
                </div>

                {/* K Selector */}
                <div className="mt-4 rounded-xl border border-white/10 bg-white/[0.02] p-4">
                  <div className="mb-3 flex items-center justify-between">
                    <label className="text-xs font-semibold uppercase tracking-wider text-white/60">
                      Number of Clusters (k)
                    </label>
                    <div className="rounded-md border border-violet-500/30 bg-violet-500/10 px-2.5 py-1 font-mono text-sm font-bold text-violet-300">
                      k = {k}
                    </div>
                  </div>

                  <input
                    type="range"
                    min={1}
                    max={10}
                    value={k}
                    onChange={(e) => setK(parseInt(e.target.value))}
                    className="h-2 w-full cursor-pointer appearance-none rounded-full bg-white/10 accent-violet-500"
                  />

                  <div className="mt-2 flex justify-between text-[10px] text-white/40">
                    <span>Underfit (k=1)</span>
                    <span>Overfit (k=10)</span>
                  </div>

                  <div className="mt-3 flex flex-wrap gap-1.5">
                    {[1, 2, 3, 4, 5, 6, 8, 10].map((kVal) => (
                      <button
                        key={kVal}
                        onClick={() => setK(kVal)}
                        className={`rounded-md border px-3 py-1 text-xs font-mono font-semibold transition ${
                          kVal === k
                            ? 'border-violet-500 bg-violet-500/20 text-violet-300'
                            : 'border-white/10 bg-white/[0.02] text-white/60 hover:border-white/20 hover:text-white'
                        }`}
                      >
                        k={kVal}
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            </div>

            {/* Right Panel */}
            <div className="space-y-6 lg:col-span-2">
              {/* Cluster Legend */}
              <div className="rounded-2xl border border-white/10 bg-black/40 p-5">
                <div className="mb-3 flex items-center gap-2">
                  <Target className="h-4 w-4 text-violet-400" />
                  <h3 className="text-[11px] font-semibold uppercase tracking-wider text-white/60">
                    Clusters
                  </h3>
                </div>
                <div className="space-y-1.5">
                  {stats.map((s, i) => (
                    <div
                      key={i}
                      className="flex items-center justify-between rounded-lg border border-white/5 bg-white/[0.02] px-3 py-2"
                    >
                      <div className="flex items-center gap-2">
                        <div
                          className="h-3 w-3 rounded-full"
                          style={{ backgroundColor: CLUSTER_COLORS[i] }}
                        />
                        <span className="text-xs font-semibold">C{i}</span>
                      </div>
                      <div className="flex items-center gap-3 text-xs">
                        <span className="font-mono text-white/60">
                          n={s.size}
                        </span>
                        <span className="font-mono text-white/40">
                          ({((s.size / scaled.length) * 100).toFixed(0)}%)
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Metrics */}
              <div className="rounded-2xl border border-white/10 bg-black/40 p-5">
                <div className="mb-3 flex items-center gap-2">
                  <Activity className="h-4 w-4 text-pink-400" />
                  <h3 className="text-[11px] font-semibold uppercase tracking-wider text-white/60">
                    Model Metrics
                  </h3>
                </div>
                <div className="grid grid-cols-2 gap-3">
                  <div className="rounded-lg border border-violet-500/20 bg-violet-500/5 p-3">
                    <p className="text-[10px] uppercase tracking-wider text-violet-400/70">
                      Inertia
                    </p>
                    <p className="mt-1 font-mono text-xl font-bold text-violet-300">
                      {result.inertia.toFixed(0)}
                    </p>
                    <p className="text-[10px] text-white/50">
                      Within-cluster variance
                    </p>
                  </div>
                  <div className="rounded-lg border border-pink-500/20 bg-pink-500/5 p-3">
                    <p className="text-[10px] uppercase tracking-wider text-pink-400/70">
                      Silhouette
                    </p>
                    <p className="mt-1 font-mono text-xl font-bold text-pink-300">
                      {currentSilhouette.toFixed(3)}
                    </p>
                    <p className="text-[10px] text-white/50">
                      Separation quality
                    </p>
                  </div>
                </div>
                <div className="mt-3 rounded-lg border border-white/10 bg-white/[0.02] p-3">
                  <div className="flex items-center justify-between text-xs">
                    <span className="text-white/60">Iterations</span>
                    <span className="font-mono text-cyan-300">
                      {result.iterations}
                    </span>
                  </div>
                </div>
              </div>

              {/* Elbow Plot */}
              <div className="rounded-2xl border border-white/10 bg-black/40 p-5">
                <div className="mb-3 flex items-center gap-2">
                  <Activity className="h-4 w-4 text-cyan-400" />
                  <h3 className="text-[11px] font-semibold uppercase tracking-wider text-white/60">
                    Elbow Method
                  </h3>
                </div>
                <svg
                  width={elbowWidth}
                  height={elbowHeight}
                  className="w-full h-auto"
                >
                  <rect
                    x={elbowPadding.left}
                    y={elbowPadding.top}
                    width={elbowInnerWidth}
                    height={elbowInnerHeight}
                    fill="rgba(255,255,255,0.02)"
                  />
                  {/* Line */}
                  <polyline
                    points={elbowData
                      .map((d) => `${elbowXScale(d.k)},${elbowYScale(d.inertia)}`)
                      .join(' ')}
                    fill="none"
                    stroke="url(#elbowGradient)"
                    strokeWidth={2}
                  />
                  {/* Points */}
                  {elbowData.map((d) => (
                    <g key={d.k}>
                      <circle
                        cx={elbowXScale(d.k)}
                        cy={elbowYScale(d.inertia)}
                        r={d.k === k ? 5 : 3}
                        fill={d.k === k ? '#06b6d4' : 'rgba(255,255,255,0.4)'}
                        stroke="white"
                        strokeWidth={d.k === k ? 2 : 0.5}
                      />
                    </g>
                  ))}
                  {/* X labels */}
                  {elbowData
                    .filter((_, i) => i % 2 === 0 || i === 9)
                    .map((d) => (
                      <text
                        key={d.k}
                        x={elbowXScale(d.k)}
                        y={elbowHeight - 5}
                        textAnchor="middle"
                        fill="rgba(255,255,255,0.5)"
                        fontSize="9"
                        fontFamily="monospace"
                      >
                        {d.k}
                      </text>
                    ))}
                  <defs>
                    <linearGradient
                      id="elbowGradient"
                      x1="0%"
                      y1="0%"
                      x2="100%"
                      y2="0%"
                    >
                      <stop offset="0%" stopColor="#a855f7" />
                      <stop offset="50%" stopColor="#ec4899" />
                      <stop offset="100%" stopColor="#06b6d4" />
                    </linearGradient>
                  </defs>
                </svg>
                <p className="mt-2 text-[10px] text-center text-white/40">
                  k → Inertia curve (look for elbow)
                </p>
              </div>
            </div>
          </div>

          {/* Silhouette Analysis */}
          <div className="mt-6 rounded-2xl border border-white/10 bg-black/40 p-6">
            <div className="mb-4 flex items-center gap-3">
              <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-gradient-to-br from-emerald-500/20 to-cyan-500/20 ring-1 ring-emerald-400/30">
                <Activity className="h-4 w-4 text-emerald-300" />
              </div>
              <div>
                <h3 className="text-sm font-semibold">
                  Silhouette Analysis
                </h3>
                <p className="text-xs text-white/50">
                  Optimal k by silhouette coefficient
                </p>
              </div>
            </div>

            <div className="grid grid-cols-9 gap-2">
              {silhouetteData.map((d) => {
                const bestK = silhouetteData.reduce((a, b) =>
                  a.score > b.score ? a : b
                ).k;
                const isBest = d.k === bestK;
                const isCurrent = d.k === k;
                const maxScore = Math.max(...silhouetteData.map((s) => s.score));
                const height = (d.score / maxScore) * 100;
                return (
                  <div
                    key={d.k}
                    className="flex flex-col items-center gap-1"
                  >
                    <div className="flex h-24 w-full items-end">
                      <div
                        className={`w-full rounded-t transition-all ${
                          isBest
                            ? 'bg-gradient-to-t from-emerald-600 to-emerald-400'
                            : isCurrent
                            ? 'bg-gradient-to-t from-violet-600 to-pink-500'
                            : 'bg-white/10'
                        }`}
                        style={{ height: `${height}%` }}
                      />
                    </div>
                    <span
                      className={`font-mono text-[10px] ${
                        isBest
                          ? 'font-bold text-emerald-300'
                          : isCurrent
                          ? 'text-violet-300'
                          : 'text-white/50'
                      }`}
                    >
                      {d.k}
                    </span>
                    <span
                      className={`font-mono text-[9px] ${
                        isBest ? 'text-emerald-400' : 'text-white/40'
                      }`}
                    >
                      {d.score.toFixed(2)}
                    </span>
                  </div>
                );
              })}
            </div>
            <p className="mt-3 text-center text-xs text-white/50">
              Best k:{' '}
              <span className="font-mono font-bold text-emerald-300">
                {silhouetteData.reduce((a, b) => (a.score > b.score ? a : b)).k}
              </span>{' '}
              · Current k:{' '}
              <span className="font-mono font-bold text-violet-300">{k}</span>
            </p>
          </div>

          {/* Code Preview */}
          {showCode && (
            <div className="mt-6 rounded-2xl border border-white/10 bg-black/60">
              <div className="flex items-center gap-3 border-b border-white/5 px-6 py-4">
                <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-gradient-to-br from-violet-500/20 to-pink-500/20 ring-1 ring-violet-400/30">
                  <Code2 className="h-4 w-4 text-violet-300" />
                </div>
                <div>
                  <h3 className="text-sm font-semibold">Python Source</h3>
                  <p className="text-xs text-white/50">
                    sklearn KMeans on breast cancer data
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
            <div className="group relative overflow-hidden rounded-xl border border-white/10 bg-black/40 p-5 transition hover:border-violet-500/30">
              <div className="absolute -right-4 -top-4 h-16 w-16 rounded-full bg-violet-500/10 transition group-hover:bg-violet-500/20" />
              <div className="relative">
                <div className="mb-2 flex h-8 w-8 items-center justify-center rounded-lg bg-violet-500/20 ring-1 ring-violet-400/30">
                  <Aperture className="h-4 w-4 text-violet-300" />
                </div>
                <h4 className="text-[10px] font-semibold uppercase tracking-wider text-white/40">
                  Algorithm
                </h4>
                <p className="mt-1 font-semibold text-white">K-Means++</p>
                <p className="mt-1 text-xs text-white/50">
                  Smart centroid initialization
                </p>
              </div>
            </div>

            <div className="group relative overflow-hidden rounded-xl border border-white/10 bg-black/40 p-5 transition hover:border-pink-500/30">
              <div className="absolute -right-4 -top-4 h-16 w-16 rounded-full bg-pink-500/10 transition group-hover:bg-pink-500/20" />
              <div className="relative">
                <div className="mb-2 flex h-8 w-8 items-center justify-center rounded-lg bg-pink-500/20 ring-1 ring-pink-400/30">
                  <Target className="h-4 w-4 text-pink-300" />
                </div>
                <h4 className="text-[10px] font-semibold uppercase tracking-wider text-white/40">
                  Objective
                </h4>
                <p className="mt-1 font-semibold text-white">
                  Minimize Inertia
                </p>
                <p className="mt-1 text-xs text-white/50">
                  Sum of squared distances
                </p>
              </div>
            </div>

            <div className="group relative overflow-hidden rounded-xl border border-white/10 bg-black/40 p-5 transition hover:border-cyan-500/30">
              <div className="absolute -right-4 -top-4 h-16 w-16 rounded-full bg-cyan-500/10 transition group-hover:bg-cyan-500/20" />
              <div className="relative">
                <div className="mb-2 flex h-8 w-8 items-center justify-center rounded-lg bg-cyan-500/20 ring-1 ring-cyan-400/30">
                  <Activity className="h-4 w-4 text-cyan-300" />
                </div>
                <h4 className="text-[10px] font-semibold uppercase tracking-wider text-white/40">
                  Complexity
                </h4>
                <p className="mt-1 font-mono font-semibold text-white">
                  O(n · k · i · d)
                </p>
                <p className="mt-1 text-xs text-white/50">
                  n points, k clusters
                </p>
              </div>
            </div>

            <div className="group relative overflow-hidden rounded-xl border border-white/10 bg-black/40 p-5 transition hover:border-amber-500/30">
              <div className="absolute -right-4 -top-4 h-16 w-16 rounded-full bg-amber-500/10 transition group-hover:bg-amber-500/20" />
              <div className="relative">
                <div className="mb-2 flex h-8 w-8 items-center justify-center rounded-lg bg-amber-500/20 ring-1 ring-amber-400/30">
                  <Info className="h-4 w-4 text-amber-300" />
                </div>
                <h4 className="text-[10px] font-semibold uppercase tracking-wider text-white/40">
                  Preprocessing
                </h4>
                <p className="mt-1 font-semibold text-white">
                  StandardScaler
                </p>
                <p className="mt-1 text-xs text-white/50">
                  μ=0, σ=1 per feature
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
