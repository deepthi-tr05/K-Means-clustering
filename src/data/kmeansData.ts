// Breast Cancer Dataset & K-Means Clustering Implementation

export interface DataPoint {
  x: number;
  y: number;
  cluster?: number;
  originalRadius?: number;
  originalTexture?: number;
}

export interface KMeansResult {
  labels: number[];
  centroids: [number, number][];
  inertia: number;
  iterations: number;
}

// Seeded random
function seededRandom(seed: number) {
  let state = seed;
  return function () {
    state = (state * 9301 + 49297) % 233280;
    return state / 233280;
  };
}

function normalRandom(rng: () => number, mean = 0, std = 1): number {
  const u1 = rng();
  const u2 = rng();
  const z0 = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  return z0 * std + mean;
}

// Generate breast cancer-like data (mean radius and mean texture)
export function generateBreastCancerData(): DataPoint[] {
  const rng = seededRandom(42);
  const data: DataPoint[] = [];
  
  // Malignant class (tends to have larger radius, higher texture)
  for (let i = 0; i < 212; i++) {
    const radius = 17 + normalRandom(rng, 0, 3);
    const texture = 21 + normalRandom(rng, 0, 4);
    data.push({
      x: radius,
      y: texture,
      originalRadius: radius,
      originalTexture: texture,
    });
  }
  
  // Benign class (smaller radius, lower texture)
  for (let i = 0; i < 357; i++) {
    const radius = 12 + normalRandom(rng, 0, 2);
    const texture = 17 + normalRandom(rng, 0, 3);
    data.push({
      x: radius,
      y: texture,
      originalRadius: radius,
      originalTexture: texture,
    });
  }
  
  return data;
}

// StandardScaler
export function standardScaler(data: DataPoint[]): {
  scaled: DataPoint[];
  meanX: number;
  stdX: number;
  meanY: number;
  stdY: number;
} {
  const n = data.length;
  const meanX = data.reduce((s, d) => s + d.x, 0) / n;
  const meanY = data.reduce((s, d) => s + d.y, 0) / n;
  const stdX = Math.sqrt(
    data.reduce((s, d) => s + Math.pow(d.x - meanX, 2), 0) / n
  );
  const stdY = Math.sqrt(
    data.reduce((s, d) => s + Math.pow(d.y - meanY, 2), 0) / n
  );
  
  const scaled = data.map((d) => ({
    ...d,
    x: (d.x - meanX) / stdX,
    y: (d.y - meanY) / stdY,
  }));
  
  return { scaled, meanX, stdX, meanY, stdY };
}

// Euclidean distance
function euclidean(p1: [number, number], p2: [number, number]): number {
  return Math.sqrt(Math.pow(p1[0] - p2[0], 2) + Math.pow(p1[1] - p2[1], 2));
}

// K-Means algorithm (Lloyd's)
export function kMeans(
  data: DataPoint[],
  k: number,
  maxIter = 100,
  seed = 0
): KMeansResult {
  const rng = seededRandom(seed);
  const n = data.length;
  const points: [number, number][] = data.map((d) => [d.x, d.y]);
  
  // Initialize centroids using k-means++ 
  const centroids: [number, number][] = [];
  const firstIdx = Math.floor(rng() * n);
  centroids.push(points[firstIdx]);
  
  for (let c = 1; c < k; c++) {
    const distances = points.map((p) => {
      const minDist = Math.min(
        ...centroids.map((cent) => euclidean(p, cent))
      );
      return minDist * minDist;
    });
    const sum = distances.reduce((s, d) => s + d, 0);
    let r = rng() * sum;
    for (let i = 0; i < n; i++) {
      r -= distances[i];
      if (r <= 0) {
        centroids.push(points[i]);
        break;
      }
    }
  }
  
  let labels = new Array(n).fill(0);
  let iterations = 0;
  
  // Iterate
  for (let iter = 0; iter < maxIter; iter++) {
    iterations = iter + 1;
    
    // Assign points to nearest centroid
    const newLabels = points.map((p) => {
      let bestCluster = 0;
      let bestDist = Infinity;
      centroids.forEach((cent, c) => {
        const d = euclidean(p, cent);
        if (d < bestDist) {
          bestDist = d;
          bestCluster = c;
        }
      });
      return bestCluster;
    });
    
    // Check convergence
    const changed = newLabels.some((l, i) => l !== labels[i]);
    labels = newLabels;
    
    if (!changed) break;
    
    // Update centroids
    for (let c = 0; c < k; c++) {
      const clusterPoints = points.filter((_, i) => labels[i] === c);
      if (clusterPoints.length > 0) {
        centroids[c] = [
          clusterPoints.reduce((s, p) => s + p[0], 0) / clusterPoints.length,
          clusterPoints.reduce((s, p) => s + p[1], 0) / clusterPoints.length,
        ];
      }
    }
  }
  
  // Compute inertia
  const inertia = points.reduce((sum, p, i) => {
    const cent = centroids[labels[i]];
    return sum + Math.pow(euclidean(p, cent), 2);
  }, 0);
  
  return { labels, centroids, inertia, iterations };
}

// Silhouette score (simplified)
export function silhouetteScore(
  data: DataPoint[],
  labels: number[],
  k: number
): number {
  if (k < 2) return 0;
  
  const points: [number, number][] = data.map((d) => [d.x, d.y]);
  const n = points.length;
  let totalSilhouette = 0;
  
  // Sample for efficiency
  const sampleSize = Math.min(n, 100);
  const step = Math.floor(n / sampleSize);
  
  for (let i = 0; i < n; i += step) {
    const p = points[i];
    const cluster = labels[i];
    
    // Average distance to same cluster
    let a = 0;
    let sameCount = 0;
    for (let j = 0; j < n; j++) {
      if (labels[j] === cluster && j !== i) {
        a += euclidean(p, points[j]);
        sameCount++;
      }
    }
    a = sameCount > 0 ? a / sameCount : 0;
    
    // Minimum average distance to other clusters
    let b = Infinity;
    for (let c = 0; c < k; c++) {
      if (c === cluster) continue;
      let dist = 0;
      let count = 0;
      for (let j = 0; j < n; j++) {
        if (labels[j] === c) {
          dist += euclidean(p, points[j]);
          count++;
        }
      }
      if (count > 0) {
        b = Math.min(b, dist / count);
      }
    }
    
    if (b === Infinity) b = 0;
    
    const s = (b - a) / Math.max(a, b);
    totalSilhouette += s;
  }
  
  return totalSilhouette / sampleSize;
}

// Compute cluster stats
export function clusterStats(
  data: DataPoint[],
  labels: number[],
  k: number
): { size: number; meanX: number; meanY: number; stdX: number; stdY: number }[] {
  const stats = [];
  for (let c = 0; c < k; c++) {
    const clusterData = data.filter((_, i) => labels[i] === c);
    const size = clusterData.length;
    const meanX = clusterData.reduce((s, d) => s + d.x, 0) / size;
    const meanY = clusterData.reduce((s, d) => s + d.y, 0) / size;
    const stdX = Math.sqrt(
      clusterData.reduce((s, d) => s + Math.pow(d.x - meanX, 2), 0) / size
    );
    const stdY = Math.sqrt(
      clusterData.reduce((s, d) => s + Math.pow(d.y - meanY, 2), 0) / size
    );
    stats.push({ size, meanX, meanY, stdX, stdY });
  }
  return stats;
}
