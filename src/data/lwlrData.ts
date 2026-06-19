// Locally Weighted Linear Regression Implementation in TypeScript

export interface DataPoint {
  x: number;
  y: number;
}

// Seeded random number generator
function seededRandom(seed: number) {
  let state = seed;
  return function() {
    state = (state * 9301 + 49297) % 233280;
    return state / 233280;
  };
}

// Box-Muller transform for normal distribution
function normalRandom(rng: () => number, mean = 0, std = 1): number {
  const u1 = rng();
  const u2 = rng();
  const z0 = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  return z0 * std + mean;
}

export function generateLWLRData(): DataPoint[] {
  const rng = seededRandom(0);
  const points: DataPoint[] = [];
  
  for (let i = 0; i < 100; i++) {
    const x = -3 + (6 * i) / 99; // linspace(-3, 3, 100)
    const y = Math.sin(x) + normalRandom(rng, 0, 0.1);
    points.push({ x, y });
  }
  
  return points;
}

// Gaussian kernel
function gaussianKernel(x0: number, X: DataPoint[], tau: number): number[] {
  return X.map(p => {
    const diff = p.x - x0;
    return Math.exp(-(diff * diff) / (2 * tau * tau));
  });
}

// Matrix operations
function multiplyMatrixVector(A: number[][], v: number[]): number[] {
  return A.map(row => row.reduce((sum, val, i) => sum + val * v[i], 0));
}

function transposeMatrix(A: number[][]): number[][] {
  const rows = A.length;
  const cols = A[0].length;
  const result: number[][] = Array.from({ length: cols }, () => Array(rows).fill(0));
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      result[j][i] = A[i][j];
    }
  }
  return result;
}

function multiplyMatrices(A: number[][], B: number[][]): number[][] {
  const rowsA = A.length;
  const colsA = A[0].length;
  const colsB = B[0].length;
  const result: number[][] = Array.from({ length: rowsA }, () => Array(colsB).fill(0));
  
  for (let i = 0; i < rowsA; i++) {
    for (let j = 0; j < colsB; j++) {
      for (let k = 0; k < colsA; k++) {
        result[i][j] += A[i][k] * B[k][j];
      }
    }
  }
  return result;
}

// Pseudo-inverse using SVD approximation (simplified for 2x2)
function pseudoInverse2x2(A: number[][]): number[][] {
  // For 2x2 matrices, use direct formula
  const det = A[0][0] * A[1][1] - A[0][1] * A[1][0];
  
  if (Math.abs(det) < 1e-10) {
    // Singular matrix, return pseudo-inverse approximation
    const [[a, b], [c, d]] = A;
    const denom = a * a + b * b + c * c + d * d;
    if (denom < 1e-10) return [[0, 0], [0, 0]];
    return [
      [a / denom, c / denom],
      [b / denom, d / denom]
    ];
  }
  
  return [
    [A[1][1] / det, -A[0][1] / det],
    [-A[1][0] / det, A[0][0] / det]
  ];
}

// Predict using LWLR
export function lwlrPredict(x0: number, data: DataPoint[], tau: number): number {
  // Get kernel weights
  const weights = gaussianKernel(x0, data, tau);
  
  // Create design matrix X1 = [1, x]
  const X1 = data.map(p => [1, p.x]);
  const y = data.map(p => p.y);
  
  // Create diagonal weight matrix W
  const W = weights.map((w, i) => {
    const row = Array(data.length).fill(0);
    row[i] = w;
    return row;
  });
  
  // Compute theta = (X1^T @ W @ X1)^-1 @ X1^T @ W @ y
  const X1T = transposeMatrix(X1);
  const X1TW = multiplyMatrices(X1T, W);
  const X1TWX1 = multiplyMatrices(X1TW, X1);
  const X1TWX1_inv = pseudoInverse2x2(X1TWX1);
  const X1TWy = multiplyMatrixVector(X1TW, y);
  const theta = multiplyMatrixVector(X1TWX1_inv, X1TWy);
  
  // Predict: [1, x0] @ theta
  return theta[0] + theta[1] * x0;
}

// Predict for multiple points
export function lwlrPredictAll(
  X_test: number[],
  data: DataPoint[],
  tau: number
): number[] {
  return X_test.map(x => lwlrPredict(x, data, tau));
}

// Generate test points
export function generateTestPoints(n = 300): number[] {
  return Array.from({ length: n }, (_, i) => -3 + (6 * i) / (n - 1));
}

// Get tau values to test
export const TAU_VALUES = [0.1, 0.3, 0.5, 1.0, 2.0, 5.0];
