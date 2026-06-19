// Linear & Polynomial Regression Implementation in TypeScript

export interface DataPoint {
  x: number;
  y: number;
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

// ============================================================
// LINEAR REGRESSION — California Housing (MedInc -> Price)
// ============================================================

// Synthetic California-like housing data: MedIncome (in 10k) -> Price (in 100k)
export function generateHousingData(): {
  train: DataPoint[];
  test: DataPoint[];
} {
  const rng = seededRandom(42);
  const data: DataPoint[] = [];
  for (let i = 0; i < 200; i++) {
    const medinc = 0.5 + rng() * 14.5; // 0.5 to 15 (10k units)
    const price =
      0.5 + 0.3 * medinc + normalRandom(rng, 0, 0.6); // Correlated with noise
    data.push({ x: medinc, y: Math.max(0.15, price) });
  }
  // 80/20 split
  const split = Math.floor(data.length * 0.8);
  return { train: data.slice(0, split), test: data.slice(split) };
}

// OLS Linear Regression
export function fitLinearRegression(data: DataPoint[]): { m: number; b: number } {
  const n = data.length;
  let sumX = 0,
    sumY = 0,
    sumXY = 0,
    sumX2 = 0;
  for (const p of data) {
    sumX += p.x;
    sumY += p.y;
    sumXY += p.x * p.y;
    sumX2 += p.x * p.x;
  }
  const m = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
  const b = (sumY - m * sumX) / n;
  return { m, b };
}

export function linearPredict(
  x: number,
  params: { m: number; b: number }
): number {
  return params.m * x + params.b;
}

// ============================================================
// POLYNOMIAL REGRESSION — MPG dataset (Horsepower -> MPG)
// ============================================================

export function generateMPGData(): DataPoint[] {
  const rng = seededRandom(123);
  const data: DataPoint[] = [];
  for (let i = 0; i < 150; i++) {
    const hp = 40 + rng() * 200; // 40 to 240 horsepower
    // Non-linear inverse relationship with noise
    const mpg = 45 - 0.12 * hp + 0.0003 * hp * hp + normalRandom(rng, 0, 2.5);
    data.push({ x: hp, y: Math.max(8, mpg) });
  }
  return data;
}

// Polynomial regression (degree 2): y = a*x^2 + b*x + c
export function fitPolynomialRegression(
  data: DataPoint[],
  degree = 2
): number[] {
  const n = data.length;
  const d = degree + 1;

  // Build X matrix: [[1, x, x^2, ...], ...]
  const X: number[][] = data.map((p) => {
    const row: number[] = [];
    for (let j = 0; j < d; j++) row.push(Math.pow(p.x, j));
    return row;
  });

  const y = data.map((p) => p.y);

  // Compute X^T X
  const XtX: number[][] = Array.from({ length: d }, () => Array(d).fill(0));
  for (let i = 0; i < d; i++) {
    for (let j = 0; j < d; j++) {
      let sum = 0;
      for (let k = 0; k < n; k++) sum += X[k][i] * X[k][j];
      XtX[i][j] = sum;
    }
  }

  // Compute X^T y
  const Xty: number[] = Array(d).fill(0);
  for (let i = 0; i < d; i++) {
    let sum = 0;
    for (let k = 0; k < n; k++) sum += X[k][i] * y[k];
    Xty[i] = sum;
  }

  // Solve XtX * coef = Xty using Gaussian elimination
  return solveLinearSystem(XtX, Xty);
}

function solveLinearSystem(A: number[][], b: number[]): number[] {
  const n = A.length;
  const M = A.map((row, i) => [...row, b[i]]);

  for (let i = 0; i < n; i++) {
    // Partial pivoting
    let maxRow = i;
    for (let k = i + 1; k < n; k++) {
      if (Math.abs(M[k][i]) > Math.abs(M[maxRow][i])) maxRow = k;
    }
    [M[i], M[maxRow]] = [M[maxRow], M[i]];

    const pivot = M[i][i];
    if (Math.abs(pivot) < 1e-12) continue;

    for (let j = i; j <= n; j++) M[i][j] /= pivot;

    for (let k = 0; k < n; k++) {
      if (k === i) continue;
      const factor = M[k][i];
      for (let j = i; j <= n; j++) M[k][j] -= factor * M[i][j];
    }
  }

  return M.map((row) => row[n]);
}

export function polynomialPredict(x: number, coefficients: number[]): number {
  return coefficients.reduce(
    (sum, coef, i) => sum + coef * Math.pow(x, i),
    0
  );
}

// ============================================================
// Locally Weighted Linear Regression
// ============================================================

export function generateLWLRData(): DataPoint[] {
  const rng = seededRandom(0);
  const points: DataPoint[] = [];
  for (let i = 0; i < 100; i++) {
    const x = -3 + (6 * i) / 99;
    const y = Math.sin(x) + normalRandom(rng, 0, 0.2);
    points.push({ x, y });
  }
  return points;
}

export function lwlrPredict(
  x0: number,
  data: DataPoint[],
  tau: number
): number {
  const weights = data.map((p) => {
    const diff = p.x - x0;
    return Math.exp(-(diff * diff) / (2 * tau * tau));
  });

  const X1 = data.map((p) => [1, p.x]);
  const y = data.map((p) => p.y);

  // Compute X1^T W X1 (2x2)
  const XtWX: number[][] = [
    [0, 0],
    [0, 0],
  ];
  for (let i = 0; i < data.length; i++) {
    const w = weights[i];
    XtWX[0][0] += w * X1[i][0] * X1[i][0];
    XtWX[0][1] += w * X1[i][0] * X1[i][1];
    XtWX[1][0] += w * X1[i][1] * X1[i][0];
    XtWX[1][1] += w * X1[i][1] * X1[i][1];
  }

  // Compute X1^T W y (2x1)
  const XtWy = [0, 0];
  for (let i = 0; i < data.length; i++) {
    const w = weights[i];
    XtWy[0] += w * X1[i][0] * y[i];
    XtWy[1] += w * X1[i][1] * y[i];
  }

  // Solve 2x2 system
  const det = XtWX[0][0] * XtWX[1][1] - XtWX[0][1] * XtWX[1][0];
  if (Math.abs(det) < 1e-10) return y.reduce((s, v) => s + v, 0) / y.length;

  const theta0 =
    (XtWX[1][1] * XtWy[0] - XtWX[0][1] * XtWy[1]) / det;
  const theta1 =
    (XtWX[0][0] * XtWy[1] - XtWX[1][0] * XtWy[0]) / det;

  return theta0 + theta1 * x0;
}

export function lwlrPredictAll(
  X_test: number[],
  data: DataPoint[],
  tau: number
): number[] {
  return X_test.map((x) => lwlrPredict(x, data, tau));
}
