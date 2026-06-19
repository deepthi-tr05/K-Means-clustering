// KNN Implementation in TypeScript

export interface DataPoint {
  x: number;
  y: number;
  actual?: number;
  predicted?: number;
}

// Generate random data with seed
function seededRandom(seed: number) {
  let state = seed;
  return function() {
    state = (state * 9301 + 49297) % 233280;
    return state / 233280;
  };
}

export function generateKNNData(): { train: DataPoint[]; test: DataPoint[] } {
  const rng = seededRandom(0);
  const points: DataPoint[] = [];
  
  // Generate 100 random points
  for (let i = 0; i < 100; i++) {
    points.push({ x: rng(), y: 0 });
  }
  
  // Split into train (first 50) and test (next 50)
  const train = points.slice(0, 50).map(p => ({
    ...p,
    y: p.x <= 0.5 ? 1 : 2,
  }));
  
  const test = points.slice(50).map(p => ({
    ...p,
    y: 0, // No actual label for test
  }));
  
  return { train, test };
}

// Calculate Euclidean distance
function euclideanDistance(p1: DataPoint, p2: DataPoint): number {
  return Math.abs(p1.x - p2.x);
}

// KNN prediction
export function knnPredict(
  train: DataPoint[],
  testPoint: DataPoint,
  k: number
): number {
  // Calculate distances to all training points
  const distances = train.map(p => ({
    point: p,
    distance: euclideanDistance(p, testPoint),
  }));
  
  // Sort by distance
  distances.sort((a, b) => a.distance - b.distance);
  
  // Get k nearest neighbors
  const kNearest = distances.slice(0, k);
  
  // Count votes for each class
  const votes = new Map<number, number>();
  kNearest.forEach(n => {
    const label = n.point.y;
    votes.set(label, (votes.get(label) || 0) + 1);
  });
  
  // Return class with most votes
  let maxVotes = 0;
  let predictedClass = 1;
  votes.forEach((count, label) => {
    if (count > maxVotes) {
      maxVotes = count;
      predictedClass = label;
    }
  });
  
  return predictedClass;
}

// Predict for all test points
export function knnPredictAll(
  train: DataPoint[],
  test: DataPoint[],
  k: number
): DataPoint[] {
  return test.map(point => ({
    ...point,
    predicted: knnPredict(train, point, k),
  }));
}

// Get k values to test
export const K_VALUES = [1, 2, 3, 4, 5, 20, 30];
