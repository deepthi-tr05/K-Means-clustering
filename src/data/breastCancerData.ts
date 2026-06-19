// Breast Cancer Dataset & Decision Tree Implementation

export interface DataPoint {
  features: number[];
  label: number; // 0 = malignant, 1 = benign
}

export interface TreeNode {
  featureIndex?: number;
  featureName?: string;
  threshold?: number;
  left?: TreeNode;
  right?: TreeNode;
  value?: number[]; // class distribution [malignant_count, benign_count]
  prediction?: number;
  isLeaf: boolean;
  gini?: number;
  samples?: number;
}

// Feature names from breast cancer dataset (subset for visualization)
export const FEATURE_NAMES = [
  'mean radius',
  'mean texture',
  'mean perimeter',
  'mean area',
  'mean smoothness',
  'mean compactness',
  'mean concavity',
  'mean concave points',
  'mean symmetry',
  'mean fractal dimension',
  'radius error',
  'texture error',
  'perimeter error',
  'area error',
  'smoothness error',
  'compactness error',
  'concavity error',
  'concave points error',
  'symmetry error',
  'fractal dimension error',
  'worst radius',
  'worst texture',
  'worst perimeter',
  'worst area',
  'worst smoothness',
  'worst compactness',
  'worst concavity',
  'worst concave points',
  'worst symmetry',
  'worst fractal dimension',
];

export const TARGET_NAMES = ['malignant', 'benign'];
export const CLASS_COLORS: Record<number, string> = {
  0: '#ef4444', // malignant - red
  1: '#14b8a6', // benign - teal
};

// Seeded random for reproducible data
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

// Generate synthetic breast cancer-like data
export function generateBreastCancerData(): DataPoint[] {
  const rng = seededRandom(42);
  const data: DataPoint[] = [];
  
  // Malignant class (class 0) - tends to have higher values
  for (let i = 0; i < 212; i++) {
    const features = [
      17 + normalRandom(rng, 0, 3),   // mean radius
      21 + normalRandom(rng, 0, 4),   // mean texture
      115 + normalRandom(rng, 0, 20), // mean perimeter
      950 + normalRandom(rng, 0, 200), // mean area
      0.1 + normalRandom(rng, 0, 0.02), // mean smoothness
      0.17 + normalRandom(rng, 0, 0.06), // mean compactness
      0.19 + normalRandom(rng, 0, 0.1), // mean concavity
      0.09 + normalRandom(rng, 0, 0.05), // mean concave points
      0.18 + normalRandom(rng, 0, 0.03), // mean symmetry
      0.06 + normalRandom(rng, 0, 0.01), // mean fractal dimension
    ];
    // Add remaining features with some correlation
    for (let j = 10; j < 30; j++) {
      features.push(features[0] * 0.8 + normalRandom(rng, 0, 2));
    }
    data.push({ features, label: 0 });
  }
  
  // Benign class (class 1) - tends to have lower values
  for (let i = 0; i < 357; i++) {
    const features = [
      12 + normalRandom(rng, 0, 2),   // mean radius
      17 + normalRandom(rng, 0, 3),   // mean texture
      78 + normalRandom(rng, 0, 12),  // mean perimeter
      460 + normalRandom(rng, 0, 130), // mean area
      0.09 + normalRandom(rng, 0, 0.01), // mean smoothness
      0.06 + normalRandom(rng, 0, 0.02), // mean compactness
      0.03 + normalRandom(rng, 0, 0.03), // mean concavity
      0.02 + normalRandom(rng, 0, 0.02), // mean concave points
      0.17 + normalRandom(rng, 0, 0.02), // mean symmetry
      0.06 + normalRandom(rng, 0, 0.008), // mean fractal dimension
    ];
    for (let j = 10; j < 30; j++) {
      features.push(features[0] * 0.7 + normalRandom(rng, 0, 1));
    }
    data.push({ features, label: 1 });
  }
  
  return data;
}

// Gini impurity
function giniImpurity(labels: number[]): number {
  if (labels.length === 0) return 0;
  const counts = [0, 0];
  labels.forEach((l) => counts[l]++);
  const total = labels.length;
  return 1 - counts.reduce((sum, c) => sum + Math.pow(c / total, 2), 0);
}

// Find best split
function findBestSplit(
  data: DataPoint[],
  featureIndices: number[]
): { featureIndex: number; threshold: number; gini: number } | null {
  let bestSplit = null;
  let bestGini = Infinity;
  
  for (const fi of featureIndices) {
    const values = data.map((d) => d.features[fi]).sort((a, b) => a - b);
    
    // Try thresholds between consecutive values
    const uniqueVals = [...new Set(values)];
    for (let i = 0; i < uniqueVals.length - 1; i++) {
      const threshold = (uniqueVals[i] + uniqueVals[i + 1]) / 2;
      
      const leftLabels = data
        .filter((d) => d.features[fi] <= threshold)
        .map((d) => d.label);
      const rightLabels = data
        .filter((d) => d.features[fi] > threshold)
        .map((d) => d.label);
      
      if (leftLabels.length === 0 || rightLabels.length === 0) continue;
      
      const gini =
        (leftLabels.length * giniImpurity(leftLabels) +
          rightLabels.length * giniImpurity(rightLabels)) /
        data.length;
      
      if (gini < bestGini) {
        bestGini = gini;
        bestSplit = { featureIndex: fi, threshold, gini };
      }
    }
  }
  
  return bestSplit;
}

// Build decision tree
export function buildDecisionTree(
  data: DataPoint[],
  maxDepth = 4,
  minSamples = 5
): TreeNode {
  const labels = data.map((d) => d.label);
  const counts = [0, 0];
  labels.forEach((l) => counts[l]++);
  
  const prediction = counts[0] > counts[1] ? 0 : 1;
  const gini = giniImpurity(labels);
  
  // Leaf conditions
  if (
    maxDepth === 0 ||
    data.length < minSamples ||
    counts[0] === 0 ||
    counts[1] === 0
  ) {
    return {
      isLeaf: true,
      prediction,
      value: counts,
      gini,
      samples: data.length,
    };
  }
  
  // Find best split (use subset of features for efficiency)
  const featureIndices = Array.from({ length: 10 }, (_, i) => i);
  const split = findBestSplit(data, featureIndices);
  
  if (!split) {
    return {
      isLeaf: true,
      prediction,
      value: counts,
      gini,
      samples: data.length,
    };
  }
  
  const leftData = data.filter((d) => d.features[split.featureIndex] <= split.threshold);
  const rightData = data.filter((d) => d.features[split.featureIndex] > split.threshold);
  
  return {
    isLeaf: false,
    featureIndex: split.featureIndex,
    featureName: FEATURE_NAMES[split.featureIndex],
    threshold: split.threshold,
    gini: split.gini,
    samples: data.length,
    value: counts,
    left: buildDecisionTree(leftData, maxDepth - 1, minSamples),
    right: buildDecisionTree(rightData, maxDepth - 1, minSamples),
  };
}

// Predict single sample
export function predict(tree: TreeNode, features: number[]): number {
  if (tree.isLeaf) return tree.prediction!;
  
  if (features[tree.featureIndex!] <= tree.threshold!) {
    return predict(tree.left!, features);
  } else {
    return predict(tree.right!, features);
  }
}

// Compute feature importance
export function computeFeatureImportance(tree: TreeNode): Map<number, number> {
  const importance = new Map<number, number>();
  
  function traverse(node: TreeNode, weight: number) {
    if (node.isLeaf) return;
    
    const impDecrease =
      (node.samples! * node.gini!) -
      (node.left!.samples! * node.left!.gini!) -
      (node.right!.samples! * node.right!.gini!);
    
    const current = importance.get(node.featureIndex!) || 0;
    importance.set(node.featureIndex!, current + impDecrease * weight);
    
    traverse(node.left!, weight);
    traverse(node.right!, weight);
  }
  
  traverse(tree, 1);
  return importance;
}

// Compute confusion matrix
export function computeConfusionMatrix(
  tree: TreeNode,
  data: DataPoint[]
): number[][] {
  const matrix = [
    [0, 0], // [actual_mal_pred_mal, actual_mal_pred_ben]
    [0, 0], // [actual_ben_pred_mal, actual_ben_pred_ben]
  ];
  
  data.forEach((d) => {
    const pred = predict(tree, d.features);
    matrix[d.label][pred]++;
  });
  
  return matrix;
}

// Compute accuracy
export function computeAccuracy(tree: TreeNode, data: DataPoint[]): number {
  let correct = 0;
  data.forEach((d) => {
    if (predict(tree, d.features) === d.label) correct++;
  });
  return correct / data.length;
}
