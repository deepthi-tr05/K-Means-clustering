// Olivetti Faces Dataset Simulation & Gaussian Naive Bayes Implementation

export interface Face {
  pixels: number[]; // 64x64 = 4096 grayscale values (0-1)
  label: number;
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

// Generate stylized face images (simplified Olivetti-like)
// Each subject has unique "face features" (position of eyes, nose, mouth)
export function generateOlivettiFaces(
  numSubjects = 10,
  imagesPerSubject = 10
): Face[] {
  const rng = seededRandom(42);
  const faces: Face[] = [];
  const size = 64;

  // Generate subject-specific face templates
  const subjectTemplates: {
    eyeY: number;
    eyeX: number;
    eyeSize: number;
    noseY: number;
    mouthY: number;
    faceSize: number;
    brightness: number;
  }[] = [];

  for (let s = 0; s < numSubjects; s++) {
    subjectTemplates.push({
      eyeY: 0.35 + rng() * 0.1,
      eyeX: 0.3 + rng() * 0.05,
      eyeSize: 0.04 + rng() * 0.02,
      noseY: 0.5 + rng() * 0.05,
      mouthY: 0.7 + rng() * 0.05,
      faceSize: 0.5 + rng() * 0.2,
      brightness: 0.3 + rng() * 0.4,
    });
  }

  for (let s = 0; s < numSubjects; s++) {
    const template = subjectTemplates[s];
    
    for (let img = 0; img < imagesPerSubject; img++) {
      const pixels: number[] = [];
      
      // Add per-image variation
      const variation = {
        eyeOffset: normalRandom(rng, 0, 0.01),
        brightnessVar: normalRandom(rng, 0, 0.05),
        noise: 0.03 + rng() * 0.02,
      };
      
      for (let y = 0; y < size; y++) {
        for (let x = 0; x < size; x++) {
          const nx = x / size;
          const ny = y / size;
          
          // Face oval
          const cx = 0.5;
          const cy = 0.5;
          const rx = template.faceSize * 0.4;
          const ry = template.faceSize * 0.55;
          const dist = Math.sqrt(
            Math.pow((nx - cx) / rx, 2) + Math.pow((ny - cy) / ry, 2)
          );
          
          let val = 0;
          
          if (dist < 1) {
            // Inside face
            val = template.brightness + variation.brightnessVar;
            val *= 1 - dist * 0.3; // Slight vignette
            
            // Eyes
            const eyeL = {
              x: template.eyeX + variation.eyeOffset,
              y: template.eyeY,
            };
            const eyeR = {
              x: 1 - template.eyeX - variation.eyeOffset,
              y: template.eyeY,
            };
            const dEyeL = Math.sqrt(
              Math.pow(nx - eyeL.x, 2) + Math.pow(ny - eyeL.y, 2)
            );
            const dEyeR = Math.sqrt(
              Math.pow(nx - eyeR.x, 2) + Math.pow(ny - eyeR.y, 2)
            );
            if (dEyeL < template.eyeSize || dEyeR < template.eyeSize) {
              val *= 0.3; // Dark eyes
            }
            
            // Nose (subtle shadow)
            const dNose = Math.abs(nx - 0.5);
            if (
              ny > template.noseY - 0.05 &&
              ny < template.noseY + 0.1 &&
              dNose < 0.03
            ) {
              val *= 0.85;
            }
            
            // Mouth
            const dMouthY = Math.abs(ny - template.mouthY);
            const dMouthX = Math.abs(nx - 0.5);
            if (dMouthY < 0.015 && dMouthX < 0.08) {
              val *= 0.7;
            }
          } else {
            // Background (dark)
            val = 0.1 + rng() * 0.05;
          }
          
          // Add noise
          val += normalRandom(rng, 0, variation.noise);
          val = Math.max(0, Math.min(1, val));
          
          pixels.push(val);
        }
      }
      
      faces.push({ pixels, label: s });
    }
  }
  
  return faces;
}

// Gaussian Naive Bayes implementation
export class GaussianNB {
  private means: number[][] = []; // [class][feature]
  private variances: number[][] = []; // [class][feature]
  private priors: number[] = [];
  private numClasses: number = 0;

  fit(X: number[][], y: number[]): void {
    const classes = [...new Set(y)].sort((a, b) => a - b);
    this.numClasses = classes.length;
    const nFeatures = X[0].length;
    
    this.means = Array.from({ length: this.numClasses }, () =>
      Array(nFeatures).fill(0)
    );
    this.variances = Array.from({ length: this.numClasses }, () =>
      Array(nFeatures).fill(0)
    );
    this.priors = Array(this.numClasses).fill(0);
    
    // Compute mean and variance per class
    const counts = Array(this.numClasses).fill(0);
    
    for (let i = 0; i < X.length; i++) {
      const c = y[i];
      counts[c]++;
      for (let j = 0; j < nFeatures; j++) {
        this.means[c][j] += X[i][j];
      }
    }
    
    // Normalize means
    for (let c = 0; c < this.numClasses; c++) {
      for (let j = 0; j < nFeatures; j++) {
        this.means[c][j] /= counts[c];
      }
      this.priors[c] = counts[c] / X.length;
    }
    
    // Compute variances
    for (let i = 0; i < X.length; i++) {
      const c = y[i];
      for (let j = 0; j < nFeatures; j++) {
        const diff = X[i][j] - this.means[c][j];
        this.variances[c][j] += diff * diff;
      }
    }
    
    for (let c = 0; c < this.numClasses; c++) {
      for (let j = 0; j < nFeatures; j++) {
        this.variances[c][j] =
          this.variances[c][j] / counts[c] + 1e-9; // Add small epsilon
      }
    }
  }

  // Log probability for numerical stability
  private logProbability(x: number[], c: number): number {
    let logP = Math.log(this.priors[c]);
    const nFeatures = x.length;
    
    for (let j = 0; j < nFeatures; j++) {
      const mean = this.means[c][j];
      const variance = this.variances[c][j];
      const diff = x[j] - mean;
      logP += -0.5 * Math.log(2 * Math.PI * variance);
      logP += -(diff * diff) / (2 * variance);
    }
    
    return logP;
  }

  predict(X: number[][]): number[] {
    return X.map((x) => {
      let bestClass = 0;
      let bestLogP = -Infinity;
      
      for (let c = 0; c < this.numClasses; c++) {
        const logP = this.logProbability(x, c);
        if (logP > bestLogP) {
          bestLogP = logP;
          bestClass = c;
        }
      }
      
      return bestClass;
    });
  }

  predictProba(X: number[][]): number[][] {
    return X.map((x) => {
      const logPs = Array.from({ length: this.numClasses }, (_, c) =>
        this.logProbability(x, c)
      );
      
      // Convert to probabilities using softmax
      const maxLogP = Math.max(...logPs);
      const exps = logPs.map((lp) => Math.exp(lp - maxLogP));
      const sum = exps.reduce((a, b) => a + b, 0);
      return exps.map((e) => e / sum);
    });
  }

  getMeans(): number[][] {
    return this.means;
  }

  getVariances(): number[][] {
    return this.variances;
  }

  getPriors(): number[] {
    return this.priors;
  }
}

// Accuracy
export function accuracy(yTrue: number[], yPred: number[]): number {
  let correct = 0;
  for (let i = 0; i < yTrue.length; i++) {
    if (yTrue[i] === yPred[i]) correct++;
  }
  return correct / yTrue.length;
}

// Confusion matrix
export function confusionMatrix(
  yTrue: number[],
  yPred: number[],
  numClasses: number
): number[][] {
  const matrix = Array.from({ length: numClasses }, () =>
    Array(numClasses).fill(0)
  );
  for (let i = 0; i < yTrue.length; i++) {
    matrix[yTrue[i]][yPred[i]]++;
  }
  return matrix;
}
