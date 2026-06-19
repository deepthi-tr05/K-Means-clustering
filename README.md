<<<<<<< HEAD
# Gaussian-Naive-Bayes
=======
<<<<<<< HEAD
# 👤 Face Recognition Visualizer

A **beautiful, interactive visualization** of face recognition using **Gaussian Naive Bayes** on the Olivetti Faces dataset. Watch the model classify 64×64 grayscale face images in real-time!
=======
# 🌳 Decision Tree Classifier Visualizer

A **beautiful, interactive visualization** of Decision Tree classification on the Breast Cancer dataset. Watch the tree make predictions step-by-step and explore feature importance, confusion matrices, and decision paths in real-time!
>>>>>>> dcf5a6c (Update project files)

![React](https://img.shields.io/badge/React-19-61dafb?style=flat&logo=react)
![TypeScript](https://img.shields.io/badge/TypeScript-5.7-3178c6?style=flat&logo=typescript)
![Tailwind CSS](https://img.shields.io/badge/Tailwind-4.0-38bdf8?style=flat&logo=tailwindcss)

## 🎯 Overview

<<<<<<< HEAD
This dashboard demonstrates **Gaussian Naive Bayes** for face classification. Each face is a 64×64 grayscale image (4096 pixel features). The model learns the mean and variance of each pixel per subject and uses Bayes' theorem for classification.

## ✨ Features

### 🖼️ Interactive Face Gallery
- **2×5 grid** of test samples (like the matplotlib output)
- **Canvas-rendered** grayscale face images
- **Color-coded borders**: Green for correct, red for wrong
- **Label overlays** showing actual → predicted
- **Click-to-inspect** functionality

### 🔍 Sample Explorer
- **Navigate** through all test samples
- **Large face preview** (192×192)
- **Three-panel display**:
  - Result (Correct/Incorrect)
  - Actual Subject ID
  - Predicted Subject ID
- **Probability distribution** (Top 5 classes)
- **Confidence score**

### 📊 Mean Faces per Subject
- **Average face** learned for each subject
- **10 mean images** in a grid
- Shows what the model "sees" per class

### 🎯 Confusion Matrix
- **10×10 heatmap** of predictions
- **Color-coded**:
  - Green diagonal = correct
  - Red off-diagonal = errors
- **Intensity** shows count magnitude
- **Tooltips** with exact values

### 📈 Per-Class Accuracy
- **Bar chart** showing accuracy per subject
- **Gradient bars** (amber → orange → rose)
- **Numeric values** in monospace

### 🎨 Beautiful Design
- **Warm amber/orange/rose** theme (portrait studio aesthetic)
- **Animated gradient orbs** in background
- **Gradient text** and accents
- **Canvas-rendered** faces with anti-aliasing
- **Smooth transitions** throughout
=======
This dashboard visualizes a **Decision Tree Classifier** trained on the Breast Cancer Wisconsin dataset. The tree recursively splits the data using Gini impurity to classify tumors as **malignant** or **benign**.

### Classes
- 🔴 **Malignant** (red) - Cancerous tumors
- 🟢 **Benign** (teal) - Non-cancerous tumors

## ✨ Features

### 🌳 Interactive Tree Visualization
- **Full SVG tree rendering** with proper layout
- **Color-coded nodes** showing prediction class
- **Node information**: feature name, threshold, Gini impurity, sample count
- **Class distribution** shown in each node
- **Curved edges** with Yes/No labels

### 📊 Confusion Matrix
- **Visual matrix** showing TP, FP, TN, FN
- **Color-coded** for easy interpretation
- **Overall accuracy** with gradient progress bar

### 🎯 Feature Importance
- **Bar chart** of top 8 most important features
- **Gradient bars** showing relative importance
- **Sorted** by importance value

### 🔍 Sample Explorer
- **Navigate** through all 569 samples
- **See actual vs predicted** diagnosis
- **Decision path** visualization showing which splits were made
- **Color-coded** correct/incorrect predictions

### 📱 Rich Statistics
- Total samples: 569
- Features: 30
- Accuracy percentage
- Tree depth: 4
- Node and leaf counts

### 💻 Code Preview
- Toggle to view Python source code
- sklearn's DecisionTreeClassifier implementation

### 🎨 Beautiful Design
- **Rose/teal/purple** color scheme (medical theme)
- **Animated gradient orbs** in background
- **Gradient hero section**
- **Smooth transitions** throughout
- **Dark theme** with professional medical aesthetic
>>>>>>> dcf5a6c (Update project files)

## 🚀 Getting Started

```bash
<<<<<<< HEAD
# Clone
git clone https://github.com/yourusername/face-recognition-visualizer.git
cd face-recognition-visualizer
=======
# Clone the repository
git clone https://github.com/yourusername/decision-tree-visualizer.git
cd decision-tree-visualizer
>>>>>>> dcf5a6c (Update project files)

# Install dependencies
npm install

# Start development server
npm run dev
```

<<<<<<< HEAD
## 📚 The Algorithm

### Gaussian Naive Bayes

For each class c and feature j:
```
μ_cj = mean of feature j in class c
σ²_cj = variance of feature j in class c
```

Prediction using Bayes' theorem:
```
P(class|x) ∝ P(class) × ∏ P(x_j|class)
```

Where each P(x_j|class) is a Gaussian distribution.

### Why Naive Bayes for Faces?
- **Fast training** - closed-form solution
- **Scales well** to high dimensions (4096 features)
- **Works surprisingly well** despite independence assumption
- **Probabilistic outputs** (confidence scores)
=======
Open [http://localhost:5173](http://localhost:5173)

### Build for Production

```bash
npm run build
```

## 📚 The Algorithm

### CART (Classification And Regression Tree)

1. **Start** with all data at the root
2. **Find the best split**: feature and threshold that minimizes Gini impurity
3. **Split** data into left (≤ threshold) and right (> threshold)
4. **Repeat** recursively until:
   - Max depth reached
   - Node is pure (all same class)
   - Too few samples

### Gini Impurity

```
Gini = 1 - Σ p(i)²
```

Where p(i) is the proportion of samples belonging to class i.
>>>>>>> dcf5a6c (Update project files)

## 🛠️ Technology Stack

- **React 19** - UI framework
- **TypeScript** - Type safety
- **Vite** - Build tool
- **Tailwind CSS** - Styling
- **Lucide React** - Icons
<<<<<<< HEAD
- **HTML5 Canvas** - Face image rendering
- **Custom Gaussian NB** implementation
=======
- **Custom SVG** - Tree visualization
- **Custom CART implementation** in TypeScript
>>>>>>> dcf5a6c (Update project files)

## 📁 Project Structure

```
<<<<<<< HEAD
face-recognition-visualizer/
├── src/
│   ├── data/
│   │   └── olivettiFaces.ts  # Data & Gaussian NB
│   ├── App.tsx               # Main dashboard
│   ├── main.tsx              # Entry point
│   └── index.css             # Global styles
=======
decision-tree-visualizer/
├── src/
│   ├── data/
│   │   └── breastCancerData.ts  # Dataset & tree algorithm
│   ├── App.tsx                  # Main dashboard
│   ├── main.tsx                 # Entry point
│   └── index.css                # Global styles
>>>>>>> dcf5a6c (Update project files)
├── index.html
├── package.json
└── README.md
```

## 🎓 Educational Value

### Learn About
<<<<<<< HEAD
- **Image classification** fundamentals
- **Gaussian distributions** in ML
- **Bayes' theorem** in practice
- **Confusion matrices** interpretation
- **Feature independence** assumption
- **Generative vs discriminative** models

### Key Insights
- Naive Bayes works well even for correlated features (pixels)
- Mean faces reveal what the model learns per class
- Confidence scores help understand uncertainty
- Confusion matrix shows systematic errors
=======
- **Decision tree construction** from scratch
- **Gini impurity** as a split criterion
- **Overfitting** and max depth control
- **Feature importance** derivation
- **Confusion matrix** interpretation
- **Model interpretability** (white-box models)

### Key Concepts
- **Parametric vs non-parametric**: Trees are non-parametric
- **White-box models**: Easy to interpret and explain
- **Greedy algorithm**: Makes locally optimal splits
- **Pruning**: Prevents overfitting
>>>>>>> dcf5a6c (Update project files)

## 🔬 Algorithm Details

### Time Complexity
<<<<<<< HEAD
- **Training**: O(n · d) where n = samples, d = features
- **Prediction**: O(c · d) where c = classes

### Space Complexity
- O(c · d) to store means and variances

### Features
- **Input**: 64×64 grayscale images
- **Representation**: 4096 pixel intensity values (0-1)
- **Classes**: 10 subjects (10 images each)

## 🌐 Live Demo

[View Live Demo](https://face-recognition-visualizer.vercel.app)

## 🤝 Contributing

Contributions welcome!

### Ideas for Enhancement
- [ ] Load real Olivetti dataset
- [ ] Add data augmentation visualization
- [ ] Compare with other classifiers (SVM, KNN, Neural Nets)
- [ ] PCA visualization for dimensionality reduction
- [ ] Interactive feature selection
- [ ] Eigenfaces implementation
=======
- **Training**: O(n · m · log n)
- **Prediction**: O(log n) for balanced trees

### Space Complexity
- O(n) to store the tree

### Hyperparameters
- **Max depth**: Controls tree complexity
- **Min samples**: Minimum samples per node
- **Criterion**: Gini or entropy

## 🌐 Live Demo

[View Live Demo](https://decision-tree-visualizer.vercel.app)

## 🤝 Contributing

Contributions are welcome!

### Ideas for Enhancement
- [ ] Interactive max-depth slider
- [ ] Animation of tree growing
- [ ] Export tree to DOT format
- [ ] Compare with Random Forest
- [ ] Feature selection tool
- [ ] Custom datasets
>>>>>>> dcf5a6c (Update project files)

## 📝 License

MIT License

## 🙏 Acknowledgments

- Dataset: Wisconsin Breast Cancer (UCI ML Repository)
- Algorithm: sklearn's CART implementation
- Design inspiration: medical data visualization

---

**Made with ❤️ for educational purposes**

<<<<<<< HEAD
*Understanding face recognition through beautiful visualizations*
=======
*Understanding decision trees through beautiful visualizations*
>>>>>>> dcf5a6c (Update project files)
>>>>>>> c04665d (Initial project upload)
