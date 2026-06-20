# 🎯 K-Means Clustering Visualizer

A **beautiful, interactive visualization** of K-Means clustering on the Breast Cancer dataset. Explore how different values of **k** affect cluster formation with real-time updates, elbow analysis, and silhouette scores!

![React](https://img.shields.io/badge/React-19-61dafb?style=flat&logo=react)
![TypeScript](https://img.shields.io/badge/TypeScript-5.7-3178c6?style=flat&logo=typescript)
![Tailwind CSS](https://img.shields.io/badge/Tailwind-4.0-38bdf8?style=flat&logo=tailwindcss)

## 🌐 Live Demo

[View Live Demo](https://k-means-clustering-hruxoukfo-deepthi-portfolio.vercel.app/)

## 🎯 Overview

This dashboard demonstrates **K-Means clustering** on the Breast Cancer Wisconsin dataset using two features: **mean radius** and **mean texture**. The data is standardized and clustered into k groups, with interactive controls to explore different values of k.

## ✨ Features

### 🎨 Interactive Scatter Plot
- **569 data points** plotted in 2D
- **Color-coded clusters** with 10 distinct colors
- **Animated centroids** with glow effects and X markers
- **Cluster labels** (C0, C1, C2, ...)
- **Grid lines** and proper axis labels
- **Standardized coordinates** (mean=0, std=1)

### 🎛️ Interactive K Selector
- **Range slider** (k=1 to k=10)
- **Quick-select buttons** for common values
- **Real-time updates** as k changes
- **Visual feedback** on active k value

### 📊 Elbow Method Plot
- **Inertia vs k** curve
- **Gradient line** (violet → pink → cyan)
- **Highlighted current k**
- Helps identify optimal number of clusters

### 📈 Silhouette Analysis
- **Bar chart** showing silhouette scores for k=2 to 10
- **Best k highlighted** in emerald
- **Current k highlighted** in violet/pink gradient
- **Numeric scores** displayed

### 🎯 Cluster Statistics
- **Cluster list** with:
  - Color-coded indicators
  - Cluster size (n)
  - Percentage of total
- **Model metrics**:
  - Inertia (within-cluster variance)
  - Silhouette score (separation quality)
  - Iterations to converge

### 💻 Code Preview
- Toggle to view Python source code
- Shows sklearn KMeans implementation
- Syntax highlighted with line numbers

### 🎨 Beautiful Design
- **Violet/pink/cyan** color scheme (multi-cluster theme)
- **Animated gradient orbs** in background
- **Gradient text** for emphasis
- **Smooth transitions** on k changes
- **Professional dark theme**

## 🚀 Getting Started

```bash
# Clone
git clone https://github.com/yourusername/kmeans-visualizer.git
cd kmeans-visualizer

# Install & run
npm install
npm run dev
```

## 📚 The Algorithm

### K-Means (Lloyd's Algorithm)

1. **Initialize** k centroids (using K-Means++ in this implementation)
2. **Assign** each point to nearest centroid
3. **Update** centroids to cluster means
4. **Repeat** until convergence

### K-Means++ Initialization

Selects initial centroids to be far apart:
1. Choose first centroid randomly
2. For each subsequent centroid:
   - Compute distance to nearest existing centroid
   - Choose next centroid with probability proportional to distance²

### Silhouette Coefficient

Measures how similar a point is to its own cluster vs other clusters:
```
s = (b - a) / max(a, b)
```
Where:
- **a** = average distance to same cluster
- **b** = minimum average distance to other clusters

## 🛠️ Technology Stack

- **React 19** - UI framework
- **TypeScript** - Type safety
- **Vite** - Build tool
- **Tailwind CSS** - Styling
- **Lucide React** - Icons
- **Custom SVG** - Data visualization
- **Custom K-Means** implementation in TypeScript

## 📁 Project Structure

```
kmeans-visualizer/
├── src/
│   ├── data/
│   │   └── kmeansData.ts  # K-Means algorithm & data
│   ├── App.tsx            # Main dashboard
│   ├── main.tsx           # Entry point
│   └── index.css          # Global styles
├── index.html
├── package.json
└── README.md
```

## 🎓 Educational Value

### Learn About
- **Unsupervised learning** concepts
- **K-Means algorithm** mechanics
- **K-Means++** initialization benefits
- **Elbow method** for choosing k
- **Silhouette analysis** for cluster quality
- **Standardization** importance

### Key Insights
- K-Means converges to local optima
- K-Means++ gives better initial centroids
- Elbow method helps choose optimal k
- Silhouette score validates clustering quality
- Standardization is crucial for distance-based methods

## 🔬 Algorithm Details

### Time Complexity
**O(n · k · i · d)** where:
- n = number of points
- k = number of clusters
- i = iterations
- d = dimensions

### Space Complexity
**O(n · d + k · d)** for data and centroids

### Features Used
- **Mean Radius**: Size of the cell nucleus
- **Mean Texture**: Standard deviation of gray-scale values

### Preprocessing
- **StandardScaler**: Zero mean, unit variance per feature
- Critical for distance-based algorithms

## 🤝 Contributing

Contributions welcome!

### Ideas for Enhancement
- [ ] Load full 30-feature dataset
- [ ] PCA for higher dimensions
- [ ] Compare with hierarchical clustering
- [ ] DBSCAN implementation
- [ ] Animated convergence process
- [ ] Export cluster assignments
- [ ] Custom dataset upload

## 📝 License

MIT License

---

**Made with ❤️ for educational purposes**

*Understanding unsupervised learning through beautiful visualizations*
