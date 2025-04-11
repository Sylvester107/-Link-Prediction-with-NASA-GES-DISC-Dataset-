
# 🔍 Link Prediction Using GCN and Random Walk (PPR)


## 📁 **1. Data Preparation**

The dataset consists of:
- **Nodes file** with columns: `id`, `label`, `properties`
- **Edges (train, validation, test)** with columns: `source`, `target`, `relationship_type`

### Steps:
1. **Node Features**:
   - **Labels**: One-hot encoded (e.g., ‘Dataset’, ‘Publication’, etc.)
   - **Properties**: Encoded using TF-IDF on textual content (like abstract, name, year).
   - **Final Feature**: Concatenation of label vector + TF-IDF vector → Dense [N, D] matrix.

2. **Graph Representation (PyG)**:
   - Convert all edges to `edge_index` tensors of shape [2, num_edges].
   - Encode features into `x` matrix.
   - Ensure all source/target node IDs are within bounds (0 to 25023) and avoid hanging edges.

3. **Negative Sampling**:
   - Randomly generate node pairs not in the existing edge list.
   - Label: 1 (real edge), 0 (negative edge).

---

## 🧠 **2. Models**

### A. **Graph Convolutional Network (GCN)**

- **Architecture**:
  - `GCN Layer 1`: `ReLU` activation
  - `GCN Layer 2`: No activation → Outputs 32-dim embeddings per node
  - `Link Predictor (MLP)`: 
    - Linear → ReLU → Linear → Sigmoid

- **Input to MLP**:
  - Concatenated embeddings of source and target nodes → Shape [num_edges, 64]

- **Output**:
  - Probabilities ∈ [0, 1] → Likelihood of an edge existing between two nodes

---

### B. **Random Walk + Personalized PageRank**

- **Steps**:
  1. Convert graph to `networkx`
  2. For each node:
     - Perform `num_walks` walks of `walk_length`
     - Use `restart_probability` to restart walks at the origin node
  3. Collect visit stats → Normalize to get **proximity scores**
  4. Predict links based on similarity/proximity

- **Evaluation**:
  - Concatenate real and predicted links
  - Label real as 1, random as 0
  - Score using `ROC-AUC` and `Average Precision`

---

## 🎓 **3. Learning & Training**

### GCN Training Flow:

1. Forward:
   - Pass node features + edge index → GCN
   - Get embeddings `z`
   - For each (src, tgt) edge: concatenate `z[src]` + `z[tgt]`
   - Pass through MLP → Get probability via sigmoid

2. Loss:
   - Binary Cross Entropy between predicted probs and true labels

3. Backprop:
   - Compute gradients → Update GCN + MLP weights via optimizer

### Random Walk:
- No deep learning training
- Hyperparameters like walk length, restart probability, number of walks influence performance

---

## 🧪 **4. Evaluation**

| Model               | ROC-AUC (Eval) | ROC-AUC (Test) | Average Precision (Eval) | Average Precision (Test) |
|--------------------|----------------|----------------|---------------------------|---------------------------|
| GCN + MLP          | 0.9949         | 0.9963         | 0.9958                    | 0.9961                    |
| Random Walk (No tuning) | 0.6859         | 0.6874         | 0.6819                    | 0.6786                    |
| Random Walk (Tuned) | 0.7051         | 0.7103         | 0.7018                    | 0.7163                    |

---

## 💬 **5. Comments, Reflections & Insights**

### Strengths:
- **GCN** outperformed significantly due to its ability to learn from both **node features and graph structure**.
- **Random Walk** is lightweight and unsupervised, making it appealing for scenarios with limited resources.

### Challenges:
- TF-IDF on free-text properties produced **sparse representations**.
- Negative sampling required care to avoid bias in training.
- Random Walk performance was heavily dependent on **hyperparameters** (restart prob, walk length, etc.).



