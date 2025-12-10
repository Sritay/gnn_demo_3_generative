# GNN Demo 3: Generative Defect Healing

> **Note:** This repository serves as a code demonstration for the Research Associate position in "Defect-Tolerant Materials for Energy" at Imperial College London.

---

## 1. Project Overview
This repository implements a **Generative Denoising GNN** designed to "heal" structural defects in crystalline materials. Unlike regression models that predict a global property (e.g., Energy) from a structure, this model learns a **local displacement field** to map disordered atomic environments back to their ground-state crystalline lattice.

This approach serves as a foundational step toward **defect foundation models**, capable of identifying and correcting non-equilibrium structures in real-time molecular dynamics simulations.

### Key Objectives
* **Generative Restoration:** Train a model to reverse entropy by mapping noisy coordinates $X_{noisy}$ to clean crystal coordinates $X_{clean}$.
* **Local Action:** Learn strictly local atomic interactions (via RBF cutoffs) to ensure linear scaling $O(N)$ with system size.
* **Vacuum Stability:** Demonstrate the ability to distinguish between "real" vacuum (a vacancy) and "fake" vacuum (noise), preserving the defect while healing the lattice around it.

---

## 2. Methodology

### 2.1 The Generative Task
We frame the healing process as learning a displacement vector field $\Delta \mathbf{r}_i$ for every atom $i$. The model acts as a learned force field that minimizes the difference between the current state and the ground truth:

$$ \mathcal{L} = \frac{1}{N} \sum_{i=1}^N \| \mathbf{f}_{\theta}(\mathbf{r}_i + \mathbf{\epsilon}) - (-\mathbf{\epsilon}) \|^2 $$

Where:
* $\mathbf{r}_i$ is the ground truth position.
* $\mathbf{\epsilon}$ is injected Gaussian noise (the "damage").
* $\mathbf{f}_{\theta}$ is the GNN prediction.
* Target: The model must predict $-\mathbf{\epsilon}$ to cancel the noise.

### 2.2 RBF-Interaction Architecture
To maintain rotational invariance and handle periodic boundary conditions (PBC) rigorously, the network uses a **Radial Basis Function (RBF) Expansion**:

* **Pairwise Distances:** Calculate Euclidean distances $d_{ij}$ respecting the Minimum Image Convention.

* **Gaussian Expansion:** Expand distances into a high-dimensional feature vector:
  $$ \phi_k(d_{ij}) = \exp(-\gamma (d_{ij} - \mu_k)^2) $$

* **Learned Interaction Weights:** An MLP transforms these geometric features into scalar interaction weights $w_{ij}$.

* **Vector Aggregation:** The final displacement for atom $i$ is the weighted sum of unit vectors to its neighbors:
  $$ \Delta \mathbf{r}_i = \sum_{j \in \mathcal{N}(i)} w_{ij} \cdot \frac{\mathbf{r}_j - \mathbf{r}_i}{\| \mathbf{r}_j - \mathbf{r}_i \|} $$

This architecture ensures that the predicted displacements are always geometrically consistent with the local environment.

---

## 3. Repository Structure

```
gnn_demo_3/
├── data/
│   └── training_data.xyz       # Ground truth platinum crystal structures
├── train_gen_2.py              # Denoising Autoencoder training logic
├── final_demo.py               # Vacancy healing simulation (Inference)
├── best_denoise_model.pth      # Pre-trained model weights
├── final_demo_vacancy_healing.lammpstrj  # Output trajectory (Visualizable in OVITO)
└── README.md                   # Project documentation
```

---

## 4. Usage Instructions

### Step 1: Train the Denoising Model
The training script loads clean crystal structures, injects synthetic noise on-the-fly, and trains the GNN to reverse it.
```bash
python train_gen_2.py
```
* **Input:** `data/training_data.xyz`
* **Output:** `best_denoise_model.pth`
* **Hyperparameters:** $\sigma \in [0.01, 0.10]$ (Noise Scale), Learning Rate $= 10^{-3}$.

### Step 2: Run the Healing Demo
The demo script creates a **Vacancy Defect** (removes one atom), applies severe damage to the surrounding lattice, and uses the trained model to "snap" the atoms back into place without filling the hole.
```bash
python final_demo.py
```
* **Output:** `final_demo_vacancy_healing.lammpstrj`

### Step 3: Visualization
Open the generated `.lammpstrj` file in **OVITO** or **VMD** to watch the lattice heal in real-time.

---

## 5. Implementation Details & Choices

### Why RBF Expansion?
Standard Multi-Layer Perceptrons (MLPs) struggle with the continuous nature of atomic distances. By expanding distances into a Gaussian basis set (implemented in `GaussianSmearing`), we provide the neural network with a "soft histogram" of the local environment. This allows the model to easily distinguish between first-neighbor shells (bonding) and second-neighbor shells.

### Periodic Boundary Conditions (PBC)
Handling PBC is critical for bulk materials. The custom `get_pbc_displacement` function ensures that edge atoms correctly perceive neighbors on the opposite side of the simulation box. This prevents surface artifacts and ensures the model is valid for bulk crystal simulations.

### Vacancy Preservation
A naive denoising model might try to "fill" the vacancy by pulling all atoms toward the center. However, by training on **local** relative displacements rather than absolute positions, the model learns the *local symmetry* of the fcc lattice. Consequently, it restores the lattice structure *around* the vacancy, preserving the defect as a stable feature rather than treating it as noise to be erased.

---

## 6. Dependencies
* **Python 3.8+**
* **PyTorch:** Core deep learning framework.
* **NumPy:** Tensor manipulations and data loading.

```bash
pip install torch numpy
```
