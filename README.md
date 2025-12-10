# GNN Demo 3: Generative Defect Healing & Crystal Restoration

> **DISCLAIMER:** This repository is a technical proof-of-concept developed to demonstrate the architectural implementation of Generative Graph Neural Networks (GNNs) for materials science. It focuses on the application of geometric deep learning to "inverse" problems—mapping disordered states back to equilibrium—rather than standard property prediction.

---

## 1. Project Overview
This repository implements a **Generative Denoising GNN** designed to "heal" structural defects in crystalline materials. 

While standard GNNs in computational chemistry are **discriminative** (predicting a scalar property $y$ from a structure $X$), this model is **generative**: it learns a local displacement field to map disordered atomic environments ($X_{noisy}$) back to their ground-state crystalline lattice ($X_{clean}$).

By treating the "healing" process as a denoising task, the model effectively learns the **Force Field** of the crystal lattice without requiring explicit energy labels. This approach serves as a foundational step toward **defect foundation models**, capable of identifying and correcting non-equilibrium structures in real-time molecular dynamics simulations.

### Key Objectives
* **Generative Restoration:** Train a model to reverse entropy. The network acts as a Maxwell's Demon, sorting atoms from high-disorder states back to low-disorder lattice sites.
* **Local Action:** Interactions are strictly local (defined by radial cutoffs), ensuring the model scales linearly $O(N)$ and can be deployed on systems of arbitrary size.
* **Vacuum Stability:** A critical challenge in generative modeling for crystals is distinguishing between "real" vacuum (a vacancy defect) and "fake" vacuum (noise). This model successfully preserves structural vacancies while healing the lattice around them.

---

## 2. Key Features

* **Equivariant Vector Output:** The model predicts vector displacements ($\Delta \mathbf{r}$) constructed from the geometric difference of atomic positions, ensuring that if the crystal rotates, the restoration vectors rotate with it.
* **RBF-Interaction Layer:** Standard MLPs struggle with the continuous nature of atomic distances. We utilize a **Gaussian Radial Basis Function (RBF)** expansion to project continuous distances into a high-dimensional feature space, acting as a "soft histogram" of the local environment.
* **Vacancy Preservation:** Unlike naive denoising which might collapse a hole, this model learns the *local symmetry* of the fcc lattice. It restores the lattice structure *around* the vacancy, preserving the defect as a stable feature.
* **Periodic Boundary Conditions (PBC):** Custom logic ensures that edge atoms correctly perceive neighbors across the simulation box boundaries, preventing surface artifacts.

---

## 3. Repository Structure

```text
gnn_demo_3_generative/
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
The training script loads clean crystal structures, exploits a data augmentation pipeline to inject synthetic Gaussian noise on-the-fly, and trains the GNN to reverse it.
```bash
python train_gen_2.py
```
* **Input:** `data/training_data.xyz`
* **Output:** `best_denoise_model.pth`
* **Hyperparameters:** Noise Scale $\sigma \in [0.01, 0.10]$, Learning Rate $= 10^{-3}$.

### Step 2: Run the Healing Demo (Inference)
The demo script creates a **Vacancy Defect** (removes one atom), applies severe damage to the surrounding lattice, and uses the trained model to "snap" the atoms back into place without filling the hole.
```bash
python final_demo.py
```
* **Output:** `final_demo_vacancy_healing.lammpstrj`

### Step 3: Visualization
The output is a LAMMPS trajectory file. Open the generated `.lammpstrj` file in **OVITO** or **VMD** to watch the lattice heal in real-time.

---

## 5. Technical Details (The Math)

The architecture is motivated by **Denoising Score Matching**, where the goal is to learn the gradient of the data distribution.

### 5.1 The Generative Task
We frame the healing process as learning a displacement vector field $\Delta \mathbf{r}_i$ for every atom $i$. The model minimizes the difference between the predicted restoration vector and the inverse of the applied noise:

$$ \mathcal{L} = \frac{1}{N} \sum_{i=1}^N \| \mathbf{f}_{\theta}(\mathbf{r}_i + \mathbf{\epsilon}) - (-\mathbf{\epsilon}) \|^2 $$

**Variable Definitions:**
* $\mathbf{r}_i$: Ground truth position.
* $\mathbf{\epsilon}$: Injected Gaussian noise (the "damage").
* $\mathbf{f}_{\theta}$: GNN prediction.
* **Target:** The model must predict $-\mathbf{\epsilon}$ to cancel the noise.

### 5.2 RBF-Interaction Architecture
To maintain rotational invariance and handle periodic boundary conditions (PBC) rigorously, the network uses a **Radial Basis Function (RBF) Expansion**.

**1. Pairwise Distances**
First, we calculate Euclidean distances $d_{ij}$ respecting the Minimum Image Convention:

$$ \vec{\delta}_{ij} = (\vec{x}_j - \vec{x}_i) - \text{Box} \cdot \text{round}\left( \frac{\vec{x}_j - \vec{x}_i}{\text{Box}} \right) $$

**2. Gaussian Expansion**
Next, we expand these distances into a learnable basis set:

$$ \phi_k(d_{ij}) = \exp(-\gamma (d_{ij} - \mu_k)^2) $$

**3. Vector Aggregation**
Finally, the displacement for atom $i$ is the weighted sum of unit vectors to its neighbors, scaled by the learned interaction weights $w_{ij}$:

$$ \Delta \mathbf{r}_i = \sum_{j \in \mathcal{N}(i)} w_{ij} \cdot \frac{\mathbf{r}_j - \mathbf{r}_i}{\| \mathbf{r}_j - \mathbf{r}_i \|} $$

This architecture ensures that the predicted displacements are always geometrically consistent with the local environment.

---

## 6. Dependencies
* **Python 3.8+**
* **PyTorch:** Core deep learning framework.
* **NumPy:** Tensor manipulations and data loading.

```bash
pip install torch numpy
```
