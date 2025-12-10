import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple

# --- Configuration ---
DATA_PATH = 'data/training_data.xyz'
MODEL_SAVE_PATH = 'best_denoise_model.pth'
BOX_SIZE = 11.76
LEARNING_RATE = 0.001
EPOCHS = 100
BATCH_SIZE = 16
# We train on a mix of noise levels so the model handles both 
# "badly broken" and "slightly off" atoms.
SIGMA_MIN = 0.01
SIGMA_MAX = 0.10 

# --- Architecture: RBF Displacement GNN ---
class GaussianSmearing(nn.Module):
    """
    Expands distances into a set of Gaussian basis functions.
    """
    def __init__(self, start=0.0, stop=5.0, n_gaussians=50):
        super().__init__()
        offset = torch.linspace(start, stop, n_gaussians)
        self.coeff = -0.5 / ((stop - start) / (n_gaussians - 1))**2
        self.register_buffer('offset', offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))

def get_pbc_distances(pos: torch.Tensor, box_dims: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes pairwise displacement vectors and distances under PBC.
    """
    # Vector from j to i
    delta = pos.unsqueeze(0) - pos.unsqueeze(1) # Shape (N, N, 3)
    box = box_dims.view(1, 1, 3)
    delta = delta - box * torch.round(delta / box)
    dist = torch.sqrt((delta**2).sum(dim=2) + 1e-8)
    return delta, dist

class DenoiseGNN(nn.Module):
    """
    Graph Neural Network for denoising atomic coordinates via learned displacement fields.
    Predicts a correction vector for every atom based on its local environment.
    """
    def __init__(self, n_gaussians=50, hidden_dim=128):
        super(DenoiseGNN, self).__init__()
        self.rbf = GaussianSmearing(start=0.0, stop=6.0, n_gaussians=n_gaussians)
        
        # Input: RBF Features of neighbor distance
        self.mlp = nn.Sequential(
            nn.Linear(n_gaussians, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1) 
        )

    def forward(self, positions: torch.Tensor, box_dims: torch.Tensor) -> torch.Tensor:
        N = positions.shape[0]
        # delta_vecs[i, j] is vector from j -> i
        delta_vecs, dists = get_pbc_distances(positions, box_dims)
        
        # 1. RBF Expansion
        flat_dists = dists.view(-1)
        rbf_feats = self.rbf(flat_dists)
        
        # 2. Learn Pairwise Weights
        # "How much should neighbor j pull/push atom i?"
        weights = self.mlp(rbf_feats).view(N, N)
        
        # 3. Aggregate Displacement Vectors
        norm_delta = delta_vecs / (dists.unsqueeze(2) + 1e-8)
        
        # Mask self and far atoms
        mask = 1.0 - torch.eye(N).to(positions.device)
        cutoff_mask = (dists < 5.0).float()
        
        # Sum weighted vectors
        # If weight > 0, atom i moves AWAY from j (Repulsion)
        # If weight < 0, atom i moves TOWARDS j (Attraction)
        # The MLP learns the sign automatically to minimize loss.
        weighted_vectors = norm_delta * weights.unsqueeze(2) * mask.unsqueeze(2) * cutoff_mask.unsqueeze(2)
        displacement = torch.sum(weighted_vectors, dim=1)
        
        return displacement

def load_clean_structures(filename: str) -> List[torch.Tensor]:
    print(f"Loading ground truth crystals from {filename}...")
    with open(filename, 'r') as f:
        lines = f.readlines()
    structures = []
    lines_per_frame = 9 + 108
    for i in range(0, len(lines), lines_per_frame):
        atom_lines = lines[i + 9 : i + lines_per_frame]
        data = np.loadtxt(atom_lines)
        pos = torch.tensor(data[:, 2:5], dtype=torch.float32)
        structures.append(pos)
    return structures

def train_denoiser():
    clean_data = load_clean_structures(DATA_PATH)
    clean_data = clean_data[:50] # Train on first 50 frames
    box_dims = torch.tensor([BOX_SIZE, BOX_SIZE, BOX_SIZE])
    
    model = DenoiseGNN()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print("\n--- Starting Denoising Autoencoder Training ---")
    
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        np.random.shuffle(clean_data)
        
        for pos_clean in clean_data:
            optimizer.zero_grad()
            
            # 1. Create Input (Noisy) and Target (Displacement)
            sigma = torch.rand(1) * (SIGMA_MAX - SIGMA_MIN) + SIGMA_MIN
            noise_vector = torch.randn_like(pos_clean) * sigma
            pos_noisy = pos_clean + noise_vector
            
            # Ideally, the model should predict a vector that cancels the noise
            # Target Displacement = pos_clean - pos_noisy = -noise_vector
            target_displacement = -noise_vector
            
            # 2. Predict Displacement
            pred_displacement = model(pos_noisy, box_dims)
            
            # 3. Loss (MSE)
            loss = torch.mean((pred_displacement - target_displacement)**2)
            
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d}: MSE Loss = {epoch_loss / len(clean_data):.6f}")

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print("Training Complete.")

if __name__ == "__main__":
    train_denoiser()
