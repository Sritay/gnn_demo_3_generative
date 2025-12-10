import torch
import numpy as np
from typing import TextIO
from train_gen_2 import DenoiseGNN, BOX_SIZE, load_clean_structures

# --- Config ---
MODEL_PATH = 'best_denoise_model.pth'
OUTPUT_FILE = 'final_demo_vacancy_healing.lammpstrj'
INPUT_DATA_PATH = 'data/training_data.xyz'
HEALING_STEPS = 40
UPDATE_RATE = 0.6  # Elevated update rate for visualization efficiency

def get_pbc_displacement(pos1: torch.Tensor, pos2: torch.Tensor, box_size: float) -> torch.Tensor:
    """Calculates the minimum image displacement vector from pos1 to pos2."""
    diff = pos2 - pos1
    diff = diff - box_size * torch.round(diff / box_size)
    return diff

def save_lammps_frame(file_handle: TextIO, atoms: np.ndarray, step_num: int) -> None:
    file_handle.write("ITEM: TIMESTEP\n")
    file_handle.write(f"{step_num}\n")
    file_handle.write("ITEM: NUMBER OF ATOMS\n")
    file_handle.write(f"{len(atoms)}\n")
    file_handle.write("ITEM: BOX BOUNDS pp pp pp\n")
    file_handle.write(f"0.0 {BOX_SIZE}\n")
    file_handle.write(f"0.0 {BOX_SIZE}\n")
    file_handle.write(f"0.0 {BOX_SIZE}\n")
    file_handle.write("ITEM: ATOMS id type x y z\n")
    for i, atom in enumerate(atoms):
        file_handle.write(f"{i+1} 1 {atom[0]:.4f} {atom[1]:.4f} {atom[2]:.4f}\n")

def run_final_demo():
    print(f"Loading model from {MODEL_PATH}...")
    model = DenoiseGNN()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    
    box_dims = torch.tensor([BOX_SIZE, BOX_SIZE, BOX_SIZE])

    # 1. Load Perfect Crystal
    data = load_clean_structures(INPUT_DATA_PATH)
    pos_clean = data[0]

    # 2. Create the "Vacancy Scenario"
    # Find center atom
    center = torch.mean(pos_clean, dim=0)
    dists_from_center = torch.norm(pos_clean - center, dim=1)
    center_idx = torch.argmin(dists_from_center)
    
    # Identify neighbors (atoms close to the center)
    neighbor_mask = dists_from_center < 4.0 
    
    # Remove the center atom (Vacancy)
    mask = torch.ones(len(pos_clean), dtype=torch.bool)
    mask[center_idx] = False
    
    pos_start = pos_clean[mask]
    neighbor_mask = neighbor_mask[mask] # Adjust mask for removed atom
    
    # 3. Add Selective Noise
    # Background atoms: Tiny noise (looks like crystal)
    # Defect neighbors: High noise (looks like local damage)
    noise = torch.randn_like(pos_start)
    
    # Scale noise: 0.02 for bulk, 0.20 for defect region
    sigma_per_atom = torch.ones(len(pos_start), 1) * 0.02
    sigma_per_atom[neighbor_mask] = 0.20 
    
    pos = pos_start + noise * sigma_per_atom
    pos = torch.remainder(pos, BOX_SIZE)

    print(f"Created demo with 1 Vacancy and localized damage.")
    print(f"Saving to {OUTPUT_FILE}...")

    # 4. Run Healing
    with open(OUTPUT_FILE, 'w') as f_out:
        # Save Frame 0
        save_lammps_frame(f_out, pos.numpy(), 0)
        
        for step in range(1, HEALING_STEPS + 1):
            with torch.no_grad():
                # Model Predicts Correction
                displacement = model(pos, box_dims)
            
            # Apply update
            pos = pos + (displacement * UPDATE_RATE)
            pos = torch.remainder(pos, BOX_SIZE)
            
            save_lammps_frame(f_out, pos.numpy(), step)
            
            # Calculate REAL error (PBC aware)
            # We compare to the CLEAN crystal (minus the vacancy atom)
            real_diff = get_pbc_displacement(pos, pos_clean[mask], BOX_SIZE)
            mae = real_diff.norm(dim=1).mean().item()
            
            if step % 5 == 0:
                print(f"Step {step:02d} | Avg Error: {mae:.4f} A")

    print("\nDONE. Open 'final_demo_vacancy_healing.lammpstrj' in OVITO.")
    print("You will see a clean crystal with a wobbly hole in the middle that snaps shut.")

if __name__ == "__main__":
    run_final_demo()
