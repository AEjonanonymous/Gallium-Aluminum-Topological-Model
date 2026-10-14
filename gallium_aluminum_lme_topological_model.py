# ===========
# MIT LICENSE 
# ===========

import numpy as np
from scipy.spatial import KDTree
from scipy.linalg import eigh
from scipy.sparse.csgraph import connected_components
import scipy.sparse as sp
import matplotlib.pyplot as plt

# ==================================
# 1. GENERATE CLEAN PLANAR INTERFACE
# ==================================
a0 = 4.05  # FCC lattice parameter for Aluminum in Angstroms

def generate_fcc_block(size_x, size_y, size_z):
    unit_cell = np.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5]
    ]) * a0
    
    atoms = []
    for x in range(size_x):
        for y in range(size_y):
            for z in range(size_z):
                shift = np.array([x, y, z], dtype=float) * a0
                atoms.append(unit_cell + shift)
    return np.vstack(atoms)

bottom_block = generate_fcc_block(4, 3, 4)
top_block = generate_fcc_block(4, 3, 4)
interface_plane_y = 3.0 * a0
top_block[:, 1] += interface_plane_y

all_atoms = np.vstack([bottom_block, top_block])
full_tree = KDTree(all_atoms)

interface_mask = (all_atoms[:, 1] > (interface_plane_y - 0.5 * a0)) & (all_atoms[:, 1] < (interface_plane_y + 0.5 * a0))
interface_indices = np.where(interface_mask)[0]
interface_atoms = all_atoms[interface_indices]

print(f"System locked: {len(all_atoms)} total atoms, {len(interface_atoms)} interface atoms.")


# ===================================
# 2. DYNAMIC NETWORK & HESSIAN SWEEP
# ===================================
def evaluate_interface_state(softening_factor, rupture_threshold=0.3):
    N = len(interface_atoms)
    H = np.zeros((3 * N, 3 * N))
    k_base = 10.0  
    cutoff_radius = 3.3
    
    # Build adjacency matrix dynamically based on bond survival
    # Edges are kept only if their effective spring constant > rupture threshold
    row_indices = []
    col_indices = []

    for i in range(N):
        pos_i = interface_atoms[i]
        neighbor_idxs = full_tree.query_ball_point(pos_i, r=cutoff_radius)
        
        for j_global in neighbor_idxs:
            if j_global == interface_indices[i]:
                continue
                
            pos_j = all_atoms[j_global]
            diff = pos_j - pos_i
            r = np.linalg.norm(diff)
            
            if r < 1e-5:
                continue
                
            unit_vec = diff / r
            
            # Check if bond crosses the interface plane
            is_cross_interface = (pos_i[1] < interface_plane_y and pos_j[1] >= interface_plane_y) or \
                                 (pos_i[1] >= interface_plane_y and pos_j[1] < interface_plane_y)
            
            current_k = k_base * softening_factor if is_cross_interface else k_base
            
            # PHYSICAL RUPTURE CRITERION: If bond strength drops below threshold, it snaps (edge removed)
            if current_k >= (k_base * rupture_threshold):
                # Add to Hessian
                sub_block = -current_k * np.outer(unit_vec, unit_vec)
                H[3*i:3*i+3, 3*i:3*i+3] -= sub_block
                if j_global in interface_indices:
                    j_local = np.where(interface_indices == j_global)[0][0]
                    H[3*i:3*i+3, 3*j_local:3*j_local+3] += sub_block
                
                # Track for Topology (if both atoms are in interface)
                if j_global in interface_indices:
                    j_local = np.where(interface_indices == j_global)[0][0]
                    row_indices.append(i)
                    col_indices.append(j_local)

    # Solve Eigenspectrum
    evals = eigh(H, eigvals_only=True)
    lowest_eval = np.sort(evals)[0]
    
    # Compute Topological Invariants (Betti Numbers) for this state
    data = np.ones(len(row_indices))
    adj_sparse = sp.csr_matrix((data, (row_indices, col_indices)), shape=(N, N))
    connectivity = (adj_sparse + adj_sparse.T > 0).astype(int)
    
    num_components, _ = connected_components(csgraph=connectivity, directed=False, return_labels=True)
    b0 = num_components
    num_edges = int(connectivity.nnz / 2)
    b1 = num_edges - N + b0
    euler_char = b0 - b1  # assuming beta_2 = 0 for 1-skeleton graph

    return lowest_eval, b0, b1, num_edges, euler_char


# ==================================
# 3. RUN THE REVISED PARAMETRIC SWEEP
# ==================================
softening_steps = np.linspace(1.0, 0.0, 11)

print(f"\n--- Running Dynamic Topology & Spectral Sweep ---")
print(f"{'Factor':<8} | {'Lowest Eval':<12} | {'Edges (E)':<10} | {'Betti-1 (b1)':<12} | {'Euler (chi)':<10}")
print("-" * 65)

for factor in softening_steps:
    low_ev, b0, b1, edges, chi = evaluate_interface_state(factor, rupture_threshold=0.2)
    print(f"{factor:<8.1f} | {low_ev:<12.4f} | {edges:<10} | {b1:<12} | {chi:<10}")

print(">>> Dynamic Network Pipeline Executed Successfully.")

# ==========================================
# 4. PLOTTING THE PHASE TRANSITION (MATPLOTLIB)
# ==========================================

# Collect the sweep data into arrays for plotting
factors = []
lowest_evals = []
edge_counts = []
betti_1_counts = []

for factor in softening_steps:
    low_ev, b0, b1, edges, chi = evaluate_interface_state(factor, rupture_threshold=0.2)
    factors.append(factor)
    lowest_evals.append(low_ev)
    edge_counts.append(edges)
    betti_1_counts.append(b1)

# Set up a clean, publication-style two-panel figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

# Panel A: Topological Invariants (The Cliff)
ax1.plot(factors, betti_1_counts, 'o-', color='#1f77b4', linewidth=2, markersize=6, label=r'Betti-1 ($\beta_1$ loops)')
ax1.plot(factors, [e/2 for e in edge_counts], 's--', color='#ff7f0e', linewidth=1.5, markersize=5, label='Scaled Edges ($E/2$)')
ax1.axshaded = ax1.axvspan(0.2, 0.3, color='red', alpha=0.15, label='Critical Transition Zone')
ax1.invert_xaxis()  # So softening goes from left (1.0) to extreme weak (0.0)
ax1.set_xlabel('Bond Softening Factor', fontsize=11)
ax1.set_ylabel('Network Topological Count', fontsize=11)
ax1.set_title('(A) Interfacial Topological Collapse', fontsize=12, fontweight='bold')
ax1.legend(loc='lower left')
ax1.grid(True, linestyle=':', alpha=0.6)

# Panel B: Spectral Response (Lowest Hessian Eigenvalue)
ax2.plot(factors, lowest_evals, 'd-', color='#2ca02c', linewidth=2, markersize=6, label=r'Lowest Eigenvalue ($\lambda_0$)')
ax2.axvspan(0.2, 0.3, color='red', alpha=0.15, label='Critical Transition Zone')
ax2.invert_xaxis()
ax2.set_xlabel('Bond Softening Factor', fontsize=11)
ax2.set_ylabel('Lowest Hessian Eigenvalue ($lambda_0$)', fontsize=11)
ax2.set_title('(B) Interfacial Spectral Eigenspectrum', fontsize=12, fontweight='bold')
ax2.legend(loc='upper left')
ax2.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.savefig('lme_topological_transition.png', dpi=300)
plt.show()

print(">>> Visualization Generated and Saved Successfully.")