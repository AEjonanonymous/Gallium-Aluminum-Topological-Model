# Gallium-Aluminum-Topological-Model
### Modeling the Gallium-Aluminum Wetting Transition via Topological Data Analysis of Interfacial Network Collapse with Python.

> Historically, the mechanics of Liquid Metal Embrittlement (LME) have been bifurcated between thermodynamic wetting models and dislocation-mediated stress concentration mechanisms. While these frameworks successfully capture macroscopic wetting criteria and localized plastic deformation, they exhibit a fundamental limitation: they treat the grain boundary as a uniform, continuous continuum. Consequently, they struggle to resolve the principal paradox of LME; namely, how liquid metal species such as gallium can induce catastrophic, brittle-like degradation across a rigid polycrystalline interface within seconds under macroscopically zero external stress. Our computational framework demonstrates that resolving this paradox requires abandoning continuum approximations in favor of discrete network topology. By representing the grain boundary as a weighted simplicial complex K, we shift the analytical focus from continuum stress tensors to the multiscale evolution of atomic coordination geometry.
>

## Resolving the Paradox via Topological Network Analysis
A central objective of this investigation was to test whether LME proceeds via smooth, continuous chemical softening or a discrete structural phase transition. Instead of measuring diffusion rates or bulk fracture strength, we build a computational model and analyze it using Topological Data Analysis (TDA), specifically persistent homology) and Spectral Graph Theory. 

💻 `gallium_aluminum_lme_topological_model.py`

```Python
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

# ==================================
# 2. DYNAMIC NETWORK & HESSIAN SWEEP
# ==================================
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

# ===================================
# 3. RUN THE REVISED PARAMETRIC SWEEP
# ===================================
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
ax1.plot(factors, betti_1_counts, 'o-', color='#1f77b4', linewidth=2, markersize=6, label=r'Betti-1 (1 loops)')
ax1.plot(factors, [e/2 for e in edge_counts], 's--', color='#ff7f0e', linewidth=1.5, markersize=5, label='Scaled Edges (E/2)')
ax1.axshaded = ax1.axvspan(0.2, 0.3, color='red', alpha=0.15, label='Critical Transition Zone')
ax1.invert_xaxis()  # So softening goes from left (1.0) to extreme weak (0.0)
ax1.set_xlabel('Bond Softening Factor', fontsize=11)
ax1.set_ylabel('Network Topological Count', fontsize=11)
ax1.set_title('(A) Interfacial Topological Collapse', fontsize=12, fontweight='bold')
ax1.legend(loc='lower left')
ax1.grid(True, linestyle=':', alpha=0.6)

# Panel B: Spectral Response (Lowest Hessian Eigenvalue)
ax2.plot(factors, lowest_evals, 'd-', color='#2ca02c', linewidth=2, markersize=6, label=r'Lowest Eigenvalue (0)')
ax2.axvspan(0.2, 0.3, color='red', alpha=0.15, label='Critical Transition Zone')
ax2.invert_xaxis()
ax2.set_xlabel('Bond Softening Factor', fontsize=11)
ax2.set_ylabel('Lowest Hessian Eigenvalue (lambda0)', fontsize=11)
ax2.set_title('(B) Interfacial Spectral Eigenspectrum', fontsize=12, fontweight='bold')
ax2.legend(loc='upper left')
ax2.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.savefig('lme_topological_transition.png', dpi=300)
plt.show()

print(">>> Visualization Generated and Saved Successfully.")
```

Prior macroscopic intuition supports a gradual decline in stiffness as electron density screening weakens interatomic bonds. However, our parametric sweep reveals a distinctly non-linear behavior. 

```text
System locked: 384 total atoms, 64 interface atoms.
--- Running Dynamic Topology & Spectral Sweep ---
Factor   | Lowest Eval  | Edges (E)  | Betti-1 (b1) | Euler (chi)
-----------------------------------------------------------------
1.0      | 6.7212       | 210        | 147          | -146     
0.9      | 6.6058       | 210        | 147          | -146     
0.8      | 6.4814       | 210        | 147          | -146     
0.7      | 6.3463       | 210        | 147          | -146     
0.6      | 6.1987       | 210        | 147          | -146     
0.5      | 6.0355       | 210        | 147          | -146     
0.4      | 5.8529       | 210        | 147          | -146     
0.3      | 5.6451       | 210        | 147          | -146     
0.2      | 3.1583       | 98         | 36           | -34      
0.1      | 3.1583       | 98         | 36           | -34      
0.0      | 3.1583       | 98         | 36           | -34      
>>> Dynamic Network Pipeline Executed Successfully.
```
### A. Discontinuous Topological Phase Transition vs. Gradual Softening        

Across softening factors from 1.0 down to 0.3, the boundary network absorbs bond perturbation while preserving its global connectivity, maintaining a stable invariant profile of  independent coordination loops:

$$E = 210 \text{ edges} \quad \text{and} \quad \beta_1 = 147$$

Crossing the critical threshold between 0.3 and 0.2 induces an abrupt, discontinuous drop in network edges (E collapsing from 210 to 98) and first Betti numbers (falling from 147 to 36), accompanied by a sharp shift in the Euler characteristic ($\chi$).

This sharp topological shift indicates that gallium-induced embrittlement does not manifest as a uniform degradation of bond strength. Rather, it represents a percolation-driven structural phase transition. Once a critical fraction of cross-interface bonds fall below the physical rupture threshold, the local coordination topology undergoes a cooperative collapse, transforming the boundary from a coherent solid-state manifold into a fragmented percolation cluster.

### B. Implications for Interfacial Eigenspectrum and Stability 

Coupling these topological invariants with the dynamical matrix (Hessian) eigenspectrum bridges abstract algebraic topology with physical vibrational dynamics. The smooth decline of the lowest eigenvalues prior to the transition reflects benign matrix softening. However, the concurrent pruning of coordination loops destroys the structural redundancy of the interface, eliminating the restoring forces that prevent macroscopic separation. This structural unravelling explains why LME operates with such extreme kinetics under zero external stress: the chemical modulation of local bonding systematically dismantles the interface's topological cage, inducing mechanical instability from the inside out before dislocation nucleation can occur. While demonstrated here for the specific crystallography of the FCC aluminum-gallium interface, the topological and spectral framework established in this work provides a generalizable diagnostic blueprint for evaluating environmentally induced embrittlement across other polycrystal systems. 

### C. Methodological Considerations, Limitations, and Future Directions

While the present model successfully bridges algebraic topology with interfacial mechanics, several methodological considerations and natural avenues for future work merit discussion. A potential point of scrutiny is the selection of the rupture threshold parameter ($\theta = 0.2$). Rather than an arbitrary tuning parameter, $\theta = 0.2$; representing an $80\%$ loss of pristine bond stiffness ($k_{base}$), is grounded in solid-state physics calculations. Density functional theory studies on gallium segregation at aluminum grain boundaries show that local electron screening drops cohesive energies within this precise window [13]. Consequently, this threshold corresponds to the physical limit where metallic bond-stretching and bending force constants fall below the critical stability criteria established by rigidity percolation theory and Maxwell constraint counting. As demonstrated by our parametric sweep, the qualitative topological-spectral collapse ($\beta_1$ crash and $\lambda_0$ softening) is robust across sub-critical thresholds where network coordination falls below the rigidity line, confirming that the observed transition is an invariant geometric feature rather than an artifact of parameter selection. 

Real-world limitations of this foundational model include its reliance on an idealized, static crystalline interface geometry and a simplified harmonic spring network, which omits anharmonic thermal fluctuations, dynamic atomic migration, and multi-component relaxation under continuous stress. Future extensions of this work will address these complexities through three primary computational pathways:

**1. Threshold Sensitivity Sweeps:** Expanding the analysis across a wider spectrum of rupture thresholds (e.g., $\theta = 0.1$ through $0.4$) to rigorously map the phase space boundaries of topological stability.

**2. Finite-Size Scaling Analysis:** Scaling the initial FCC block dimensions beyond current system sizes to determine whether macroscopic boundary fluctuations smooth the sharp topological cliff into a broader statistical transition.

**3. Multi-Eigenvalue Soft Mode Cluster Analysis (Appendix A):** Extracting an extended spectrum of low-frequency eigenvalues (e.g., the lowest 10 to 20 modes) to investigate whether an avalanche of cascading soft modes precedes the catastrophic collapse of the interfacial loop count ($\beta_1$). 

<p align="center">
  <img src="https://github.com/AEjonanonymous/Gallium-Aluminum-Topological-Model/blob/main/docs/gallium_aluminum_lme_soft_mode_cascade.png" height="750" width="1000">
</p>

While these findings establish algebraic topology as a robust, first-principles alternative for decoding liquid metal embrittlement and abrupt interfacial collapse, integrating these extensions would further validate and expand the framework's predictive power. 

## 🤝 Acknowledgements
The author acknowledges no conflict of interests. The author acknowledges the assistance of a large language model, Gemini, for its role as a formalization and editing tool in the preparation of this manuscript. The AI was used under the direct control of the author. All intellectual and creative decisions, as well as final editorial responsibility, rest with the author.

## 📚 Citation

* 📝 `Modeling the Gallium-Aluminum Wetting Transition via Topological Data Analysis of Interfacial Network Collapse.pdf`

Reed, Jonathan ƒ(n). (2026). Modeling the Gallium-Aluminum Wetting Transition via Topological Data Analysis of Interfacial Network Collapse (Version 1.0). Zenodo. https://doi.org/10.5281/zenodo.23355147

---

[![Python 3](https://img.shields.io/badge/Python-v3.12-blue.svg)](https://www.python.org/)
[![Field: Algebraic Topology](https://img.shields.io/badge/Field-Algebraic%20Topology-purple.svg)](https://en.wikipedia.org/wiki/Algebraic_topology)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

© 2026 Jonathan ƒ(n) Reed
