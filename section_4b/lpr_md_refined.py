from LE_ACE import LE_ACE
import torch
from extract import extract_environment
from embed_FG import get_local_cube_frame
from embed_SC import extract_local_sphere, embed_into_HS_struc
import tqdm
import sys

import ase.io
import numpy as np
np.random.seed(123)

torch.set_default_dtype(torch.float64)

device = "cuda" if torch.cuda.is_available() else "cpu"

batch_size = 10
alpha = 1e-5

le_ace = LE_ACE(
    r_cut_rs=3.7,
    r_cut=3.7,
    E_max=[0.0, 1000.0, 300.0, 200.0],
    all_species=[6],
    le_type="physical",
    factor=1.5,
    factor2=-1.0,
    cost_trade_off=False,
    fixed_stoichiometry=False,
    is_trace=False,
    n_trace=-1,
    device=device
)
n_feat = sum(tensor.shape[0] for tensor in le_ace.extended_LE_energies)

def get_batches(list: list, batch_size: int) -> list:
    batches = []
    n_full_batches = len(list)//batch_size
    for i_batch in range(n_full_batches):
        batches.append(list[i_batch*batch_size:(i_batch+1)*batch_size])
    if len(list) % batch_size != 0:
        batches.append(list[n_full_batches*batch_size:])
    return batches

def add_features_to_covariance(calculator, batch, covariance):
    features = calculator.compute_features(batch)
    features /= features[:, 0].clone().unsqueeze(1) # take per-atom features
    covariance += features.T @ features


full_dataset = ase.io.read("datasets/volker_carbon/train.xyz", ":")

np.random.shuffle(full_dataset)
training_set = full_dataset[:500]

covariance = torch.zeros(n_feat, n_feat, device=device)
for batch in get_batches(training_set, batch_size):
    add_features_to_covariance(le_ace, batch, covariance)
# covariance /= n_atoms_in_training_set  # this step makes sure that the regularizer has the same effect as later
covariance = covariance + alpha * torch.eye(covariance.shape[0], device=covariance.device, dtype=covariance.dtype)
inv_covariance = torch.linalg.inv(covariance)

eval_dataset = ase.io.read("MD_structure.xyz", ":")

dia = ase.io.read("dia.xyz")  # diamond config taken from materials project (#66)
dia.set_positions(
    dia.positions - dia.positions[0]
)
dia.wrap()

features_per_atom = le_ace.compute_features(eval_dataset, per_atom=True)

lprs = 1.0 / torch.einsum("ij, jk, ik -> i", features_per_atom, inv_covariance, features_per_atom)
argmin = int(torch.argmin(lprs))
initial_lpr = lprs[argmin].item()

features = le_ace.compute_features(eval_dataset, per_atom=False)
pr = 1.0 / torch.einsum("ij, jk, ik -> i", features, inv_covariance, features)
initial_pr = pr.item()

def get_structure_and_atom(structures, index):
    n_atoms = 0
    for s in structures:
        if index - n_atoms < len(s):
            return s, index - n_atoms
        n_atoms += len(s)
    raise ValueError("Index out of range")

s, a = get_structure_and_atom(eval_dataset, argmin)

n_structures_list = [1, 2, 5, 10, 20, 50, 100]
max_strucs = 100

r_cut_sg = 3.75
contact_dist = 1.0
sigma = float(sys.argv[1])
buffer_length = float(sys.argv[2])

amo_embeddings = get_local_cube_frame(s, a, r_cut_sg, buffer_length, contact_dist, sigma, max_strucs)[0]
hs_embeddings = embed_into_HS_struc(dia, s, a, r_cut_sg, buffer_length, contact_dist, sigma, max_strucs)[0]
cluster_carvings = [extract_environment(s, a, r_cut_sg)] * max_strucs

 
methods = ["orig_struc", "carve", "embed_amo", "embed_hs"]
results_lpr = {method: [] for method in methods}
results_pr = {method: [] for method in methods}

for n_structures in n_structures_list:

    for method in methods:

        print(method)

        if method == "carve":
            add_set = cluster_carvings[:n_structures]

        elif method == "embed_amo":
            add_set = amo_embeddings[:n_structures]

        elif method == "embed_hs":
            add_set = [dia] + hs_embeddings[:n_structures-1]

        temp_covariance = covariance.clone()

        if method == "orig_struc":
            traj_features = le_ace.compute_features(eval_dataset)
            traj_features /= traj_features[:, 0].clone().unsqueeze(1) # take per-atom features
            temp_covariance += (traj_features.T @ traj_features) * n_structures

        else:
            for batch in get_batches(add_set, batch_size):
                add_features_to_covariance(le_ace, batch, temp_covariance)

        inv_covariance = torch.linalg.inv(temp_covariance)

        features_per_atom = le_ace.compute_features(eval_dataset, per_atom=True)
        lprs = 1.0/torch.einsum("ij, jk, ik -> i", features_per_atom, inv_covariance, features_per_atom)
        final_lpr = lprs[argmin].item()
        results_lpr[method].append(final_lpr/initial_lpr)

        features = le_ace.compute_features(eval_dataset, per_atom=False)
        pr = 1.0/torch.einsum("ij, jk, ik -> i", features, inv_covariance, features)
        final_pr = pr.item()
        results_pr[method].append(final_pr/initial_pr)

print(results_pr)
print(results_lpr)

for method in methods:
    with open(f"results_{method}_sigma{sigma}_buffer{buffer_length}.txt", "w") as f:
        for n_structures, lpr, pr in zip(n_structures_list, results_lpr[method], results_pr[method]):
            f.write(f"{n_structures} {lpr} {pr}\n")
