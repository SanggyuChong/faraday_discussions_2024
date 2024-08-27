from LE_ACE import LE_ACE
import torch
from extract import extract_environment
from embed_FG import get_local_cube_frame
from embed_SC import extract_local_sphere, embed_into_HS_struc
import tqdm
import sys
import ase.io

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


eval_dataset = ase.io.read("MD_structure.xyz", ":")


dia = ase.io.read("dia.xyz")  # diamond config taken from materials project (#66)
dia.set_positions(
    dia.positions - dia.positions[0]
)
dia.wrap()


features_per_atom = le_ace.compute_features(eval_dataset, per_atom=True)
features_dia = le_ace.compute_features([dia], per_atom=True)

argmin = 13

def get_structure_and_atom(structures, index):
    n_atoms = 0
    for s in structures:
        if index - n_atoms < len(s):
            return s, index - n_atoms
        n_atoms += len(s)
    raise ValueError("Index out of range")

s, a = get_structure_and_atom(eval_dataset, argmin)

n_structures_list = [1, 2, 5, 10]
max_strucs = 10

r_cut = 3.75
contact_dist = 1.0
sigma = float(sys.argv[1])
buffer_length = float(sys.argv[2])

amo_embeddings = get_local_cube_frame(s, a, r_cut, buffer_length, contact_dist, sigma, max_strucs)[0]
hs_embeddings = embed_into_HS_struc(dia, s, a, r_cut, buffer_length, contact_dist, sigma, max_strucs)[0]
cluster_carvings = [extract_environment(s, a, 3.75)] * max_strucs


features_carve = le_ace.compute_features(cluster_carvings, per_atom=True)
features_amo = le_ace.compute_features(amo_embeddings, per_atom=True)
features_hs = le_ace.compute_features(hs_embeddings, per_atom=True)

diff_carve = features_carve - features_per_atom[argmin]
diff_amo = features_amo - features_per_atom[argmin]
diff_hs = features_hs - features_per_atom[argmin]

print(min(torch.norm(diff_carve, dim=1)))
print(min(torch.norm(diff_amo, dim=1)))
print(min(torch.norm(diff_hs, dim=1)))


diff_dia_hs = features_hs - features_dia[0]
print(min(torch.norm(diff_dia_hs, dim=1)))

