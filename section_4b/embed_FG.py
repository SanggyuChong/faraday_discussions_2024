import numpy as np
import ase

def extract_local_cube(frame, atom_idx, r_cut, buffer_length):

    # detect sphere that circumscribes the cube
    dists_from_atom = frame.get_distances(atom_idx, np.arange(len(frame)), mic=True)
    r_cut_eff = np.sqrt(3) * (r_cut + buffer_length)
    hit_list = (0 < dists_from_atom).astype(int) * (dists_from_atom < r_cut_eff).astype(int)
    neigh_idx = np.array([ii for ii, v in enumerate(hit_list) if v > 0])
    local_env_sphere = frame[np.concatenate([np.array([atom_idx]), neigh_idx])]

    # center the spherical env to the cell center
    local_env_sphere.set_positions(local_env_sphere.positions - local_env_sphere.positions[0])
    cell_center_pos = local_env_sphere.cell.cartesian_positions([0.5, 0.5, 0.5])
    local_env_sphere.set_positions(local_env_sphere.positions + cell_center_pos)
    local_env_sphere.wrap()

    # carve out the cube
    half_cube_l = r_cut + buffer_length
    hit_list = np.ones(len(local_env_sphere))
    for i in range(3):
        blw = cell_center_pos[i] - half_cube_l
        abv = cell_center_pos[i] + half_cube_l
        hit_list *= (local_env_sphere.positions[:, i] > blw).astype(int)
        hit_list *= (local_env_sphere.positions[:, i] < abv).astype(int)
    keep_idx = np.array([ii for ii, v in enumerate(hit_list) if v > 0])
    local_env_cube = local_env_sphere[keep_idx]

    return local_env_cube, cell_center_pos

def get_local_cube_frame(frame, atom_idx, r_cut, buffer_length, contact_dist, sigma, num_configs):

    cube, center = extract_local_cube(frame, atom_idx, r_cut, buffer_length)
    cube_l = (r_cut + buffer_length) * 2 + contact_dist + (sigma * 1.5)
    new_center = np.ones(3) * cube_l / 2
    new_positions = cube.positions - center + new_center
    cube_cell = ase.Atoms(
        symbols=cube.symbols,
        cell=[cube_l, cube_l, cube_l],
        positions=new_positions,
        pbc=True,
    )
    cube_cell.wrap()

    # save idx of atoms in the buffer region
    dists = cube_cell.get_distances(0, np.arange(len(cube_cell)), mic=True)
    hit_list = (dists > r_cut).astype(int)
    buffer_idx = np.array([ii for ii, v in enumerate(hit_list) if v > 0])

    cube_cells = []
    for _ in range(num_configs):
        close_contact = True
        while close_contact:
            cur_cube_cell = cube_cell.copy()
            for ii in buffer_idx:
                cur_cube_cell.positions[ii] += np.random.normal(scale=sigma, size=3)
            dists = cur_cube_cell.get_all_distances(mic=True)
            np.fill_diagonal(dists, 10000)
            print(np.min(dists))
            if np.min(dists) > contact_dist:
                close_contact = False
        cur_cube_cell.wrap()
        cube_cells.append(cur_cube_cell)

    return cube_cells, buffer_idx, hit_list

