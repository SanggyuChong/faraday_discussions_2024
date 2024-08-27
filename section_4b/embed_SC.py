import numpy as np
import ase

def extract_local_sphere(frame, atom_idx, r_cut, buffer_length):

    # detect local env sphere
    dists_from_atom = frame.get_distances(atom_idx, np.arange(len(frame)), mic=True)
    r_cut_eff = r_cut + buffer_length
    hit_list = (0 < dists_from_atom).astype(int) * (dists_from_atom < r_cut_eff).astype(int)
    neigh_idx = np.array([ii for ii, v in enumerate(hit_list) if v > 0])
    local_env_sphere = frame[np.concatenate([np.array([atom_idx]), neigh_idx])]

    # center the spherical env to the cell center
    local_env_sphere.set_positions(local_env_sphere.positions - local_env_sphere.positions[0])
    cell_center_pos = local_env_sphere.cell.cartesian_positions([0.5, 0.5, 0.5])
    local_env_sphere.set_positions(local_env_sphere.positions + cell_center_pos)
    local_env_sphere.wrap()

    return local_env_sphere, cell_center_pos


def carve_sphere_from_center(frame, r_carve):
    # carve out center
    center_pos = frame.cell.cartesian_positions([0.5, 0.5, 0.5])
    frame_new = frame + ase.Atom(position=center_pos)
    dists_from_center = frame_new.get_distances(-1, np.arange(len(frame_new)), mic=True)
    hit_list = (dists_from_center > r_carve).astype(int)
    keep_idx = np.array([ii for ii, v in enumerate(hit_list) if v > 0])
    frame_carved = frame_new[keep_idx]

    return frame_carved, center_pos


def embed_into_HS_struc(HS_frame, frame, atom_idx, r_cut, buffer_length, contact_dist, sigma, num_configs):

    # get cell expansion factors -- assume cubic HS structure
    # TODO: consider cases where non-cubic HS structure are given
    cube_l = (2 * r_cut + buffer_length + contact_dist + sigma * 1.5) / np.sqrt(3) * 2
    hs_l = HS_frame.cell.cellpar()[0]
    mult = (cube_l // hs_l) + (cube_l % hs_l > 0).astype(int)
    hs_new = ase.build.make_supercell(HS_frame, [[mult, 0, 0],
                                                 [0, mult, 0],
                                                 [0, 0, mult]])

    # get carved HS struc
    r_carve = r_cut + buffer_length + contact_dist + sigma * 1.5
    hs_carved, hs_center_pos = carve_sphere_from_center(hs_new, r_carve)

    local_env_sphere, sphere_center_pos = extract_local_sphere(frame, atom_idx, r_cut, buffer_length)
    cell_centered_positions = local_env_sphere.positions - sphere_center_pos    
    hs_centered_positions = cell_centered_positions + hs_center_pos
    
    sphere_in_cube = ase.Atoms(
        symbols=local_env_sphere.symbols,
        cell=hs_carved.cell,
        positions=hs_centered_positions,
        pbc=True,
    )
    embedded_cube = sphere_in_cube + hs_carved
    embedded_cube.wrap()
    dia_center_idx = len(sphere_in_cube)

    # detect and pass the idx of atoms in the buffer region for relaxation
    target_env_dists = embedded_cube.get_distances(0, np.arange(len(embedded_cube)), mic=True)
    hs_env_dists = embedded_cube.get_distances(dia_center_idx, np.arange(len(embedded_cube)), mic=True)
    hit_list = (target_env_dists > r_cut).astype(int) * (hs_env_dists > r_cut).astype(int)
    buffer_idx = np.array([ii for ii, v in enumerate(hit_list) if v > 0])    

    cube_cells = []
    for _ in range(num_configs):        
        close_contact = True
        while close_contact:
            cur_cube_cell = embedded_cube.copy()
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

