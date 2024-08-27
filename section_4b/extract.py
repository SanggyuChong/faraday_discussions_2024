import numpy as np
import ase


def extract_environment(structure, index, cutoff):
    """
    Select atoms from a structure by extracting the atoms that are within a certain cutoff distance from the atom at a
    given index.

    Args:
        structure (ase.Atoms): The structure from which to extract atoms.
        index (int): The index of the atom around which to extract atoms.
        cutoff (float): The cutoff distance within which to extract atoms.

    Returns:
        ase.Atoms: The extracted atoms.
    """
    if np.any(structure.cell.diagonal() < cutoff):
        raise ValueError("The cutoff distance is larger than the smallest periodic distance in the structure.")
    interatomic_vectors = structure.get_distances(index, np.arange(len(structure)), mic=True, vector=True)
    indices = np.where(np.sqrt(np.sum(interatomic_vectors ** 2, axis=1)) < cutoff)[0]
    return ase.Atoms(
        numbers=structure.numbers[indices],
        positions=interatomic_vectors[indices],
    )
