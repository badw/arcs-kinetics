#!/usr/bin/env python
import numpy as np
import itertools as it

# from reactit.reactit import ReactionGenerator
from ase import Atoms
from ase.io import write
from rdkit import Chem
from rdkit.Chem import AllChem
import os
import subprocess
import tqdm
from monty.serialization import dumpfn
compounds = {
    "NO2":  "[N+](=O)[O-]",
    "NO":   "[N]=O",
    # "N2":   "N#N",
    "O2":   "O=O",
    # "HNO3": "O[N+](=O)[O-]",
    "HNO2": "ON=O",
    # "H2":   "[H][H]",
    "H2O":  "O",
    # "H2O2": "OO",
    "SO2": "O=S=O",
    "H2S": "S",
    "H2SO4": "OS(=O)(=O)O"
}
input_file = """
    $md
       temp=1200
       time=10.0
       step=0.5
       dump=5.0
       nvt=true
       shake=0
       restart=false
    $end
    $scc
       temp=3000.0
    $end
    $wall
       potential=logfermi
       beta=6.0
       temp=6000.0
       sphere: 14.0, all
    $end
    """


def smiles_to_atoms(smi, seed=42):
    """Embed a SMILES string in 3D with ETKDG + MMFF and return (Atoms, charge)."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smi!r}")
    mol = Chem.AddHs(mol)
    charge = Chem.GetFormalCharge(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    if AllChem.EmbedMolecule(mol, params) != 0:
        raise RuntimeError(f"Embedding failed for {smi!r}")
    AllChem.MMFFOptimizeMolecule(mol)
    atoms = Atoms(
        symbols=[a.GetSymbol() for a in mol.GetAtoms()],
        positions=mol.GetConformer().GetPositions(),
    )
    atoms.positions -= atoms.get_center_of_mass()
    return atoms, charge


def _random_rotation(rng):
    """Uniformly distributed rotation matrix via a random unit quaternion."""
    q = rng.normal(size=4)
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _point_in_ball(rng, radius):
    v = rng.normal(size=3)
    return v / np.linalg.norm(v) * radius * rng.random() ** (1 / 3)


def pack_smiles(smiles, counts=None, radius=None, min_dist=2.0, contact_dist=4.0,
                max_tries=300, seed=None, filename="structure.xyz", write_chrg=True):
    """Randomly pack molecules into a connected, non-periodic cluster.

    smiles       : list of SMILES strings
    counts       : copies of each (default: one each)
    radius       : cluster radius in Angstrom (default: estimated from molecule sizes)
    min_dist     : minimum interatomic distance between molecules
    contact_dist : each new molecule must come within this of the existing cluster so
                   nothing is left floating in vacuum (set None to disable)

    Returns (Atoms, total_charge).
    """
    rng = np.random.default_rng(seed)
    counts = [1] * len(smiles) if counts is None else list(counts)

    templates = []
    for smi in smiles:
        mol, chg = smiles_to_atoms(smi, seed=int(rng.integers(1 << 30)))
        templates.append(
            (mol, chg, np.linalg.norm(mol.positions, axis=1).max()))

    if radius is None:  # sphere sized for a loose ~35% packing of bounding spheres
        vol = sum(n * (r + min_dist / 2) ** 3 for (_, _, r),
                  n in zip(templates, counts))
        radius = (vol / 0.35) ** (1 / 3)

    system = Atoms()
    charge = 0

    for (template, chg, mol_r), n in zip(templates, counts):
        for copy in range(n):
            for attempt in range(max_tries):
                pos = template.positions @ _random_rotation(rng).T
                pos = pos + _point_in_ball(rng, max(radius - mol_r, 0.0))

                if len(system) == 0:
                    break
                dmin = np.linalg.norm(
                    system.positions[None, :, :] - pos[:, None, :], axis=-1
                ).min()
                if dmin >= min_dist and (contact_dist is None or dmin <= contact_dist):
                    break
                if attempt and attempt % 100 == 0:
                    radius *= 1.1  # too tight, let the cluster breathe
            else:
                raise RuntimeError(
                    f"Could not place copy {copy + 1} of a molecule after {max_tries} "
                    f"tries - lower min_dist or raise radius/contact_dist"
                )

            mol = template.copy()
            mol.positions = pos
            system += mol
            charge += chg

    system.positions -= system.get_center_of_mass()
    write(filename, system, format="xyz",
          comment=f"random cluster, {len(system)} atoms, charge {charge}")
    if write_chrg and charge != 0:
        with open(".CHRG", "w") as f:
            f.write(f"{charge}\n")
    return system, charge


if __name__ == "__main__":
    combinations = list(it.combinations(compounds, 2))
    for r_index, combination in tqdm.tqdm(enumerate(combinations), total=len(combinations), desc="combination", position=0, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}', leave=True):
        _pwd = os.getcwd()

        directory = os.path.join(_pwd, f"combination_{r_index}")
        os.makedirs(directory)
        # make a json with the combination for reference
        dumpfn(
            {r_index: combination},
            os.path.join(directory, "combination.json"), indent=4
        )
        # change directory to the combination you want to run
        os.chdir(os.path.join(_pwd, f"combination_{r_index}"))
        # get the smiles strings
        smiles_list = [compounds[p] for p in combination]
        # write the input file
        with open(os.path.join(directory, "md.inp"), "w") as f:
            f.write(input_file)

        charges = [
            pack_smiles(
                smiles_list,
                counts=[1, 1],
                min_dist=2.2,
                seed=i,
                filename=f"{directory}/structure_{i}.xyz"
            )[1] for i in range(0, 7)
        ]
        #  set up 4 MD runs
        procs = []
        for i in range(4):
            log = open(f"./out_{i}.log", "w")
            p = subprocess.Popen(
                ["xtb", f"structure_{i}.xyz", "--md", "--input", "md.inp",
                 "--gfn", "2", "--chrg", str(charges[0]), "--uhf", "0", "-P", "4", "--namespace", str(i)],
                stdout=log, stderr=subprocess.STDOUT,
            )
            procs.append((p, log))

        for p, log in procs:
            p.wait()
            log.close()
        # go back
        os.chdir(_pwd)
