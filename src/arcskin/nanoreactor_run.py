import numpy as np
import itertools as it
from ase import Atoms
from ase.io import write, read
from rdkit import Chem
from rdkit.Chem import AllChem
import os
import subprocess
import tqdm
from monty.serialization import dumpfn
from ase.units import Bohr
import shutil
import subprocess


class SMILESPack:

    def __init__(self, smiles, **kws):
        self.kws = kws
        self.smiles = smiles

    def smiles_to_atoms(self, smi, seed=42):
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

    def _random_rotation(self, rng):
        """Uniformly distributed rotation matrix via a random unit quaternion."""
        q = rng.normal(size=4)
        w, x, y, z = q / np.linalg.norm(q)
        return np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),
             2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ])

    def _point_in_ball(self, rng, radius):
        v = rng.normal(size=3)
        return v / np.linalg.norm(v) * radius * rng.random() ** (1 / 3)

    def pack_smiles(self, counts=None, radius=None, min_dist=2.0, contact_dist=4.0,
                    max_tries=300, seed=None, filename="structure.xyz", write_file=False):
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
        counts = [1] * len(self.smiles) if counts is None else list(counts)

        templates = []
        for smi in self.smiles:
            mol, chg = self.smiles_to_atoms(
                smi, seed=int(rng.integers(1 << 30)))
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
                    pos = template.positions @ self._random_rotation(rng).T
                    pos = pos + \
                        self._point_in_ball(rng, max(radius - mol_r, 0.0))

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
        if write_file:
            write(filename, system, format="xyz",
                  comment=f"random cluster, {len(system)} atoms, charge {charge}")

        return system, charge

    def run_multiple(self, num_cells, **kws):
        random_seeds = np.random.random_integers(
            low=1, high=42, size=num_cells)

        cells = []
        charges = []
        for i, cell in enumerate(np.arange(num_cells)):
            c, ch = self.pack_smiles(seed=random_seeds[i], **kws)
            cells.append(c)
            charges.append(ch)
        self.cells = cells
        self.charges = charges

        return cells, charges


class XTBNanoReactor:

    def __init__(self, combination: dict, cells: list[Atoms], charges: list[float], temperature: int = 1200, time: float = 10.0, step: float = 0.5):

        self.xtb = shutil.which("xtb")
        if self.xtb is None:
            raise FileNotFoundError("xtb executable not found in PATH")

        self.cells = cells
        self.num_cells = len(self.cells)
        self.charges = charges
        self.spheres = [self.wall_radius(cell) for cell in cells]
        self.combination = combination
        self.input_file = f"""
    $md
       temp={temperature}
       time={time}
       step={step}
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
       sphere: {np.max(self.spheres):.2F}, all
    $end
    """

    def wall_radius(self, atoms, buffer=2.0):  # buffer in Å
        com = atoms.get_center_of_mass()
        rmax = np.linalg.norm(atoms.positions - com, axis=1).max()
        return (rmax + buffer) / Bohr   # Å → Bohr

    def make_directories(self, location='.', exist_ok=True):
        '''
        need to make directories initially 
        '''
        self.directories = []
        self.location = location
        for i in range(self.num_cells):
            _dir = os.path.join(location, f"seed_{i}")
            self.directories.append(_dir)
            os.makedirs(
                _dir,
                exist_ok=exist_ok
            )

    def setup_xtb(self):
        for cell, directory in zip(self.cells, self.directories):
            print(cell, directory)
            with open(os.path.join(directory, "md.inp"), "w") as f:
                f.write(self.input_file)
            cell.write(os.path.join(directory, "structure.xyz"))

    def save_xtb_run(self):
        self.trajectories["combinations"] = self.combination
        dumpfn(self.trajectories, os.path.join(self.location,
               "trajectories.json")) if self.trajectories else None

    def run_xtb(self, **kws):
        procs = []
        for directory in self.directories:
            outfile = open(os.path.join(directory, "xtb.out"), "w")
            p = subprocess.Popen(
                [self.xtb, "structure.xyz", "--md", "--input", "md.inp"],
                cwd=directory, stdout=outfile, stderr=subprocess.STDOUT,
                env={**os.environ, "OMP_NUM_THREADS": "4"},  # seems to be best
            )
            procs.append((p, outfile))

        for p, out in procs:
            p.wait()
            out.close()

        self.trajectories = {
            f"seed_{i}": [x.todict() for x in read(os.path.join(directory, "xtb.trj"), index=":", format="xyz")] for i, directory in enumerate(self.directories)
        }

        self.save_xtb_run()

    def run(self, location="."):
        self.make_directories(location=location)
        self.setup_xtb()
        self.run_xtb()
