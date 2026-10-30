import pandas as pd
from ase import Atoms
from ase.io import read
from openbabel import openbabel as ob
from monty.serialization import loadfn
from collections import Counter, defaultdict
from python_utils import raise_exception
from tqdm import tqdm
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import PointGroupAnalyzer, iterative_symmetrize
from arcskin.nanoreactor_run import smiles_to_atoms
import warnings
from tblite.ase import TBLite
from ase.optimize import BFGS
from monty.serialization import loadfn
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
import itertools as it
import numpy as np
import tqdm_pathos
import multiprocess as mp
import warp as wp
from aimnet.calculators import AIMNet2ASE
import warnings


from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit import rdBase

rdBase.DisableLog("rdApp.error")
rdBase.DisableLog("rdApp.warning")
_calc = None


def _get_calc():
    global _calc
    if _calc is None:
        import warp as wp

        wp.config.quiet = True
        import torch

        torch.set_num_threads(1)
        from aimnet.calculators import AIMNet2ASE

        _calc = AIMNet2ASE("aimnet2")
    return _calc


def compare_molecules(
    smiles1, smiles2, near_dup_threshold=0.9, ignore_stereo=False, small_mol_atoms=10
):
    """EDIT!!!!
    Compare two molecules.

    Returns a dict with:
      - identical:      True if they are the same molecule (InChIKey match)
      - similarity:     Tanimoto similarity of Morgan fingerprints (0.0-1.0)
      - near_duplicate: True if identical, or similar above the threshold
                        (small molecules must also share a molecular formula)
    """
    _morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

    # 1. Parse SMILES strings into RDKit molecule objects
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)
    if mol1 is None or mol2 is None:
        raise ValueError("One or both SMILES strings are invalid.")

    # 2. Exact identity via InChIKey
    key1 = Chem.MolToInchiKey(mol1)
    key2 = Chem.MolToInchiKey(mol2)
    if ignore_stereo:
        key1, key2 = key1.split("-")[0], key2.split("-")[0]
    identical = key1 == key2

    # 3. Tanimoto similarity on Morgan fingerprints
    fp1 = _morgan_gen.GetFingerprint(mol1)
    fp2 = _morgan_gen.GetFingerprint(mol2)
    similarity = DataStructs.TanimotoSimilarity(fp1, fp2)

    # 4. Near-duplicate decision
    near_duplicate = identical or similarity >= near_dup_threshold
    # Small molecules have sparse fingerprints, so also require the same formula
    if near_duplicate and not identical:
        if min(mol1.GetNumHeavyAtoms(), mol2.GetNumHeavyAtoms()) <= small_mol_atoms:
            near_duplicate = CalcMolFormula(mol1) == CalcMolFormula(mol2)

    return {
        "identical": identical,
        "similarity": similarity,
        "near_duplicate": near_duplicate,
    }


class AnalyseNanoReactor:
    def __init__(
        self, reactants: tuple[str, str] = None, skip_step: int = 5, calculator=None
    ):
        warnings.simplefilter("ignore")
        self.trajectory = defaultdict(dict)
        self.unique_structures = defaultdict(lambda: defaultdict(list))
        self.unique_smiles_counter = defaultdict(lambda: defaultdict(list))
        wp.config.quiet = True
        mp.set_start_method("spawn", force=True)
        if not calculator:
            print("calculating with default calculator: aimnet")
            self.calculator = AIMNet2ASE("aimnet2")
        elif calculator == "tblite":
            print("calculating with TBLite")
            self.calculator = calculator
        else:
            self.calculator = calculator

        self.reactants = reactants

        self.skip_step = skip_step
        if self.reactants:
            self.get_num_reactant_atoms()

    def get_num_atoms(self, smiles_string):
        return Counter(
            smiles_to_atoms(smiles_string, return_charge=False).get_chemical_symbols()
        )

    def get_num_reactant_atoms(self):
        self.num_reactant_atoms = sum(
            (self.get_num_atoms(a) for a in self.reactants), Counter()
        )

    def read_trajectory(self, trajectory: list | str, format=None):
        "reads one trajectory of multiple trajectories in xyz format (or xtb's .trj format )"
        if isinstance(trajectory, list):
            for i, traj in enumerate(trajectory):
                self.trajectory[i] = read(
                    traj, format=format, index=f"::{self.skip_step}"
                )
        elif isinstance(trajectory, str):
            self.trajectory[0] = read(
                trajectory, format=format, index=f"::{self.skip_step}"
            )

        if not self.reactants:
            first_frame = self.trajectory[0][0]
            self.reactants = list(Counter(self.fragment_smiles(first_frame)))
            self.get_num_reactant_atoms()
            print(f"reactants found = {self.reactants}")

    def read_arcskintrajectories(self, data: dict):
        #  add skip step.
        del data["combinations"]
        for i, seed in enumerate(data):
            self.trajectory[i] = [
                Atoms.fromdict(at) for at in data[seed][:: self.skip_step]
            ]

    def relax_atoms(self, atoms: Atoms, fmax=0.05, steps=500, calculator=None):
        if calculator == "tblite":
            # use TBLIte xtb
            mult = 2 if sum(atoms.numbers) % 2 else 1
            atoms.calc = TBLite(
                method="GFN2-xTB", multiplicity=mult, max_iterations=1000, verbosity=0
            )
        else:
            atoms.calc = calculator
        BFGS(atoms, logfile=None).run(fmax=fmax, steps=steps)
        return atoms

    def seperate_atoms(self, atoms: Atoms) -> list[Atoms]:
        """Split an ase.Atoms object into its disconnected molecular fragments."""
        mol = ob.OBMol()
        for Z, pos in zip(atoms.numbers, atoms.positions):
            a = mol.NewAtom()
            a.SetAtomicNum(int(Z))
            a.SetVector(*map(float, pos))
        mol.ConnectTheDots()  # guess bonds from interatomic distances

        return [
            Atoms(
                numbers=[a.GetAtomicNum() for a in ob.OBMolAtomIter(frag)],
                positions=[
                    (a.GetX(), a.GetY(), a.GetZ()) for a in ob.OBMolAtomIter(frag)
                ],
            )
            for frag in mol.Separate()
        ]

    def to_obmol(self, atoms: Atoms) -> ob.OBMol:
        mol = ob.OBMol()
        for Z, pos in zip(atoms.numbers, atoms.positions):
            a = mol.NewAtom()
            a.SetAtomicNum(int(Z))
            a.SetVector(*map(float, pos))
        mol.ConnectTheDots()
        mol.PerceiveBondOrders()  # needed for correct double/aromatic bonds
        return mol

    def fragment_smiles(self, atoms: Atoms, fmt: str = "can") -> list[str]:
        """Canonical SMILES for each disconnected fragment."""
        conv = ob.OBConversion()
        conv.SetOutFormat(fmt)
        return [
            conv.WriteString(frag).split()[0]
            for frag in self.to_obmol(atoms).Separate()
        ]

    def symmetrize(self, atoms: Atoms, tolerance: float = 0.3) -> tuple[Atoms, str]:
        """Return a symmetrised copy of a molecule and its point group."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", np.exceptions.ComplexWarning)
            mol = AseAtomsAdaptor.get_molecule(atoms, charge_spin_check=False)
            sym_mol = iterative_symmetrize(mol, max_n=10, tolerance=tolerance)[
                "sym_mol"
            ]
            pg = PointGroupAnalyzer(sym_mol, tolerance=tolerance).sch_symbol
        return AseAtomsAdaptor.get_atoms(sym_mol), pg

    def analyse_trajectory(
        self,
        index: int = 0,
        tolerance: float = 0.1,
        fmax=0.05,
        relax_frame=True,
        multiprocess=True,
        tqdm_kws={"disable": True},
    ):
        for i, frame in tqdm(
            enumerate(self.trajectory[index]),
            total=len(self.trajectory[index]),
            **tqdm_kws,
        ):
            if relax_frame:
                self.relax_atoms(
                    frame, fmax=0.1, calculator=self.calculator
                )  # test relaxing the frame first.
            atoms_list = self.seperate_atoms(frame)
            for at in atoms_list:
                try:
                    at = self.relax_atoms(at, fmax=fmax, calculator=self.calculator)
                    sym, _ = self.symmetrize(at, tolerance=tolerance)
                    frags = self.fragment_smiles(sym)

                    for frag in frags:
                        if frag not in self.reactants:
                            num_atoms_frag = self.get_num_atoms(frag)
                            sensible = Counter(num_atoms_frag) <= Counter(
                                self.num_reactant_atoms
                            )
                            if sensible:
                                self.unique_structures[index]["smiles"].append(frag)

                except Exception as e:
                    print(e)
                    pass
        if multiprocess:
            return Counter(self.unique_structures[index]["smiles"])
        else:
            self.unique_smiles_counter[index] = Counter(
                self.unique_structures[index]["smiles"]
            )

    def analyse_frame(
        self,
        frame,
        tolerance=0.1,
        fmax=0.05,
        relax_frame=True,
    ):
        calc = _get_calc()
        if relax_frame:
            self.relax_atoms(frame, fmax=0.1, calculator=calc)
        atoms_list = self.seperate_atoms(frame)
        sensible_structures = []
        for at in atoms_list:
            try:
                at = self.relax_atoms(at, fmax=fmax, calculator=calc)
                sym, _ = self.symmetrize(at, tolerance=tolerance)
                frags = self.fragment_smiles(sym)

                for frag in frags:
                    if frag not in self.reactants:
                        num_atoms_frag = self.get_num_atoms(frag)
                        sensible = Counter(num_atoms_frag) <= Counter(
                            self.num_reactant_atoms
                        )
                        if sensible:
                            sensible_structures.append(
                                frag
                            )  # self.unique_structures[index]["smiles"].append(frag)

            except Exception as e:
                pass
        return sensible_structures

    def _analyse_trajectories(
        self,
        tolerance: float = 0.3,
        relax_frame=True,
        multiprocess=False,
        tqdm_kws={"disable": False},
    ):
        if len(self.trajectory) > 1:
            for ii in self.trajectory:
                self.analyse_trajectory(
                    index=ii, tolerance=tolerance, relax_frame=True, tqdm_kws=tqdm_kws
                )
        else:
            self.analyse_trajectory(tolerance=tolerance, tqdm_kws=tqdm_kws)

        self.remove_similars()

    def analyse_trajectories(
        self, tolerance: float = 0.3, n_cpus=4, tqdm_kws={"disable": False}
    ):
        frames = list(it.chain(*list(self.trajectory.values())))
        d = tqdm_pathos.map(
            self.analyse_frame,
            frames,
            tolerance=tolerance,
            tqdm_kwargs=tqdm_kws,
            n_cpus=n_cpus,
        )
        self.unique_smiles_counter = Counter(it.chain(*d))
        self.remove_similars()
        self.get_sensible_smiles()

    def remove_similars(self):
        try:  # from previous format... will deprecate later
            unique_smiles = list(
                pd.DataFrame(self.unique_smiles_counter)
                .sum(axis=1)
                .sort_values(ascending=False)
                .index
            )
        except Exception:
            unique_smiles = list(
                pd.Series(self.unique_smiles_counter).sort_values(ascending=False).index
            )
        similar = self.find_similar(unique_smiles + self.reactants)
        if similar:
            for sim in similar:
                mol1, mol2 = sim["molecules"]
                if mol1 in unique_smiles:
                    unique_smiles.remove(mol1)
                elif mol2 in unique_smiles:
                    unique_smiles.remove(mol2)

        self.unique_smiles = unique_smiles

    def get_sensible_smiles(self):

        ranked_smiles = sorted(self.unique_smiles)
        possible = [self.get_num_atoms(smiles) for smiles in ranked_smiles]
        possible = [Counter(p) for p in possible]
        results = []

        def search(remaining, start, chosen):
            if not remaining:  # everything used up
                results.append(chosen)
                return
            for i in range(start, len(possible)):
                if possible[i] <= remaining:  # enough of each element left
                    search(remaining - possible[i], i, chosen + [i])

        search(Counter(self.num_reactant_atoms), 0, [])
        self.sensible_outcomes = [
            [ranked_smiles[i] for i in combination] for combination in results
        ]

    def find_similar(self, smileslist):
        combinations = it.combinations(smileslist, 2)
        similar = []
        for i, j in combinations:
            similarity = compare_molecules(i, j)
            similarity["molecules"] = i, j
            if similarity["identical"] or similarity["near_duplicate"]:
                similar.append(similarity)
        return similar

    def get_possible_paths_dict(self):

        if self.sensible_outcomes:
            possible_paths = {
                i: {"reactants": self.reactants, "products": prod}
                for i, prod in enumerate(self.sensible_outcomes)
            }
            return self.relax_all_paths(possible_paths)
        else:
            return None

    def relax_all_paths(self, possible_paths, fmax=0.05):
        for k, v in possible_paths.items():
            reactants = v["reactants"]
            products = v["products"]
            possible_paths[k]["reactants_energies"] = []
            possible_paths[k]["products_energies"] = []
            for r in reactants:
                r_atoms = smiles_to_atoms(r)[0]
                r_relaxed = self.relax_atoms(
                    r_atoms, fmax=fmax, calculator=self.calculator
                )
                possible_paths[k]["reactants_energies"].append(
                    r_relaxed.get_potential_energy()
                )
            for p in products:
                p_atoms = smiles_to_atoms(p)[0]
                p_relaxed = self.relax_atoms(
                    p_atoms, fmax=fmax, calculator=self.calculator
                )
                possible_paths[k]["products_energies"].append(
                    p_relaxed.get_potential_energy()
                )
            possible_paths[k]["diff"] = np.sum(
                possible_paths[k]["products_energies"]
            ) - np.sum(possible_paths[k]["reactants_energies"])

        return possible_paths
