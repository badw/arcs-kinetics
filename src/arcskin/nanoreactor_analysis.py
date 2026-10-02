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
# from tblite.ase import TBLite
from ase.optimize import BFGS
from aimnet.calculators import AIMNet2ASE

import copy
# uses the same as XTB


class AnalyseNanoReactor:
    def __init__(self, reactants: tuple[str, str] = (None, None), skip_step: int = 5, calculator=None):
        warnings.simplefilter("ignore")
        self.trajectory = defaultdict(dict)
        self.unique_structures = defaultdict(lambda: defaultdict(list))
        self.unique_smiles_counter = defaultdict(lambda: defaultdict(list))
        # self.unique_compositions_counter = defaultdict(
        #    lambda: defaultdict(list))
        # self.unique_point_group_counter = defaultdict(
        #   lambda: defaultdict(list))
        #  needs to automatically determine.
        self.calculator = AIMNet2ASE("aimnet2")
        self.skip_step = skip_step
        if reactants:
            self.reactants = reactants
            self.get_num_reactant_atoms()

    def get_num_atoms(self, smiles_string):
        return Counter(
            smiles_to_atoms(
                smiles_string, return_charge=False).get_chemical_symbols()
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
                    traj, format=format, index=f"::{self.skip_step}")
        elif isinstance(trajectory, str):
            self.trajectory[0] = read(
                trajectory, format=format, index=f"::{self.skip_step}")

    def read_arcskintrajectories(self, data: dict):
        #  add skip step.
        del data["combinations"]
        for i, seed in enumerate(data):
            self.trajectory[i] = [Atoms.fromdict(
                at) for at in data[seed][::self.skip_step]]

    def relax_atoms(self, atoms: Atoms, fmax=0.05, steps=200):
        #  use TBLIte xtb
        # mult = 2 if sum(atoms.numbers) % 2 else 1
        # atoms.calc = TBLite(method="GFN2-xTB",
        #                    multiplicity=mult, verbosity=0)
        atoms.calc = self.calculator
        BFGS(atoms, logfile=None).run(fmax=fmax, steps=200)
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
            Atoms(numbers=[a.GetAtomicNum() for a in ob.OBMolAtomIter(frag)],
                  positions=[(a.GetX(), a.GetY(), a.GetZ()) for a in ob.OBMolAtomIter(frag)])
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
        return [conv.WriteString(frag).split()[0] for frag in self.to_obmol(atoms).Separate()]

    def symmetrize(self, atoms: Atoms, tolerance: float = 0.3) -> tuple[Atoms, str]:
        """Return a symmetrised copy of a molecule and its point group."""
        mol = AseAtomsAdaptor.get_molecule(atoms, charge_spin_check=False)
        sym_mol = iterative_symmetrize(
            mol, max_n=10, tolerance=tolerance)["sym_mol"]
        pg = PointGroupAnalyzer(sym_mol, tolerance=tolerance).sch_symbol
        return AseAtomsAdaptor.get_atoms(sym_mol), pg

    def analyse_trajectory(self, index: int = 0, tolerance: float = 0.1, tqdm_kws={"disable": False}):
        for i, frame in tqdm(enumerate(self.trajectory[index]), total=len(self.trajectory[index]), **tqdm_kws):
            self.relax_atoms(frame, fmax=0.1)  # test relaxing the frame first.
            atoms_list = self.seperate_atoms(frame)
            for at in atoms_list:
                try:
                    at = self.relax_atoms(at)
                    sym, pg = self.symmetrize(at, tolerance=tolerance)
                    frags = self.fragment_smiles(sym)

                    for frag in frags:
                        if frag not in self.reactants:
                            num_atoms_frag = self.get_num_atoms(frag)
                            sensible = Counter(num_atoms_frag) <= Counter(
                                self.num_reactant_atoms)
                            if sensible:
                                self.unique_structures[index]["smiles"].append(
                                    frag)

                except Exception:
                    pass
                    # self.unique_structures[index]["snapshot_atoms"].append(at)
                    # self.unique_structures[index]["symmetrised_atoms"].append(
                    #    sym)
                    # self.unique_structures[index]["compositions"].append(
                    #    sym.get_chemical_formula())

        self.unique_smiles_counter[index] = Counter(
            self.unique_structures[index]["smiles"]
        )
#        self.unique_compositions_counter[index] = Counter(
#            self.unique_structures[index]["compositions"])
#        self.unique_point_group_counter[index] = Counter(
#            self.unique_structures[index]["point_group"])

#        if self.reactants:
#            #  remove any of the reactants
#            indexes_a = [i for i, v in enumerate(
#                self.unique_structures[0]["compositions"]) if v == self.reactants[0]]
#            indexes_b = [i for i, v in enumerate(
#                self.unique_structures[0]["compositions"]) if v == self.reactants[1]]
#            indexes_to_delete = indexes_a + indexes_b
#            df = pd.DataFrame(self.unique_structures[0])
#            df = df.drop(index=indexes_to_delete)#

#            #  remove any fragments which have too many atoms compared to the reactants#

#            # self.unique_structures[index]["snapshot_atoms"] = df["snapshot_atoms"].values
#            self.unique_structures[index]["symmetrised_atoms"] = df["symmetrised_atoms"].values
#            self.unique_structures[index]["compositions"] = df["compositions"].values
#            self.unique_structures[index]["smiles"] = df["smiles"].values
#            self.unique_structures[index]["point_group"] = df["point_group"].values#

#            self.unique_smiles_counter[index] = Counter(
#                self.unique_structures[index]["smiles"])
#            self.unique_compositions_counter[index] = Counter(
#                self.unique_structures[index]["compositions"])
#            self.unique_point_group_counter[index] = Counter(
#                self.unique_structures[index]["point_group"])

    def analyse_trajectories(self, tolerance: float = 0.3, tqdm_kws={"disable": False}):
        if len(self.trajectory) > 1:
            for ii in self.trajectory:
                self.analyse_trajectory(
                    index=ii, tolerance=tolerance, tqdm_kws=tqdm_kws)
        else:
            self.analyse_trajectory(tolerance=tolerance, tqdm_kws=tqdm_kws)
