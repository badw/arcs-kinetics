from arcskin.nanoreactor_run import SMILESPack
from arcskin.nanoreactor_run import XTBNanoReactor
from arcskin.nanoreactor_analysis import AnalyseNanoReactor
from arcskin.make_neb_endpoints import make_endpoints
from arcskin.run_neb import run_all
import seaborn as sns
import matplotlib.pyplot as plt
from ase.mep import NEBTools
from ase.utils import forcecurve
import seaborn as sns
import matplotlib.pyplot as plt
from ase.mep import NEBTools
from ase.utils import forcecurve
import collections
import pandas as pd
from monty.serialization import dumpfn

num_molecules = 4
num_runs = 4
min_distance = 2.2
nano_reactor_temperature = 1200
total_time = 5
directory = "./so3_h2o/"

molecules = {"SO3": "O=S(=O)=O", "H2O": "O"}

sp = SMILESPack(molecules.values())
cells, charges = sp.run_multiple(
    num_runs, counts=[num_molecules, num_molecules], min_dist=min_distance
)

xnr = XTBNanoReactor(
    combination=tuple(molecules.values()),
    cells=cells,
    charges=charges,
    temperature=nano_reactor_temperature,
    time=total_time,
)

xnr.run(location=directory)

anr = AnalyseNanoReactor(skip_step=10, reactants=list(molecules.values()))

anr.read_trajectory(xnr.trajectory_locations, format="xyz")
anr.analyse_trajectories(n_cpus=10)

df = pd.DataFrame(anr.unique_smiles_counter).sum(axis=1).sort_values(ascending=False)
print(df)


print(anr.unique_smiles)

anr.get_sensible_smiles()
possible_paths = anr.get_possible_paths_dict()
print(possible_paths)


if possible_paths:
    reaction_endpoints = {}
    for k, v in possible_paths.items():
        try:
            reactants, products = make_endpoints(v["reactants"], v["products"])
            reaction_endpoints[k] = {
                "reactants": reactants,
                "products": products,
                "overall": possible_paths[k],
            }
        except Exception:
            print(f"{k} failed")
            pass

    results = run_all(reaction_endpoints, n_images=16, climb=True, fmax=0.05)

    print(results)

    dumpfn(results, f"{directory}results.json", indent=2)
