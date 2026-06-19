"""
reaction_network.py
-------------------
Builds a reaction network graph from SpeciesTracker output.

Each node is a unique molecular formula (intermediate, reactant, or product).
Each directed edge A → B means: species A was observed to disappear and B
appeared within `transition_window_frames` frames, in at least one trajectory
frame sequence.  Edge weight = number of times this transition was observed.

The output graph is a networkx DiGraph, ready for:
  - Visualisation (networkx / matplotlib / pyvis)
  - NEB job list generation (get_neb_pairs)
  - Export to standard formats (GEXF, JSON, adjacency list)
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

try:
    import networkx as nx
    HAS_NX = True
except ImportError:
    HAS_NX = False

from species_tracker import SpeciesRecord


@dataclass
class ReactionEdge:
    source: str
    target: str
    count: int = 0           # number of observed transitions
    first_frame: int = -1
    last_frame: int = -1


class ReactionNetwork:
    """
    Build a reaction network from per-frame species lists.

    Parameters
    ----------
    transition_window_frames : a disappearance of A and appearance of B within
                               this many frames counts as one A→B transition.
    min_edge_count           : prune edges seen fewer times than this.
    """

    def __init__(
        self,
        transition_window_frames: int = 10,
        min_edge_count: int = 2,
        verbose: bool = True,
    ):
        self.transition_window_frames = transition_window_frames
        self.min_edge_count = min_edge_count
        self.verbose = verbose

        self._frame_species: list[set[str]] = []   # per-frame set of formulae
        self._edges: dict[tuple[str, str], ReactionEdge] = {}
        self._nodes: set[str] = set()

    # ------------------------------------------------------------------
    # Ingestion (mirrors SpeciesTracker interface)
    # ------------------------------------------------------------------

    def add_frame_species(self, frame_idx: int, formulae: list[str]) -> None:
        """Record which species are present in one frame."""
        while len(self._frame_species) <= frame_idx:
            self._frame_species.append(set())
        self._frame_species[frame_idx] = set(formulae)
        self._nodes.update(formulae)

    def build_from_tracker_records(
        self,
        records: dict[str, SpeciesRecord],
        total_frames: int,
    ) -> None:
        """
        Reconstruct per-frame species sets from SpeciesRecord frame_indices,
        then derive transition edges.

        Parameters
        ----------
        records       : dict of formula -> SpeciesRecord (from tracker.get_intermediates
                        or tracker._records — pass all records to capture reactant→
                        intermediate transitions too)
        total_frames  : total number of frames in the trajectory
        """
        # Reconstruct per-frame presence
        self._frame_species = [set() for _ in range(total_frames)]
        for formula, rec in records.items():
            for fidx in rec.frame_indices:
                if fidx < total_frames:
                    self._frame_species[fidx].add(formula)
                    self._nodes.add(formula)

        self._derive_edges()

    # ------------------------------------------------------------------
    # Edge detection
    # ------------------------------------------------------------------

    def _derive_edges(self) -> None:
        """
        Scan the frame-by-frame species timeline for transitions.

        For every frame pair (t, t+w) within the transition window:
          - species in t but NOT in t+w  → "disappeared" set D
          - species in t+w but NOT in t  → "appeared" set A
          - record a transition D_i → A_j for every combination
            (this is conservative; you may want to restrict to 1:1 pairs
             if you know the stoichiometry)
        """
        n_frames = len(self._frame_species)
        w = self.transition_window_frames

        for t in range(n_frames - w):
            present_now = self._frame_species[t]
            present_later = self._frame_species[t + w]

            disappeared = present_now - present_later
            appeared = present_later - present_now

            if not disappeared or not appeared:
                continue

            for src in disappeared:
                for tgt in appeared:
                    if src == tgt:
                        continue
                    key = (src, tgt)
                    if key not in self._edges:
                        self._edges[key] = ReactionEdge(
                            source=src, target=tgt, first_frame=t)
                    edge = self._edges[key]
                    edge.count += 1
                    edge.last_frame = t + w

        if self.verbose:
            raw = len(self._edges)
            pruned = sum(1 for e in self._edges.values()
                         if e.count >= self.min_edge_count)
            print(f"[ReactionNetwork] {raw} raw transitions → "
                  f"{pruned} after min_count≥{self.min_edge_count} filter")

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def to_networkx(self) -> "nx.DiGraph":
        """Return a weighted directed networkx graph (requires networkx)."""
        if not HAS_NX:
            raise ImportError("networkx is required: pip install networkx")

        G = nx.DiGraph()
        G.add_nodes_from(self._nodes)

        for (src, tgt), edge in self._edges.items():
            if edge.count >= self.min_edge_count:
                G.add_edge(src, tgt, weight=edge.count,
                           first_frame=edge.first_frame,
                           last_frame=edge.last_frame)
        return G

    def get_neb_pairs(
        self,
        reactants: Optional[set[str]] = None,
        products: Optional[set[str]] = None,
    ) -> list[tuple[str, str, int]]:
        """
        Return list of (source, target, count) pairs that should be
        submitted as NEB jobs.

        Excludes self-loops and optionally excludes reactant/product nodes
        that don't need a TS search (direct reactant→product goes in, but
        you may want to run those anyway).
        """
        exclude = (reactants or set()) | (products or set())
        pairs = []
        for (src, tgt), edge in sorted(
            self._edges.items(), key=lambda x: -x[1].count
        ):
            if edge.count < self.min_edge_count:
                continue
            # Keep reactant/product endpoints — the caller can filter further
            pairs.append((src, tgt, edge.count))
        return pairs

    def print_network(self, top_n: int = 30) -> None:
        """Print the top edges sorted by observation count."""
        pairs = sorted(self._edges.items(), key=lambda x: -x[1].count)
        print(f"\n{'Source':>14}  →  {'Target':<14}  count")
        print("-" * 46)
        for (src, tgt), edge in pairs[:top_n]:
            if edge.count < self.min_edge_count:
                continue
            print(f"  {src:>12}  →  {tgt:<12}  {edge.count}")

    def plot_network(
        self,
        reactants: Optional[set[str]] = None,
        products: Optional[set[str]] = None,
        figsize: tuple = (10, 8),
        min_count: Optional[int] = None,
    ):
        """
        Draw the reaction network with matplotlib + networkx.
        Node colour: green = reactant, red = product, blue = intermediate.
        Edge width scales with observation count.
        """
        if not HAS_NX:
            raise ImportError("networkx is required for plot_network()")
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for plot_network()")

        threshold = min_count if min_count is not None else self.min_edge_count
        G = self.to_networkx()
        edges_to_remove = [(u, v) for u, v, d in G.edges(data=True)
                           if d["weight"] < threshold]
        G.remove_edges_from(edges_to_remove)
        # Remove isolated nodes
        G.remove_nodes_from(list(nx.isolates(G)))

        reactants = reactants or set()
        products = products or set()

        node_colors = []
        for node in G.nodes():
            if node in reactants:
                node_colors.append("#4CAF50")   # green
            elif node in products:
                node_colors.append("#F44336")   # red
            else:
                node_colors.append("#2196F3")   # blue

        weights = [G[u][v]["weight"] for u, v in G.edges()]
        max_w = max(weights) if weights else 1
        edge_widths = [1.0 + 4.0 * w / max_w for w in weights]

        pos = nx.spring_layout(G, seed=42, k=2.5)

        fig, ax = plt.subplots(figsize=figsize)
        nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                               node_size=1200, ax=ax, alpha=0.9)
        nx.draw_networkx_labels(G, pos, font_size=9, ax=ax)
        nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.7,
                               edge_color="#555", arrows=True,
                               arrowstyle="-|>", arrowsize=15,
                               connectionstyle="arc3,rad=0.1", ax=ax)
        edge_labels = {(u, v): d["weight"] for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels,
                                     font_size=7, ax=ax)

        from matplotlib.patches import Patch
        legend = [
            Patch(color="#4CAF50", label="Reactant"),
            Patch(color="#F44336", label="Product"),
            Patch(color="#2196F3", label="Intermediate"),
        ]
        ax.legend(handles=legend, loc="upper left", fontsize=9)
        ax.set_title("Reaction network  (edge width ∝ transition count)")
        ax.axis("off")
        plt.tight_layout()
        return fig, ax

    def export_neb_input(
        self,
        output_file: str = "neb_jobs.txt",
        reactants: Optional[set[str]] = None,
        products: Optional[set[str]] = None,
    ) -> None:
        """Write a simple text file listing all NEB pairs to run."""
        pairs = self.get_neb_pairs(reactants=reactants, products=products)
        with open(output_file, "w") as fh:
            fh.write("# NEB job list generated by ReactionNetwork\n")
            fh.write("# source  target  observation_count\n")
            for src, tgt, count in pairs:
                fh.write(f"{src}  {tgt}  {count}\n")
        if self.verbose:
            print(
                f"[ReactionNetwork] Wrote {len(pairs)} NEB pairs to {output_file}")
