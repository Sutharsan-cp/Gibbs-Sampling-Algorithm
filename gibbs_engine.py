"""
Gibbs Sampling Engine for Homophily-Based Label Propagation
Based on Algorithm 2: GS (Gibbs Sampling Algorithm)
"""

import numpy as np
import networkx as nx
from collections import defaultdict, Counter
from typing import Dict, List, Optional, Tuple, Set
import random


class GibbsSampler:
    """
    Implements Gibbs Sampling for label propagation on graphs.
    
    Core idea: Uses the homophily principle — nodes tend to have
    the same label as their neighbors. Iteratively samples labels
    for each node conditioned on its neighborhood.
    """

    def __init__(
        self,
        graph: nx.Graph,
        labels: List[str],
        observed: Dict[int, str],
        burn_in: int = 100,
        num_samples: int = 200,
        alpha: float = 1.0,
    ):
        """
        Args:
            graph:       The input graph (NetworkX)
            labels:      All possible label values, e.g. ['Tech', 'Finance', 'HR']
            observed:    Dict mapping node_id -> known label (seed nodes)
            burn_in:     Number of burn-in iterations (B)
            num_samples: Number of sampling iterations after burn-in (S)
            alpha:       Dirichlet smoothing parameter
        """
        self.G = graph
        self.labels = labels
        self.label_index = {l: i for i, l in enumerate(labels)}
        self.L = len(labels)
        self.observed = observed  # fixed seed nodes
        self.B = burn_in
        self.S = num_samples
        self.alpha = alpha

        self.nodes = list(graph.nodes())
        self.N = len(self.nodes)

        # Current label assignment Y_i for each node
        self.Y: Dict[int, str] = {}
        # Sample counts c[i, l]
        self.counts: Dict[int, np.ndarray] = {
            n: np.zeros(self.L) for n in self.nodes
        }
        # History of label assignments per iteration (for animation)
        self.history: List[Dict[int, str]] = []

        # Bootstrap: initialize labels
        self._bootstrap()

    # ------------------------------------------------------------------ #
    #  Step 1 – Bootstrapping                                              #
    # ------------------------------------------------------------------ #

    def _bootstrap(self):
        """Initialize Y using only observed neighbours (Algorithm line 1-4)."""
        for node in self.nodes:
            if node in self.observed:
                self.Y[node] = self.observed[node]
            else:
                # Compute label using only observed nodes in N_i
                obs_neighbors = [
                    self.observed[nb]
                    for nb in self.G.neighbors(node)
                    if nb in self.observed
                ]
                if obs_neighbors:
                    # Pick the most common label among observed neighbors
                    self.Y[node] = Counter(obs_neighbors).most_common(1)[0][0]
                else:
                    # Fallback: random label
                    self.Y[node] = random.choice(self.labels)

    # ------------------------------------------------------------------ #
    #  Core computation: a_i (label distribution for node i)               #
    # ------------------------------------------------------------------ #

    def _compute_a(self, node: int) -> np.ndarray:
        """
        Compute the conditional distribution a_i for node `node`
        given current neighbor assignments (with Dirichlet smoothing).
        """
        counts = np.full(self.L, self.alpha)  # smoothing prior
        for nb in self.G.neighbors(node):
            lbl = self.Y.get(nb)
            if lbl is not None:
                counts[self.label_index[lbl]] += 1
        # Normalize to get probability distribution
        return counts / counts.sum()

    def _sample_label(self, node: int) -> str:
        """Sample Y_i ~ a_i (multinomial draw)."""
        a = self._compute_a(node)
        idx = np.random.choice(self.L, p=a)
        return self.labels[idx]

    # ------------------------------------------------------------------ #
    #  Main algorithm                                                       #
    # ------------------------------------------------------------------ #

    def run(self, progress_callback=None) -> Dict[int, str]:
        """
        Execute the full Gibbs Sampling algorithm.
        Returns final labels for all nodes.
        """
        total_iters = self.B + self.S

        # ---------- Burn-in (lines 5-11) ----------
        for n in range(1, self.B + 1):
            order = self.nodes[:]
            random.shuffle(order)
            for node in order:
                if node not in self.observed:
                    self.Y[node] = self._sample_label(node)
            self.history.append(dict(self.Y))
            if progress_callback:
                progress_callback(n, total_iters, phase="burn-in")

        # Reset counts (lines 12-16)
        for node in self.nodes:
            self.counts[node] = np.zeros(self.L)

        # ---------- Sample collection (lines 17-25) ----------
        for n in range(1, self.S + 1):
            order = self.nodes[:]
            random.shuffle(order)
            for node in order:
                if node not in self.observed:
                    self.Y[node] = self._sample_label(node)
                # Accumulate counts c[i, y_i] += 1
                self.counts[node][self.label_index[self.Y[node]]] += 1
            self.history.append(dict(self.Y))
            if progress_callback:
                progress_callback(self.B + n, total_iters, phase="sampling")

        # ---------- Final labels: argmax_l c[i, l] (lines 26-28) ----------
        final: Dict[int, str] = {}
        for node in self.nodes:
            if node in self.observed:
                final[node] = self.observed[node]
            else:
                final[node] = self.labels[int(np.argmax(self.counts[node]))]

        return final

    def get_confidence(self) -> Dict[int, float]:
        """Return confidence score (max count / total samples) per node."""
        conf = {}
        for node in self.nodes:
            total = self.counts[node].sum()
            if total > 0:
                conf[node] = float(self.counts[node].max() / total)
            else:
                conf[node] = 1.0  # observed nodes: 100% confident
        return conf

    def get_label_distribution(self, node: int) -> Dict[str, float]:
        """Return normalized label probability distribution for a node."""
        total = self.counts[node].sum()
        if total == 0:
            return {l: 1.0 / self.L for l in self.labels}
        return {
            self.labels[i]: float(self.counts[node][i] / total)
            for i in range(self.L)
        }


# ------------------------------------------------------------------ #
#  Graph generators                                                    #
# ------------------------------------------------------------------ #

def generate_linkedin_graph(
    n_nodes: int = 50,
    n_communities: int = 3,
    homophily: float = 0.8,
    seed: int = 42,
) -> Tuple[nx.Graph, Dict[int, str], List[str]]:
    """
    Generate a synthetic LinkedIn-like graph with community structure.
    
    Args:
        n_nodes:       Total number of professional nodes
        n_communities: Number of job sectors (labels)
        homophily:     Probability [0,1] that edges connect same-sector nodes
        seed:          Random seed
    
    Returns:
        (graph, true_labels, label_names)
    """
    rng = np.random.default_rng(seed)
    random.seed(seed)

    SECTORS = ["Tech", "Finance", "Healthcare", "Marketing", "Legal", "Education"]
    label_names = SECTORS[:n_communities]

    # Assign ground-truth labels
    true_labels: Dict[int, str] = {}
    community_nodes: Dict[str, List[int]] = defaultdict(list)
    for node in range(n_nodes):
        lbl = label_names[node % n_communities]
        true_labels[node] = lbl
        community_nodes[lbl].append(node)

    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))

    # Add edges with homophily bias
    target_edges = int(n_nodes * 2.5)
    attempts = 0
    while G.number_of_edges() < target_edges and attempts < target_edges * 20:
        attempts += 1
        u = rng.integers(n_nodes)
        if rng.random() < homophily:
            # Same community edge
            candidates = community_nodes[true_labels[u]]
        else:
            # Cross-community edge
            candidates = list(range(n_nodes))
        v = rng.choice(candidates)
        if u != v and not G.has_edge(u, v):
            G.add_edge(u, v)

    return G, true_labels, label_names


def load_graph_from_edges(edges: List[Tuple[int, int]]) -> nx.Graph:
    G = nx.Graph()
    G.add_edges_from(edges)
    return G


def evaluate(
    true_labels: Dict[int, str],
    predicted: Dict[int, str],
    observed: Set[int],
) -> Dict[str, float]:
    """Compute accuracy on unobserved (unlabelled) nodes only."""
    total = correct = 0
    for node, pred in predicted.items():
        if node not in observed:
            total += 1
            if true_labels.get(node) == pred:
                correct += 1
    return {
        "accuracy": correct / total if total else 0.0,
        "total_unlabeled": total,
        "correct": correct,
    }
