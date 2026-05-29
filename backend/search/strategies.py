import numpy as np
from backend.bench.search_space import (
    OPERATIONS, NUM_EDGES, NUM_OPS,
    index_to_ops, ops_to_index,
    is_valid_arch, TOTAL_ARCHS
)
from backend.search.darts import ArchitectureWeights


class RandomSearch:
    """
    Pure random search — the baseline every NAS method must beat.

    No learning. Every architecture has equal probability of selection.
    We include this for two reasons:
      1. Scientific integrity — your visualizer shows all three strategies
         side by side so users can see DARTS outperforming random search
      2. It serves as the bootstrap phase for evolutionary search

    Despite its simplicity, random search is a surprisingly strong
    baseline on small search spaces. On NAS-Bench-201 it finds
    top-10% architectures roughly 10% of the time by definition.
    DARTS should find top-10% architectures far more often.
    """

    name = 'random'

    def __init__(self, seed: int = 42):
        self.rng     = np.random.RandomState(seed)
        self.visited = set()

    def suggest(self, n: int) -> list:
        """
        Return n unique valid architecture indices chosen uniformly
        at random from the unvisited portion of the search space.
        """
        results  = []
        attempts = 0

        while len(results) < n and attempts < n * 30:
            idx = int(self.rng.randint(0, TOTAL_ARCHS))
            if idx not in self.visited and is_valid_arch(idx):
                results.append(idx)
                self.visited.add(idx)
            attempts += 1

        return results

    def update(self, evaluated: list, **kwargs):
        """Random search doesn't learn — just record visited indices."""
        for item in evaluated:
            self.visited.add(item['arch_index'])

    def state(self) -> dict:
        return {
            'name':    self.name,
            'visited': len(self.visited),
        }


class EvolutionarySearch:
    """
    Regularized evolutionary search (Real et al., AAAI 2019).

    One of the most cited NAS algorithms. The idea is simple but powerful:
    maintain a population of good architectures, mutate the best ones
    to produce children, evaluate children, keep the good ones.

    'Regularized' means we remove the OLDEST member when the population
    is full — not the worst. This preserves diversity and prevents the
    search from collapsing to a local optimum.

    The mutation operation is minimal: change exactly one edge's operation
    to a different randomly chosen operation. This makes child architectures
    close neighbors of their parents in the search space, allowing fine-grained
    local exploration around good regions.
    """

    name = 'evolutionary'

    def __init__(self, population_size: int = 64, seed: int = 42):
        self.population_size = population_size
        self.rng             = np.random.RandomState(seed)
        # Population stored as list of (arch_index, score) tuples
        # Kept in insertion order for regularized (oldest-first) removal
        self.population      = []
        self.visited         = set()

    def suggest(self, n: int) -> list:
        """
        Bootstrap with random archs until population is large enough,
        then generate children by mutating the best members.
        """
        results = []

        if len(self.population) < 16:
            # Bootstrap phase — need enough diversity before evolving
            attempts = 0
            while len(results) < n and attempts < n * 30:
                idx = int(self.rng.randint(0, TOTAL_ARCHS))
                if idx not in self.visited and is_valid_arch(idx):
                    results.append(idx)
                    self.visited.add(idx)
                attempts += 1

        else:
            # Evolution phase — mutate top-k members
            sorted_pop = sorted(
                self.population, key=lambda x: x[1], reverse=True
            )
            # Tournament pool: top third of population
            pool    = sorted_pop[:max(8, len(sorted_pop) // 3)]
            attempts = 0

            while len(results) < n and attempts < n * 30:
                # Pick a random parent from the tournament pool
                parent_idx, _ = pool[self.rng.randint(0, len(pool))]
                child_idx     = self._mutate(parent_idx)

                if child_idx not in self.visited and is_valid_arch(child_idx):
                    results.append(child_idx)
                    self.visited.add(child_idx)
                attempts += 1

        return results

    def _mutate(self, arch_index: int) -> int:
        """
        Produce a child architecture by changing exactly one edge.

        Why exactly one? Larger mutations produce architectures too
        different from the parent — you lose the locality that makes
        evolution effective. One-step mutations let the search do
        fine-grained hill climbing around good solutions.
        """
        ops            = index_to_ops(arch_index)[:]
        edge_to_mutate = self.rng.randint(0, NUM_EDGES)
        current_op_idx = OPERATIONS.index(ops[edge_to_mutate])

        # All operations except the current one
        alternatives   = [i for i in range(NUM_OPS) if i != current_op_idx]
        new_op_idx     = alternatives[self.rng.randint(0, len(alternatives))]

        ops[edge_to_mutate] = OPERATIONS[new_op_idx]
        return ops_to_index(ops)

    def update(self, evaluated: list, **kwargs):
        """
        Add evaluated architectures to the population.
        Remove oldest member if population exceeds max size.
        Uses val_accuracy if available, otherwise proxy_score.
        """
        for item in evaluated:
            score = item.get('val_accuracy') or item.get('proxy_score', 0.0)
            self.population.append((item['arch_index'], float(score)))
            self.visited.add(item['arch_index'])

        # Regularized removal: drop oldest when over capacity
        while len(self.population) > self.population_size:
            self.population.pop(0)

    def state(self) -> dict:
        if not self.population:
            return {'name': self.name, 'visited': 0, 'population': 0}

        scores   = [s for _, s in self.population]
        best_idx = max(self.population, key=lambda x: x[1])[0]

        return {
            'name':       self.name,
            'visited':    len(self.visited),
            'population': len(self.population),
            'best_arch':  best_idx,
            'best_score': round(float(max(scores)), 4),
            'avg_score':  round(float(np.mean(scores)), 4),
        }


class DARTSSearch:
    """
    DARTS-style weighted architecture search.

    This is the flagship strategy. It wraps ArchitectureWeights and
    adds the full search loop logic:

      Round 1 (proxy phase):
        - Sample 500 architectures using current weights (uniform at start)
        - Score all with zero-cost proxies (instant, no training)
        - Update weights from proxy scores
        - Anneal temperature

      Round 2 (training phase):
        - Sample top 20 candidates from updated weights
        - Train each for 20 epochs on CIFAR-10
        - Update weights from training results (stronger signal)
        - Anneal temperature again

    After two rounds the weight matrix has converged enough that
    best_architecture() reliably returns a strong candidate.

    Why is this better than random or evolutionary?
    - It learns across the entire 6-edge structure simultaneously
    - Operations that consistently appear in good architectures
      get reinforced on ALL edges at once
    - Temperature annealing focuses the search as it gets smarter
    """

    name = 'darts'

    def __init__(self, temperature: float = 1.0, seed: int = 42):
        self.weights = ArchitectureWeights(temperature=temperature, seed=seed)
        self.visited = set()
        self.round   = 0

    def suggest(self, n: int) -> list:
        """
        Sample n architectures using current architecture weights.
        Automatically excludes already-visited architectures.
        """
        batch = self.weights.sample_batch(n, exclude=self.visited)
        self.visited.update(batch)
        return batch

    def update(self, evaluated: list, source: str = 'proxy'):
        """
        Update architecture weights and anneal temperature.

        source: 'proxy' for zero-cost scores, 'training' for real accuracy
        The learning rate inside ArchitectureWeights.update() is
        automatically higher for training results (0.15 vs 0.05).
        """
        self.weights.update(evaluated, source=source)
        self.weights.anneal()
        self.round += 1

        for item in evaluated:
            self.visited.add(item['arch_index'])

    def state(self) -> dict:
        return {
            'name':           self.name,
            'round':          self.round,
            'visited':        len(self.visited),
            'temperature':    round(float(self.weights.temperature), 4),
            'best_arch':      self.weights.best_architecture(),
            'weight_summary': self.weights.to_dict(),
        }


# --------------------------------------------------------------------
# Factory
# --------------------------------------------------------------------

STRATEGIES = {
    'random':       RandomSearch,
    'evolutionary': EvolutionarySearch,
    'darts':        DARTSSearch,
}


def get_strategy(name: str, seed: int = 42):
    """
    Return a fresh strategy instance by name.
    Raises ValueError for unknown names.
    """
    if name not in STRATEGIES:
        raise ValueError(
            f"Unknown strategy '{name}'. "
            f"Choose from: {list(STRATEGIES.keys())}"
        )
    return STRATEGIES[name](seed=seed)