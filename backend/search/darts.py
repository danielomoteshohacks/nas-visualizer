import numpy as np
from backend.bench.search_space import (
    OPERATIONS, EDGES, NUM_OPS, NUM_EDGES,
    index_to_ops, ops_to_index, is_valid_arch
)


class ArchitectureWeights:
    """
    Manages a learnable weight matrix over the search space.

    The core idea from DARTS (Liu et al., ICLR 2019):
    represent every possible architecture simultaneously as a
    weighted mixture. Each edge in the cell DAG has a probability
    distribution over its 5 possible operations.

    Shape: [6 edges × 5 operations]

    weights[i][j] answers: "how promising is operation j on edge i?"

    These weights start uniform and get updated every time we observe
    proxy scores or real training results. Over time the distribution
    sharpens — the search converges on what works.
    """

    def __init__(self, temperature: float = 1.0, seed: int = 42):
        self.rng         = np.random.RandomState(seed)
        self.temperature = temperature

        # Raw logits — unnormalized scores per (edge, operation).
        # We store logits not probabilities so updates are unbounded.
        # Initialized to zero = uniform distribution to start.
        self.logits = np.zeros((NUM_EDGES, NUM_OPS), dtype=np.float64)

        # Tracks how many top-scoring architectures used each operation
        # on each edge. Used for reporting and visualization.
        self.op_wins = np.zeros((NUM_EDGES, NUM_OPS), dtype=np.float64)

        # Full history of every update for debugging and visualization
        self.update_log = []

    # ------------------------------------------------------------------
    # Probability computation
    # ------------------------------------------------------------------

    def probabilities(self) -> np.ndarray:
        """
        Convert logits → probabilities via temperature-scaled softmax.

        Softmax(x_i) = exp(x_i / T) / Σ exp(x_j / T)

        Temperature T controls exploration vs exploitation:
          T = 1.0  → standard softmax, balanced exploration
          T = 0.5  → sharper, more focused on best operations
          T = 0.1  → near-deterministic, almost always picks the best
          T = 2.0  → flatter, more random exploration

        We subtract the row max before exp() — this is the
        log-sum-exp trick that prevents numerical overflow without
        changing the output values.
        """
        scaled = self.logits / max(self.temperature, 1e-8)
        scaled -= scaled.max(axis=1, keepdims=True)   # numerical stability
        exp    = np.exp(scaled)
        return exp / exp.sum(axis=1, keepdims=True)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample_one(self) -> int:
        """
        Sample one architecture index using current weights.

        For each of the 6 edges, sample one operation according
        to that edge's probability distribution. Combine into index.

        This is weighted random search — biased toward promising
        operations but still capable of exploring novel combinations.
        """
        probs    = self.probabilities()
        ops_list = []

        for edge_idx in range(NUM_EDGES):
            # np.random.choice with p= does weighted sampling
            op_idx = self.rng.choice(NUM_OPS, p=probs[edge_idx])
            ops_list.append(OPERATIONS[op_idx])

        return ops_to_index(ops_list)

    def sample_batch(self, n: int, exclude: set = None) -> list:
        """
        Sample n unique valid architectures.

        exclude: set of arch indices already visited — we never
                 suggest the same architecture twice in a run.
        """
        exclude  = exclude or set()
        results  = []
        attempts = 0

        while len(results) < n and attempts < n * 50:
            idx = self.sample_one()
            if idx not in exclude and is_valid_arch(idx):
                results.append(idx)
                exclude.add(idx)
            attempts += 1

        # Safety fallback: if weighted sampling gets stuck,
        # fill remaining slots with uniform random valid archs
        while len(results) < n:
            idx = int(self.rng.randint(0, 15625))
            if idx not in exclude and is_valid_arch(idx):
                results.append(idx)
                exclude.add(idx)

        return results

    # ------------------------------------------------------------------
    # Weight updates
    # ------------------------------------------------------------------

    def update(self, evaluated: list, source: str = 'proxy',
               lr: float = None):
        """
        Update logits based on evaluation results.

        evaluated: list of dicts, each must have:
                   'arch_index' and either 'proxy_score' or 'val_accuracy'
        source:    'proxy' or 'training' — determines score key and lr
        lr:        learning rate override

        Logic:
          1. Compute z-score of each architecture's performance
             relative to this batch (mean=0, std=1)
          2. For each architecture, add z_score × lr to the logit
             of every operation it uses
          3. Good architectures (z > 0) increase their operations' weights
             Bad architectures (z < 0) decrease their operations' weights

        Using z-scores (not raw scores) makes updates scale-invariant —
        it doesn't matter if proxy_score is in [0,1] or val_accuracy
        is in [0,100]. Only relative rank within the batch matters.
        """
        if not evaluated:
            return

        # Choose score key and learning rate based on source
        if source == 'proxy':
            score_key  = 'proxy_score'
            lr         = lr or 0.05   # conservative — proxy is noisy
        else:
            score_key  = 'val_accuracy'
            lr         = lr or 0.15   # stronger — training is reliable

        scores = np.array([
            a.get(score_key, 0.0) for a in evaluated
        ], dtype=np.float64)

        # Handle edge case: all scores identical → no useful signal
        if scores.std() < 1e-8:
            return

        # Normalize to z-scores
        z_scores = (scores - scores.mean()) / (scores.std() + 1e-8)

        for arch, z in zip(evaluated, z_scores):
            ops = index_to_ops(arch['arch_index'])

            for edge_idx, op_name in enumerate(ops):
                op_idx = OPERATIONS.index(op_name)
                self.logits[edge_idx][op_idx] += lr * z

                # Track wins (architectures significantly above average)
                if z > 0.5:
                    self.op_wins[edge_idx][op_idx] += 1

        self.update_log.append({
            'source':    source,
            'n':         len(evaluated),
            'mean':      float(scores.mean()),
            'best':      float(scores.max()),
            'lr':        lr,
        })

    # ------------------------------------------------------------------
    # Temperature annealing
    # ------------------------------------------------------------------

    def anneal(self, factor: float = 0.92):
        """
        Reduce temperature by factor to shift toward exploitation.

        Called at the end of each search round. As temperature drops,
        the softmax sharpens — the search increasingly commits to
        the operations it believes are best.

        Floor at 0.1 to prevent complete determinism (always leave
        some exploration capacity).
        """
        self.temperature = max(self.temperature * factor, 0.1)

    # ------------------------------------------------------------------
    # Discretization — extract final architecture
    # ------------------------------------------------------------------

    def best_architecture(self) -> int:
        """
        Extract the single best architecture from current weights.

        This is the DARTS 'discretization' step: take the continuous
        weight matrix and snap it to one architecture by picking the
        highest-weight operation on each edge (argmax).

        Returns the arch index of the predicted best architecture.
        """
        best_ops = [
            OPERATIONS[np.argmax(self.logits[edge_idx])]
            for edge_idx in range(NUM_EDGES)
        ]
        return ops_to_index(best_ops)

    # ------------------------------------------------------------------
    # Serialization for frontend
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """
        Serialize current state for the frontend visualization.

        Returns everything the React graph component needs to render
        the live architecture weight heatmap.
        """
        probs = self.probabilities()

        edge_data = []
        for edge_idx, (src, tgt) in enumerate(EDGES):
            best_op_idx = int(np.argmax(probs[edge_idx]))
            edge_data.append({
                'edge_index':  edge_idx,
                'source':      src,
                'target':      tgt,
                'weights':     {
                    op: round(float(probs[edge_idx][i]), 4)
                    for i, op in enumerate(OPERATIONS)
                },
                'best_op':     OPERATIONS[best_op_idx],
                'confidence':  round(float(probs[edge_idx][best_op_idx]), 4),
                'op_wins':     {
                    op: int(self.op_wins[edge_idx][i])
                    for i, op in enumerate(OPERATIONS)
                },
            })

        return {
            'temperature':    round(float(self.temperature), 4),
            'edges':          edge_data,
            'best_arch':      self.best_architecture(),
            'total_updates':  len(self.update_log),
        }