# search_space.py
#
# This file defines the universe of all possible neural networks
# our system can consider — the search space.
#
# We implement the exact NAS-Bench-201 search space:
#   - A cell represented as a Directed Acyclic Graph (DAG)
#   - 4 nodes, 6 directed edges (every pair where source < target)
#   - 5 possible operations on each edge
#   - Total: 5^6 = 15,625 unique architectures
#
# Every architecture in our system is identified by a single integer
# from 0 to 15,624. This file converts between that integer and the
# actual list of operations it represents.

OPERATIONS = [
    'none',           # no connection — edge carries nothing
    'skip_connect',   # identity — input passes through unchanged
    'nor_conv_1x1',   # 1×1 convolution — mixes channels only
    'nor_conv_3x3',   # 3×3 convolution — spatial + channel mixing
    'avg_pool_3x3',   # 3×3 average pooling — spatial downsampling
]

# The 6 edges of the DAG as (source_node, target_node) pairs.
# Node 0 = input, Node 3 = output.
EDGES = [
    (0, 1),
    (0, 2),
    (0, 3),
    (1, 2),
    (1, 3),
    (2, 3),
]

NUM_OPS     = len(OPERATIONS)               # 5
NUM_EDGES   = len(EDGES)                    # 6
TOTAL_ARCHS = NUM_OPS ** NUM_EDGES          # 15,625


def index_to_ops(arch_index: int) -> list:
    """
    Convert an integer index into a list of 6 operation names.

    Think of arch_index as a base-5 number with 6 digits.
    Each digit (0-4) maps to one of the 5 operations.
    Each position corresponds to one of the 6 edges.

    index 0     → [none, none, none, none, none, none]
    index 1     → [skip_connect, none, none, none, none, none]
    index 15624 → [avg_pool_3x3, avg_pool_3x3, ..., avg_pool_3x3]
    """
    if not 0 <= arch_index < TOTAL_ARCHS:
        raise IndexError(
            f"arch_index must be 0 to {TOTAL_ARCHS - 1}, got {arch_index}"
        )
    ops = []
    remaining = arch_index
    for _ in range(NUM_EDGES):
        ops.append(OPERATIONS[remaining % NUM_OPS])
        remaining //= NUM_OPS
    return ops


def ops_to_index(ops: list) -> int:
    """
    Convert a list of 6 operation names back to its unique integer index.
    Exact inverse of index_to_ops().
    """
    if len(ops) != NUM_EDGES:
        raise ValueError(f"Need exactly {NUM_EDGES} ops, got {len(ops)}")
    index = 0
    for i, op in enumerate(ops):
        if op not in OPERATIONS:
            raise ValueError(f"Unknown op '{op}'")
        index += OPERATIONS.index(op) * (NUM_OPS ** i)
    return index


def ops_to_arch_string(ops: list) -> str:
    """
    Convert operations list to the standard NAS-Bench-201 string format.
    This is the format used in all published research papers.

    Output example:
    '|nor_conv_3x3~0|+|none~0|skip_connect~1|+|nor_conv_1x1~0|none~1|skip_connect~2|'

    The format groups edges by their target node.
    """
    # edges by target: node1 gets edge0, node2 gets edges1+3, node3 gets edges2+4+5
    groups = [
        [(ops[0], 0)],
        [(ops[1], 0), (ops[3], 1)],
        [(ops[2], 0), (ops[4], 1), (ops[5], 2)],
    ]
    parts = []
    for group in groups:
        parts.append('|' + '|'.join(f'{op}~{src}' for op, src in group) + '|')
    return '+'.join(parts)


def get_arch_info(arch_index: int) -> dict:
    """
    Return the full description of an architecture.
    This is the main function the rest of the system uses.
    """
    ops = index_to_ops(arch_index)
    return {
        'arch_index':  arch_index,
        'operations':  ops,
        'arch_string': ops_to_arch_string(ops),
        'edges': [
            {
                'source':    src,
                'target':    tgt,
                'operation': op,
                'op_index':  OPERATIONS.index(op),
            }
            for (src, tgt), op in zip(EDGES, ops)
        ],
        'nodes': [
            {'id': 0, 'label': 'input'},
            {'id': 1, 'label': 'h1'},
            {'id': 2, 'label': 'h2'},
            {'id': 3, 'label': 'output'},
        ]
    }


def get_random_indices(n: int, seed: int = None) -> list:
    """Return n random architecture indices."""
    import random
    rng = random.Random(seed)
    return rng.sample(range(TOTAL_ARCHS), min(n, TOTAL_ARCHS))


def is_valid_arch(arch_index: int) -> bool:
    """
    Check if an architecture is non-trivial.
    Architectures where ALL edges are 'none' are useless —
    no information flows from input to output.
    """
    ops = index_to_ops(arch_index)
    # At least one path from node 0 to node 3 must exist
    # Direct path: edge (0,3) = ops[2]
    # Path via node1: edges (0,1) and (1,3) = ops[0] and ops[4]
    # Path via node2: edges (0,2) and (2,3) = ops[1] and ops[5]
    # Path via node1+2: ops[0], ops[3], ops[5]
    has_direct     = ops[2] != 'none'
    has_via_node1  = ops[0] != 'none' and ops[4] != 'none'
    has_via_node2  = ops[1] != 'none' and ops[5] != 'none'
    has_via_both   = ops[0] != 'none' and ops[3] != 'none' and ops[5] != 'none'
    return has_direct or has_via_node1 or has_via_node2 or has_via_both