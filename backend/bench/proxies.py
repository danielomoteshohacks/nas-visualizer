# proxies.py
#
# Zero-cost proxies — the secret weapon of this project.
#
# These functions score an architecture's quality using only
# a single forward+backward pass at random initialization.
# No training. No epochs. Results in milliseconds.
#
# We implement three proxies and combine them into an ensemble:
#
#   1. grad_norm  — sum of gradient magnitudes after one backward pass.
#                   Measures how much the network can learn from data.
#                   Higher = better signal flow.
#
#   2. synflow    — product of all weight norms through the network.
#                   Data-independent (no labels needed).
#                   Detects layer collapse — architectures where
#                   gradients vanish before reaching early layers.
#
#   3. jacob_cov  — covariance of the Jacobian (output w.r.t. input).
#                   Measures how differently the network responds to
#                   different inputs. Low variance = network is ignoring
#                   the data. High variance = network is expressive.
#
# Research basis: Abdelfattah et al., ICLR 2021 — "Zero-Cost Proxies
# for Lightweight NAS". All three proxies are from this paper and are
# evaluated on the NAS-Bench-201 search space specifically.

import torch
import torch.nn as nn
import numpy as np
from .cell import NASCell   # we'll build this next


def compute_grad_norm(model: nn.Module, data: torch.Tensor,
                      targets: torch.Tensor) -> float:
    """
    Gradient norm proxy.

    Run one forward pass, compute cross-entropy loss, backpropagate.
    Sum the L2 norm of all parameter gradients.
    A larger total gradient norm means the network has stronger
    learning signal — its weights are more responsive to the data.

    Returns a float score. Higher = better architecture.
    """
    model.zero_grad()
    model.train()

    output = model(data)
    loss = nn.CrossEntropyLoss()(output, targets)
    loss.backward()

    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            # .norm(2) computes the L2 (Euclidean) norm of this gradient tensor
            total_norm += p.grad.norm(2).item() ** 2

    # Return the square root to get the true L2 norm of the full gradient vector
    return float(total_norm ** 0.5)


def compute_synflow(model: nn.Module, data: torch.Tensor) -> float:
    """
    SynFlow proxy (Tanaka et al., 2020).

    Key insight: instead of using real data and labels, SynFlow uses
    all-ones inputs and computes a special loss = sum of all outputs.
    This makes the proxy completely data-independent.

    Why this works: it measures the product of all weight magnitudes
    along every path through the network. Architectures with at least
    one near-zero path (collapsed layers) get a very low score.
    It specifically catches the 'layer collapse' failure mode in NAS
    where some layers become effectively dead.

    Returns a float score. Higher = better architecture.
    """
    model.eval()

    # Replace all parameters with their absolute values temporarily.
    # This ensures gradients flow through even negative weights.
    @torch.no_grad()
    def linearize(m):
        signs = {}
        for name, param in m.named_parameters():
            signs[name] = torch.sign(param.data)
            param.data.abs_()
        return signs

    @torch.no_grad()
    def nonlinearize(m, signs):
        for name, param in m.named_parameters():
            param.data.mul_(signs[name])

    signs = linearize(model)

    # All-ones input — data-independent
    ones_input = torch.ones_like(data)
    ones_input.requires_grad_(False)

    model.zero_grad()
    output = model(ones_input)

    # Loss = sum of all output values (no labels needed)
    loss = output.sum()
    loss.backward()

    # SynFlow score = sum of (parameter * its gradient)
    # This computes the product along paths through the network
    score = 0.0
    for p in model.parameters():
        if p.grad is not None:
            score += (p.data * p.grad).sum().item()

    nonlinearize(model, signs)
    model.zero_grad()

    return float(score)


def compute_jacob_cov(model: nn.Module, data: torch.Tensor) -> float:
    """
    Jacobian covariance proxy (Mellor et al., 2021).

    Computes how differently the network responds to different inputs
    by measuring the covariance of the input-output Jacobian matrix.

    Low covariance = the network maps all inputs to similar outputs
                     = not expressive = bad architecture
    High covariance = the network distinguishes inputs well = good

    We use the log determinant of the covariance matrix as the score,
    which is a standard measure of matrix "volume" / expressiveness.

    Returns a float score. Higher = better architecture.
    """
    model.eval()
    model.zero_grad()

    data = data.clone().requires_grad_(True)
    output = model(data)

    # Compute the Jacobian: how does each output dimension respond
    # to changes in the input? We sum outputs to get a scalar,
    # then backpropagate to get gradients w.r.t. inputs.
    output.sum().backward()

    # data.grad is now shape [batch_size, C, H, W]
    # Flatten to [batch_size, -1] to get one vector per sample
    jacobian = data.grad.view(data.size(0), -1).cpu().numpy()

    # Compute covariance matrix: [batch_size, batch_size]
    # Each entry measures how similarly two samples activate the network
    K = jacobian @ jacobian.T

    # Add small diagonal to avoid numerical instability (singular matrix)
    K += 1e-5 * np.eye(K.shape[0])

    # Log determinant = log of the "volume" of the covariance ellipsoid
    # Higher = more diverse responses = more expressive network
    sign, logdet = np.linalg.slogdet(K)

    if sign <= 0:
        # Negative determinant means something went wrong numerically
        return -float('inf')

    return float(logdet)


def score_architecture(arch_index: int, num_classes: int = 10,
                       batch_size: int = 32,
                       input_size: int = 32) -> dict:
    """
    Compute all three proxy scores for a given architecture index.
    Returns a combined ensemble score and the individual scores.

    This is the main function called by the search algorithm.
    It takes ~50-200ms per architecture on CPU.
    """
    from .search_space import get_arch_info
    from .cell import build_model_from_ops

    arch_info = get_arch_info(arch_index)

    # Build the tiny model for this architecture
    model = build_model_from_ops(arch_info['operations'], num_classes=num_classes)
    model.eval()

    # Create a single random mini-batch — this is the only "data" we need
    # for zero-cost proxies. No real dataset required for scoring.
    torch.manual_seed(42)  # fixed seed for reproducibility
    dummy_data    = torch.randn(batch_size, 3, input_size, input_size)
    dummy_targets = torch.randint(0, num_classes, (batch_size,))

    scores = {}

    # --- Proxy 1: gradient norm ---
    try:
        model_gn = build_model_from_ops(arch_info['operations'], num_classes=num_classes)
        scores['grad_norm'] = compute_grad_norm(model_gn, dummy_data, dummy_targets)
    except Exception:
        scores['grad_norm'] = 0.0

    # --- Proxy 2: synflow ---
    try:
        model_sf = build_model_from_ops(arch_info['operations'], num_classes=num_classes)
        scores['synflow'] = compute_synflow(model_sf, dummy_data)
    except Exception:
        scores['synflow'] = 0.0

    # --- Proxy 3: jacobian covariance ---
    try:
        model_jc = build_model_from_ops(arch_info['operations'], num_classes=num_classes)
        scores['jacob_cov'] = compute_jacob_cov(model_jc, dummy_data)
    except Exception:
        scores['jacob_cov'] = -float('inf')

    # --- Ensemble score ---
    # We can't average the three scores directly because they live on
    # different scales (grad_norm might be 0.5, synflow might be 1e6).
    # Solution: rank-normalize each score across a sample, then average.
    # For a single architecture we store raw scores and normalize later
    # when we have a population to compare against.
    scores['arch_index']  = arch_index
    scores['arch_string'] = arch_info['arch_string']
    scores['operations']  = arch_info['operations']
    scores['edges']       = arch_info['edges']

    return scores