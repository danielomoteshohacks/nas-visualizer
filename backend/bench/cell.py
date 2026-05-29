# cell.py
#
# This file builds actual PyTorch neural network models from
# an architecture description (list of 6 operations).
#
# The architecture is one "cell" — a small computational block.
# In the real NAS-Bench-201 setup, a full network stacks this cell
# multiple times with downsampling between stages.
# We build a simplified but faithful version: 3 stacked cells
# with a final linear classifier.
#
# This is the model that gets instantiated and evaluated by the
# zero-cost proxies and the short training runs.

import torch
import torch.nn as nn
import torch.nn.functional as F
from .search_space import OPERATIONS, EDGES


class ConvBnRelu(nn.Module):
    """
    Convolution → BatchNorm → ReLU block.

    This is the standard building block for all conv operations
    in the NAS-Bench-201 search space. BatchNorm stabilizes training
    and ReLU introduces the non-linearity that makes deep networks work.
    """
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, stride: int = 1, padding: int = 0):
        super().__init__()
        self.op = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.op(x)


class NASCellOp(nn.Module):
    """
    A single operation on one edge of the cell DAG.

    Given an operation name (e.g. 'nor_conv_3x3'), this module
    applies the corresponding transformation to the input tensor.
    """
    def __init__(self, op_name: str, channels: int):
        super().__init__()
        self.op_name = op_name

        if op_name == 'none':
            # No operation — return zeros. This edge carries nothing.
            self.op = None

        elif op_name == 'skip_connect':
            # Identity — pass input through unchanged.
            # No parameters. Just returns x.
            self.op = nn.Identity()

        elif op_name == 'nor_conv_1x1':
            # 1×1 convolution — mixes channel information only.
            # Doesn't look at spatial neighbors, just combines channels.
            self.op = ConvBnRelu(channels, channels, kernel_size=1, padding=0)

        elif op_name == 'nor_conv_3x3':
            # 3×3 convolution — the workhorse of vision networks.
            # Looks at each pixel and its 8 neighbors. padding=1 keeps
            # the spatial dimensions the same (same-padding).
            self.op = ConvBnRelu(channels, channels, kernel_size=3, padding=1)

        elif op_name == 'avg_pool_3x3':
            # 3×3 average pooling — replaces each pixel with the average
            # of its 3×3 neighborhood. No learned parameters.
            # padding=1 keeps spatial dimensions the same.
            self.op = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

        else:
            raise ValueError(f"Unknown operation: {op_name}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.op_name == 'none':
            # Return a zero tensor of the same shape as input
            return torch.zeros_like(x)
        return self.op(x)


class NASCell(nn.Module):
    """
    One cell of the NAS architecture — a DAG with 4 nodes and 6 edges.

    Each node aggregates (sums) all incoming edge outputs.
    The cell output is node 3.

    Node 0 = input to the cell
    Node 1, 2 = intermediate nodes
    Node 3 = cell output
    """
    def __init__(self, operations: list, channels: int):
        """
        operations: list of 6 operation names, one per edge
        channels:   number of channels for all feature maps in this cell
        """
        super().__init__()

        # Build one op module per edge, stored in a ModuleList so PyTorch
        # tracks their parameters for gradient computation
        self.ops = nn.ModuleList([
            NASCellOp(op_name, channels)
            for op_name in operations
        ])

        # Store edge connectivity for the forward pass
        # EDGES = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
        self.edges = EDGES

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # node_outputs[i] holds the tensor at node i
        node_outputs = [None] * 4
        node_outputs[0] = x   # node 0 is always the cell input

        for edge_idx, (src, tgt) in enumerate(self.edges):
            # Apply the operation on this edge
            edge_output = self.ops[edge_idx](node_outputs[src])

            # Aggregate into the target node by summation
            if node_outputs[tgt] is None:
                node_outputs[tgt] = edge_output
            else:
                node_outputs[tgt] = node_outputs[tgt] + edge_output

        # If node 3 received nothing (all paths had 'none'), return zeros
        if node_outputs[3] is None:
            return torch.zeros_like(x)

        return node_outputs[3]


class NASNetwork(nn.Module):
    """
    A full network built from NAS cells.

    Architecture (faithful to NAS-Bench-201):
        stem → [cell × N → downsample] × 3 stages → global avg pool → classifier

    Simplified from the original (16 channels instead of 64, 3 cells per stage
    instead of 5) so it runs fast on CPU for zero-cost proxy scoring and
    short training runs.
    """
    def __init__(self, operations: list, num_classes: int = 10,
                 channels: int = 16, cells_per_stage: int = 3):
        super().__init__()

        # Stem: initial feature extraction before the cells
        # Takes 3-channel RGB input → channels feature maps
        self.stem = ConvBnRelu(3, channels, kernel_size=3, padding=1)

        # Stage 1: cells at full resolution
        self.stage1 = nn.Sequential(*[
            NASCell(operations, channels)
            for _ in range(cells_per_stage)
        ])

        # Downsample 1: halve spatial dimensions, double channels
        self.down1 = ConvBnRelu(channels, channels * 2,
                                kernel_size=3, stride=2, padding=1)
        channels *= 2

        # Stage 2: cells at half resolution
        self.stage2 = nn.Sequential(*[
            NASCell(operations, channels)
            for _ in range(cells_per_stage)
        ])

        # Downsample 2
        self.down2 = ConvBnRelu(channels, channels * 2,
                                kernel_size=3, stride=2, padding=1)
        channels *= 2

        # Stage 3: cells at quarter resolution
        self.stage3 = nn.Sequential(*[
            NASCell(operations, channels)
            for _ in range(cells_per_stage)
        ])

        # Global average pooling: collapses spatial dimensions to 1×1
        # This is how modern CNNs go from feature maps to a flat vector
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Final linear classifier
        self.classifier = nn.Linear(channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.down1(x)
        x = self.stage2(x)
        x = self.down2(x)
        x = self.stage3(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)   # flatten [batch, channels, 1, 1] → [batch, channels]
        x = self.classifier(x)
        return x


def build_model_from_ops(operations: list, num_classes: int = 10) -> NASNetwork:
    """
    Build a NASNetwork from a list of 6 operation names.
    This is the single entry point for creating models.
    """
    return NASNetwork(operations, num_classes=num_classes)