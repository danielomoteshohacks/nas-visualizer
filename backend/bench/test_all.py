# test_all.py — run this to verify the entire data layer works

import sys
sys.path.insert(0, 'C:/Users/Daniel/nas-visualizer/backend')

print("Test 1: Search space...")
from bench.search_space import index_to_ops, ops_to_index, get_arch_info, TOTAL_ARCHS
assert TOTAL_ARCHS == 15625
info = get_arch_info(42)
assert len(info['operations']) == 6
assert len(info['edges']) == 6
assert ops_to_index(index_to_ops(42)) == 42
print(f"  PASS — {TOTAL_ARCHS} architectures, round-trip index check passed")
print(f"  Sample arch 42: {info['arch_string']}")

print("\nTest 2: Cell model builds correctly...")
from bench.cell import build_model_from_ops
import torch
ops = index_to_ops(42)
model = build_model_from_ops(ops, num_classes=10)
dummy = torch.randn(2, 3, 32, 32)
out = model(dummy)
assert out.shape == (2, 10), f"Expected (2,10), got {out.shape}"
print(f"  PASS — model output shape: {out.shape}")

print("\nTest 3: Zero-cost proxies run without error...")
from bench.proxies import compute_grad_norm, compute_synflow, compute_jacob_cov
from bench.cell import build_model_from_ops
data    = torch.randn(8, 3, 32, 32)
targets = torch.randint(0, 10, (8,))

m1 = build_model_from_ops(ops)
gn = compute_grad_norm(m1, data, targets)
assert gn >= 0, "grad_norm should be non-negative"
print(f"  grad_norm:  {gn:.4f}")

m2 = build_model_from_ops(ops)
sf = compute_synflow(m2, data)
print(f"  synflow:    {sf:.4f}")

m3 = build_model_from_ops(ops)
jc = compute_jacob_cov(m3, data)
print(f"  jacob_cov:  {jc:.4f}")
print("  PASS — all three proxies computed")

print("\nTest 4: Database initializes correctly...")
from bench.database import init_db, upsert_proxy_scores, get_stats
init_db()
upsert_proxy_scores({
    'arch_index':  42,
    'arch_string': info['arch_string'],
    'operations':  ops,
    'grad_norm':   gn,
    'synflow':     sf,
    'jacob_cov':   jc,
    'proxy_score': 0.5,
})
stats = get_stats()
assert stats['total_scored'] >= 1
print(f"  PASS — database has {stats['total_scored']} architecture(s) stored")

print("\nAll tests passed. Data layer is solid.")