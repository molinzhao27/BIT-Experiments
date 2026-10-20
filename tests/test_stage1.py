"""Stage 1 objective: GradCache must give exactly the full-batch gradient; false negatives must be masked."""

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kunzdecoder.stage1 import AlignHead, gradcache_step  # noqa: E402


def _setup(seed=0, B=24, d_in=12, d_t=10):
    torch.manual_seed(seed)
    enc = nn.Sequential(nn.Linear(d_in, 16), nn.Dropout(0.3), nn.ReLU(), nn.Linear(16, 20))
    head = AlignHead(dim_neural=20, dim_text=d_t, proj_dim=8)
    x = torch.randn(B, 5, d_in)
    mask = torch.ones(B, 5, dtype=torch.long)
    mask[::3, 3:] = 0
    t = torch.randn(B, d_t)
    sid = torch.arange(B)
    sid[5] = sid[2]  # duplicated sentence -> masked false negative
    return enc, head, x, mask, t, sid


def _grads(*mods):
    return [p.grad.clone() for m in mods for p in m.parameters() if p.grad is not None]


def test_gradcache_equals_full_batch():
    enc, head, x, mask, t, sid = _setup()
    embed = lambda ch: head.embed_neural(enc(ch[0]), ch[1])  # noqa: E731
    dev = torch.device("cpu")
    # GradCache with 3 chunks
    loss_gc, stats = gradcache_step(embed, [(x[i:i + 8], mask[i:i + 8]) for i in range(0, 24, 8)],
                                    head.embed_text(t), sid, head, dev)
    g_gc = _grads(enc, head)
    # full batch with ordinary backprop, from the same RNG state the GradCache call started from
    enc.zero_grad(), head.zero_grad()
    _setup()  # re-seeds and consumes the RNG exactly as before the GradCache call (its modules are discarded)
    zs = [embed((x[i:i + 8], mask[i:i + 8])) for i in range(0, 24, 8)]
    loss_full, _ = head.loss(torch.cat(zs), head.embed_text(t), sid)
    loss_full.backward()
    g_full = _grads(enc, head)
    assert abs(loss_gc - float(loss_full)) < 1e-6
    for a, b in zip(g_gc, g_full):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
    assert stats["distinct_in_denominator"] == 23 and stats["masked_false_negatives"] == 1


def test_cosine_loss_has_no_text_head():
    head = AlignHead(dim_neural=20, dim_text=10, loss="cosine")
    assert head.text is None and head.neural.out_features == 10
    z = torch.nn.functional.normalize(torch.randn(4, 10), dim=-1)
    loss, _ = head.loss(z, z, torch.arange(4))
    assert float(loss) < 1e-6
