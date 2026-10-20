"""Each stacked model must equal the standard torch.nn network with the same weights,
and stacked models must train independently of each other."""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kunzdecoder import seqmodels  # noqa: E402


def test_cnn_matches_conv1d():
    torch.manual_seed(0)
    X = torch.randn(6, 40, 10)
    net = seqmodels.StackedCNN(3, 10, 7, channels=4)
    out = net(X, train=False)
    P = net.params
    for m in range(3):
        w1 = P["W1"][m].reshape(10, 5, 4).permute(2, 0, 1)  # (C, F, k)
        h = F.gelu(F.conv1d(X.transpose(1, 2), w1, P["b1"][m].reshape(-1)))
        h = F.gelu(F.conv1d(h, P["W2"][m], P["b2"][m], dilation=2))
        feat = torch.cat([h.mean(-1), h.amax(-1)], -1)
        assert torch.allclose(out[m], feat @ P["Wo"][m] + P["bo"][m], atol=1e-5)


def test_gru_matches_nn_gru():
    torch.manual_seed(0)
    X = torch.randn(5, 12, 8)
    net = seqmodels.StackedGRU(2, 8, 7, proj=6, hidden=4)
    out = net(X, train=False)
    P = net.params
    for m in range(2):
        gru = torch.nn.GRU(6, 4, batch_first=True)
        with torch.no_grad():
            gru.weight_ih_l0.copy_(P["Wih"][m].T)
            gru.weight_hh_l0.copy_(P["Whh"][m].T)
            gru.bias_ih_l0.copy_(P["bih"][m].reshape(-1))
            gru.bias_hh_l0.copy_(P["bhh"][m].reshape(-1))
        z = X @ P["Wp"][m] + P["bp"][m, 0]
        seq, h_last = gru(z)
        feat = torch.cat([h_last[0], seq.mean(1)], -1)
        assert torch.allclose(out[m], feat @ P["Wo"][m] + P["bo"][m], atol=1e-5)


def test_stacked_models_learn_and_respect_masks():
    rng = np.random.default_rng(0)
    y = np.repeat(np.arange(7), 12)
    X = rng.normal(size=(84, 30, 10)).astype(np.float32)
    X[:, 10:20, :7] += 1.5 * np.eye(7)[y][:, None, :]
    X = torch.as_tensor(X)
    mask = torch.ones(2, 84)
    mask[1, ::2] = 0  # model 1 trains on half the trials
    logits = seqmodels.train_eval("cnn", {"channels": 8}, X, torch.as_tensor(y), mask, torch.tensor([0.01, 0.01]),
                                  [60], seed=0, jitter=2)
    acc = (logits[0].argmax(-1) == y).mean(-1)
    assert acc[0] > 0.8 and acc[1] > 0.6
    held_out = (logits[0, 1, ::2].argmax(-1) == y[::2]).mean()
    assert held_out > 0.5  # generalizes to its unseen trials
