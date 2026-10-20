"""Small CPU-only audit reproductions. These assert existing defects, not fixes.

Run from the project root:
  CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 .venv/bin/python docs/audits/2026-10-19/reproduce.py
No models are downloaded, no remote commands run, and all scratch writes use a temporary directory.
"""
from __future__ import annotations

import ast
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))

import editdistance
import pandas as pd
import torch
from safetensors.torch import save_file
from kunzdecoder import phase4_eval as ev
from kunzdecoder.stage2 import load_model_state, slim_checkpoint

torch.set_num_threads(1)
torch.set_default_device('cpu')
result = {'device': 'cpu'}

with tempfile.TemporaryDirectory(prefix='kunz-audit-repro-') as scratch:
    tmp = Path(scratch)

    # Execute the real small BIT metric functions without importing its training stack.
    tree = ast.parse((ROOT / 'external/bit/utils/eval_llm.py').read_text())
    names = {'word_edit_distance', 'word_error_count', 'format_ctc', 'per'}
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    ns = {'torch': torch, 'json': json, 'deepcopy': deepcopy, 'editdistance': editdistance}
    exec(compile(ast.Module(body=funcs, type_ignores=[]), '<actual-bit-metrics>', 'exec'), ns)
    vocab = ['BLANK', 'AA', 'BB']
    decoded = ns['format_ctc']([1, 0, 1], vocab, 0)
    assert decoded == ['AA']
    result['ctc_blank_repeat'] = {'actual': decoded, 'correct': ['AA', 'AA']}
    vocab_file = tmp / 'vocab.json'
    vocab_file.write_text(json.dumps(vocab))
    cfg = SimpleNamespace(method=SimpleNamespace(model_kwargs=SimpleNamespace(blank_id=0)),
                          data=SimpleNamespace(vocab_file=str(vocab_file)))
    logits = torch.nn.functional.one_hot(torch.tensor([[1, 2, 2], [1, 2, 0]]), 3).float()
    per = ns['per'](None, {'spikes': torch.zeros(2, 3, 1), 'spikes_lengths': torch.tensor([1, 3])},
                    {'phonemes': [['AA'], ['AA', 'BB']]}, {'preds': logits}, cfg, n_print=0)
    assert abs(float(per) - 1 / 3) < 1e-6
    result['ctc_padded_validation'] = {'actual_per': float(per), 'correct_per': 0.0}

    # Two distinct current experiment names with the same arm and seed.
    cfg = ev.load_config()
    scored = pd.DataFrame({'trial': [0], 'reference': ['hello'], 'decoded': ['hello'],
                           'word_errors': [0], 'ref_words': [1], 'bs_p': [1.0], 'bs_r': [1.0], 'bs_f1': [1.0]})
    source = tmp / 'decodes.csv'
    scored.assign(bertscore_hash=ev.bertscore_settings(cfg)['bertscore_hash']).to_csv(source, index=False)
    meta = {'arm': 'B', 'seed': 42, 'split': 'brandman_2024_text:test', 'decoding': 'greedy', 'condition': 'real'}
    for model in ['stage2 B_ssl_s42', 'stage2 B_ssl_qwen25-1.5b-instruct_s42']:
        ev.evaluate_file(source, meta | {'model': model}, cfg, tmp / 'eval')
    metrics = pd.read_csv(tmp / 'eval/metrics.csv')
    artifacts = list((tmp / 'eval/per_trial').glob('*.csv'))
    assert len(metrics) == len(artifacts) == 1
    result['evaluator_collision'] = {'models_scored': 2, 'metric_rows': len(metrics), 'per_trial_files': len(artifacts)}

    # Local row numbers are not sufficient evidence of the same trial/reference.
    real = pd.concat([scored] * 3, ignore_index=True).assign(trial=[0, 1, 2])
    unrelated = real.assign(reference='unrelated', decoded='', word_errors=1, bs_p=0.0, bs_r=0.0, bs_f1=0.0)
    nulls = ev.null_controls(real, {name: unrelated for name in ev.NULLS}, n_boot=32)
    assert bool(nulls.beats_all_nulls.all())
    result['unrelated_nulls_accepted'] = {'beats_all_nulls': bool(nulls.beats_all_nulls.all())}

    # Unverified cached numeric columns override contradictory actual text.
    scored.assign(decoded='wrong').to_csv(tmp / 'unverified.csv', index=False)
    accepted = ev.load_or_score(tmp / 'unverified.csv', cfg)
    assert int(accepted.word_errors.iloc[0]) == 0
    result['unverified_scores_accepted'] = {'reported_word_errors': 0, 'correct_word_errors': 1}

    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Linear(1, 1, bias=False)
            self.llm = torch.nn.Linear(1, 1, bias=False)
            self.llm.weight.requires_grad_(False)

    model = Toy()
    ckpt = tmp / 'BEST'
    ckpt.mkdir()
    with torch.no_grad():
        model.encoder.weight.fill_(1)
    slim_checkpoint(model, ckpt)
    with torch.no_grad():
        model.encoder.weight.fill_(2)
    save_file({k: v.detach().contiguous() for k, v in model.state_dict().items()}, str(ckpt / 'model.safetensors'))
    loaded = load_model_state(model, ckpt)
    weight = float(model.encoder.weight.detach())
    assert weight == 1
    result['stale_slim_checkpoint'] = {'new_full_weight': 2, 'loaded_weight': weight, 'chosen_file': loaded['file']}

print(json.dumps(result, indent=2))
