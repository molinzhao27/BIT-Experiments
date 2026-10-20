"""CPU-only reproduction of the exact BIT merge/loss methods, without checkpoint loads.
Run from repo root with CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 .venv/bin/python <this file>.
Local transformer version differs from the pinned GPU environment. The target-to-query index
and padding gradient conclusion are structural consequences of the repo's merge/loss code.
"""
import ast
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Tuple

import torch
import transformers
from transformers import Qwen2Config, Qwen2ForCausalLM

source = Path('external/bit/models/bci.py').read_text()
cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == 'BCI_AudioLLM')
ns = {'torch': torch, 'Optional': Optional, 'Tuple': Tuple, 'BCIOutput': SimpleNamespace}
for name in ['merge_embeds', 'forward']:
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), '<repo-bci-method>', 'exec'), ns)

class TinyBCI(torch.nn.Module):
    merge_embeds = ns['merge_embeds']
    forward = ns['forward']

    def __init__(self):
        super().__init__()
        config = Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                             num_attention_heads=4, num_key_value_heads=2, attention_dropout=0.0)
        self.llm = Qwen2ForCausalLM(config).cpu().eval()
        self.llm_config = SimpleNamespace(text_config=self.llm.config)
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')

    def prepare_embeds(self, input_ids, spikes, spikes_mask, spikes_timestamp, dataset_name, day_idx):
        return spikes, self.llm.get_input_embeddings()(input_ids), spikes_mask, None

torch.set_num_threads(1)
torch.manual_seed(17)
model = TinyBCI().eval()
real = torch.randn(1, 4, 32) * 0.1
padding = torch.randn(1, 20, 32) * 0.1

def run(padded, prompt):
    x = (torch.cat([real, padding], 1) if padded else real).clone().requires_grad_()
    ids = torch.tensor([[9, 11, 12, 2]] if prompt else [[11, 12, 2]])
    targets = ids.clone()
    if prompt:
        targets[:, 0] = -100
    mask = torch.ones(x.shape[:2], dtype=torch.long)
    if padded:
        mask[:, 4:] = 0
    split = torch.tensor([[0]])
    kwargs = dict(input_ids=ids, attention_mask=torch.ones_like(ids), input_split=split, spikes=x,
                  spikes_mask=mask, spikes_timestamp=torch.arange(x.shape[1]).view(1, -1),
                  spikes_lengths=torch.tensor([4]), targets=targets)
    output = model(**kwargs)
    first_label = int(torch.where(output.targets[0] != -100)[0][0])
    _, merged_mask, _ = model.merge_embeds(model.llm.get_input_embeddings()(ids), x, mask,
                                          torch.ones_like(ids), split, targets)
    first_logits = output.preds[0, first_label - 1]
    first_loss = torch.nn.functional.cross_entropy(first_logits.view(1, -1), output.targets[0, first_label].view(1))
    model.zero_grad()
    first_loss.backward()
    stats = {'first_label_index': first_label, 'predicting_position': first_label - 1,
             'predicting_position_mask': int(merged_mask[0, first_label - 1]),
             'first_word_loss': round(float(first_loss.detach()), 6),
             'masked_pad_grad_abs_sum': round(float(x.grad[:, 4:].abs().sum()) if padded else 0.0, 6)}
    return first_logits.detach(), stats

results = []
for with_prompt in [False, True]:
    a, unpadded_stats = run(False, with_prompt)
    b, padded_stats = run(True, with_prompt)
    results.append({'prompt': with_prompt, 'alone': unpadded_stats, 'padded': padded_stats,
                    'first_word_logit_max_abs_diff': round(float((a - b).abs().max()), 6)})
print(json.dumps({'device': 'cpu', 'torch': torch.__version__, 'transformers': transformers.__version__,
                  'results': results}, indent=2))
