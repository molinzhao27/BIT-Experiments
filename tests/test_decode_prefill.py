"""Skipping the unused vocabulary projection during diffusion prefix prefill, with real tiny-model decodes.

The reference runtime switch restores the original prefill projection, including revision and score recording.
Independent head-call checks verify the expected removed work. Needs BIT's training environment.
"""

import pytest
import torch

pytest.importorskip("peft")

from test_diffusion_revision import BS, D, EOS, GREEDY, NUCLEUS, bci, tiny_llm


def decode(model, prefix, options, seed=72):
    model.trace = []
    model.commit_logprobs = []
    calls = []
    hook = model._parts()[1].register_forward_pre_hook(lambda _, args: calls.append(tuple(args[0].shape)))
    try:
        torch.manual_seed(seed)
        with torch.no_grad(), model._layout(training=False):
            ids = model._denoise(prefix, eos=EOS, **options)
        return ids, model.trace, model.commit_logprobs, torch.get_rng_state().clone(), calls
    finally:
        hook.remove()


@pytest.mark.parametrize("length", [17, 32, 33, 63, 64, 97])
@pytest.mark.parametrize("sampler", [
    {},
    {"editing_threshold": 0.5, "max_post_steps": 2, "max_edits": 2},
    {"remask_threshold": 0.95, "remask_rounds": 1},
], ids=["plain", "editing", "remasking"])
def test_prefill_keeps_decodes_scores_trace_and_rng(length, sampler):
    llm = tiny_llm().eval()
    optimized, reference = bci(llm, **sampler), bci(llm, **sampler)
    optimized._skip_unused_prefill_logits = True
    reference._skip_unused_prefill_logits = False
    prefix = torch.randn(length, D, generator=torch.Generator().manual_seed(length))
    for options in (GREEDY, NUCLEUS, GREEDY | {"max_new_tokens": 70}):
        actual = decode(optimized, prefix, options)
        expected = decode(reference, prefix, options)
        assert actual[:3] == expected[:3]  # tokens, every proposal/edit event, exact recorded log-probabilities
        assert torch.equal(actual[3], expected[3])
        omitted = 1 if length >= BS and length % BS else 0
        assert actual[4] == expected[4][omitted:]
        if omitted:
            assert expected[4][0] == (1, length // BS * BS, D)


@pytest.mark.parametrize("length", [17, 32, 33, 63, 64, 97])
def test_prefill_projects_only_consumed_hidden_states(length):
    model = bci(tiny_llm().eval())
    model._skip_unused_prefill_logits = True
    prefix = torch.randn(length, D, generator=torch.Generator().manual_seed(length))
    decoder_calls = []
    hook = model._parts()[0].register_forward_pre_hook(
        lambda _, args, kwargs: decoder_calls.append((tuple(kwargs["inputs_embeds"].shape),
                                                     kwargs["update_past_key_values"])), with_kwargs=True)
    try:
        result = decode(model, prefix, GREEDY)
    finally:
        hook.remove()
    skipped = length >= BS and length % BS != 0
    assert len(result[4]) == len(decoder_calls) - int(skipped)
    if length >= BS:
        assert decoder_calls[0] == ((1, length // BS * BS, D), True)
    expected_head_shapes = [shape for shape, _ in decoder_calls][int(skipped):]
    assert result[4] == expected_head_shapes
