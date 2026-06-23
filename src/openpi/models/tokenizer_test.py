import numpy as np

from openpi.models import tokenizer as _tokenizer


class _FakeSentencePiece:
    def __init__(self):
        self.prompts = []

    def encode(self, prompt, add_bos=False):
        self.prompts.append((prompt, add_bos))
        return [2, 10, 11] if add_bos else [10, 11]


def test_tokenize():
    tokenizer = _tokenizer.PaligemmaTokenizer(max_len=10)
    tokens, masks = tokenizer.tokenize("Hello, world!")

    assert tokens.shape == (10,)
    assert masks.shape == (10,)


def test_fast_tokenizer():
    prompt = "Hello, world!"
    state = np.random.rand(5).astype(np.float32)
    action = np.random.rand(3, 2).astype(np.float32)
    tokenizer = _tokenizer.FASTTokenizer(max_len=256)
    tokens, token_masks, ar_masks, loss_masks = tokenizer.tokenize(prompt, state, action)

    assert tokens.shape == (256,)
    assert token_masks.shape == (256,)
    assert ar_masks.shape == (256,)
    assert loss_masks.shape == (256,)

    act = tokenizer.extract_actions(tokens, 3, 2)
    assert act.shape == (3, 2)


def test_tokenize_high_low_prompt_infer_builds_subtask_prefix_without_loss_tokens():
    tokenizer = _tokenizer.PaligemmaTokenizer.__new__(_tokenizer.PaligemmaTokenizer)
    tokenizer._max_len = 8
    tokenizer._tokenizer = _FakeSentencePiece()

    tokens, token_mask, ar_mask, loss_mask = tokenizer.tokenize_high_low_prompt_infer(
        "Spray_water!",
        np.array([-1.0, 0.0, 1.0], dtype=np.float32),
    )

    prompt, add_bos = tokenizer._tokenizer.prompts[0]
    assert prompt.startswith("Task: spray water.; State: ")
    assert prompt.endswith("; Subtask: ")
    assert add_bos is True
    assert tokens.tolist() == [2, 10, 11, False, False, False, False, False]
    assert token_mask.tolist() == [True, True, True, False, False, False, False, False]
    assert ar_mask.tolist() == [1, 1, 1, 0, 0, 0, 0, 0]
    assert loss_mask.tolist() == [False, False, False, False, False, False, False, False]
