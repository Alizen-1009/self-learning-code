import torch

from python.LLM.MHA import MultiHeadAttention


def test_mha_returns_expected_shapes():
    mha = MultiHeadAttention(embed_dim=8, num_heads=2)
    x = torch.randn(3, 4, 8)

    y, attn = mha(x)

    assert y.shape == (3, 4, 8)
    assert attn.shape == (3, 2, 4, 4)
