import torch

from python.LLM.MLA import MultiHeadLatentAttention


def test_mla_returns_expected_shapes():
    mla = MultiHeadLatentAttention(embed_dim=8, num_heads=2)
    x = torch.randn(3, 4, 8)

    y, attn = mla(x)

    assert y.shape == (3, 4, 8)
    assert attn.shape == (3, 2, 4, 4)
