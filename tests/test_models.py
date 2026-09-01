"""Tests for model components."""
import torch
from coffe.models.components import (
    ChannelTokenizer,
    SpatialTokenizer,
    AuxTokenizer,
)


class TestTokenizers:
    def test_channel_tokenizer(self):
        tok = ChannelTokenizer(in_channels=144, embed_dim=128)
        x = torch.randn(2, 144, 11, 11)
        out = tok(x)
        assert out.shape == (2, 128, 11, 11)

    def test_spatial_tokenizer(self):
        tok = SpatialTokenizer(embed_dim=128)
        x = torch.randn(2, 128, 11, 11)
        out = tok(x)
        assert out.shape == (2, 121, 128)

    def test_aux_tokenizer(self):
        tok = AuxTokenizer(in_channels=1, patch_size=11, embed_dim=128)
        x = torch.randn(2, 1, 11, 11)
        out = tok(x)
        assert out.shape == (2, 1, 128)
