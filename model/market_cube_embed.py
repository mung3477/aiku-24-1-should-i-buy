"""
    https://pytorch.org/tutorials/beginner/translation_transformer.html?highlight=nn%20transformer
"""

import math

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Transformer


# helper Module that adds positional encoding to the token embedding to introduce a notion of word order.
class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])

# helper Module to convert tensor of input market cubes into corresponding tensor of temporal embeddings
class MarketCubeEmbedding(nn.Module):
    def __init__(self, stock_size: int = 507, indctr_size: int = 7, emb_size: int = 192):
        super(MarketCubeEmbedding, self).__init__()
        self.h = stock_size
        self.w = indctr_size
        self.conv = nn.Conv2d(1, emb_size, (self.h, self.w))
        self.btc_token = nn.Parameter(torch.zeros(1, 1, emb_size))

    def forward(self, market_cube: Tensor):
        """
            param:
                market_cube - Tensor[batch_size, temporal_size, stock_size, indctr_size]
            return:
                Tensor[batch_size, temporal_size, emb_size]
        """
        batch_size = market_cube.size(0)
        temporal_size = market_cube.size(1)
        market_cube_emb = self.conv(market_cube.view(-1, 1, self.h, self.w))
        market_cube_emb = market_cube_emb.view(batch_size, temporal_size, -1)
        # add BTC token
        btc_tokens = self.btc_token.expand(batch_size, -1, -1)
        market_cube_emb = torch.cat((btc_tokens, market_cube_emb), dim=1)

        return market_cube_emb

# Seq2Seq Network
class Seq2SeqTransformer(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 stock_size: int,
                 indctr_size: int,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1,
                 DEVICE = 'cpu'):
        super(Seq2SeqTransformer, self).__init__()
        self.device = DEVICE
        self.transformer = Transformer(d_model=emb_size,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout,
                                       batch_first=True)
        self.market_cube_emb = MarketCubeEmbedding(
                                        stock_size=stock_size,
                                        indctr_size=indctr_size,
                                        emb_size=emb_size)
        self.positional_encoding = PositionalEncoding(
                                        emb_size,
                                        dropout=dropout)

    def create_mask(self, src: Tensor):
        """
            param:
                src - Tensor[batch_size, temporal_size, stock_size, indicator_size]
            return:
                Tensor[temporal_size, temporal_size]
        """
        emb_seq_len = src.shape[1] + 1

        src_mask = torch.zeros((emb_seq_len, emb_seq_len),device=self.device).type(torch.bool)
        return src_mask

    def forward(self, src: Tensor):
        """
            return:
                Tensor[batch_size, temporal_size + 1, emb_size]
        """
        return self.transformer.encoder(
                    self.positional_encoding(self.market_cube_emb(src)),
                    self.create_mask(src)
        )

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x)
