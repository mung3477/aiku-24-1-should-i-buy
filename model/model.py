import torch
from torch import Tensor, nn

from .market_cube_embed import Seq2SeqTransformer
from .news_embed import NewsInterpreter


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

class BTCPricePredictor(nn.Module):
    def __init__(
            self,
            mrkt_cube_emb_size: int,
            stock_size: int,
            indctr_size: int,
            nli_model, nli_tokenizer, roberta_model, roberta_tokenizer, device):
        super(BTCPricePredictor, self).__init__()
        self.mrkt_transformer = Seq2SeqTransformer(
                                6,
                                6,
                                nhead=8,
                                emb_size=mrkt_cube_emb_size,
                                stock_size=stock_size,
                                indctr_size=indctr_size)
        self.FEDFormer = None
        self.news_interpreter = NewsInterpreter(nli_model, nli_tokenizer, roberta_model, roberta_tokenizer, device)
        self.mlp = FeedForward(
            dim=mrkt_cube_emb_size + 768 * 2,
            hidden_dim=128,
            dropout=0.1
        )

    def forward(self, btc_chart: Tensor, market_cube: Tensor, news: list):
        mrkt_emb = self.mrkt_transformer(market_cube)[:, 0]
        # btc_emb = self.FEDFormer(btc_chart)
        news_emb = self.news_interpreter(news)
        emb = torch.cat((mrkt_emb, news_emb), dim=1)
        return self.mlp(emb)
