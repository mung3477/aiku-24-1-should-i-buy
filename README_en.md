# <div align="center"> So... Should I buy? </div>

## 🤗 Introduction

This repository is an implementation of the project "So.. should I buy?" by [AIKU](https://aiku.notion.site/AIKU-b614c69220704b848758e5cf21a54238?pvs=74) members, [mung3477](https://github.com/mung3477), [je1att0](https://github.com/je1att0), [iamnotwhale](https://github.com/iamnotwhale), [delaykimm](https://github.com/delaykimm).
<br>

We implemented a model that predicts a bitcoin price fluctuation using the historical data and bitcoin related articles. The model is based on the Transformer architecture and BERT based models.

### Dataset

We crawled news headlines from [cointelegraph](https://cointelegraph.com/tags/bitcoin) and [coindesk](https://www.coindesk.com/), and reddit threads from [bitcoin subreddit](https://www.reddit.com/r/Bitcoin/) using Reddit API. We used a twitter dataset from [kaggle](https://www.kaggle.com/datasets/kaushiksuresh147/bitcoin-tweets).
[TBD]

### Related works

We referred to the following papers and repositories for the implementation.<br>
[This paper](https://arxiv.org/pdf/2311.14759) proposed a combination of `Twitter-RoBERTa` & `BART MNLI` for predicting the bitcoin price based on the news articles.<br>
[This paper](https://arxiv.org/pdf/1809.03684.pdf) proposed a attention-based model that predicts the bitcoin price based on a cube shaped historical data, called a market cube.

### Model

-   We used the `Twitter-RoBERTa` & `BERT MNLI` for the news articles
-   We embedded a market cube with convolutional layers and added position embeddings, and fed the resulting sequence of vectors to a `standard transformer encoder`. In order to perform prediction, we used the standard approach of adding an extra learnable "prediction token" to the sequence.
-   We used a MLP to aggregate embeddings and predict a bitcoin price fluctuation range.

## 🎯 Performance

<div align="center">
TBD
</div>

## ⚙ Install Dependencies

This code is tested with python 3.11.0, torch 2.3.1

```
python -m pip install -r requirements.txt
```

## 🧱 Train

### Data Preparation

Download sampled news data and coin market data in the repository.

### Training

You can change hyperparameters in `train.py`.

```
python3 train.py
```

### 🔍 Test

The code will plot the predicted price fluctuation and the actual price fluctuation between <i>July 2023 and May 2024</i> in `btc-predicted.png`.

```
python3 test.py
```

### 🧶 Checkpoints

TBD

## 🔔 Note

-   The data we used is not permitted to be shared or used for commercial purposes.
