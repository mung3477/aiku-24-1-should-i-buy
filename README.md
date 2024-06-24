# ... 그래서 사요?

📢 2024년 1학기 [AIKU](https://github.com/AIKU-Official) 활동으로 진행한 프로젝트입니다<br>
🎉 2024년 1학기 AIKU Conference 열심히상 수상!

## 소개

이 레포지토리는 [고려대학교 인공지능 학회 AIKU](https://aiku.notion.site/AIKU-b614c69220704b848758e5cf21a54238?pvs=74)의 2024년 1학기 프로젝트로 진행된 **"그래서 사요?"** 프로젝트의 구현입니다. <br>

저희는 역사적 차트 데이터와 비트코인 관련 기사를 사용하여 비트코인 가격 변동을 예측하는 모델을 구현했습니다. 모델은 Transformer 아키텍처와 BERT 기반 모델을 기반으로 합니다.

## 방법론

저희가 구현한 모델은 하나의 기사 또는 텍스트와 10일치 암호화폐 500 여 종목의 차트 데이터를 입력으로 받아, 10일 후의 비트코인 가격 변동을 예측하는 모델입니다. 모델은 `Twitter-RoBERTa` & `BERT MNLI` 기반의 텍스트 임베딩 모델과 `transformer encoder` 기반의 암호화폐 시장 상황 임베딩 모델을 MLP를 이용해 결합하여 구현했습니다.

-   Twitter-RoBERTa: SNS 데이터에 특화된 RoBERTa 모델을 사용하여 텍스트 데이터를 임베딩합니다.
-   BERT MNLI: 텍스트 데이터와 `"The Bitcoin Price is likely to continue rising."` 이라는 가설 문장이 논리적으로 부합하는지를 판단합니다.
-   Market cube embedding: 암호화폐 시장 상황을 3차원 큐브로 표현하여, `conv2d`를 통해 각 날짜 별 시장 상황을 임베딩하고, 위치 임베딩을 추가하여 시퀀스 벡터를 얻어 `standard transformer encoder`에 입력합니다. 예측을 수행하기 위해 시퀀스에 추가적인 학습 가능한 "예측 토큰"을 추가하는 표준적인 방법을 사용했습니다.

### 관련 연구

아래와 같은 연구들을 참조하였습니다.

-   [이 논문](https://arxiv.org/pdf/2311.14759)은 뉴스 기사를 바탕으로 비트코인 가격을 예측하기 위해 `Twitter-RoBERTa` & `BERT MNLI`를 결합한 모델을 제안했습니다.
-   [이 논문](https://arxiv.org/pdf/1809.03684.pdf)은 역사적 데이터를 바탕으로 큐브 모양으로 쌓아, 비트코인 가격을 예측하기 위한 어텐션 기반 모델을 제안했습니다.

### 데이터셋

-   뉴스 데이터: [cointelegraph](https://cointelegraph.com/tags/bitcoin), [coindesk](https://www.coindesk.com/)에서 뉴스 헤드라인을 크롤링하였습니다.
-   트위터 데이터: [kaggle](https://www.kaggle.com/datasets/kaushiksuresh147/bitcoin-tweets)에서 제공하고 있는 비트코인 관련 트위터 데이터를 사용했습니다.
-   레딧 데이터: Reddit API를 사용하여 [bitcoin subreddit](https://www.reddit.com/r/Bitcoin/)에서 레딧 쓰레드를 크롤링하였습니다.
-   암호화폐 가격 데이터: [binance](https://www.binance.com/en/landing/data)에서 제공하고 있는 K-lines, candlestick 데이터를 사용했습니다.
    -   거래량, 시초가, 고가, 저가, 종가, [MACD](https://en.wikipedia.org/wiki/MACD), [RSI](https://en.wikipedia.org/wiki/Relative_strength_index)를 사용하였습니다. 두 지표의 계산 방법은 [여기](https://github.com/mung3477/AIKU_BTC_Project/blob/main/src/stocks/indicators.py)에서 확인할 수 있습니다.
    -   10일치 데이터를 하나의 큐브로 표현하였습니다.
-   성능 검증 단계의 정확성을 위해, test 과정에서 학습에 사용된 데이터와 다른 시기의 데이터를 사용했습니다.
    -   학습 데이터: 2017년 8월부터 2022년 6월까지의 데이터
    -   validation 데이터: 2022년 7월부터 2023년 6월까지의 데이터
    -   검증 데이터: 2023년 7월부터 2024년 5월까지의 데이터

## 환경 설정

이 프로젝트는 python 3.11.0, torch 2.3.1 버전에서 테스트되었습니다.
아래 코드를 실행하면 필요한 라이브러리를 설치할 수 있습니다.

```
python -m pip install -r requirements.txt
```

## 사용 방법

-   이 레포지토리에 있는 샘플 뉴스 데이터와 코인 시장 데이터를 다운로드합니다.
-   학습을 진행하려면 `train.py`를 실행합니다. 해당 코드에서 하이퍼파라미터를 변경할 수 있습니다.
-   성능을 테스트하려면 `test.py`를 실행합니다. 코드는 *2023년 7월부터 2024년 5월*까지의 예측된 가격 변동과 실제 가격 변동을 `btc-predicted.png`에 그려냅니다.

## 예시 결과

(사용 방법을 실행했을 때 나타나는 결과나 시각화 이미지를 보여주세요) TBD

## 팀원

-   [mung3477](https://github.com/mung3477): <br>
    팀장, 프로젝트 코드 전반 및 마켓 큐브 임베딩 모델 구현, 나스닥 데이터 수집
-   [je1att0](https://github.com/je1att0): <br>
    암호화폐 데이터 수집, 모델 학습 및 차트 패턴 임베딩 모델 구현 시도
-   [iamnotwhale](https://github.com/iamnotwhale), [delaykimm](https://github.com/delaykimm): <br>
    뉴스, 트위터, 레딧 데이터 수집, 텍스트 데이터 임베딩 모델 구현
