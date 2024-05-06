from datetime import date, timedelta

import pandas as pd
from torch.utils.data import DataLoader, Dataset, random_split

from lib import get_all_files, tail_from_row
from stocks import make_market_cube


class CustomCoinMarketDataset(Dataset):
    """Coin market dataset to make a dataloader

    Args:
        crypto_chart_fps (list[str]) : list of filepaths of cryptocurrency chart data files
        BTC_fp (str) : a filepath of BTC chart data file
        news_fps (list[str]) : list of filepaths of news files
        t (int) : time range to collect data


    return:
        x (???, Tensor[t, m, n], ???) : (BTC chart data for t days, coin market cube for t days, a news)
        y (float) : percantage of difference on close price after 7 days

    self:
        t (int) : time range to collect data
        crypto_chart_fps (list[str]) : list of filepaths of cryptocurrency chart data files
        BTC_chart (pd.DataFrame) : BTC chart data, indexed with date string. Can be accessed using .loc["2024-05-06"]
        news_fps (list[str]) : list of filepaths of news files


    """
    def __init__(self, crypto_chart_fps: list[str], BTC_fp: str, news_fps: list[str], t: int = 10):
        self.t = t
        self.crypto_chart_fps = crypto_chart_fps
        self.BTC_chart = pd.read_csv(BTC_fp).set_index("date")
        self.news_fps = news_fps

    def __len__(self):
        return len(self.news_fps)

    def __getitem__(self, idx):
        news = pd.read_csv(self.news_fps[idx])
        today_date = news["date"]

        market_cube = make_market_cube(self.crypto_chart_fps, self.t, target_date=today_date)

        btc_chart = tail_from_row(self.BTC_chart, self.t, lambda df: df.index.to_series() == today_date)

        today_price = self.BTC_chart.loc[today_date]
        next_w = date.fromisoformat(today_date) + timedelta(days=7)
        next_w_price = self.BTC_chart.loc[str(next_w)]
        label = (next_w_price - today_price) / today_price


        return btc_chart, market_cube, news, label

def make_dataloaders():
	# https://076923.github.io/posts/Python-pytorch-11/
	crypto_chart_fps = get_all_files("./full_history")
	news_fps = get_all_files("./news")


	dataset = CustomCoinMarketDataset(crypto_chart_fps, "./full_history/BTC.csv", news_fps)
	dataset_size = len(dataset)
	train_size = int(dataset_size * 0.8)
	validation_size = int(dataset_size * 0.1)
	test_size = dataset_size - train_size - validation_size

	train_dataset, validation_dataset, test_dataset = random_split(dataset, [train_size, validation_size, test_size])

	print(f"Training Data Size : {len(train_dataset)}")
	print(f"Validation Data Size : {len(validation_dataset)}")
	print(f"Testing Data Size : {len(test_dataset)}")

	train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, drop_last=True)
	validation_dataloader = DataLoader(validation_dataset, batch_size=4, shuffle=True, drop_last=True)
	test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True, drop_last=True)

	return train_dataloader, validation_dataloader, test_dataloader

