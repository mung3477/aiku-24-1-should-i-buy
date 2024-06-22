import os
from datetime import date, timedelta
from typing import Callable

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from lib import get_all_files, tail_from_row
from stocks import make_market_cube, use_market_image_from_date


class CustomCoinMarketDataset(Dataset):
	"""Coin market dataset to make a dataloader

	Args:
		crypto_chart_fps (list[str]) : list of filepaths of cryptocurrency chart data files
		BTC_fp (str) : a filepath of BTC chart data file
		news_fp (str) : a filepath to the news csvfile
		t (int) : time range to collect data
		chart_columns (list) : list of chart_columns to use

	return:
		x (Tenser[t, 1, n], Tensor[t, m, n], ???) : (BTC chart data for t days, coin market cube for t days, a news)
		y (float) : percantage of difference on close price after 7 days

	self:
		t (int) : time range to collect data
		crypto_chart_fps (list[str]) : list of filepaths of cryptocurrency chart data files
		BTC_chart (pd.DataFrame) : BTC chart data, indexed with date string. Can be accessed using .loc["2024-05-06"]
		news_df (pd.DataFrame): a dataframe that contains news data.


	"""
	def __init__(
		self,
		crypto_chart_fps: list[str],
		BTC_fp: str,
		news_fp: str,
		t: int,
		chart_columns: list
	):
		self.t = t
		self.crypto_chart_fps = crypto_chart_fps
		self.chart_columns = chart_columns
		BTC_chart = pd.read_csv(BTC_fp)

		news_df = pd.read_csv(news_fp, usecols=["text", "date", "category"])
		#news_df = pd.read_csv(news_fp)
		news_date_limit = str(date.fromisoformat(BTC_chart.sort_values("date").iloc[0]["date"]) + timedelta(days=t))
		self.news_df = news_df.loc[news_df["date"] >= news_date_limit]
		self.BTC_chart = BTC_chart.set_index("date")

	def __len__(self):
		return len(self.news_df)

	def __getitem__(self, idx):
		news = self.news_df.iloc[idx]
		today_date = news["date"]

		cube_file_name = f"./market_cube/{today_date}.pt"
		market_cube = torch.load(cube_file_name) \
						if os.path.isfile(cube_file_name) \
						else make_market_cube(
							filepaths=self.crypto_chart_fps,
							t=self.t,
							chart_columns=self.chart_columns,
							tail_f=use_market_image_from_date(today_date)
						)

		market_cube = torch.nan_to_num(market_cube)


		today_price = self.BTC_chart.loc[today_date]["close"]
		next_w = date.fromisoformat(today_date) + timedelta(days=7)
		next_w_price = self.BTC_chart.loc[str(next_w)]["close"]
		label = (next_w_price - today_price) / today_price

		btc_chart = tail_from_row(self.BTC_chart, self.t, lambda df: df.index.to_series() == today_date)
		btc_chart = torch.Tensor(btc_chart.values)

		return btc_chart, market_cube, news.to_list(), label

def split_news_data(condition: Callable[[pd.DataFrame], bool], save_name: str):
	assert os.path.exists("./10k_sampled_news_dataset_balanced.csv"), "10k_sampled_news_dataset_balanced.csv does not exist"
	news = pd.read_csv("./10k_sampled_news_dataset_balanced.csv", usecols=["text", "date", "category"], dtype={"text": str, "date": str, "category": str})
	news = news.loc[condition(news) & (news["category"] != "category"), :]
	news.to_csv(save_name, index=False)

def make_dataloaders(chart_columns: list)->dict:
	split_news_data(lambda news: news["date"] < "2022-07-01", "./news_data_train.csv")
	split_news_data(lambda news: (news["date"] >= "2022-07-01") & (news["date"] < "2023-07-01"), "./news_data_val.csv")
	split_news_data(lambda news: news["date"] >= "2023-07-01", "./news_data_test.csv")

	# https://076923.github.io/posts/Python-pytorch-11/
	crypto_chart_fps = get_all_files("./coin_data")

	train_dataset = CustomCoinMarketDataset(crypto_chart_fps, "./coin_data/BTC.csv", "./news_data_train.csv", t=10, chart_columns=chart_columns)
	validation_dataset = CustomCoinMarketDataset(crypto_chart_fps, "./coin_data/BTC.csv", "./news_data_val.csv", t=10, chart_columns=chart_columns)
	test_dataset = CustomCoinMarketDataset(crypto_chart_fps, "./coin_data/BTC.csv", "./news_data_test.csv", t=10, chart_columns=chart_columns)

	print(f"Training Data Size : {len(train_dataset)}")
	print(f"Validation Data Size : {len(validation_dataset)}")
	print(f"Testing Data Size : {len(test_dataset)}")

	train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, drop_last=True)
	validation_dataloader = DataLoader(validation_dataset, batch_size=4, shuffle=True, drop_last=True)
	test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True, drop_last=True)

	dataloaders = dict()
	dataloaders["train"] = train_dataloader
	dataloaders["validation"] = validation_dataloader
	dataloaders["test"] = test_dataloader
	dataloaders["crypto_chart_fps"] = crypto_chart_fps

	return dataloaders

