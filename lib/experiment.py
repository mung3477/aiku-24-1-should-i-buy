import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader


def train_epoch(model, optimizer: Optimizer, loss_fn, train_dataloader: DataLoader, device):
	model.train()
	losses = 0

	for btc_chart, market_cube, news, train_labels in train_dataloader:
		btc_chart = btc_chart.to(device)
		market_cube = market_cube.to(device)
		train_labels = train_labels.float().to(device)

		logits = model(btc_chart, market_cube, news)

		optimizer.zero_grad()

		loss = loss_fn(logits.squeeze(), train_labels) * 100
		loss.backward()

		optimizer.step()
		losses += loss.item()

	return losses / len(train_dataloader)


def evaluate(model, loss_fn, validation_dataloader: DataLoader, device):
	model.eval()
	losses = 0
	with torch.no_grad():
		for btc_chart, market_cube, news, val_labels in validation_dataloader:
			btc_chart = btc_chart.to(device)
			market_cube = market_cube.to(device)
			val_labels = val_labels.float().to(device)

			logits = model(btc_chart, market_cube, news)

			loss = loss_fn(logits.squeeze(), val_labels) * 100
			losses += loss.item()

	return losses / len(validation_dataloader)

def test(model, test_dataloader: DataLoader, loss_fn, device):
	from datetime import date, timedelta

	import pandas as pd
	from tqdm import tqdm


	model.eval()
	losses = 0
	correct = 0

	BTC_Close = pd.read_csv("./coin_data/BTC.csv", usecols=["date", "close"])
	BTC_Close["date"] = pd.to_datetime(BTC_Close["date"])
	BTC_Close = BTC_Close.set_index("date")

	dates = list()
	preds = list()

	with torch.no_grad():
		for btc_chart, market_cube, news, test_labels in tqdm(test_dataloader):
			btc_chart = btc_chart.to(device)
			market_cube = market_cube.to(device)
			test_labels = test_labels.float().to(device)

			logits = model(btc_chart, market_cube, news)

			loss = loss_fn(logits.squeeze(), test_labels) * 100
			losses += loss.item()

			correct += (torch.sub(logits.squeeze(), test_labels).abs() <= 0.01).sum().item()

			batch_dates = list(map(lambda dt: str(date.fromisoformat(dt) + timedelta(days=10)), news[1]))
			batch_prices = list(map(lambda dt: BTC_Close.loc[dt]["close"], batch_dates))
			dates += batch_dates
			preds += (torch.Tensor(batch_prices) + torch.Tensor(batch_prices) * logits.squeeze().cpu()).tolist()

	accuracy = correct / len(test_dataloader) * 100.0

	return losses / len(test_dataloader), accuracy, pd.DataFrame(list(zip(dates, preds)), columns=["date", "price"])
