import os

import torch
from transformers import (AutoModel, AutoTokenizer, RobertaModel,
                          RobertaTokenizer)

from dataset import make_dataloaders
from lib import test
from model import BTCPricePredictor


def plot_prediction(result):
	# https://matplotlib.org/stable/gallery/showcase/stock_prices.html
	import matplotlib.pyplot as plt
	import matplotlib.transforms as mtransforms
	import numpy as np
	import pandas as pd

	stock_data = pd.read_csv("./coin_data/BTC.csv", usecols=["date", "close"])
	stock_data["date"] = pd.to_datetime(stock_data["date"])

	fig, ax = plt.subplots(1, 1, figsize=(9, 12), layout='constrained')

	# These are the colors that will be used in the plot
	ax.set_prop_cycle(color=[
	    '#1f77b4', '#ff9896'])

	stocks_name = ['Bitcoin', 'Predicted']
	stocks_ticker = ['BTC', 'OURS']

	"""
	Plot btc data
	"""
	# Plot each line separately with its own color.
	# don't include any data with NaN.
	good = np.nonzero(np.isfinite(stock_data["close"]))
	line, = ax.plot(stock_data['date'].loc[good], stock_data["close"].loc[good], lw=2.5)

	# Add a text label to the right end of every line. Most of the code below
	# is adding specific offsets y position because some labels overlapped.
	y_pos = stock_data["close"].iloc[-1]

	# Use an offset transform, in points, for any text that needs to be nudged
	# up or down.
	offset = 0 / 72
	trans = mtransforms.ScaledTranslation(0, offset, fig.dpi_scale_trans)
	trans = ax.transData + trans

	# Again, make sure that all labels are large enough to be easily read
	# by the viewer.
	ax.text(np.datetime64('2024-05-22'), y_pos, "Bitcoin",
	        color=line.get_color(), transform=trans)

	"""
	Plot test set result
	"""
	result["date"] = pd.to_datetime(result["date"])
	line, = ax.plot(result['date'], result["price"], 's', marker='.')

	"""
	Format the plots
	"""
	ax.set_xlim(np.datetime64('2023-07-01'), np.datetime64('2024-05-20'))

	fig.suptitle("Bitcoin close prices (2023.07-2024.05)",
	             ha="center")

	# Remove the plot frame lines. They are unnecessary here.
	ax.spines[:].set_visible(False)

	# Ensure that the axis ticks only show up on the bottom and left of the plot.
	# Ticks on the right and top of the plot are generally unnecessary.
	ax.xaxis.tick_bottom()
	ax.yaxis.tick_left()
	# ax.set_yscale('log')

	# Provide tick lines across the plot to help your viewers trace along
	# the axis ticks. Make sure that the lines are light and small so they
	# don't obscure the primary data lines.
	ax.grid(True, 'major', 'both', ls='--', lw=.5, c='k', alpha=.3)

	# Remove the tick marks; they are unnecessary with the tick lines we just
	# plotted. Make sure your axis ticks are large enough to be easily read.
	# You don't want your viewers squinting to read your plot.
	ax.tick_params(axis='both', which='both', labelsize='large',
	               bottom=False, top=False, labelbottom=True,
	               left=False, right=False, labelleft=True)

	# Finally, save the figure as a PNG.
	# You can also save it as a PDF, JPEG, etc.
	# Just change the file extension in this call.
	# fig.savefig('stock-prices.png', bbox_inches='tight')
	plt.show()

def test_model(model: BTCPricePredictor, test_dataloader, loss_fn, device):

	_, accuracy, result = test(model, test_dataloader, loss_fn, device)
	print(f"accuracy: {accuracy:.2f}%")
	plot_prediction(result);

if __name__ == "__main__":
	assert os.path.exists("model.pth"), "model.pth does not exist"

	MRKT_CUBE_EMB_SIZE = 192

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	chart_columns = ["date", "volume", "open", "high", "low", "close", "MACD", "RSI"]
	dataloaders = make_dataloaders(chart_columns=chart_columns)
	test_dataloader = dataloaders["test"]

	nli_tokenizer = AutoTokenizer.from_pretrained("NbAiLab/nb-bert-base-mnli")
	nli_model = AutoModel.from_pretrained("NbAiLab/nb-bert-base-mnli")
	roberta_tokenizer = RobertaTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
	roberta_model = RobertaModel.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")

	model = BTCPricePredictor(
		mrkt_cube_emb_size=MRKT_CUBE_EMB_SIZE,
		stock_size=len(dataloaders["crypto_chart_fps"]),
		indctr_size=len(chart_columns) - 1,
		nli_tokenizer=nli_tokenizer,
		nli_model=nli_model,
		roberta_tokenizer=roberta_tokenizer,
		roberta_model=roberta_model,
		device=device)

	model = model.to(device)

	model.load_state_dict(torch.load("model.pth"))

	loss_fn = torch.nn.HuberLoss()
	test_model(model, test_dataloader, loss_fn, device)
