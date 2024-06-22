import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (AutoModel, AutoTokenizer, RobertaModel,
                          RobertaTokenizer)

from src.dataset import make_dataloaders
from src.lib import evaluate, train_epoch
from src.model import BTCPricePredictor


def train(hyperparams: dict, stock_size: int, chart_columns: list, train_dataloader: DataLoader, val_dataloader:DataLoader,  device="cpu"):
	from timeit import default_timer as timer

	nli_tokenizer = AutoTokenizer.from_pretrained("NbAiLab/nb-bert-base-mnli")
	nli_model = AutoModel.from_pretrained("NbAiLab/nb-bert-base-mnli")
	roberta_tokenizer = RobertaTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
	roberta_model = RobertaModel.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")

	model = BTCPricePredictor(
		mrkt_cube_emb_size=hyperparams["mrkt_cube_emb_size"],
		stock_size=stock_size,
		indctr_size=len(chart_columns) - 1,
		nli_tokenizer=nli_tokenizer,
		nli_model=nli_model,
		roberta_tokenizer=roberta_tokenizer,
		roberta_model=roberta_model,
		device=device)

	model = model.to(device)

	loss_fn = torch.nn.HuberLoss()

	optimizer = hyperparams["optim"](model.parameters(), lr=hyperparams["lr"], betas=hyperparams["betas"], eps=hyperparams["eps"] )

	NUM_EPOCHS = 0

	for epoch in tqdm(range(1, NUM_EPOCHS+1), desc="Train Epoch"):
		start_time = timer()
		train_loss = train_epoch(model, optimizer, loss_fn, train_dataloader=train_dataloader, device=device)
		if train_loss is None:
			break;
		end_time = timer()

		val_loss = evaluate(model, optimizer, loss_fn, validation_dataloader=val_dataloader, device=device)
		print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))

	# Save the model
	torch.save(model.state_dict(), "model.pth")

if __name__ == "__main__":
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	chart_columns = ["date", "volume", "open", "high", "low", "close", "MACD", "RSI"]
	dataloaders = make_dataloaders(chart_columns=chart_columns)

	hyperparams = dict()
	hyperparams["mrkt_cube_emb_size"] = 192
	hyperparams["optim"] = torch.optim.Adam
	hyperparams["lr"] = 0.0001
	hyperparams["betas"] = (0.9, 0.98)
	hyperparams["eps"] = 1e-9
	train(
		hyperparams=hyperparams,
		stock_size=len(dataloaders["crypto_chart_fps"]),
		chart_columns=chart_columns,
		train_dataloader=dataloaders["train"],
		val_dataloader=dataloaders["validation"],
		device=device
	)
