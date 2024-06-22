from os import listdir

import pandas as pd


def MACD(data: pd.DataFrame):
	"""
		https://medium.com/@financial_python/building-a-macd-indicator-in-python-190b2a4c1777
	"""
	# Calculate the 12-period, 26-period EMA
	EMA12 = data['close'].ewm(span=12, adjust=False).mean()
	EMA26 = data['close'].ewm(span=26, adjust=False).mean()
	# Calculate MACD (the difference between 12-period EMA and 26-period EMA)
	MACD = EMA12 - EMA26
	return round(MACD, 2)

def RSI(dataset: pd.DataFrame, n:int = 14):
	"""
		https://stackoverflow.com/a/73115577/11004209
		SMA based RSI
	"""
	delta = dataset.loc[:, 'close'].diff()

	dUp, dDown = delta.copy(), delta.copy()
	dUp[dUp < 0] = 0
	dDown[dDown > 0] = 0

	RolUp = pd.Series(dUp).rolling(window=n).mean()
	RolDown = pd.Series(dDown).rolling(window=n).mean().abs()
	RS = RolUp / RolDown
	rsi= 100.0 - (100.0 / (1.0 + RS))
	return round(rsi, 2)

def add_MACD_RSI(filepath: str, dtype: dict):
	"""
		Add MACD, RSI field in csv file and save it
	"""
	dataset = pd.read_csv(filepath, dtype=dtype).iloc[::-1]
	dataset['MACD'] = MACD(dataset)
	dataset['RSI'] = RSI(dataset)
	dataset.to_csv(filepath)




if __name__ == "__main__":
	dtype = {'date': str, 'volume': float, 'open': float, 'high': float, 'low': float, 'close': float, 'adj close': float}

	filepaths = listdir("./full_history")
	filepaths = list(map(lambda fp: "./full_history/" + fp, filepaths))

	for fp in filepaths:
		add_MACD_RSI(fp, dtype=dtype)

