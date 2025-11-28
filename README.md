# Bitcoin Candle Predictor

Simple neural network that tries to predict if the next BTC hourly candle will be green or red. Pulls data from Binance and trains on historical patterns.

## What it does

Looks at the last 10 hourly candles and uses 3 features per candle:
- Return (open to close movement)
- High/Low range relative to open
- Volume normalized against 20-period average

Gets around 60-85% accuracy which is honestly pretty decent for something this simple.

## Setup

Install dependencies:
```bash
pip install torch pandas numpy scikit-learn requests matplotlib
```

Then run in order:
```bash
python download_data.py    # grabs historical data
python train_model.py       # trains the model
python use_model_live.py    # make predictions
```

The live prediction script lets you either predict once or keep it running to update every hour.

## Model

Pretty straightforward neural net:
- Input: 30 features (10 candles × 3 features)
- Hidden layers: 128 → 64 → 32 neurons with dropout
- Output: sigmoid for binary classification

I tried adding more features like momentum and volatility but they actually made it worse. Also bigger networks just overfit.

## Warning

Don't actually trade with this. It's just a fun project to mess around with ML and crypto data. If you do trade based on this and lose money that's on you.