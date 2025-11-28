import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import requests
import pickle
import os
import time
from datetime import datetime

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"device={device}")


class BitcoinPredictor(nn.Module):
    def __init__(self, input_size):
        #plus rapide et quasiment aussi performant
        super(BitcoinPredictor, self).__init__()
        
        self.fc1 = nn.Linear(input_size, 128)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(64, 32)
        self.relu3 = nn.ReLU()
        self.drop3 = nn.Dropout(0.2)
        
        self.fc4 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()
        
        """
        super(BitcoinPredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(128, 64)
        self.relu3 = nn.ReLU()
        self.drop3 = nn.Dropout(0.2)
        
        self.fc4 = nn.Linear(64, 32)
        self.relu4 = nn.ReLU()
        self.drop4 = nn.Dropout(0.1)
        
        self.fc5 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()"""
    
    def forward(self, x):
        x = self.drop1(self.relu1(self.fc1(x)))
        x = self.drop2(self.relu2(self.fc2(x)))
        x = self.drop3(self.relu3(self.fc3(x)))
        x = self.sigmoid(self.fc4(x))
        return x


# charger le modèle et le scaler
def load_model_and_scaler():
    # vérifier les fichiers
    if not os.path.exists('bitcoin_model.pth'):
        print("Error: bitcoin_model.pth not found")
        return None, None, None, None
    
    if not os.path.exists('scaler.pkl'):
        print("Error: scaler.pkl not found")
        return None, None, None, None
        
    checkpoint = torch.load('bitcoin_model.pth')
    input_size =checkpoint['input_size']
    lookback = checkpoint['lookback']
    
    model = BitcoinPredictor(input_size).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    print("Model loaded")
    return model, scaler, input_size, lookback


# récupérer les données en temps réel
def fetch_live_data(lookback=10):
    url = "https://api.binance.com/api/v3/klines"
    params = {
        'symbol': 'BTCUSDT',
        'interval': '1h',
        'limit': lookback + 20  # +20 for rolling window calc
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        print(f"fetched {len(data)} candles")
        
        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore' 
        ])
        # garder seulement les colonnes importantes
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        
        cols_numeriques=['open', 'high', 'low', 'close', 'volume']
        for col in cols_numeriques:
            if cols_numeriques.count(col)==1:
                df[col] =pd.to_numeric(df[col])
        
        return df
    
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None


def prepare_live_features(df, lookback=10):
    # mêmes features que dans training
    df['return'] = (df['close'] - df['open']) / df['open']
    df['hl_range'] = (df['high'] - df['low']) / df['open']
    df['vol_norm'] = df['volume'] / df['volume'].rolling(window=20).mean()
    
    # remplir les valeurs manquantes
    df = df.bfill()
    df = df.fillna(1.0)  # fallback
    
    # prendre les dernières bougies
    window = df.iloc[-lookback:][['return', 'hl_range', 'vol_norm']].values
    features = window.flatten()
    
    bougie_actuelle = df.iloc[-1]  # current candle
    
    return features, bougie_actuelle


# prédire la prochaine bougie
def predict_next_candle(model, scaler, features):
    features_norm = scaler.transform([features])
    
    with torch.no_grad():
        tensor = torch.FloatTensor(features_norm).to(device)
        pred = model(tensor)
        prob = pred.item()
    
    # threshold at 0.5 - tried 0.55 but worse results
    if prob>0.5:
        resultat="GREEN" 
    else:
        resultat="RED" 
    if resultat=="GREEN":
        conf=prob
    else:
        conf=1-prob
    
    return resultat, conf


def display_prediction(current, prediction, confidence): 
    # déterminer la couleur actuelle
    couleur_actuelle="GREEN"
    if current['close']<current['open']:
        couleur_actuelle="RED"
    
    print(f"\nCurrent candle: {couleur_actuelle}")
    print(f"Prediction: {prediction} ({confidence*100:.1f}% confidence)\n")


# mode continu - refresh every hour
def continuous_mode(model, scaler, lookback):
    while True:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Fetching data...")
        
        df = fetch_live_data(lookback=lookback)

        if df is not None: 
            features, current = prepare_live_features(df, lookback=lookback)
            prediction, confidence = predict_next_candle(model, scaler, features)
            display_prediction(current, prediction, confidence)
        else:
            print("Failed to fetch data")

        print(f"Waiting until {(datetime.now() + pd.Timedelta(hours=1)).strftime('%H:%M:%S')}")
        time.sleep(3600)  # 1 heure


if __name__ == "__main__": 
    model, scaler, input_size, lookback = load_model_and_scaler()
    
    # vérifier si le chargement a fonctionné
    if model is None:
        exit(1)
    
    print("\n1. Single prediction")
    print("2. Continuous mode (every hour)")
    choix = input("\nChoice: ")
    
    if choix =='1': 
        df = fetch_live_data(lookback=lookback)
        
        if df is not None:
            features, current = prepare_live_features(df, lookback=lookback)
            prediction, confidence = predict_next_candle(model, scaler, features)
            display_prediction(current, prediction, confidence)
        else:
            print("Failed to fetch data")
    
    elif choix=='2': 
        continuous_mode(model, scaler, lookback)
    
    else:
        print("Invalid choice")
