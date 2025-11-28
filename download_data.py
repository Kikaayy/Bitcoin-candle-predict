import pandas as pd
import requests
from datetime import datetime

# fonction pour télécharger les données
def fetch_bitcoin_data(limit=2000):
    print("Fetching Bitcoin data...")
    
    url = "https://api.binance.com/api/v3/klines"
    params = {
        'symbol': 'BTCUSDT',
        'interval': '1h',
        'limit': limit
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        print(f"got {len(data)} records")
        
        # créer dataframe à partir de la réponse
        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])

        # garder seulement les colonnes nécessaires
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        
        # convert to numeric
        cols=['open', 'high', 'low', 'close', 'volume']
        for col in cols:
            if cols.count(col)==1:  # check if column exists
                df[col] = pd.to_numeric(df[col])
            
        df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        print(f"Got {len(df)} rows from {df['date'].iloc[0]} to {df['date'].iloc[-1]}")
        return df
    
    except Exception as e:
        print(f"Error: {e}")
        return None


if __name__ == "__main__":
    df = fetch_bitcoin_data(limit=2000)
    
    if df is not None:
        filename = 'bitcoin_data.csv'
        df.to_csv(filename, index=False)
        print(f"Saved to {filename}")
        print("Run train_model.py next")
    else:
        print("Failed to download")
