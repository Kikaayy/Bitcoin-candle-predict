import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import os
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# charger les données depuis csv
def load_data(filename='bitcoin_data.csv'):
    # vérifier si le fichier existe
    if not os.path.exists(filename):
        print(f"Error: {filename} not found")
        return None
    
    df = pd.read_csv(filename)
    print(f"Loaded {len(df)} rows")
    return df


def prepare_features(df, lookback=10):
    # features engineering - tried different combinations
    #######     "0123456789012345678901234567890123456789"
    df['return'] = (df['close'] - df['open']) / df['open']
    df['hl_range'] = (df['high'] - df['low']) / df['open']
    df['vol_norm'] = df['volume'] / df['volume'].rolling(window=20).mean()
    # df['momentum'] = df['close'].pct_change(5)  # didn't help much
    # df['volatility'] = df['return'].rolling(10).std()  # made it worse
    
    # target: haussière ?
    df['target'] = ((df['close'].shift(-1) - df['open'].shift(-1)) > 0).astype(int)
    
    df = df.bfill()
    df = df.fillna(1.0)
    
    X = []
    y = []
    
    for i in range(lookback, len(df) - 1):
        window = df.iloc[i-lookback:i][['return', 'hl_range', 'vol_norm']].values
        features = window.flatten()
        
        label = df.iloc[i]['target']
        
        # skip les données invalides
        is_valid=True
        for f in features:
            if str(f)=='nan' or str(f)=='inf' or str(f)=='-inf':  # manual check
                is_valid=False
                break
        if is_valid:
            X.append(features)
            y.append(label)
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"Created {len(X)} training examples")
    
    return X, y


# modèle de prédiction
class BitcoinPredictor(nn.Module):
    def __init__(self, input_size):
        super(BitcoinPredictor, self).__init__()
        
        # tried 256,128,64 too slow and overfitting
        # tried 64,32,16 accuracy was bad (<50%)
        # Lui return en mouyenne 60-65%
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
    
    def forward(self, x):
        x = self.drop1(self.relu1(self.fc1(x)))
        x = self.drop2(self.relu2(self.fc2(x)))
        x = self.drop3(self.relu3(self.fc3(x)))
        x = self.sigmoid(self.fc4(x))
        return x


def train_model(model, X_train, y_train, X_val, y_val, epochs=100, batch_size=64):
    # entraînement du modèle
    criterion = nn.BCELoss()
    
    # SGD me provoque des erreurs - lr=0.01 was too high, 0.0001 too slow
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    X_train = torch.FloatTensor(X_train).to(device)
    y_train = torch.FloatTensor(y_train).unsqueeze(1).to(device)
    X_val = torch.FloatTensor(X_val).to(device)
    y_val = torch.FloatTensor(y_val).unsqueeze(1).to(device)
    
    meilleure_acc = 0  # best accuracy tracking
    
    for epoch in range(epochs):
        model.train()
        
        perm = torch.randperm(X_train.size(0))
        
        total_loss=0
        for i in range(0, X_train.size(0), batch_size):
            idx = perm[i:i+batch_size]
            batch_x = X_train[idx]
            batch_y = y_train[idx]
            
            optimizer.zero_grad()
            out = model(batch_x)
            loss = criterion(out, batch_y)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # vérifier la validation
        model.eval()
        with torch.no_grad():
            val_out = model(X_val)
            preds = (val_out > 0.5).float()
            acc = (preds == y_val).float().mean()

            if acc>=meilleure_acc:
                meilleure_acc = acc
                
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Val Acc: {acc:.4f}")
            # print(f"Loss: {total_loss}") 
    
    print(f"\nBest validation accuracy: {meilleure_acc:.4f}")
    return model


if __name__ == "__main__":
    df = load_data('bitcoin_data.csv')
    if df is None:
        exit(1)

    lookback = 10  # nombre de bougies à regarder en arrière - moins provoque des bug et plus improve pas
    X, y = prepare_features(df, lookback=lookback)
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"\nTrain: {len(X_train)}, Val: {len(X_val)}")
    print(f"Green: {y.sum()}, Red: {len(y) - y.sum()}")
    
    input_size = X.shape[1]
    model = BitcoinPredictor(input_size).to(device)
    
    params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {params:,}\n")

    model = train_model(model, X_train, y_train, X_val, y_val, epochs=100, batch_size=64)
    
    # save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_size': input_size,
        'lookback': lookback
    }, 'bitcoin_model.pth')
    
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    print("\nModel saved to bitcoin_model.pth")
    print("Scaler saved to scaler.pkl")
