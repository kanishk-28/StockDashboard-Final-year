import pandas as pd
import yfinance as yf
import numpy as np
import joblib

from ta.trend import MACD, EMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# ================= DATA =================
df = yf.download("RELIANCE.NS", period="2y")

close = df['Close']

df['RSI'] = RSIIndicator(close).rsi()
df['EMA20'] = EMAIndicator(close).ema_indicator()

macd = MACD(close)
df['MACD'] = macd.macd()
df['MACD_SIGNAL'] = macd.macd_signal()

bb = BollingerBands(close)
df['BB_HIGH'] = bb.bollinger_hband()
df['BB_LOW'] = bb.bollinger_lband()

# ================= TARGET =================
df['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)

df.dropna(inplace=True)

X = df[['RSI','EMA20','MACD','MACD_SIGNAL','BB_HIGH','BB_LOW']]
y = df['Target']

# ================= SPLIT =================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# ================= SCALE =================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ================= MODEL =================
model = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05)
model.fit(X_train, y_train)

# ================= EVAL =================
pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))

# ================= SAVE =================
joblib.dump(model, "ml_model.pkl")
joblib.dump(scaler, "scaler.pkl")