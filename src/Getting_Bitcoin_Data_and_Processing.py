import yfinance as yf
import pandas as pd

# 1. Definir o "ticker" do Bitcoin
ticker = "BTC-USD"

# 2. Criar o objeto
btc = yf.Ticker(ticker)

# 3. Baixar o histórico
hist_preco = btc.history(period="max", interval="1d")

# 4. Limpar e preparar os dados
df_preco = hist_preco[['Close', 'Volume']].copy()
df_preco.index = df_preco.index.tz_localize(None)  # remove fuso horário
df_preco.rename(columns={'Close': 'btc_price'}, inplace=True)

print("Dados de preço baixados:")
print(df_preco.head())

# 🔥 5. SALVAR COMO CSV
df_preco.to_csv("Data/raw/bitcoin_price_data.csv", index=True)
print("Arquivo salvo em Data/raw/bitcoin_price_data.csv")