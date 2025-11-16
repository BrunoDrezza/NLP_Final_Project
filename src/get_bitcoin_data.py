# src/get_bitcoin_data.py

import yfinance as yf
import pandas as pd
from src.paths import RAW_DIR, PROCESSED_DIR

def main():
    ticker = "BTC-USD"
    btc = yf.Ticker(ticker)

    # histórico diário completo
    hist_preco = btc.history(period="max", interval="1d")

    # mantém só Close e Volume
    df_preco = hist_preco[["Close", "Volume"]].copy()
    df_preco.index = pd.DatetimeIndex(df_preco.index).tz_localize(None)
    df_preco.rename(columns={"Close": "btc_price"}, inplace=True)

    print("Dados de preço baixados (head):")
    print(df_preco.head())

    # RAW: cópia 'crua'
    raw_path = RAW_DIR / "bitcoin_price_data_raw.csv"
    df_preco.to_csv(raw_path, index_label="Date")
    print("Arquivo RAW salvo em:", raw_path)

    # PROCESSED: nome esperado pelo script de treino
    processed_path = PROCESSED_DIR / "btc_price_data.csv"
    df_preco.to_csv(processed_path, index_label="Date")
    print("Arquivo PROCESSED salvo em:", processed_path)


if __name__ == "__main__":
    main()
