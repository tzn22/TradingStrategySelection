from preprocessing import download_data, preprocess_data
from data_config import (
    TRAIN_START_DATE, 
    TRAIN_END_DATE,
    TRADE_START_DATE, 
    TRADE_END_DATE
)
from finrl.config import INDICATORS
import pandas as pd

if __name__ == "__main__":
    tickers = pd.read_csv('prepared_csv/rusell1000_tickers.csv')
    tickers = tickers["Name"].tolist()

    download_output_path = "prepared_csv/russel1000.csv"

    download_data(tickers, 
                  output_path=download_output_path,
                  start_date=TRAIN_START_DATE,
                  end_date=TRADE_END_DATE)
    

    preprocess_data(df_path=download_output_path,
                    save_path_train="prepared_csv/russel1000_train_data.csv",
                    save_path_trade="prepared_csv/russel1000_trade_data.csv",
                    indicators=INDICATORS,
                    train_start_date=TRAIN_START_DATE,
                    train_end_date=TRAIN_END_DATE,
                    trade_start_date=TRADE_START_DATE,
                    trade_end_date=TRADE_END_DATE)