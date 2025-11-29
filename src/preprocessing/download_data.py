import pandas as pd
import yfinance as yf

from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from src.data_config import *


def download_data(
        tickers: list[str],
        output_path: str,
        start_date=TRAIN_START_DATE, 
        end_date=TRADE_END_DATE,
    ):

    df_finrl = YahooDownloader(
        start_date = start_date,
        end_date = end_date,
        ticker_list = tickers
    ).fetch_data()


    df_finrl.to_csv(output_path, index=False)

    