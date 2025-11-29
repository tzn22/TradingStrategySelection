import pandas as pd

from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.config import INDICATORS
from src.data_config import *
import itertools


def preprocess_data(
        df_path: str,
        save_path_train: str,
        save_path_trade: str,
        indicators=INDICATORS,
        train_start_date=TRAIN_START_DATE,
        train_end_date=TRAIN_END_DATE,
        trade_start_date=TRADE_START_DATE,
        trade_end_date=TRADE_END_DATE
        
):
    df_finrl = pd.read_csv(df_path)

    fe = FeatureEngineer(use_technical_indicator=True,
                        tech_indicator_list = indicators,
                        use_vix=True,
                        use_turbulence=True,
                        user_defined_feature = False)

    processed = fe.preprocess_data(df_finrl)


    list_ticker = processed["tic"].unique().tolist()
    list_date = list(pd.date_range(processed['date'].min(),processed['date'].max()).astype(str))
    combination = list(itertools.product(list_date,list_ticker))

    processed_full = pd.DataFrame(combination,columns=["date","tic"]).merge(processed,on=["date","tic"],how="left")
    processed_full = processed_full[processed_full['date'].isin(processed['date'])]
    processed_full = processed_full.sort_values(['date','tic'])

    train = data_split(processed_full, train_start_date, train_end_date)
    trade = data_split(processed_full, trade_start_date, trade_end_date)
    print(len(train))
    print(len(trade))


    train.to_csv(save_path_train)
    trade.to_csv(save_path_trade)
