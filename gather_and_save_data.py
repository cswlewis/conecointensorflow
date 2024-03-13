import numpy as np
import pandas as pd
from TickerData import *

def gather_and_save_data():
    # Load and prepare the testing data
    df = TickerData().df

    df2 = pd.DataFrame()

    columns_to_normalize = [ 'volume_adi', 'volume_obv', 'volume_cmf', 'volume_fi', 'volume_em',
       'volume_sma_em', 'volume_vpt', 'volume_vwap', 'volume_mfi',
       'volume_nvi', 'volatility_bbm', 'volatility_bbh', 'volatility_bbl',
       'volatility_bbw', 'volatility_bbp', 'volatility_bbhi',
       'volatility_bbli', 'volatility_kcc', 'volatility_kch', 'volatility_kcl',
       'volatility_kcw', 'volatility_kcp', 'volatility_kchi',
       'volatility_kcli', 'volatility_dcl', 'volatility_dch', 'volatility_dcm',
       'volatility_dcw', 'volatility_dcp', 'volatility_atr', 'volatility_ui',
       'trend_macd', 'trend_macd_signal', 'trend_macd_diff', 'trend_sma_fast',
       'trend_sma_slow', 'trend_ema_fast', 'trend_ema_slow',
       'trend_vortex_ind_pos', 'trend_vortex_ind_neg', 'trend_vortex_ind_diff',
       'trend_trix', 'trend_mass_index', 'trend_dpo', 'trend_kst',
       'trend_kst_sig', 'trend_kst_diff', 'trend_ichimoku_conv',
       'trend_ichimoku_base', 'trend_ichimoku_a', 'trend_ichimoku_b',
       'trend_stc', 'trend_adx', 'trend_adx_pos', 'trend_adx_neg', 'trend_cci',
       'trend_visual_ichimoku_a', 'trend_visual_ichimoku_b', 'trend_aroon_up',
       'trend_aroon_down', 'trend_aroon_ind', 'trend_psar_up',
       'trend_psar_down', 'trend_psar_up_indicator',
       'trend_psar_down_indicator', 'momentum_rsi', 'momentum_stoch_rsi',
       'momentum_stoch_rsi_k', 'momentum_stoch_rsi_d', 'momentum_tsi',
       'momentum_uo', 'momentum_stoch', 'momentum_stoch_signal', 'momentum_wr',
       'momentum_ao', 'momentum_roc', 'momentum_ppo', 'momentum_ppo_signal',
       'momentum_ppo_hist', 'momentum_pvo', 'momentum_pvo_signal',
       'momentum_pvo_hist', 'momentum_kama', 'others_dr', 'others_dlr',
       'others_cr'] 
    
    lookbacks = 10
    minutesToDecideOver = 6 

    X = np.zeros((df[columns_to_normalize].shape[0]-lookbacks, lookbacks, df[columns_to_normalize].shape[1]))
    y = np.zeros((df[columns_to_normalize].shape[0]-lookbacks,))

    dataLength = df[columns_to_normalize].shape[0]
    df[columns_to_normalize] = df[columns_to_normalize].astype(np.float32)

    for i in range(lookbacks+minutesToDecideOver+1000, df[columns_to_normalize].shape[0]-lookbacks-minutesToDecideOver-1000):
        if(i % 100 == 0 or i ==lookbacks+minutesToDecideOver+1000):
            print(i/dataLength*100)
            df4 = df[columns_to_normalize].iloc[i-lookbacks:i+100]
            df3 = df[columns_to_normalize].iloc[i-lookbacks-1000:i]
        X[i-1] = (df4[columns_to_normalize].iloc[i] - df3[columns_to_normalize].mean()) / df3[columns_to_normalize].std()
        if df["Close"].values[i: (i-1)+minutesToDecideOver].sum() > df["Close"].values[i-minutesToDecideOver:i-1].sum():
            y[i-1] = 1  # True
        else: 
            y[i-1] = 0  # False

    # Save X and y to CSV files
    np.savetxt('X_data_full50-1.csv', X.reshape((X.shape[0], -1)), delimiter=',')
    np.savetxt('y_data_full50-1.csv', y, delimiter=',')

if __name__ == "__main__":
    gather_and_save_data()