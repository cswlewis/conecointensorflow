from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import ta
import pandas as pd

app = Flask(__name__)

# Load the saved model
model = tf.keras.models.load_model('model-gpu-rnn-1m-weighted-3')

# Define an API endpoint for predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Parse the input data from the request
    data = request.json['data']
    data = np.array([data[0], data[1], data[2], data[3], data[4]])
    # data = data[:, :-1]

    # Create a DataFrame with column names
    columns = ['Low', 'High', 'Open', 'Close', 'Volume_BTC']
    data = pd.DataFrame(data.T, columns=columns)

    # Calculate technical indicators
    df = ta.add_all_ta_features(data, open="Open", high="High", low="Low", close="Close", volume="Volume_BTC", fillna=True)

    # Extract the required indicators for normalization
    #selected_indicators = ['trend_adx_neg', 'trend_adx_pos', 'momentum_rsi', 'trend_macd_diff', 'trend_kst_diff', 'trend_aroon_ind', 'trend_psar_up_indicator', 'trend_psar_down_indicator', 'momentum_stoch_rsi_k', 'momentum_stoch_rsi_d', 'volatility_bbw']
    #normalized_data = (indicators[selected_indicators] - indicators[selected_indicators].mean()) / indicators[selected_indicators].std()

    df2 = pd.DataFrame()

    columns_to_normalize = [ 'volume_obv', 'volume_cmf', 'volume_fi', 'volume_em',
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
    
    lookbacks = 10;

    # df2.insert(loc = 0, column="Low", value=df["Low"])
    # df2.insert(loc = 1, column="High", value=df["High"])
    # # df2.insert(loc = 2, column="Open", value=df["Open"])
    # df2.insert(loc = 2, column="Close", value=df["Close"])
    # df2.insert(loc = 3, column="trend_ema_slow", value=(df["Close"].values-df["trend_ema_slow"].values))
    # df2.insert(loc = 4, column="Close2", value=(df["Close"].values-(df["trend_ema_fast"].values+df["trend_ema_slow"].values)/2))
    # df2.insert(loc = 5, column="Volume_BTC", value=df["Volume_BTC"])
    # df2.insert(loc = 1, column="trend_ema_fast", value=(df["Close"].values-df["trend_ema_fast"].values))
    # df2.insert(loc = 6, column="trend_adx_neg", value=df["trend_adx_neg"].values)
    # df2.insert(loc = 7, column="trend_adx_pos", value=df["trend_adx_pos"].values)
    # df2.insert(loc = 8, column="momentum_rsi", value=df["momentum_rsi"].values)
    # df2.insert(loc = 9, column="trend_macd_diff", value=df["trend_macd_diff"].values) 
    # df2.insert(loc = 11, column="trend_kst_diff", value=df["trend_kst_diff"].values)
    # df2.insert(loc = 12, column="trend_aroon_ind", value=df["trend_aroon_ind"].values)
    # df2.insert(loc = 13, column="trend_psar_up_indicator", value=df["trend_psar_up_indicator"].values)
    # df2.insert(loc = 14, column="trend_psar_down_indicator", value=df["trend_psar_down_indicator"].values)
    # df2.insert(loc = 15, column="momentum_stoch_rsi_k", value=df["momentum_stoch_rsi_k"].values)
    # df2.insert(loc = 16, column="momentum_stoch_rsi_d", value=df["momentum_stoch_rsi_d"].values)
    # df2.insert(loc = 17, column="volatility_bbw", value=df["volatility_bbw"].values)
    # df2.insert(loc = 5, column="momentum_stoch", value=df["Low"].shift(1))
    # df2.insert(loc = 6, column="trend_adx_neg", value=df["High"].shift(1))
    # df2.insert(loc = 7, column="trend_adx_pos", value=df["Open"].shift(1))
    # df2.insert(loc = 8, column="momentum_rsi", value=df["Close"].shift(1))
    # df2.insert(loc = 9, column="trend_macd_diff", value=df["Volume_BTC"].shift(1))
    #df2=(df2-df2.mean())/df2.std()
    # X = np.zeros((df[columns_to_normalize].shape[0]-lookbacks, lookbacks, df[columns_to_normalize].shape[1]))
    # y = np.zeros((df[columns_to_normalize].shape[0]-lookbacks,))
    # print(df[columns_to_normalize].isna().sum())
    # print(df[columns_to_normalize].shape)
    # print(df[columns_to_normalize].mean())
    # print(df[columns_to_normalize].std())

    # Create a LearningRateScheduler callback
    # lr_scheduler = LearningRateScheduler(lr_schedule)
    
    normalised_df = df[columns_to_normalize] = (df[columns_to_normalize] - df[columns_to_normalize].mean()) / df[columns_to_normalize].std()
    df3 = normalised_df.tail(lookbacks)
    #print(df3);
    # Reshape the input data
    data2 = np.array([df3[col] for col in columns_to_normalize])
    
    input_data = data2.T.reshape(1, lookbacks, 85)
    # print(input_data[-1])
    # Make predictions on the new data
    predictions = model.predict(input_data, batch_size = 1)
    shouldBuy = predictions
    print(shouldBuy)
    # Return the predicted labels as a JSON response
    return jsonify(bool((shouldBuy[0]) > 0.5))

if __name__ == '__main__':
    app.run(debug=False, host="0.0.0.0", port=3000)



#Low                          22353.193753
#High                         22370.028702
#Open                         22361.624856
#Close                        22361.628185
#Volume_BTC                      68.819553
#trend_ema_fast               22361.505227
#trend_adx_neg                   21.854995
#trend_adx_pos                   21.602683
#momentum_rsi                    50.173137
#trend_macd_diff                  0.000106
#trend_ema_slow               22361.348869
#trend_kst_diff                   0.001526
#trend_aroon_ind                  0.376166
#trend_psar_up_indicator          0.043940
#trend_psar_down_indicator        0.043943
#momentum_stoch_rsi_k             0.501339
#momentum_stoch_rsi_d             0.501338
#volatility_bbw                   0.320141
#dtype: float64
#Low                          4503.333745
#High                         4506.249493
#Open                         4504.837954
#Close                        4504.838677
#Volume_BTC                     89.214623
#trend_ema_fast               4504.737748
#trend_adx_neg                   8.807192
#trend_adx_pos                   8.812875
#momentum_rsi                   11.020468
#trend_macd_diff                 5.058524
#trend_ema_slow               4504.615429
#trend_kst_diff                  0.807071
#trend_aroon_ind                58.907726
#trend_psar_up_indicator         0.204963
#trend_psar_down_indicator       0.204969
#momentum_stoch_rsi_k            0.325099
#momentum_stoch_rsi_d            0.311512
#volatility_bbw                  0.360557