from pickle import TRUE
from subprocess import _USE_POSIX_SPAWN
from urllib.parse import uses_fragment
import numpy as np
import pandas as pd
from TickerData import *
import os
# from numba import cuda 
from tensorflow.keras.callbacks import LearningRateScheduler

# device = cuda.get_current_device()
# device.reset()

# set the GPU device
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import tensorflow as tf
def gather_and_save_data():
    # Load and prepare the testing data
    df = TickerData().df

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
    
    lookbacks = 10
    minutesToDecideOver = 6 

    X = np.zeros((df[columns_to_normalize].shape[0]-lookbacks, lookbacks, df[columns_to_normalize].shape[1]))
    y = np.zeros((df[columns_to_normalize].shape[0]-lookbacks,))

    dataLength = df[columns_to_normalize].shape[0]
    df[columns_to_normalize] = df[columns_to_normalize].astype(np.float32)
    close_values = df["Close"]
    df3 = (df[columns_to_normalize]-df[columns_to_normalize].mean())/df[columns_to_normalize].std()
    for i in range(lookbacks+minutesToDecideOver+1000, df[columns_to_normalize].shape[0]-lookbacks-minutesToDecideOver):

        if(i % 200 == 0 or i ==lookbacks+minutesToDecideOver+1000):
            print(i/dataLength*100)
            
        X[i] = df3[columns_to_normalize].iloc[i-lookbacks:i]
        
        y[i]= (df["Close"].iloc[i]-df["Open"].iloc[i])/df["Open"].iloc[i]
            
    # Save X and y to CSV files
    np.savetxt('X_data_full50-newest-20.csv', X.reshape((X.shape[0], -1)), delimiter=',')
    np.savetxt('y_data_full50-newest-20.csv', y, delimiter=',')

def lr_schedule(epoch):
    initial_lr = 0.0005
    decay_factor = 0.9
    decay_epochs = 3
    lr = initial_lr * (decay_factor ** (epoch // decay_epochs))
    return lr

def main():
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    #tf.debugging.set_log_device_placement(True)
    # Load and prepare the testing data
    # df = TickerData().df



    # gather_and_save_data()
    


    # df2 = pd.DataFrame()
    # print(df.columns)
    # df2.insert(loc = 0, column="Low", value=df["Low"])
    # df2.insert(loc = 1, column="High", value=df["High"])
    # # df2.insert(loc = 2, column="Open", value=df["Open"])
    # df2.insert(loc = 2, column="Close", value=df["Close"])
    # df2.insert(loc = 3, column="trend_ema_slow", value=(df["Close"].values-df["trend_ema_slow"].values))
    # df2.insert(loc = 4, column="Close2", value=(df["Close"].values-(df["trend_ema_fast"].values+df["trend_ema_slow"].values)/2))
    # # df2.insert(loc = 5, column="Volume_BTC", value=df["Volume_BTC"])
    # # df2.insert(loc = 1, column="trend_ema_fast", value=(df["Close"].values-df["trend_ema_fast"].values))
    # df2.insert(loc = 0, column="trend_adx_neg", value=df["trend_adx_neg"].values)
    # df2.insert(loc = 1, column="trend_adx_pos", value=df["trend_adx_pos"].values)
    # df2.insert(loc = 2, column="momentum_rsi", value=df["momentum_rsi"].values)
    # df2.insert(loc = 3, column="trend_macd_diff", value=df["trend_macd_diff"].values) 
    # df2.insert(loc = 4, column="trend_kst_diff", value=df["trend_kst_diff"].values)
    # df2.insert(loc = 5, column="trend_aroon_ind", value=df["trend_aroon_ind"].values)
    # df2.insert(loc = 6, column="trend_psar_up_indicator", value=df["trend_psar_up_indicator"].values)
    # df2.insert(loc = 7, column="trend_psar_down_indicator", value=df["trend_psar_down_indicator"].values)
    # df2.insert(loc = 8, column="momentum_stoch_rsi_k", value=df["momentum_stoch_rsi_k"].values)
    # df2.insert(loc=9,column="momentum_stoch_rsi_d",value=df["momentum_stoch_rsi_d"].values)
    # df2.insert(loc=10,column="volatility_bbw",value=df["volatility_bbw"].values)
    # df2.insert(loc=11,column="bb_bbm",value=df["bb_bbm"].values)
    # df2.insert(loc=12,column="bb_bbh",value=df["bb_bbh"].values)
    # df2.insert(loc=13,column="bb_bbl",value=df["bb_bbl"].values)
    # df2.insert(loc=14,column="bb_bbhi",value=df["bb_bbhi"].values)
    # df2.insert(loc=15,column="kama",value=df["kama"].values)
    # df2.insert(loc=16,column="adx",value=df["adx"].values)
    # df2.insert(loc=17,column="ichimoku_b",value=df["ichimoku_b"].values)
    # df2.insert(loc=18,column="kst_sig",value=df["kst_sig"].values)
    # df2.insert(loc=19,column="pvo",value=df["pvo"].values)
    # df2.insert(loc = 5, column="momentum_stoch", value=df["Low"].shift(1))
    # df2.insert(loc = 6, column="trend_adx_neg", value=df["High"].shift(1))
    # df2.insert(loc = 7, column="trend_adx_pos", value=df["Open"].shift(1))
    # df2.insert(loc = 8, column="momentum_rsi", value=df["Close"].shift(1))
    # df2.insert(loc = 9, column="trend_macd_diff", value=df["Volume_BTC"].shift(1))

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
    minutesToDecideOver = 6 
    numberOfIndicators = len(columns_to_normalize)
    # X = np.zeros((loaded_X.shape[0]-lookbacks, lookbacks, 86))
    # y = np.zeros((loaded_X.shape[0]-lookbacks,))
    # print(df[columns_to_normalize].isna().sum())
    # print(df[columns_to_normalize].shape)
    # print(df[columns_to_normalize].mean())
    # print(df[columns_to_normalize].std())

    # Create a LearningRateScheduler callback
    lr_scheduler = LearningRateScheduler(lr_schedule)
    # df[columns_to_normalize] = df[columns_to_normalize].astype(np.float32)
    # normalised_df = df[columns_to_normalize] = (df[columns_to_normalize] - df[columns_to_normalize].mean()) / df[columns_to_normalize].std()
    # close_values = df["Close"].values
    # for i in range(lookbacks+minutesToDecideOver+1000, normalised_df.shape[0]-lookbacks-minutesToDecideOver-1000):
    #     df3 = df[columns_to_normalize].iloc[i-lookbacks-1000:i].values
    #     if(i%100 == 0):
    #         print(i/dataLength*100)
    #     X[i-1] = (df3[1000:] - df3.mean()) / df3.std()
    #     if(close_values[i: (i-1)+minutesToDecideOver].sum()  > (close_values[i-minutesToDecideOver:i-1]).sum()):
    #         y[i-1] = True
    #     else: 
    #         y[i-1]= False


    # Save X and y to CSV files
    # np.savetxt('X_data.csv', X.reshape((X.shape[0], -1)), delimiter=',')
    # np.savetxt('y_data.csv', y, delimiter=',')

    # # Load X and y from CSV files
    print("Loading .csv data")
    loaded_X = np.loadtxt('X_data_full50-newest-20.csv', delimiter=',').reshape((-1, lookbacks, numberOfIndicators))
    loaded_y = np.loadtxt('y_data_full50-newest-20.csv', delimiter=',')
    print("Loaded .csv data")
    dataLength = loaded_X.shape[0]

    print(dataLength)
    print(len(loaded_y))

    # Split the data into testing, validation, and testing sets
    train_data_x = loaded_X[1:dataLength-20000, :]
    train_data_y = loaded_y[1:dataLength-20000]
    

    test_data_x = loaded_X[dataLength-20000:, :]
    test_data_y = loaded_y[dataLength-20000:]
    
    
    train_data_x = train_data_x.reshape(( -1, lookbacks, numberOfIndicators))
    # train_data_y = train_data_y.reshape((loaded_y.shape[0], ))
    print(train_data_x[-1])
    print(train_data_y[-1])
    # test_data_x = test_data_x.reshape((test_data_x.shape[0], test_data_x.shape[1],test_data_x.shape[2], 1 ))
    # test_data_y = test_data_y.reshape((test_data_y.shape[0], ))

    
    # train_data_y = tf.cast(train_data_y, tf.bool)
    # # test_data_x = tf.cast(test_data_x, tf.float32)
    # test_data_y = tf.cast(test_data_y, tf.bool)

    model = tf.keras.models.Sequential()    
    inputs=(tf.keras.layers.Input(shape=(lookbacks, numberOfIndicators)))

    # # Apply 1D convolutions along the time dimension using TimeDistributed
    # cnn_layer = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv1D(32, kernel_size=3, activation='relu'))(inputs)
    # cnn_layer = tf.keras.layers.TimeDistributed(tf.keras.layers.Dropout(0.7))(cnn_layer)
    # cnn_layer = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv1D(32, kernel_size=2, activation='relu'))(cnn_layer)
    # cnn_layer = tf.keras.layers.TimeDistributed(tf.keras.layers.Dropout(0.7))(cnn_layer)
    # cnn_layer = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv1D(32, kernel_size=3, activation='relu'))(cnn_layer)
    # cnn_layer = tf.keras.layers.TimeDistributed(tf.keras.layers.Dropout(0.7))(cnn_layer)
    # cnn_layer = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(inputs)
    # Apply SimpleRNN
    rnn_layer = tf.keras.layers.SimpleRNN(32, activation='relu', return_sequences=True)(inputs)
    rnn_layer = tf.keras.layers.Dropout(0.5)(rnn_layer)
    
    rnn_layer = tf.keras.layers.SimpleRNN(16, activation='relu', return_sequences=False)(rnn_layer)
    rnn_layer = tf.keras.layers.Dropout(0.5)(rnn_layer)    

    # rnn_layer = tf.keras.layers.SimpleRNN(8, activation='relu', return_sequences=True)(rnn_layer)
    # rnn_layer = tf.keras.layers.Dropout(0.3)(rnn_layer)
    # rnn_layer = tf.keras.layers.SimpleRNN(4, activation='relu')(rnn_layer)
    # rnn_layer = tf.keras.layers.Dropout(0.3)(rnn_layer)
    # rnn_layer = tf.keras.layers.Flatten()(rnn_layer)
      
    # cnn_layer = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv1D(128, kernel_size=3, activation='relu'))(rnn_layer)
    # cnn_layer = tf.keras.layers.TimeDistributed(tf.keras.layers.Dropout(0.3))(cnn_layer)

    # # Output layer
    # rnn_layer = tf.keras.layers.Dense(8, activation='sigmoid')(rnn_layer)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(rnn_layer)

    # model = tf.keras.models.Sequential()
    # inputs=(tf.keras.layers.Input(shape=(lookbacks, 11, 1)))
    # blah = tf.keras.layers.Conv2D(128, (3, 1), activation='relu')(inputs)
    # blah = tf.keras.layers.Dropout(0.3)(blah)
    # # blah = tf.keras.layers.MaxPooling2D(pool_size=(2, 1))(blah)
    # blah = tf.keras.layers.Conv2D(32, (1, 2), activation='relu')(blah)
    # blah = tf.keras.layers.Dropout(0.3)(blah)
    # blah = tf.keras.layers.Conv2D(32, (2, 1), activation='relu')(blah)
    # blah = tf.keras.layers.Dropout(0.3)(blah)
    # lah = tf.keras.layers.MaxPooling2D(pool_size=(2, 1))(blah)
    # blah = tf.keras.layers.Conv2D(32, (1, 2), activation='relu')(blah)
    # blah = tf.keras.layers.Dropout(0.3)(blah)  
    # blah = tf.keras.layers.Conv2D(32, (2, 1), activation='relu')(blah)
    # blah = tf.keras.layers.Dropout(0.3)(blah)
    # blah = tf.keras.layers.Conv2D(32, (1, 2), activation='relu')(blah) 
    # blah = tf.keras.layers.Dropout(0.3)(blah)
    # blah = tf.keras.layers.Conv2D(16, (1, 2), activation='relu')(blah) 
    # blah = tf.keras.layers.Dropout(0.3)(blah)
    # blah = tf.keras.layers.Conv2D(16, (1, 2), activation='relu')(blah) 
    # blah = tf.keras.layers.Dropout(0.3)(blah)
    # blah = tf.keras.layers.Conv2D(8, (1, 2), activation='relu')(blah) 
    # blah = tf.keras.layers.Dropout(0.3)(blah)
    # blah = tf.keras.layers.Conv2D(8, (1, 2), activation='relu')(blah)
    # blah = tf.keras.layers.Conv2D(32, (2, 1), activation='relu')(blah)
    # # blah = tf.keras.layers.Dropout(0.3)(blah)
    # # blah = tf.keras.layers.Conv2D(16, (1, 3), activation='relu')(blah)
    
    # blah = tf.keras.layers.Flatten()(blah)
    # blah = tf.keras.layers.Dense(8, activation='relu')(blah)
    # # blah = tf.keras.layers.Dense(32, activation='relu')(blah)
    # outputs=(tf.keras.layers.Dense(1, activation='sigmoid')(blah))

    
    with tf.device("/device:GPU:0"):
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
    # compile the model
    with tf.device("/device:GPU:0"):
        model.compile(optimizer=tf.optimizers.Adam(learning_rate = 0.0005), loss="mean_squared_error",
                 metrics=[tf.keras.metrics.MeanSquaredError()])
     
    model.summary()
    
    # xtrain = tf.expand_dims(train_data_x, axis=-1)
    # ytrain = tf.expand_dims(train_data_y, axis=-1)
    # xtest = tf.expand_dims(test_data_x, axis=-1)
    # ytest = tf.expand_dims(test_data_y, axis=-1)
    
    path_checkpoint = "model_flat2.h5"
    es_callback = tf.keras.callbacks.EarlyStopping(monitor="val_mean_squared_error", min_delta=0, patience=10)

    modelckpt_callback = tf.keras.callbacks.ModelCheckpoint(
        monitor="val_mean_squared_error",
        filepath=path_checkpoint,
        verbose=1,
        save_weights_only=False,
        save_best_only=True,
    )

    # Load the weights from a saved model checkpoint
    # model.load_weights(path_checkpoint)

    with tf.device("/device:GPU:0"):    
        history = model.fit(
            train_data_x,
            train_data_y,
            batch_size=512,
            epochs=2000,
            validation_split=0.1,
            validation_freq=1,
            callbacks=[modelckpt_callback, es_callback, lr_scheduler],
            validation_batch_size = 512,
            shuffle=True, 
            use_multiprocessing = True
        )
        # model.load_weights(path_checkpoint)
        model.save('model-gpu-rnn-1m-huh')
        
        test_loss,  test_accuracy = model.evaluate(test_data_x, test_data_y)
        print(f"Test loss: {test_loss:.4f}, Test accuracy: {test_accuracy:.4f}")
    
    
    #model = tf.keras.models.load_model("saved_model_bs1")
    # new_model = tf.keras.models.load_model('my_model.h5')
    #Evaluate the model
if __name__ == '__main__':
    main()