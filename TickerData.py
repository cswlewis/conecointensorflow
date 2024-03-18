import pandas as pd
import ta
class TickerData:
    def __init__(self):
        # all_files = ["c:/Projects/ConeCoin.Tensorflow/BTCFDUSD-1s-2024-01.csv"]
        all_files = [
                         # "./BTCBUSD-1m-2022-10.csv",
                         # "./BTCBUSD-1m-2022-11.csv",
                         # "./BTCBUSD-1m-2022-12.csv",
                         # "./BTCBUSD-1m-2023-01.csv",
                         # "./BTCBUSD-1m-2023-02.csv",
                         # "./BTCTUSD-1m-2023-03.csv",
                         # "./BTCTUSD-1m-2023-04.csv",
                         # "./BTCTUSD-1m-2023-05.csv",
                         # "./BTCTUSD-1m-2023-06.csv",
                         # "./BTCTUSD-1m-2023-07.csv",
                         # "./BTCFDUSD-1m-2023-08.csv",
                         # "./BTCFDUSD-1m-2023-09.csv",
                         # "./BTCFDUSD-1m-2023-10.csv",
                         # "./BTCFDUSD-1m-2023-11.csv",
                         "./BTCFDUSD-1m-2023-12.csv",
                         "./BTCFDUSD-1m-2024-01.csv"
                         ] #last file includes 02/03/04 of 2023


        self.df = pd.concat((pd.read_csv(f, index_col=0, header=0) for f in all_files), ignore_index=True)
       # self.df = pd.read_csv("C:/Projects/ConeCoin.Tensorflow/ConeCoin.TensorFlow/BTCBUSD-1m-2023-02.csv", index_col=0, header=0)
        # Convert the timestamp to a datetime object
        #self.df = self.df.sort_index(ascending=False)
        #blah = pd.DataFrame();
        blah = ta.add_all_ta_features(self.df, open="Open", high="High", low="Low", close="Close", volume="Volume_BTC", fillna=True)
        #print(self.df.columns)
        #self.df.insert(loc = 7, column="momentum_rsi", value=blah["momentum_rsi"])
        #self.df.insert(loc = 8, column="trend_macd_diff", value=blah["trend_macd_diff"])
        #self.df.insert(loc = 9, column="trend_ema_fast", value=blah["trend_ema_fast"])
        #self.df.insert(loc = 10, column="trend_adx_pos", value=blah["trend_adx_pos"])
        #self.df.insert(loc = 11, column="trend_adx_neg", value=blah["trend_adx_neg"])


   #df2.insert(loc = 4, column="Volume_BTC", value=df["Volume_BTC"])
        self.df.dropna()

        
        # self.df["Rsi"] = blah[]
        # Calculate the EMA, RSI, MACD, CCI, Stochastic Oscillator K and D, and Williams %R
