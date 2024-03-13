import numpy as np
import pandas as pd
from TickerData1s import *

def Create15sTicks(df):
    # Load and prepare the testing data
    

    df2 = pd.DataFrame()
    
    Close = df['Close'].values
    Open = df['Open'].values
    High = df['High'].values
    Low = df['Low'].values
    Volume_BTC = df['Volume_BTC'].values
    length = df.shape[0]


    close_value  = np.zeros(int((length-length%15)/15))
    open_value   = np.zeros(int((length-length%15)/15))
    low_value    = np.zeros(int((length-length%15)/15))
    high_value   = np.zeros(int((length-length%15)/15))
    volume_value = np.zeros(int((length-length%15)/15))
    

    for i in range(15, df.shape[0]-15, 15):
        if(i % 150 == 0):
            print(int(i/length*100))
        
        close_value[int(i/15)] = Close[int(i)]
        open_value[int(i/15)] = Open[int(i-15)]
        low_value[int(i/15)] = Low[int((i-15)):i].min()
        high_value[int(i/15)] = High[int((i-15)):i].max()
        volume_value[int(i/15)] = Volume_BTC[int((i-15)):i].sum()



    df2.insert(loc=0, column='Close',   value=close_value)
    df2.insert(loc=1, column='Open', value=open_value)
    df2.insert(loc=2, column='Low',  value=low_value)
    df2.insert(loc=3, column='High',  value=high_value)
    df2.insert(loc=4, column='Volume_BTC', value=volume_value)

    

    df2.to_csv('c:/Projects/ConeCoin.Tensorflow/BTCFDUSD-15s-2024-01.csv', index=False)

    return df2.dropna()

if __name__ == "__main__":
    Create15sTicks()
