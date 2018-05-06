import pandas as pd
import numpy as np 

def sharpe_calc(df):
    df["Exit"]=np.append(df.iloc[1:,:].Trade.values,None)
    df["Exit Price"]=np.append(df.iloc[1:,:].Price.values,None)
    df["Exit Time"]=np.append(df.iloc[1:,:].Time.values,None)
    df=df[(df.Trade != "TP") & (df.Trade != "SL")]
    df["PnL"] = df["Exit Price"]-df.Price
    df.loc[df.Trade=="SELL","PnL"]=df["PnL"]*-1
    df=df.dropna(axis=0)
    df["Return"]=df["PnL"]/df["Price"]
    df=df[df.PnL!=0]
    return (np.mean(df.Return)/np.std(df.Return))