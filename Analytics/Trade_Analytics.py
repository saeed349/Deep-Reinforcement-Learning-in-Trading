import pandas as pd
import numpy as np 

def sharpe_calc(df):
    try:
        df["Exit"]=np.append(df.iloc[1:,:].Trade.values,None)
        df["Exit Price"]=np.append(df.iloc[1:,:].Price.values,None)
        df["Exit Time"]=np.append(df.iloc[1:,:].Time.values,None)
        df=df[(df.Trade != "TP") & (df.Trade != "SL")]
        df["PnL"] = df["Exit Price"]-df.Price
        df.loc[df.Trade=="SELL","PnL"]=df["PnL"]*-1
        df=df.dropna(axis=0)
        df["Return"]=df["PnL"]/df["Price"]
        df=df[df.PnL!=0]
        if (np.isnan(np.mean(df.Return)) or np.isnan(np.std(df.Return))): # if missing 
            return {'strategy_sharpe':None,'num_trades':None,'position_df':None}
        elif ((np.std(df.Return))==0): # if only one round trip trade
            return {'strategy_sharpe':np.mean(df.Return),'num_trades':len(df),'position_df':df}
        else:
            return {'strategy_sharpe':(np.mean(df.Return)/np.std(df.Return)),'num_trades':len(df),'position_df':df}         
    except:
        return {'strategy_sharpe':None,'num_trades':None,'position_df':None}