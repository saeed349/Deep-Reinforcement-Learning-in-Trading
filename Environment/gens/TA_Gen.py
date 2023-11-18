import pandas as pd
import numpy as np
from sklearn import preprocessing
import talib

from Environment.core import DataGenerator

class TAStreamer(DataGenerator):
    """Data generator from csv file using TA-Lib for technical indicators.
    """

    @staticmethod
    def _generator(filename, header=False, split=0.8, mode='train', spread=.005):
        df = pd.read_csv(filename)
        if "Name" in df:
            df.drop('Name', axis=1, inplace=True)

        # Calculate indicators using TA-Lib
        cci = talib.CCI(df['high'], df['low'], df['close'], timeperiod=14)
        rsi = talib.RSI(df['close'], timeperiod=14)
        adx = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)

        # Create a DataFrame with indicators
        indicators = pd.DataFrame({
            'rsi_14': rsi,
            'cci_14': cci,
            'dx_14': adx,
            'volume': df['volume'],
            'close': df['close']
        })

        indicators = indicators.dropna(how='any')

        # Normalizing
        min_max_scaler = preprocessing.MinMaxScaler((-1, 1))
        np_scaled = min_max_scaler.fit_transform(indicators[['rsi_14', 'cci_14', 'dx_14', 'volume']])
        df_normalized = pd.DataFrame(np_scaled)
        df_normalized.columns = ['rsi_14', 'cci_14', 'dx_14', 'volume']
        df_normalized['bid'] = indicators['close'].values
        df_normalized['ask'] = df_normalized['bid'] + spread
        df_normalized['mid'] = (df_normalized['bid'] + df_normalized['ask']) / 2

        split_len = int(split * len(df_normalized))

        if mode == 'train':
            raw_data = df_normalized[['ask', 'bid', 'mid', 'rsi_14', 'cci_14', 'dx_14', 'volume']].iloc[:split_len, :]
        else:
            raw_data = df_normalized[['ask', 'bid', 'mid', 'rsi_14', 'cci_14', 'dx_14', 'volume']].iloc[split_len:, :]

        for index, row in raw_data.iterrows():
            yield row.to_numpy()

    def _iterator_end(self):
        """Rewinds if end of data reached.
        """
        super().rewind()

    def rewind(self):
        """Rewind only when the end of the data is reached.
        """
        self._iterator_end()
