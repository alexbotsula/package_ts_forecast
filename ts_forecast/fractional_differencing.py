import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller

class FractDiffTransformer:
    def __init__(self, adf_p_thres, lag_cutoff):
        self._adf_p_thres = adf_p_thres
        self._lag_cutoff = lag_cutoff
        self._columns_d = {}

    
    def _get_weights(self, d):
        '''
        Return the weights from the series expansion of the differencing operator
        for real orders d and up to lags coefficients
        '''
        w = [1]
        for k in range(1, self._lag_cutoff):
            w.append(-w[-1]*((d-k+1))/k)

        w = np.array(w).reshape(-1,1)

        return w


    def _ts_diff(self, series, d):
        '''
        Return the time series resulting from (fractional) differencing
        for real orders order up to lag_cutoff coefficients
        '''

        weights = self._get_weights(d)
        res = 0

        for k in range(self._lag_cutoff):
            res += weights[k] * series.shift(k).fillna(0)

        return res[self._lag_cutoff:]

    def _search_order(self, series):
        '''
        Return the order of differencing required to meet P-value of ADF test
        '''

        for d in np.linspace(0, 1, 11):
            ts_d = self._ts_diff(series, d)
            print([d, len(ts_d), len(series)])
            df = adfuller(ts_d, maxlag=1, regression='c', autolag=None)
            if df[1] < self._adf_p_thres:
                break
        
        return d


    def transform_data(self, df):
        '''
        Transform data using fractional difference of the order defined per self._columns_d dictionary;
        Dictionary must be initialised prior to use.
        '''

        data = dict((c, self._ts_diff(df[c], self._columns_d[c])) for c in df.columns)
        return pd.DataFrame(data)


    def fit(self, df):
        '''
        Defines required order of differencing, then transform the data.
        '''
        
        self._columns_d = dict((c, self._search_order(df[c])) for c in df.columns)
        return self
        
