import pandas as pd
import numpy as np
from datetime import datetime
import math
from sklearn.preprocessing import StandardScaler, Normalizer
from abc import ABC, abstractmethod


class DataTransformerBase(ABC):
    def __init__(self, forecast_horizon, history_used, x_variables, y_variable, lower_threshold=None, upper_threshold=None):
        '''
        Initialise data transformation parameters:
        
        Args:
            forecast_horizon (int):     defines the forecast fhorizon in minutes
            history_used (int):         defines the historic period considered for the forecast
            x_variables (list):         defines the variables used in the forecast, i.e. OHLCV
            y_variable (string):        name of the variable used for the dependent varianle in the forecast, i.e. 'O'
            lower_threshold, upper_threshold (float): defines the event predicted by the forecast, i.e. -5% return over the forecast horizon; 
                Used in the classification version of the forecast. Regression type of data is assumed if None
        '''
        self._X_normalizer = None

        self._forecast_horizon = forecast_horizon
        self._history_used = history_used
        self._x_variables = x_variables
        self._y_variable = y_variable
        self._lower_threshold = lower_threshold
        self._upper_threshold = upper_threshold


    @abstractmethod
    def _lagged_vars_x(self, df):
        pass


    @abstractmethod
    def _return_var_y(self, df):
        pass


    def _asset_return(self, df):
        ret = (df[[self._y_variable]].shift(-self._forecast_horizon) / df[[self._y_variable]] - 1.).values.flatten()

        if self._lower_threshold is not None and self._upper_threshold is not None:
            ret = pd.cut(ret, [-math.inf, self._lower_threshold, self._upper_threshold, math.inf], labels=[-1, 0, 1])        

        return ret[self._history_used:-self._forecast_horizon]


    def transform_data(self, df):
        '''
        Create predictive variables according to the parameters:
        
        Args:
            df (DataFrame):     raw data used in the modelling
        '''
        
        df_x = df.copy()
        # Normalise X values
        df_x[self._x_variables] = pd.DataFrame(self._X_normalizer.transform(df[self._x_variables]), columns=self._x_variables, index=df.index)

        _X = self._lagged_vars_x(df_x)
        _Y = self._return_var_y(df)

        return _X, _Y 


    def init_data(self, df):
        '''
        Initialise normalizer and create predictive variables according to the parameters:
        
        Args:
            df (DataFrame):     raw data used in the modelling
        '''

        # Init normalizer
        self._X_normalizer = StandardScaler().fit(df[self._x_variables])

        return self.transform_data(df)

    

class DataTransformer1D(DataTransformerBase):
    '''
    Simple data transformer class, producing a 2D table (DataFrame) with lagged variables (1D array per time point)
    '''

    def __init__(self, forecast_horizon, history_used, x_variables, y_variable, lower_threshold=None, upper_threshold=None):
        super().__init__(forecast_horizon, history_used, x_variables, y_variable, lower_threshold, upper_threshold)


    def _return_var_y(self, df):
        return self._asset_return(df)


    def _lagged_vars_x(self, df):
        x_data = pd.DataFrame(index=df.index)
        x_data[self._x_variables] = df[self._x_variables]

        for var in self._x_variables:
            varnames = ['{}_{}'.format(var, i) for i in range(1, self._history_used)]
            x_data[varnames] = pd.DataFrame(dict((varnames[i-1], 
                df[[var]].shift(i).values.flatten()) for i in range(1, self._history_used)), index=df.index)

        return x_data.iloc[self._history_used:-self._forecast_horizon]



class DataTransformer3D(DataTransformerBase):
    '''
    Data transformer class, producing a 4D table (np array), 3D array per time point:
        Features (OCHLV)
        Lags (0..forecast_horizon)
        Assets (BTC, ETH, ...)
    '''

    def __init__(self, forecast_horizon, history_used, x_variables, y_variable, x_assets, y_asset, lower_threshold=None, upper_threshold=None):
        '''
        Args (additional to base class):
            x_assets (list):    list of assets used as an input
            y_asset (string):   asset whose y_variable to be used for dependent variable 
        '''
        self._x_assets = x_assets
        self._y_asset = y_asset
        super().__init__(forecast_horizon, history_used, x_variables, y_variable, lower_threshold, upper_threshold)


    def _return_var_y(self, df):
            return self._asset_return(df[df.symb == self._y_asset])


    def _lagged_vars_x(self, df):
        '''
        Args:
            df: must be indexed by time!
        '''
        # Pre-allocate an array containing the values
        n_obs = len(df[df.symb == self._y_asset].index)
        val_array = np.zeros((n_obs, len(self._x_variables), self._history_used+1, len(self._x_assets)))

        for i_f, f in enumerate(self._x_variables):
            for i_a, a in enumerate(self._x_assets):
                for lag in range(self._history_used+1):
                    val_array[:, i_f, lag, i_a] = df[df.symb == a][f].shift(lag)  

        return val_array[self._history_used:-self._forecast_horizon,]  


def transform_date_ccxt(date_col):
    '''
    Transforms date from ccxt format into datetime

    Args:
        date_col:   column of dates as produced by ccxt lib
    '''

    return [datetime.fromtimestamp(x/1000) for x in date_col]