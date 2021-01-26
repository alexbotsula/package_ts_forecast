import pandas as pd
import numpy as np
from datetime import datetime
import math
from sklearn.preprocessing import StandardScaler, Normalizer
from abc import ABC, abstractmethod

from .fractional_differencing import FractDiffTransformer

    

class DataTransformer3D:
    '''
    Data transformer class, producing a 4D table (np array), 3D array per time point:
        Features (OCHLV)
        Lags (0..forecast_horizon)
        Assets (BTC, ETH, ...)
    ''' 

    def __init__(self, forecast_horizon, history_used, x_variables, y_variable, x_assets, y_asset, flatten_x=False, lower_threshold=None, upper_threshold=None):
        '''
        Args:
            forecast_horizon (int):     defines the forecast fhorizon in minutes
            history_used (int):         defines the historic period considered for the forecast
            x_variables (list):         defines the variables used in the forecast, i.e. OHLCV
            y_variable (string):        name of the variable used for the dependent varianle in the forecast, i.e. 'O'
            lower_threshold, upper_threshold (float): defines the event predicted by the forecast, i.e. -5% return over the forecast horizon; 
                Used in the classification version of the forecast. Regression type of data is assumed if None

            x_assets (list):    list of assets used as an input
            y_asset (string):   asset whose y_variable to be used for dependent variable 
            flatten_x (bool):   flag whether to flatten X before output
        '''

        self._X_normalizer = None

        self._forecast_horizon = forecast_horizon
        self._history_used = history_used
        self._x_variables = x_variables
        self._y_variable = y_variable
        self._lower_threshold = lower_threshold
        self._upper_threshold = upper_threshold

        # Parameters for fractional differencing
        self._X_fract_diff = None
        self._adf_p_thres = 1e-2
        self._lag_cutoff = 100

        self._x_assets = x_assets
        self._y_asset = y_asset
        self._flatten_x = flatten_x
        
    
    def _asset_return(self, df):
        ret = (df[[self._y_variable]].shift(-self._forecast_horizon) / df[[self._y_variable]] - 1.).values.flatten()

        if self._lower_threshold is not None and self._upper_threshold is not None:
            ret = pd.cut(ret, [-math.inf, self._lower_threshold, self._upper_threshold, math.inf], labels=[-1, 0, 1])        

        return ret[self._lag_cutoff:][self._history_used:-self._forecast_horizon]


    def _return_var_y(self, df): 
        return self._asset_return(df[df.symb == self._y_asset])

 
    def _init_normalizer(self, df):
        return dict((a, StandardScaler().fit(df.loc[df.symb == a, self._x_variables])) for a in self._x_assets)


    def _normalize_X(self, df):
        df_ = df.copy()
        
        for a in self._x_assets:
            df_.loc[df_.symb == a, self._x_variables] = self._X_normalizer[a].transform(df_.loc[df_.symb == a, self._x_variables])

        return df_


    def _init_fract_diff(self, df):
        return dict((a, FractDiffTransformer(self._adf_p_thres, self._lag_cutoff).fit(df[df.symb == a][self._x_variables])) for a in self._x_assets)


    def _fract_diff_X(self, df):
        df_ = pd.DataFrame()

        for a in self._x_assets:
            df_a = self._X_fract_diff[a].transform_data(df[df.symb == a][self._x_variables])
            df_a['symb'] = a
            df_a['t'] = df.loc[df.symb == a, 't']
            df_ = df_.append(df_a)
    
        return df_.set_index('t', inplace=False)
        

    def _lagged_vars_x(self, df):
        '''
        Args:
            df: must be indexed by time!
        '''
        # Pre-allocate an array containing the values
        n_obs = len(df[df.symb == self._x_assets[0]])

        val_array = np.zeros((n_obs, len(self._x_variables), self._history_used+1, len(self._x_assets)))

        for i_f, f in enumerate(self._x_variables):
            for i_a, a in enumerate(self._x_assets):
                for lag in range(self._history_used+1):
                    val_array[:, i_f, lag, i_a] = df[df.symb == a][f].shift(lag)  
        
        if self._flatten_x:
            return val_array[self._history_used:-self._forecast_horizon,].reshape(-1, len(self._x_variables)*(self._history_used+1)*len(self._x_assets))
        else:
            return np.expand_dims(val_array[self._history_used:-self._forecast_horizon,], axis=-1)
    

    def transform_data(self, df):
        '''
        Create predictive variables according to the parameters:
        
        Args:
            df (DataFrame):     raw data used in the modelling
        '''
        
        df_x = df.copy()
        
        #Perform fractional differencing
        df_x = self._fract_diff_X(df_x)
    
        # Normalise X values
        df_x[self._x_variables] = pd.DataFrame(self._normalize_X(df_x), columns=self._x_variables, index=df_x.index)
        
        _X = self._lagged_vars_x(df_x)
        _Y = self._return_var_y(df)

        return _X, _Y 


    def init_data(self, df):
        '''
        Initialise normalizer, fractional differencer, and create predictive variables according to the parameters:
        
        Args:
            df (DataFrame):     raw data used in the modelling
        '''

        df_x = df.copy()

        # Init fractional differencing
        self._X_fract_diff = self._init_fract_diff(df_x)

        #Perform fractional differencing
        df_x = self._fract_diff_X(df_x)

        # Init normalizer 
        self._X_normalizer = self._init_normalizer(df_x) 

        # Normalise X values
        df_x[self._x_variables] = pd.DataFrame(self._normalize_X(df_x), columns=self._x_variables, index=df_x.index)

        _X = self._lagged_vars_x(df_x)
        _Y = self._return_var_y(df)

        return _X, _Y 




def transform_date_ccxt(date_col):
    '''
    Transforms date from ccxt format into datetime

    Args:
        date_col:   column of dates as produced by ccxt lib
    '''

    return [datetime.fromtimestamp(x/1000) for x in date_col]