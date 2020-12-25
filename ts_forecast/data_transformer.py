import pandas as pd
import numpy as np
from datetime import datetime
import math
from sklearn.preprocessing import Normalizer


class DataTransformer:
    def __init__(self, forecast_horizon, history_used, x_variables, y_variable, lower_threshold, upper_threshold):
        '''
        Initialise data transformation parameters:
        
        Args:
            forecast_horizon (int):     defines the forecast fhorizon in minutes
            history_used (int):         defines the historic period considered for the forecast
            x_variables (list):         defines the variables used in the forecast, i.e. OHLCV
            y_variable: (string):       name of the variable used for the dependent varianle in the forecast, i.e. 'O'
            lower_threshold, upper_threshold (float): defines the event predicted by the forecast, i.e. -5% return over the forecast horizon; Used in the          classification version of the forecast
        '''

        self._X = pd.DataFrame()
        self._Y = pd.DataFrame()
        self._X_normalizer = None

        self._forecast_horizon = forecast_horizon
        self._history_used = history_used
        self._x_variables = x_variables
        self._y_variable = y_variable
        self._lower_threshold = lower_threshold
        self._upper_threshold = upper_threshold


    def __lagged_vars_x(self, df):
        x_data = pd.DataFrame(index=df.index)
        x_data[self._x_variables] = df[self._x_variables]

        for var in self._x_variables:
            varnames = ['{}_{}'.format(var, i) for i in range(1, self._history_used)]
            x_data[varnames] = pd.DataFrame(dict((varnames[i-1], 
                df[[var]].shift(i).values.flatten()) for i in range(1, self._history_used)), index=df.index)

        return x_data.iloc[self._history_used:-self._forecast_horizon]

    def __return_var_y(self, df):
        ret = (df[[self._y_variable]].shift(-self._forecast_horizon) / df[[self._y_variable]] - 1.).values.flatten()
        y_data = pd.cut(ret, [-math.inf, self._lower_threshold, self._upper_threshold, math.inf], labels=[-1, 0, 1])
        
        return y_data[self._history_used:-self._forecast_horizon]


    def init_data(self, df):
        '''
        Initialise model data and create predictive variables according to the parameters:
        
        Args:
            df (DataFrame):     raw data used in the modelling
        '''
        
        _X = self.__lagged_vars_x(df)

        # Normalise X values
        self._X_normalizer = Normalizer().fit(_X)
        self._X = pd.DataFrame(self._X_normalizer.transform(_X), columns=_X.columns, index=_X.index)

        self._Y = self.__return_var_y(df)

        return self._X, self._Y 


    def transform_data(self, df):
        '''
        Create predictive variables according to the parameters:
        
        Args:
            df (DataFrame):     raw data used in the modelling
        '''
        _X = self.__lagged_vars_x(df)
        _X = pd.DataFrame(self._X_normalizer.transform(_X), columns=_X.columns, index=_X.index)
        _Y = self.__return_var_y(df)

        return _X, _Y
        