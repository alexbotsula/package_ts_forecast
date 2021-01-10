import numpy as np
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
from typing import Iterable, Any, Tuple

def _signal_last(it):
    iterable = iter(it)
    ret_var = next(iterable)
    for val in iterable:
        yield False, ret_var
        ret_var = val
    yield True, ret_var

'''
Time series walk-forward cross-validation of a model
'''
def time_series_cv(df, data_transformer, n_fold, n_epochs, batch_size, model_func):
    '''
    Args:
        df (DataFrame):         data frame with the source data (must have time as index!)
        data_transformer (DataTransformBase): object defining the data transformation
        n_fold (int):           number of w/f CV splits
        n_epochs (int):         number of epochs per CV step
        model_func (function):  function to construct a model
    '''
    
    tscv = TimeSeriesSplit(n_splits=n_fold)
    
    perf_hist = None
    # Utility function to grow historic performance dictionary
    list_append = lambda lst, item: lst + [item] if lst else [item]


    unique_ind = df.index.unique()
    for is_last, (train_index, test_index) in _signal_last(tscv.split(unique_ind)):
        print("TRAIN:", train_index, "TEST:", test_index)
        df_train, df_test = df.loc[unique_ind[train_index]], df.loc[unique_ind[test_index]]
        
        X_train, y_train = data_transformer.init_data(df_train)
        X_val, y_val = data_transformer.transform_data(df_test)

        m = model_func(X_train)
        hist = m.fit(X_train, y_train,
                    validation_data=(X_val, y_val), 
                    batch_size=batch_size, epochs=n_epochs, verbose=0)
        
        train_metrics = [k for k in hist.history if not k.startswith('val_')]
        val_metrics = ['val_' + k for k in train_metrics] 

        # Only validation metrics are added during the cross validation to avoid double counting of the impact for expanding training set
        if not perf_hist:
            perf_hist = dict((k, []) for k in hist.history)
    
        # Add train performance metrics in the end of the CV cycle
        if not is_last:
            perf_hist = dict((key, list_append(perf_hist[key], hist.history[key]) if key in val_metrics else []) for key in hist.history)
        else: 
            perf_hist = dict((key, list_append(perf_hist[key], hist.history[key])) for key in hist.history)

    return perf_hist


'''
Plot performance history data
'''
def plot_perf_history(perf_history, n_trim):
    def plot_perf(hist, label):
        mean_perf = [np.mean([x[i] for x in hist]) for i in range(len(hist[0]))][n_trim:]

        plt.plot(range(1, len(mean_perf) + 1), mean_perf)
        plt.xlabel('Epochs')
        plt.ylabel(label)
        plt.show()

    for k in perf_history:
        plt.figure()
        plot_perf(perf_history[k], k)
        plt.show()