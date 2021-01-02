import numpy as np
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt

'''
Time series walk-forward cross-validation of a model
'''
def time_series_cv(df, data_transformer, n_fold, n_epochs, model_func):
    '''
    Args:
        df (DataFrame):         data frame with the source data
        data_transformer (DataTransformBase): object defining the data transformation
        n_fold (int):           number of w/f CV splits
        n_epochs (int):         number of epochs per CV step
        model_func (function):  function to construct a model
    '''
    
    tscv = TimeSeriesSplit(n_splits=n_fold)
    
    perf_hist = None
    # Utility function to grow historic performance dictionary
    list_append = lambda lst, item: lst + [item] if lst else [item]

    for train_index, test_index in tscv.split(df):
        print("TRAIN:", train_index, "TEST:", test_index)
        df_train, df_test = df.iloc[train_index], df.iloc[test_index]
        
        X_train, y_train = data_transformer.init_data(df_train)
        X_val, y_val = data_transformer.transform_data(df_test)

        m = model_func(X_train)
        hist = m.fit(X_train, y_train,
                    validation_data=(X_val, y_val), 
                    batch_size=4096, epochs=n_epochs, verbose=0)
        
        if not perf_hist:
            perf_hist = dict((k, []) for k in hist.history)
    
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