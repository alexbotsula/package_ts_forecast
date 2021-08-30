# Grid search walk-forward CV for parameters
# https://stats.stackexchange.com/questions/440280/choosing-model-from-walk-forward-cv-for-time-series

from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, cross_val_score



class ModelScore:
    def __init__(self, model, params):
        self.model_ = model
        self.params_ = params
        self.initialized_ = False


    def cv_score(self, X_train, y_train, X_test, y_test, n_splits=5):
        ts_split_inner = TimeSeriesSplit(n_splits = n_splits)
        ts_split_outer = TimeSeriesSplit(n_splits = n_splits)

        self.gs_ = GridSearchCV(self.model_, param_grid = self.params_, \
            cv = ts_split_inner, scoring = 'neg_mean_squared_error', n_jobs=-1)

        scores = cross_val_score(self.gs_, X_train, y_train, cv = ts_split_outer, \
            scoring = 'neg_mean_squared_error')
        self.cv_score_ = -scores.mean()
        self.gs_fit_ = self.gs_.fit(X_train, y_train)
        self.gs_score_ = -self.gs_fit_.best_score_
        self.test_score_ = -self.gs_.score(X_test, y_test)
        self.initialized_ = True
    
        return self

    def __str__(self):
        if not self.initialized_:
            return 'Not initialized'
        else:
            return ('CV score:\t{0}\nGS score:\t{1}\nTest score:\t{2}\nBest params:\t{3}'\
                .format(self.cv_score_, self.gs_score_, self.test_score_, self.gs_.best_params_))  

