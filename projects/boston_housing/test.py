import unittest
import numpy as np
import pandas as pd

from sklearn.cross_validation import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import make_scorer
from sklearn.tree import DecisionTreeRegressor
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import ShuffleSplit

class BostonHousingTest(unittest.TestCase):

    def test_explore_data(self):
        data = pd.read_csv('housing.csv')
        pricing = data['MEDV']
        features = data[['RM','LSTAT','PTRATIO']]

        return features, pricing

    def test_stats_about_prices(self):
        data = pd.read_csv('housing.csv')
        print data.sample(n=50).sort_values('PTRATIO')
        print data.sample(n=50).sort_values('RM')
        print data.sample(n=50).sort_values('LSTAT')

    def test_performance_metrics(self):
        data = pd.read_csv('housing.csv')
        pricing = data['MEDV']
        features = data[['RM','LSTAT','PTRATIO']]
        X_train, X_test, y_train, y_test = train_test_split(features,pricing, test_size=0.20, random_state=42)

    def test_r2_metrics(self):
        score = self.performance_metric([3, -0.5, 2, 7, 4.2], [2.5, 0.0, 2.1, 7.8, 5.3])
        print "Model has a coefficient of determination, R^2, of {:.3f}.".format(score)

    def performance_metric(self,y_true, y_predict):
        return r2_score(y_true,y_predict)



    def test_fit_model(self):
        data = pd.read_csv('housing.csv')
        pricing = data['MEDV']
        features = data[['RM','LSTAT','PTRATIO']]
        #tupple of (489,3)
        print features.shape[0]
        cv_sets = ShuffleSplit(features.shape[0], n_iter = 10, test_size = 0.20, random_state = 0)
        regressor = DecisionTreeRegressor()
        params = {}
        params['max_depth'] = [1,2,3,4,5,6,7,8,9,10]
        scoring_fnc = make_scorer(score_func=self.performance_metric)
        grid = GridSearchCV(estimator=regressor, param_grid=params, scoring=scoring_fnc, cv=cv_sets)
