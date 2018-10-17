import numpy as np
import pandas as pd
import math as mt
from sklearn.base import BaseEstimator


class DecisionTree(BaseEstimator):
    
    def __init__(self, max_depth=np.inf, min_samples_split=2, 
                 criterion='entropy', debug=False):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.debug = debug
        if criterion=='entropy':
            self.criterion_func = self._entropy

    def fit(self, X, y):
        data = np.c_[X, y]
        self.instruction = self._best_split(data)
    

        

    def predict(self, X):
        pass
        
    def predict_proba(self, X):
        pass

    def _entropy(self, y):
        val, count = np.unique(y, return_counts=True, axis=0)
        value_count = np.c_[val, count]
        count_y = y.shape[0]
        entropy = 0
        for x in value_count:
            p = x[1]/count_y
            entropy -= p*mt.log(p, 2)
        return(entropy)

    def _gini(self, y):
        pass

    def _variance(self, y):
        pass

    def _mad_median(self, y):
        pass


    def _best_split(self, data):
        index_y = data.shape[1]
        count_x = index_y-1
        for i in range(count_x):
            col = data[:, i]
            unique_val = np.unique(col, axis=0)
            
