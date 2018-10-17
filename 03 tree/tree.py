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
        self.instruction = self._get_instruction(data, '1')
    

        

    def predict(self, X):
        pass
        
    def predict_proba(self, X):
        pass

    def _entropy(self, y):
        proba = self._get_proba(y)
        entropy = 0
        print(proba)
        for x in proba:
            
            entropy -= x[1]*mt.log(x[1], 2)
        return(entropy)

    def _gini(self, y):
        pass

    def _variance(self, y):
        pass

    def _mad_median(self, y):
        pass

    def _get_proba(self, y):
        val, count = np.unique(y, return_counts=True, axis=0)
        value_count = np.c_[val, count]
        count_y = y.shape[0]
        print(value_count)
        proba = np._c[value_count[:,0],value_count[:,1]/count_y]
        print(proba)
        return(proba)
        

    def _information_gain(self, data, data_left, data_right):
        count = data.shape[0]
        count_left = data_left.shape[0]
        count_right = data_right.shape[0]
        fval = self.criterion_func(data)
        fval_left = self.criterion_func(data_left)
        fval_right = self.criterion_func(data_right)
        ig = fval - fval_left*count_left/count - fval_right*count_right/count
        return(ig)

    def _best_split(self, data):
        index_y = data.shape[1]
        count_x = index_y-1
        best_split = {'ig': 0, 'column': 0, 'val': 0}
        for i in range(count_x):
            unique_val = np.unique(data[:, i], axis=0)
            unique_val = np.sort(unique_val)[-1]
            for val in unique_val:
                data_left = data[data[:,i]<=val]
                data_right = data[data[:,i]>val]
                ig = self._information_gain(data[:, index_y], data_left[:, index_y], data_right[:, index_y])
                if ig > best_split['ig']:
                    best_split['ig'] = ig
                    best_split['column'] = i
                    best_split['val'] = val
        if best_split['ig'] == 0:
            return(False)
        else:
            return(best_split)

    def _get_instruction(self, data, id_instruction):
        split = self._best_split(data)
        if split:
            # Деление ветвей
            id_instr_left = id_instruction + '1'
            id_instr_right = id_instruction + '2'
            instruction = [{'id': id_instruction,
                           'column': split['column'], 
                           'val': split['val'],
                           'id_left': id_instr_left,
                           'id_right': id_instr_right,
                           'leaf_data': False}]
            data_left = data[data[:,split['column']]<=split['val']]
            data_right = data[data[:,split['column']]>split['val']]
            instruction.extend(self._get_instruction(data_left, id_instr_left))
            instruction.extend(self._get_instruction(data_right, id_instr_right))
            return(instruction)
        else:
            # Формирование листа дерева

            instruction = [{'id': id_instruction,
                           'column': None, 
                           'val': None,
                           'id_left': None,
                           'id_right': None,
                           'leaf_data': False}]
            pass
    

        