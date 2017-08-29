import pandas as pd
import os
import numpy as np
from math import floor
from sklearn import tree
HOME_PATH = str(os.path.expanduser('~')+'/')

LOAD_PATH = HOME_PATH + '/Dokumenty/analysis/data/csv_all_v4.csv'
VARS_PATH = HOME_PATH + '/Dokumenty/analysis/data/bloki_poprawione_v2.xlsx'

MAX_DELAY = 20

def main():
    data = pd.read_csv(LOAD_PATH, index_col=0)
    print('read data')
    vars = pd.read_excel(VARS_PATH, sheetname='calosc')
    print('read var names')
    vars_in = pd.concat([vars['in'], vars['control']])
    vars_out = vars['out']

    for var_out in vars_out:
        y = ((data[var_out])[MAX_DELAY:]).as_matrix()
        for var_in in vars_in:
            x = (data[var_in]).as_matrix()
            models = []
            for delay in range(0, MAX_DELAY + 1):
                x_train = (x[MAX_DELAY - delay:len(x) - delay]).reshape(-1, 1)
                model = tree.DecisionTreeRegressor()
                model.fit(x_train, y)
                models.append({'delay': delay, 'model': model})
            best_model, best_score = find_best(models, var_out, var_in)
            print('var_out: {0}  var_in: {1}  best delay: {2}  score:{3}'.format(var_out, var_in, best_model['delay'], best_score))

def find_best(models, x_test, y_test):
    best_score = None
    best_model = None
    for model in models:
        score = model['model'].score(x_test, y_test)
        if best_score is not None and best_score < score:
            best_score = score
            best_model = model
    return best_model, best_score


if __name__ == '__main__':
    main()