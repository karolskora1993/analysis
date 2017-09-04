import pandas as pd
import os
from sklearn import linear_model
from sklearn import tree
import numpy as np
from numpy import nan
import collections

HOME_PATH = str(os.path.expanduser('~')+'/')

VARS_PATH = HOME_PATH + '/Dokumenty/analysis/data/bloki_poprawione_v2.xlsx'
LOAD_PATH = HOME_PATH + '/Dokumenty/analysis/data/bloki_v4/'
SCORES_SAVE_PATH = HOME_PATH + '/Dokumenty/analysis/data/scores_v2.txt'
BLOCK_VARS_PATH_XLSX = HOME_PATH + '/Dokumenty/analysis/data/bloki_poprawione.xlsx'

names = [
    'blok I',
    'blok II',
    'blok III',
    'blok IV'
]
COL_NAMES = ['in', 'control', 'out']
MAX_DELAY = 40


def find_best(models):
    best_score = None
    best_model = None
    for model in models:
        if best_score is None or best_score < model['score']:
            best_score = model['score']
            best_model = model
    return best_model


def load_block_vars():
    df = pd.read_excel(BLOCK_VARS_PATH_XLSX, sheetname=None)
    print('blocks vars loaded')
    return df


def load_data(block_name):
    df = pd.read_csv(LOAD_PATH + block_name + '.csv', index_col=0)
    print('data loaded')
    return df


def most_common(delays):
    return collections.Counter(delays).most_common()[0][0]


def main():
    blocks_vars = load_block_vars()
    with open('delays_dt_v3.txt', 'w') as f:
        for block_name in names:
            print(block_name)
            data = load_data(block_name)
            vars = blocks_vars[block_name][COL_NAMES]
            vars_in = [x for x in vars['in'] if x is not nan]
            vars_control = [x for x in vars['control'] if x is not nan]
            vars_in += vars_control
            vars_out = [x for x in vars['out'] if x is not nan]
            for var_out in vars_out:
                y = ((data[var_out])[MAX_DELAY:-1]).as_matrix()
                delays = []
                for var_in in vars_in:
                    if var_in is not None and var_in is not nan:
                        models = []
                        x = (data[var_in]).as_matrix()
                        for delay in range(0, MAX_DELAY + 1):
                            x_train = (x[MAX_DELAY - delay:-delay - 1]).reshape(-1, 1)
                            model = tree.DecisionTreeRegressor(max_depth=5)
                            model.fit(x_train, y)
                            score = model.score(x_train, y)
                            line = 'var_out: {0}  var_in: {1}  delay: {2}  score:{3} \n'.format(var_out, var_in, delay, score)
                            print(line)
                            f.write(line)
                            models.append({'delay': delay, 'score': score})
                        best_model = find_best(models)
                        line = 'var_out: {0}  var_in: {1}  best delay: {2}  score:{3} \n'.format(var_out, var_in,
                                                                                             best_model['delay'],
                                                                                             best_model['score'])
                        print(line)
                        f.writelines(line)
                        delays.append(best_model['delay'])
                        line = 'var_out: {0} delay:{1} \n'.format(var_out, most_common(delays))
                print(line)
                f.write(line)


if __name__ == '__main__':
    main()
