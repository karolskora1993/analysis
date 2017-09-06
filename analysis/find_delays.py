import pandas as pd
import os
from sklearn import tree
from analysis import blocks
from numpy import nan
import collections
from math import floor

HOME_PATH = str(os.path.expanduser('~')+'/')

LOAD_PATH = HOME_PATH + '/Dokumenty/analysis/data/csv_all_v4.csv'
VARS_PATH = HOME_PATH + '/Dokumenty/analysis/data/bloki_poprawione_v2.xlsx'

COL_NAMES = ['in', 'control', 'out']
MAX_DELAY = 40

# def main():
#     data = pd.read_csv(LOAD_PATH, index_col=0)
#     print('read data')
#     vars = pd.read_excel(VARS_PATH, sheetname='calosc')
#     print('read var names')
#     vars_in = vars['in']
#     vars_out = vars['out']
#
#     for var_out in vars_out:
#         y = ((data[var_out])[MAX_DELAY:]).as_matrix()
#         for var_in in vars_in:
#             if var_in is not None:
#                 x = (data[var_in]).as_matrix()
#                 models = []
#                 for delay in range(0, MAX_DELAY + 1):
#                     x_train = (x[MAX_DELAY - delay:len(x) - delay]).reshape(-1, 1)
#                     model = tree.DecisionTreeRegressor()
#                     model.fit(x_train, y)
#                     models.append({'delay': delay, 'model': model})
#                 best_model, best_score = find_best(models, x_train, y)
#                 print('var_out: {0}  var_in: {1}  best delay: {2}  score:{3}'.format(var_out, var_in, best_model['delay'], best_score))

def find_best(models, x_test, y_test):
    best_score = None
    best_model = None
    for model in models:
        score = model['model'].score(x_test, y_test)
        if best_score is None or best_score < score:
            best_score = score
            best_model = model
    return best_model, best_score

def load_block_vars():
    df = pd.read_excel(blocks.BLOCK_VARS_PATH_XLSX, sheetname=None)
    print('blocks vars loaded')
    return df


def load_data(block_name):
    df = pd.read_csv(blocks.LOAD_PATH + block_name + '.csv', index_col=0)
    print('data loaded')
    return df

def most_common(list):
    return collections.Counter(list).most_common()[0][0]


def main():
    blocks_vars = load_block_vars()
    with open('delays.txt', 'w') as f:
        for block_name in blocks.names:
            print(block_name)
            data = load_data(block_name)
            vars = blocks_vars[block_name][COL_NAMES]
            vars_in = [x for x in vars['in'] if not x is nan and not '#' in x]
            vars_control = [x for x in vars['control'] if not x is nan and not '#' in x]
            vars_in = vars_in + vars_control
            vars_out = [x for x in vars['out'] if not x is nan and not '#' in x]
            for var_out in vars_out:
                y = (data[var_out]).as_matrix()
                delays = []
                for var_in in vars_in:
                    if var_in is not None:
                        x = (data[var_in]).as_matrix()
                        last_train_idx = floor(x.shape[0] * 0.875)
                        last_test_idx = x.shape[0]
                        models = []
                        for delay in range(0, MAX_DELAY + 1):
                            x_train = (x[MAX_DELAY - delay:last_train_idx - delay]).reshape(-1, 1)
                            y_train = y[MAX_DELAY:last_train_idx]
                            model = tree.DecisionTreeRegressor()
                            model.fit(x_train, y_train)
                            models.append({'delay': delay, 'model': model})
                        x_test = (x[last_train_idx + 1:last_test_idx]).reshape(-1, 1)
                        y_test = y[last_train_idx + 1 : last_test_idx]
                        best_model, best_score = find_best(models, x_test, y_test)
                        line = 'var_out: {0}  var_in: {1}  best delay: {2}  score:{3}'.format(var_out, var_in,
                                                                                             best_model['delay'],
                                                                                             best_score)
                        print(line)
                        f.write(line)
                        delays.append(best_model['delay'])
                        line = 'var_out: {0} delay:{1}'.format(var_out, most_common(delays))
                print(line)
                f.write(line)


if __name__ == '__main__':
    main()
