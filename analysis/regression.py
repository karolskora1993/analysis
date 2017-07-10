import pandas as pd
import numpy as np
from analysis import blocks
from numpy import nan
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDRegressor
from sklearn import svm

from math import floor

COL_NAMES = ['in', 'control', 'out']


def load_data(block_name):
    df = pd.read_csv(blocks.LOAD_PATH + block_name + '.csv')
    print('data loaded')
    return df


def load_block_vars():
    df = pd.read_excel(blocks.BLOCK_VARS_PATH_XLSX, sheetname=None)
    print('blocks vars loaded')
    return df

def linear_regresion(x, y):
    r = LinearRegression()
    r.fit(np.transpose(np.matrix(x)), np.transpose(np.matrix(y)))
    return r

def logistic_regresion(x, y):
    r = LogisticRegression()
    r.fit(np.transpose(np.matrix(x)), np.transpose(np.matrix(y)))
    return r

def single_output_regression(x, y):
    print('Single output regression')
    models = []
    for output in y:
        print('new model')
        r = linear_regresion(x, output)
        models.append(r)

    return models

def single_output_regression_logistic(x, y):
    print('Single output regression')
    models = []
    for output in y:
        print('new model')
        r = logistic_regresion(x, output)
        models.append(r)

    return models


def multi_output_regression(x, y):
    print('multi output regression')
    r = MultiOutputRegressor(svm.SVR())
    r.fit(np.transpose(np.matrix(x)), np.transpose(np.matrix(y)))
    return r

def multi_output_regression_(x, y, regressor = svm.SVR()):
    print('multi output regression')
    r = MultiOutputRegressor(regressor)
    r.fit(np.transpose(np.matrix(x)), np.transpose(np.matrix(y)))
    return r


def evaluate(model, x, y):
    y_pred = model.predict(x)
    r2 = model.score(x, y)
    mse = mean_squared_error(y, y_pred)
    return (r2, mse)


def main():
    blocks_vars = load_block_vars()
    for block_name in blocks.names:
        df = load_data(block_name)
        vars = blocks_vars[block_name][COL_NAMES]
        vars_in = [x for x in vars['in'] if not x is nan and not '#' in x]
        vars_out = [x for x in vars['out'] if not x is nan and not '#' in x]
        input_data = [df[x].tolist() for x in vars_in]
        output_data = [df[x].tolist() for x in vars_out]
        last_train_idx = floor(len(input_data[0]) * 0.875)
        x_train = [x[:last_train_idx] for x in input_data]
        y_train = [x[:last_train_idx] for x in output_data]
        x_test = [x[last_train_idx:len(input_data)-1] for x in input_data]
        y_test = [x[last_train_idx:len(output_data)-1] for x in output_data]
        r = single_output_regression(x_train, y_train)
        for model in r:
            r2, mse = evaluate(model, x_test, y_test)
            print('r^2: {0}\n mse: {1}'.format(r2, mse))
        r = multi_output_regression(x_train, y_train)
        r2, mse = evaluate(r, x_test, y_test)
        print('r^2: {0}\n mse: {1}'.format(r2, mse))
        r = multi_output_regression(x_train, y_train, SGDRegressor())
        r2, mse = evaluate(r, x_test, y_test)
        print('r^2: {0}\n mse: {1}'.format(r2, mse))

if __name__ == '__main__':
    main()
