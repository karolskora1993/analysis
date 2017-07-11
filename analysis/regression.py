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
import pickle
from math import floor

MODELS_PATH = '/Users/apple/Desktop/mag/dane/DANE_PO_MODERNIZACJI_VRM/modele/regresja/'

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


def single_output_logistic_regression(x, y):
    print('Single output regression')
    models = {}
    for output in y:
        print('new model')
        r = logistic_regresion(x, output)
        models[output] = r
    return models


def multi_output_regression(x, y, regressor = svm.SVR()):
    print('multi output regression')
    r = MultiOutputRegressor(regressor)
    r.fit(np.transpose(np.matrix(x)), np.transpose(np.matrix(y)))
    return r


def evaluate(model, x, y):
    y_pred = model.predict(np.transpose(np.matrix(x)))
    r2 = model.score((np.transpose(np.matrix(x)), np.transpose(np.matrix(y))))
    mse = mean_squared_error(y, y_pred)
    return (r2, mse)

def save_model(model, file_name):
    pickle.dump(model, open(file_name, 'wb'))


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
        x_test = [x[last_train_idx:len(input_data[0])-1] for x in input_data]
        y_test = [x[last_train_idx:len(output_data[0])-1] for x in output_data]

        r_list = single_output_regression(x_train, y_train)
        for i, model in enumerate(r_list):
            r2, mse = evaluate(model, x_test, y_test)
            print('r^2: {0}\n mse: {1}'.format(r2, mse))
            save_model(model, MODELS_PATH + 'single_linear_{0}'.format(i) + '.sav')

        r_list = single_output_logistic_regression(x_train, y_train)
        for i, model in enumerate(r_list):
            r2, mse = evaluate(model, x_test, y_test)
            print('r^2: {0}\n mse: {1}'.format(r2, mse))
            save_model(model, MODELS_PATH + 'single_logistic_{0}'.format(i) + '.sav')

        r = multi_output_regression(x_train, y_train)
        r2, mse = evaluate(r, x_test, y_test)
        print('r^2: {0}\n mse: {1}'.format(r2, mse))
        save_model(model, MODELS_PATH + 'multi_SVR' + '.sav')

        r = multi_output_regression(x_train, y_train, SGDRegressor())
        r2, mse = evaluate(r, x_test, y_test)
        print('r^2: {0}\n mse: {1}'.format(r2, mse))
        save_model(model, MODELS_PATH + 'multi_SGD' + '.sav')


if __name__ == '__main__':
    main()
