import pandas as pd
from analysis import blocks
from numpy import nan
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error


COL_NAMES = ['in', 'control', 'out']

def simple_model(x, y, x_test, y_test):
    scaler = StandardScaler()
    mlp = MLPRegressor(hidden_layer_sizes=(12, 12, 12), max_iter=500)

    scaler.fit(x)
    x = np.transpose(np.matrix(scaler.transform(x)))
    y = np.transpose(np.matrix(y))
    x_test = np.transpose(np.matrix(scaler.transform(x_test)))
    y_test = np.transpose(np.matrix(y_test))
    mlp.fit(x, y)

    predictions = mlp.predict(x_test)
    r2 = mlp.score(x_test, y_test)
    mse = mean_squared_error(y_test, predictions)
    print('r^2: {0}\n mse: {1}'.format(r2, mse))
    return mlp


def load_data(block_name):
    df = pd.read_csv(blocks.LOAD_PATH + block_name + '.csv')
    print('data loaded')
    return df


def load_block_vars():
    df = pd.read_excel(blocks.BLOCK_VARS_PATH_XLSX, sheetname=None)
    print('blocks vars loaded')
    return df

def evaluate(model, x, y):
    y_pred = model.predict(np.transpose(np.matrix(x)))
    r2 = model.score(np.transpose(np.matrix(x)), np.transpose(np.matrix(y)))
    mse = mean_squared_error(y, y_pred)
    return r2, mse


def save_model(model, file_name):
    pickle.dump(model, open(file_name, 'wb'))


def main():
    blocks_vars = load_block_vars()
    for block_name in blocks.names:
        df = load_data(block_name)
        vars = blocks_vars[block_name][COL_NAMES]
        vars_in = [x for x in vars['in'] if not x is nan and not '#' in x]
        vars_out = [x for x in vars['out'] if not x is nan and not '#' in x]
        input_data = [np.array(df[x].tolist()) for x in vars_in]
        output_data = [np.array(df[x].tolist()) for x in vars_out]
        last_train_idx = 5 #floor(len(input_data[0]) * 0.875)
        last_test_idx = 10 #len(input_data[0]) - 1
        x_train = [x[:last_train_idx] for x in input_data]
        y_train = [x[:last_train_idx] for x in output_data]
        x_test = [x[last_train_idx:last_test_idx] for x in input_data]
        y_test = [x[last_train_idx:last_test_idx] for x in output_data]

        #sklearn model
        print('sklearn model')
        model = simple_model(x_train, y_train, x_test, y_test)




if __name__ == '__main__':
    main()
