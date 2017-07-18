import pandas as pd
from analysis import blocks
from numpy import nan
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras import losses
from math import floor
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import classification_report


COL_NAMES = ['in', 'control', 'out']

def simple_model(x, y, x_test, y_test):
    scaler = StandardScaler()
    mlp = MLPRegressor(hidden_layer_sizes=(13, 13, 13), max_iter=500)

    scaler.fit(x)
    x = scaler.transform(x)
    x_test = scaler.transform(x_test)
    mlp.fit(np.transpose(np.matrix(x)), np.transpose(np.matrix(y)))

    predictions = mlp.predict(x_test)
    print(classification_report(y_test, predictions))


def load_data(block_name):
    df = pd.read_csv(blocks.LOAD_PATH + block_name + '.csv')
    print('data loaded')
    return df


def load_block_vars():
    df = pd.read_excel(blocks.BLOCK_VARS_PATH_XLSX, sheetname=None)
    print('blocks vars loaded')
    return df


def define_model(input_size, output_size):
    model = Sequential()
    model.add(Dense(output_size, input_dim=input_size))
    return model


def compile_model(model):
    model.compile(loss=losses.mean_squared_error, optimizer='sgd')

def train_model(model, x, y, epochs, batch_size = 32):
    model.fit(x, y, epochs=epochs, batch_size=batch_size)


def evaluate_model(model, x , y):
    scores = model.evaluate(x, y)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))


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
        x_train = [x[:last_train_idx] for x in input_data]
        y_train = [x[:last_train_idx] for x in output_data]
        x_test = [x[last_train_idx:len(input_data[0]) - 1] for x in input_data]
        y_test = [x[last_train_idx:len(output_data[0]) - 1] for x in output_data]

        #sklearn model
        simple_model(x_train, y_train, x_test, y_test)

        #Keras model
        model = define_model(len(input_data), len(output_data))
        compile_model(model)
        train_model(model, x_train, y_train, 100)
        evaluate_model(model, x_test, y_test)


if __name__ == '__main__':
    main()
