from keras.optimizers import Adadelta, Adam
from keras.activations import elu, relu
from keras.initializers import random_normal

from Models import KerasMLPModel, KerasSimpleRNNModel, KerasLSTMModel
from helpers.DataHandler import save_stats_txt, save_stats_xls, load_data, load_block_vars, save_model
import sys
import os
import pandas as pd
import numpy as np

LAST_TRAIN_IDX = 205038
LAST_VALIDATE_IDX = 257133
TIMESTAPS_TO_REMOVE = 69
COL_NAMES = ['in', 'control', 'out', 'delay']
BATCH_SIZE = 500
DROPOUT = 0.4
TIMESTEPS = 10
L = [0, 0.001, 0.01]
OPTIMIZER = [Adam(), Adadelta()]
ACTIVATION = [relu, elu]
KERNEL_INITS = [random_normal()]
HOME_PATH = str(os.path.expanduser('~')+'/')
LOAD_PATH = HOME_PATH + 'Dokumenty/analysis/data/bloki_v5/'
MODEL_SAVE_PATH = HOME_PATH + 'Dokumenty/analysis/data/models/stats/nowe_bloki/serialized/'
SCORE_SAVE_PATH = HOME_PATH + 'Dokumenty/analysis/data/models/stats/nowe_bloki/stats/'
BLOCK_VARS_PATH = HOME_PATH + 'Dokumenty/analysis/data/bloki_poprawione_v5.xlsx'
BLOCK_NAMES = [
    'blok I',
    'blok II',
    'blok III',
    'blok IV'
]


NETWORK_SHAPES = [
    (15,),
    (20, 10),
    (40, 20),
    (50, 30),
    (20, 10, 5),
    (40, 20, 10),
    (60, 40, 20),
    (80, 60, 20),
    (100, 40, 20),
    (40, 20, 10, 5),
    (50, 30, 20, 10),
    (60, 40, 20, 10),
    (70, 40, 20, 10),
    (80, 60, 40, 20),
    (100, 60, 40, 20),
]


def get_block():
    if len(sys.argv) > 1:
        block = int(sys.argv[1])
        if block > 0 and block <= 4:
            return block

    return 1


def shift_data(input_data, output_data, delay):
    input_data = input_data[: -delay]
    output_data = output_data[delay:]
    return input_data, output_data


def model_block(block_name, data, var_names):
    vars_in = var_names['in'].append(var_names['control']).dropna().tolist()
    vars_out = var_names['out'].dropna().tolist()
    delays = var_names['delay']
    block_models = []
    epochs = 300
    model_type = KerasMLPModel
    model_name = "MLP"

    input_data = data[vars_in].as_matrix()

    for i, var_out in enumerate(vars_out):
        output_data = data[var_out].values
        delay = int(delays[i]) if delays[i] >= 1 else 0
        best_r2 = (0, 0, 0)
        best_network_shape, best_model, best_activation, best_optimizer, best_kernel_init, best_l_rate = [None] * 6

        if delay > 0:
            x, y = shift_data(input_data, output_data, delay)
        else:
            x, y = input_data, output_data

        for network_shape in NETWORK_SHAPES:
            for activation in ACTIVATION:
                for optimizer in OPTIMIZER:
                    for kernel_init in KERNEL_INITS:
                        for l_rate in L:
                            model = model_type(x, y, LAST_TRAIN_IDX, LAST_VALIDATE_IDX)

                            if model_name == "RNN":
                                model.steps_back = TIMESTEPS

                            model.create_model(network_shape=network_shape,
                                               optimizer=optimizer,
                                               dropout=DROPOUT,
                                               activation=activation,
                                               l=l_rate,
                                               kernel_init=kernel_init)
                            model.train_model(epochs, batch_size=BATCH_SIZE)

                            r2 = model.test_model()

                            if r2[1] > best_r2[1] and abs(r2[1] - r2[2]) < 0.4 and r2[0] > 0.3:
                                best_network_shape = network_shape
                                best_r2 = r2
                                best_model = model
                                best_activation = activation.__name__
                                best_l_rate = l_rate
                                best_optimizer = str(optimizer)
                                best_kernel_init = str(kernel_init)

        if best_r2[1] > 0.4:
            block_models.append([var_out, best_network_shape, best_activation, best_optimizer, best_kernel_init, best_l_rate, best_r2[0], best_r2[1], best_r2[2]])
            file_name = "{0}_{1}_{2}".format(block_name, var_out, model_name)
            best_model.save_model(MODEL_SAVE_PATH, file_name)

    if block_models:
        save_stats_path = SCORE_SAVE_PATH + "{0}_{1}_{2}epochs_{3}dropout_{4}batch.xlsx".format(block_name.replace(" ", ""), model_name, epochs, DROPOUT, BATCH_SIZE)
        save_stats_xls(save_stats_path, block_models, ['var_out', 'network_shape', 'activation', 'optimizer', 'kernel_init', 'best_l_rate',  'r2_test', 'r2_validate', 'r2_train'])

def cut_data(data):
    values = data.values[0::70, :]
    df = pd.DataFrame(values, columns=data.columns)
    return df

def shuffle_in_blocks(data, groups):
    last_reshaped = (data.shape[0] // groups) * groups
    data_to_reshape = data[:last_reshaped]
    removed_data = data[last_reshaped:]
    np.random.shuffle(data_to_reshape.values.reshape(data_to_reshape.shape[0]//groups, groups, data_to_reshape.shape[1]))
    return data_to_reshape.append(removed_data)


def main():
    block_number = get_block()
    block_vars = load_block_vars(BLOCK_VARS_PATH)
    block_name = BLOCK_NAMES[block_number - 1]

    print(block_name)
    data = load_data(block_name, LOAD_PATH)
    var_names = block_vars[block_name][COL_NAMES]
    model_block(block_name, data, var_names)


if __name__ == '__main__':
    main()
