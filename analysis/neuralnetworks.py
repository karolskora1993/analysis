from keras.optimizers import adam
from DataStandarizers import SimpleStandarizer
from ModelTesters import SimpleTester, LassoTester
from Models import KerasMLPModel, KerasSimpleRNNModel, KerasConvLSTMModel, KerasLSTMModel, SklearnLasso, KerasConvModel
from helpers.DataHandler import save_stats_txt, save_stats_xls, load_data, load_block_vars, save_model
from sklearn.utils import shuffle
import sys
import os

COL_NAMES = ['in', 'control', 'out', 'delay']
LAST_TRAIN_IDX = 205038
LAST_VALIDATE_IDX = 257133
BATCH_SIZE = 500
DROPOUT = 0.4
TIMESTEPS = 10
OPTIMIZER = adam(lr=0.001)
HOME_PATH = str(os.path.expanduser('~')+'/')
LOAD_PATH = HOME_PATH + 'Dokumenty/analysis/data/bloki_v4/'
MODEL_SAVE_PATH = HOME_PATH + 'Dokumenty/analysis/data/models/'
SCORE_SAVE_PATH = HOME_PATH + 'Dokumenty/analysis/data/models/stats/nowe/'
BLOCK_VARS_PATH = HOME_PATH + 'Dokumenty/analysis/data/bloki_poprawione_v4.xlsx'
BLOCK_NAMES = [
    # 'blok I',
    'blok II',
    'blok III',
    'blok IV'
]


def get_network_shape():
    network_shape = None
    if len(sys.argv) > 1:
        cmd_line_args = []
        for i, arg in enumerate(sys.argv):
            if i > 0:
                cmd_line_args.append(int(arg))
        network_shape = tuple(cmd_line_args)
    return (network_shape[:-1], network_shape[-1]) if network_shape else (None, 1)


def shift_data(input_data, output_data, delay):
    input_data = input_data[: -delay]
    output_data = output_data[delay:]
    return input_data, output_data


def model_block(block_name, data, var_names):
    vars_in = var_names['in'].append(var_names['control']).dropna().tolist()
    vars_out = var_names['out'].dropna().tolist()
    delays = var_names['delay']

    block_models = []
    network_shape, epochs = get_network_shape()
    input_data = data[vars_in].as_matrix()

    for i, var_out in enumerate(vars_out):
        print('var_out:\t{0}'.format(var_out))
        output_data = data[var_out].values
        delay = int(delays[i]) if delays[i] >= 1 else 0
        if delay > 0:
            x, y = shift_data(input_data, output_data, delay)
        else:
            x, y = input_data, output_data

        model = KerasConvModel(x, y, LAST_TRAIN_IDX, LAST_VALIDATE_IDX, SimpleTester(), SimpleStandarizer())
        model.create_model(network_shape, optimizer=OPTIMIZER, dropout=DROPOUT)
        model.train_model(epochs, batch_size=BATCH_SIZE)

        r2 = model.test_model()
        block_models.append([var_out, r2[0], r2[1], r2[2]])
        SAVE_FILE_NAME = '{blok}_score_{network_shape}_{epochs}epochs_{model}'.format(blok=block_name,
                                                                                          network_shape=network_shape,
                                                                                          epochs=epochs,
                                                                                          model=model)
        save_stats_path = SCORE_SAVE_PATH + SAVE_FILE_NAME.format(var_out=var_out, network_shape=network_shape, epochs=epochs)
        save_stats_txt(save_stats_path + '.txt', var_out, r2)
        SAVE_FILE_NAME = '{block}_{var}_{model}_{network_shape}'.format(block=block_name, var=var_out, model=model, network_shape=network_shape)
        model.save_model(MODEL_SAVE_PATH, SAVE_FILE_NAME)
    save_stats_xls(save_stats_path + '.xlsx', block_models, ['var_out', 'r2_test', 'r2_validate', 'r2_train'])


def main():
    block_vars =load_block_vars(BLOCK_VARS_PATH)
    for block_name in BLOCK_NAMES:
        print(block_name)
        data = load_data(block_name, LOAD_PATH)
        # data = shuffle(data)
        var_names = block_vars[block_name][COL_NAMES]
        model_block(block_name, data, var_names)


if __name__ == '__main__':
    main()
