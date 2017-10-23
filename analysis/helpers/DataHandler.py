import pandas as pd
import pickle
import sys
sys.setrecursionlimit(10000)


def load_data(block_name, load_path):
        df = pd.read_csv(load_path + block_name + '.csv', index_col=0)
        print('data loaded')
        return df

def load_block_vars(path):
        df = pd.read_excel(path, sheetname=None)
        print('blocks vars loaded')
        return df

def save_model(model, save_path):
        pickle.dump(model, open(save_path, 'wb'))
        print('Model saved')


def save_stats_txt(path, var_out, r2):
    with open(path, 'a') as file:
        file.write(
            'zmienna: {0}\t\t r^2_test: {1}\t\t r^2_validate: {2}\t\t r^2_train: {3}\n'.format(var_out,
                                                                                               r2[0],
                                                                                               r2[1],
                                                                                               r2[2]))


def save_stats_xls(path, stats, labels):
    df = pd.DataFrame(stats, columns=labels)
    df.to_excel(path)
    print("data saved")

