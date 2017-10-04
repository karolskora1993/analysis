import pandas as pd
import pickle
import sys
sys.setrecursionlimit(10000)


class DataLoader(object):
    @staticmethod
    def load_data(block_name, load_path):
        df = pd.read_csv(load_path + block_name + '.csv', index_col=0)
        print('data loaded')
        return df

    @staticmethod
    def load_block_vars(path):
        df = pd.read_excel(path, sheetname=None)
        print('blocks vars loaded')
        return df

    @staticmethod
    def save_model(model, save_path):
        pickle.dump(model, open(save_path, 'wb'))
        print('Model saved')