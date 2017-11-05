import os
import pandas as pd
from data_prep import data_prep

DIR_PATH = os.path.dirname(os.path.abspath(__file__))
TEST_DATA_PATH = DIR_PATH + '/test_data/blokIII_test.csv'
TEST_MODEL_PATH = DIR_PATH + '/test_data/models/S0PC010_1.PIDA.PV.h5'
TEST_REG_MODEL_PATH = DIR_PATH + '/test_data/models/S0T302_8.DACA.PV.p'
BLOCK_VARS_PATH = DIR_PATH + '/test_data/var_names.xlsx'


def test():
    data = pd.read_csv(TEST_DATA_PATH)
    var_names = pd.read_excel(BLOCK_VARS_PATH)
    vars = var_names['in'].append(var_names['control']).dropna().tolist()
    in_data = data[vars]
    x = data_prep.prepare_data(in_data)
    x = pd.DataFrame(x, columns=var_names)
    pass
    # y = data_prep.predict(x, TEST_MODEL_PATH)
    y_reg = data_prep.predict(x, TEST_REG_MODEL_PATH)


if __name__ == '__main__':
    test()
