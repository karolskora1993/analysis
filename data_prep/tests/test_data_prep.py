import os
import pandas as pd
from data_prep.ModelsHandler import ModelsHandler
import unittest
from time import time

DIR_PATH = os.path.dirname(os.path.abspath(__file__))
TEST_DATA_PATH = DIR_PATH + '/test_data/blokIII_test.csv'
TEST_MODEL_PATH = DIR_PATH + '/test_data/models/S0PC010_1.PIDA.PV.h5'
TEST_RNN_MODEL_PATH = DIR_PATH + '/test_data/models/S0T302_5A.DACA.PV.h5'
TEST_REG_MODEL_PATH = DIR_PATH + '/test_data/models/S0N2_6.DACA.PV.p'
BLOCK_VARS_PATH = DIR_PATH + '/test_data/var_names.xlsx'

class Tests(unittest.TestCase):

    def test(self):
        data = pd.read_csv(TEST_DATA_PATH)
        var_names = pd.read_excel(BLOCK_VARS_PATH)
        vars = var_names['in'].append(var_names['control']).dropna().tolist()
        in_data = data[vars]

        handler = ModelsHandler()
        start = time()
        x = handler.prepare_data(in_data.as_matrix())
        print('preparing data time: {0}'.format(time() - start))


        org_shape = x.shape
        self.assertIsNotNone(x)
        self.assertNotEqual(0, len(x))
        self.assertEqual(x.shape, org_shape)

        start_mlp = time()
        for _ in range(1):
            y = handler.predict(x, vars, TEST_MODEL_PATH)

        self.assertNotEquals(0, len(y))
        self.assertEquals(y.shape[0], org_shape[0])

        mlp_time = time() - start_mlp
        start_reg = time()
        for _ in range(1):
            y_reg = handler.predict(x, vars, TEST_REG_MODEL_PATH)
        self.assertNotEquals(0, len(y_reg))
        print('==========================================================')
        print('forward regression prediction time (1k iterations): {0}'.format(time() - start_reg))
        print('mlp prediction time (1k iterations): {0}'.format(mlp_time))
        # y = handler.predict(x, vars, TEST_MODEL_PATH)
        # y_reg = handler.predict(x, vars, TEST_REG_MODEL_PATH)
        # y = handler.predict(x, vars, TEST_MODEL_PATH)
        # y_reg = handler.predict(x, vars, TEST_REG_MODEL_PATH)

        # in_data = in_data.as_matrix()[:11, :]
        # reshaped = in_data.reshape(1, in_data.shape[0], in_data.shape[1])
        # x = handler.prepare_data(reshaped)
        # y = handler.predict(x, vars, TEST_RNN_MODEL_PATH)



if __name__ == '__main__':
    unittest.main()
