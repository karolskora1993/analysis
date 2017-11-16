import os
import pandas as pd
from data_prep.ModelsHandler import ModelsHandler
import unittest

DIR_PATH = os.path.dirname(os.path.abspath(__file__))
TEST_DATA_PATH = DIR_PATH + '/test_data/blokIII_test.csv'
TEST_MODEL_PATH = DIR_PATH + '/test_data/models/S0PC010_1.PIDA.PV.h5'
TEST_REG_MODEL_PATH = DIR_PATH + '/test_data/models/S0N2_6.DACA.PV.p'
BLOCK_VARS_PATH = DIR_PATH + '/test_data/var_names.xlsx'

class Tests(unittest.TestCase):

    def test(self):
        data = pd.read_csv(TEST_DATA_PATH)
        var_names = pd.read_excel(BLOCK_VARS_PATH)
        vars = var_names['in'].append(var_names['control']).dropna().tolist()
        in_data = data[vars]

        handler = ModelsHandler()
        x = handler.prepare_data(in_data.as_matrix())
        org_shape = x.shape
        self.assertIsNotNone(x)
        self.assertNotEqual(0, len(x))
        self.assertEqual(x.shape, org_shape)
        y = handler.predict(x, vars, TEST_MODEL_PATH)
        self.assertNotEquals(0, len(y))
        self.assertEquals(y.shape[0], org_shape[0])
        y_reg = handler.predict(x, vars, TEST_REG_MODEL_PATH)
        self.assertNotEquals(0, len(y_reg))

        y = handler.predict(x, vars, TEST_MODEL_PATH)
        y_reg = handler.predict(x, vars, TEST_REG_MODEL_PATH)
        y = handler.predict(x, vars, TEST_MODEL_PATH)
        y_reg = handler.predict(x, vars, TEST_REG_MODEL_PATH)



if __name__ == '__main__':
    unittest.main()
