import unittest
from analysis import neuralnetworks

LAST_TRAIN_IDX = 2
LAST_VALIDATE_IDX = 3
HOME_PATH = neuralnetworks.HOME_PATH

MODEL_SAVE_PATH = HOME_PATH + 'Dokumenty/analysis/Tests/TestData/'
SCORE_SAVE_PATH = HOME_PATH + 'Dokumenty/analysis/Tests/TestData/'
BLOCK_NAMES = [
    'blok III'
]

class TestNeuralNetworks(unittest.TestCase):

    def setUp(self):
        neuralnetworks.MODEL_SAVE_PATH = MODEL_SAVE_PATH
        neuralnetworks.SCORE_SAVE_PATH = SCORE_SAVE_PATH

    def create_model(self):
        neuralnetworks.main()


def testNeuralNetworks():
    unittest.main()

if __name__ == '__main__':
    testNeuralNetworks()