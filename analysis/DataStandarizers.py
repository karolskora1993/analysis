from sklearn.preprocessing import StandardScaler
from abc import ABC, abstractmethod
import pickle
import os
HOME_PATH = str(os.path.expanduser('~')+'/')
LOAD_PATH = HOME_PATH + 'Dokumenty/analysis/data/scalers/'

class DataStandarizer(ABC):

    def fit(self, x_train, y_train):
        self._input_scaler = StandardScaler()
        self._input_scaler.fit(x_train)
        self._output_scaler = StandardScaler()
        self._output_scaler.fit(y_train.reshape(-1, 1))
        # pickle.dump(self._input_scaler, open(LOAD_PATH + 'blokIV_in_scaller.p', 'wb'))
        # pickle.dump(self._output_scaler, open(LOAD_PATH + 'blokIV_out_{0}_scaller.p'.format(self.var_out), 'wb'))
        # print("fit")

    @abstractmethod
    def standarize_data(self, input_data, output_data):
        pass


class SimpleStandarizer(DataStandarizer):
    def standarize_data(self, input_data, output_data):
        if not isinstance(input_data, list) and not isinstance(output_data, list):
            raise ValueError('input_data and output_data are not a list')
        standarized_input_data = []
        standarized_output_data = []
        for single_input in input_data:
            standarized_input_data.append(self._input_scaler.transform(single_input))
        for single_output in output_data:
            standarized_output_data.append(self._output_scaler.transform(single_output.reshape(-1, 1)))

        return standarized_input_data, standarized_output_data
