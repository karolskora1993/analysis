from sklearn.preprocessing import StandardScaler
from abc import ABC, abstractmethod

class DataStandarizer(ABC):
    def fit(self, x_train, y_train):
        self._input_scaler = StandardScaler()
        self._input_scaler.fit(x_train)
        self._output_scaler = StandardScaler()
        self._output_scaler.fit(y_train.reshape(-1, 1))

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
