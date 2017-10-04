from abc import ABC, abstractmethod

class ModelTester(ABC):
    @abstractmethod
    def test_model(self, model, x_test, y_test):
        pass


class SimpleTester(ModelTester):
    def test_model(self, model, x_test, y_test, y_train):
        predictions = model.predict(x_test)
        return self._r_2score(y_test, predictions, y_train)

    def _r_2score(y_true, y_pred, y_mean):
        mean = y_mean.mean()
        ss_res = sum((y_true - y_pred) ** 2)
        ss_tot = sum((y_true - mean) ** 2)
        return 1 - ss_res / ss_tot
