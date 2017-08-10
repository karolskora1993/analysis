import pandas as pd
import pickle
from math import floor

VARS_FILE_PATH = '/Users/apple/Desktop/mag/temperatury.xlsx'
DATA_PATH = '/Users/apple/Desktop/mag/dane/DANE_PO_MODERNIZACJI_VRM/wypelnione/csv_all_v3.csv'
SAVE_PATHS = ['/Users/apple/Desktop/mag/przestoje_polowa.p',
              '/Users/apple/Desktop/mag/przestoje_wszystkie.p'
              ]


class OutlierFinder:

    def __init__(self, vars_file_path, data_file_path, save_path):
        self._vars = pd.read_excel(vars_file_path)
        self._data = pd.read_csv(data_file_path)
        self._vars_length = len(self._vars.index)
        self._save_path = save_path
        print('data loaded')


    def find_outliers(self, tolerance = 0.5):
        outliers = 0
        outlier_indexes = []
        counter = 0
        for i, row in self._data.iterrows():
            if i != 0:
                outliers_in_row = 0
                for j, var in self._vars.iterrows():
                    if row[var['nazwa']] <= var['min'] or row[var['nazwa']] >= var['max']:
                        outliers_in_row += 1
                if outliers_in_row > floor((1-tolerance) * self._vars_length):
                    outliers += 1
                else:
                    if outliers >= 30:
                        counter += 1
                        outlier_indexes.append({'first_idx': i-outliers, 'last_idx': i})
                        print('{0}. first_idx:{1}, last_idx:{2}'.format(counter, i-outliers, i))
                    outliers = 0
        self.save(outlier_indexes)

    def save(self, outlier_indexes):
        pickle.dump(outlier_indexes, open(self._save_path, 'wb'))


def main():
    finder = OutlierFinder(VARS_FILE_PATH, DATA_PATH, SAVE_PATHS[0])
    finder.find_outliers()

if __name__ == '__main__':
    main()
