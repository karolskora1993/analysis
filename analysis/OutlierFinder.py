import pandas as pd

VARS_FILE_PATH = '/Users/apple/Desktop/mag/temperatury.xlsx'
DATA_PATH = '/Users/apple/Desktop/mag/dane/DANE_PO_MODERNIZACJI_VRM/wypelnione/csv_all_v3.csv'


class OutlierFinder:

    def __init__(self, vars_file_path, data_file_path):
        self._vars = pd.read_excel(vars_file_path)
        self._data = pd.read_csv(data_file_path)
        self._vars_length = len(self._vars.index)


    def find_outliers(self, remove = False, outliers_perc = 1):
        outliers = 0
        outlier_indexes = {}
        for i, row in self._data.iterrows():
            if i != 0:
                outliers_in_row = 0
                for j, var in self._vars.iterrows():
                    if row[var['nazwa']] <= self._vars[var['min']]:
                        outliers_in_row += 1
                if outliers_in_row >= self._vars_length * outliers_perc:
                    outliers += 1
                else:
                    if outliers >= 30:
                        outlier_indexes.append({'first_idx': i-outliers, 'last_idx:': i})



    def removeOutliers(self):
        print('TODO')





def main():
    finder = OutlierFinder(VARS_FILE_PATH, DATA_PATH)
    finder.find_outliers()

if __name__ == '__main__':
    main()