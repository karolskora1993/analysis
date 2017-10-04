import pandas as pd
from analysis import filesToCut

class DataPacker(object):

    def __init__(self):
        print("Warning! Test init!")

    def __init__(self, fileName, sheetName, gaps):
        super()
        self._open_file(fileName, sheetName)
        self._gaps = gaps
        self.createData()

    def cut(self):
        print("removing data started")
        for label in self._column_labels:
            self._new_data[label] = self._sheet[label][0::self._gaps]

        return pd.DataFrame(self._new_data)

    def _open_file(self, fileName, sheetName):
        self._sheet = pd.read_excel(fileName, sheetName)
        print("File opened")

    def createData(self):
        self._column_labels = self._sheet.keys()
        self._new_data = {label: [] for label in self._column_labels}
        print("Data created")


def _main():
    for file in filesToCut.files:
        pack = DataPacker(file['name'] + '.xlsx', file['sheet_name'], file['gaps'])

        df = pack.cut()
        df.to_csv(file['destinationName'] + '.csv')
        print('file ' + file['name'] + ' interpolated')


if __name__ == '__main__':
    _main()
