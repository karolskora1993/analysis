import pandas as pd
from analysis import filesToCut

class Pack(object):

    sheet = None
    columnLabels = None
    newData = None;
    gaps = None

    def __init__(self):
        print("Warning! Test init!")

    def __init__(self, fileName, sheetName, gaps):
        super()
        self.openFile(fileName, sheetName)
        self.gaps = gaps
        self.createData()

    def cut(self):
        print("removing data started")
        for label in self.columnLabels:
            self.newData[label] = self.sheet[label][0::self.gaps]

        return pd.DataFrame(self.newData)

    def openFile(self, fileName, sheetName):
        self.sheet = pd.read_excel(fileName, sheetName)
        print("File opened")

    def createData(self):
        self.columnLabels = self.sheet.keys()
        self.newData = {label: [] for label in self.columnLabels}
        print("Data created")


def main():
    for file in filesToCut.files:
        pack = Pack(file['name'] + '.xlsx', file['sheet_name'], file['gaps'])

        df = pack.cut()
        df.to_csv(file['destinationName'] + '.csv')
        print('file ' + file['name'] + ' interpolated')


if __name__ == '__main__':
    main()
