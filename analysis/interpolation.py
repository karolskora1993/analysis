import pandas as pd
import numpy as np
from analysis import FilesToInterpolate

class Interpolation(object):

    sheet = None
    columnLabels = None
    newElements = None
    newData = None
    gapsAfter = 0

    def __init__(self):
        print("Warning! Test init!")

    def __init__(self, fileName, sheetName, gaps, gapsAfterLast=0):
        super()
        self.openFile(fileName, sheetName)
        self.createData(gaps)
        self.gapsAfter = gapsAfterLast

    def linear(self):
        self.extendData()
        print("interpolation started")
        df = pd.DataFrame(self.newData)
        for label in self.columnLabels:
            if label == 'Timestamp':
                df[label] = self.interpolate_time(df[label])
            else:
                df[label] = self.interpolate_numbers(df[label])
        return df

    def openFile(self, fileName, sheetName):
        self.sheet = pd.read_excel(fileName, sheetName)
        print("File opened")

    def interpolate_time(self, timeSeries):
        ts = pd.Series(timeSeries.values.astype('int64'))
        ts[ts < 0] = np.nan
        ts = pd.to_datetime(ts.interpolate(), unit='ns')
        ts = ts.values.astype('<M8[m]')
        return ts

    def interpolate_numbers(self, series):
        return series.interpolate(method='linear')

    def createData(self, gaps):
        self.columnLabels = self.sheet.keys()
        self.newData = {label: [] for label in self.columnLabels}
        self.newElements = [np.NaN] * gaps
        print("Data created")


    def extendData(self):
        for label in self.columnLabels:
            self.newData[label].append(self.sheet[label][0])
            for element in self.sheet[label][1:-1]:
                self.newData[label].extend(self.newElements)
                self.newData[label].append(element)

            if self.gapsAfter != 0:
                self.newData[label].extend([np.NaN] * self.gapsAfter)
                self.newData[label].append(self.sheet[label].iloc[-1])
            else:
                self.newData[label].extend(self.newElements)
                self.newData[label].append(self.sheet[label].iloc[-1])

        print("data extended")


def main():

    for file in FilesToInterpolate.files:
        interpolation = Interpolation(file['name'] + '.xlsx', file['sheet_name'], file['gaps'], file['gapsAfter'])
        df = interpolation.linear()
        df.to_csv(file['destinationName'] + '.csv')
        print('file ' + file['name'] + ' interpolated')


if __name__ == '__main__':
    main()
