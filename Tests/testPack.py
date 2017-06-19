import unittest
from analysis.DataPacker import DataPacker

testGaps = 6
numberOfLabels = 6
numberOfTestRows = 121

class TestPack(unittest.TestCase):
    testInstance = DataPacker("./TestData/cut_test_data.xlsx", "TestSheet", testGaps)
    df = None

    def testShouldOpenFile(self):
        self.assertIsNotNone(self.testInstance.sheet, "sheet not created")

    def testShouldCreateData(self):
        self.assertIsNotNone(self.testInstance.columnLabels)
        self.assertIsNotNone(self.testInstance.newData)


    def testShouldCutData(self):
        df = self.testInstance.cut()
        for label in self.testInstance.columnLabels:
            self.assertEqual((numberOfTestRows-1)/testGaps + 1, len(df[label]))


def testPack():
    unittest.main()

if __name__ == '__main__':
    testPack()
