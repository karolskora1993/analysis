import unittest
from analysis.DataPacker import DataPacker

testGaps = 6
numberOfLabels = 6
numberOfTestRows = 121

class TestPack(unittest.TestCase):
    testInstance = DataPacker("./TestData/cut_test_data.xlsx", "TestSheet", testGaps)
    df = None

    def testShouldOpenFile(self):
        self.assertIsNotNone(self.testInstance._sheet, "sheet not created")

    def testShouldCreateData(self):
        self.assertIsNotNone(self.testInstance._column_labels)
        self.assertIsNotNone(self.testInstance._new_data)


    def testShouldCutData(self):
        df = self.testInstance.cut()
        for label in self.testInstance._column_labels:
            self.assertEqual((numberOfTestRows-1)/testGaps + 1, len(df[label]))


def testPack():
    unittest.main()

if __name__ == '__main__':
    testPack()
