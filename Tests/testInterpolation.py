import unittest
from analysis.interpolation import Interpolation

testGaps = 70-1
numberOfLabels = 6
numberOfTestRows = 6

class TestInterpolation(unittest.TestCase):
    interpolation = Interpolation("./TestData/test_data.xlsx", "TestSheet", testGaps)
    df = None

    def testShouldOpenFile(self):
        self.assertIsNotNone(self.interpolation.sheet, "sheet not created")

    def testShouldCreateData(self):
        self.assertIsNotNone(self.interpolation.columnLabels, "column labels not created")
        self.assertIsNotNone(self.interpolation.newElements, "new elements list not created")
        self.assertIsNotNone(self.interpolation.newData, "newData dict not created")


    def testShouldInterpolateNans(self):
        self.df = self.interpolation.linear()
        self.assertEqual(0, self.df.isnull().sum().sum())
        self.shouldExtendDataFrameToGivenSize()

    def shouldExtendDataFrameToGivenSize(self):
        for label in self.interpolation.columnLabels:
            self.assertEqual((numberOfTestRows - 1) * (testGaps + 1) +1, len(self.df[label]))


def testInterpolation():
    unittest.main()

if __name__ == '__main__':
    testInterpolation()
