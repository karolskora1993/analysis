import unittest

from analysis import BasicStats as bs

TEST_FILE_PATH = './TestData/basic_stats_test_data.csv'
numberOfTestColumns = 8
numberOfCorrMethods = 3
numberOfStats = 7

class TestPack(unittest.TestCase):

    testDf = None
    stats = None
    corrs = None
    def testShouldLoadDataFrame(self):
        self.testDf = bs._load_data_frame(TEST_FILE_PATH)
        self.assertIsNotNone(self.testDf)
        self.shouldCalculateStats()
        self.shouldCalculateCorr()
        # self.statsForEachVarShouldHaveGivenSize()

    def shouldCalculateStats(self):
        self.stats = bs.calculate_stats(self.testDf)
        self.assertIsNotNone(self.stats)
        self.assertEqual(numberOfTestColumns, len(self.stats))

    def shouldCalculateCorr(self):
        self.corrs = bs.calculate_corr(self.testDf)
        self.assertIsNotNone(self.corrs)
        self.assertEqual(numberOfCorrMethods, len(self.corrs))



def testPack():
    unittest.main()

if __name__ == '__main__':
    testPack()