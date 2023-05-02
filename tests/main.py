import unittest

# Import test modules
from test_alpha_vantage import TestAlphaVantage
from test_process_raw_data import TestProcessRawData
from test_train_model import TestTrainModel

def run_tests():
    # Create a test suite combining all test cases
    test_suite = unittest.TestSuite()

    test_suite.addTest(unittest.makeSuite(TestAlphaVantage))
    test_suite.addTest(unittest.makeSuite(TestProcessRawData))
    test_suite.addTest(unittest.makeSuite(TestTrainModel))

    # Run the test suite
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(test_suite)

if __name__ == '__main__':
    run_tests()