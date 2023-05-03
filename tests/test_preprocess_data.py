import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import unittest
import pandas as pd
from src.process_raw_data import process_dataframe as test_preprocess_data

class TestPreprocessData(unittest.TestCase):

    def test_preprocess_data(self):
        raw_data = pd.DataFrame({
            '1. open': [10.0, 11.0],
            '2. high': [12.0, 13.0],
            '3. low': [9.0, 10.0],
            '4. close': [11.0, 12.0],
            '5. adjusted close': [11.0, 12.0],
            '6. volume': [1000, 2000],
            '7. dividend amount': [0.0, 0.0],
            '8. split coefficient': [1.0, 1.0],
            'RSI': [30, 40]
        }, index=['2023-01-01', '2023-01-02'])


        result = test_preprocess_data(raw_data, 21)

        self.assertEqual(result.shape, (2, 6))
        self.assertEqual(result.loc['2023-01-01', 'open'], 10.0)
        self.assertEqual(result.loc['2023-01-02', 'high'], 13.0)

if __name__ == '__main__':
    unittest.main()
