import unittest
from unittest.mock import patch
from alpha_vantage import fetch_technical_data, save_data

class TestLoadData(unittest.TestCase):

    @patch('1_alpha_vantage.requests.get')
    def test_fetch_technical_data(self, mock_get):
        mock_response = mock_get.return_value
        mock_response.json.return_value = {
            'Technical Analysis: SMA': {
                '2023-01-01': {'SMA': '10.0'},
                '2023-01-02': {'SMA': '11.0'},
            }
        }
        
        symbol = 'SPY'
        tech = ['SMA', 50, 'Technical Analysis: SMA']
        result = fetch_technical_data(symbol, tech)

        self.assertEqual(result.shape, (2, 1))
        self.assertEqual(result.loc['2023-01-01', 'SMA'], 10.0)
        self.assertEqual(result.loc['2023-01-02', 'SMA'], 11.0)

if __name__ == '__main__':
    unittest.main()
