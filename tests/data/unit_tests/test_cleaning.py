import unittest

class TestDataCleaner(unittest.TestCase):
    
    def setUp(self):
        """Set up different datasets to test."""
        return super().setUp()
    
    def test_lowercase_columns(self):
        """Make sure cleaner converts column names to lower case."""
        pass