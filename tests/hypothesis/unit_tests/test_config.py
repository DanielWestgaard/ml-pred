import unittest

import tests.utils.test_hypothesis_utils


class TestConfigurationFile(unittest.TestCase):
    
    def setUp(self):
        """Set up different config files"""
        return super().setUp()
    
    def test_loading_file(self):
        """Load config file."""