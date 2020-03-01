"""
Tests for helper functions.
"""


import unittest

import helpers



class test_helpers(unittest.TestCase):


    def test_address_to_neighborhood(self):

        self.assertEqual( helpers.address_to_neighborhood('134 Mandela/Oakland / Berkeley/ San Francisco'), 'Oakland/Berkeley/San Francisco')
