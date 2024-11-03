import doctest
import unittest
import diPLSlib.models
import diPLSlib.functions
import diPLSlib.utils.misc

class TestDocstrings(unittest.TestCase):
    def test_docstrings(self):

        # Run doctests across all modules in your_package
        doctest.testmod(diPLSlib.models)
        doctest.testmod(diPLSlib.functions)
        doctest.testmod(diPLSlib.utils.misc)
        

if __name__ == '__main__':
    unittest.main()