import doctest
import unittest
import diPLSlib.models
import diPLSlib.functions
import diPLSlib.utils.misc
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import os

class TestDocstrings(unittest.TestCase):
    def test_docstrings(self):

        # Run doctests across all modules in your_package
        doctest.testmod(diPLSlib.models)
        doctest.testmod(diPLSlib.functions)
        doctest.testmod(diPLSlib.utils.misc)

    def test_notebooks(self):
        # List of notebooks to test
        notebooks = [
            './notebooks/demo_diPLS.ipynb',
            './notebooks/demo_mdiPLS.ipynb',
            './notebooks/demo_gctPLS.ipynb'
        ]
        
        for notebook in notebooks:
            with open(notebook) as f:
                nb = nbformat.read(f, as_version=4)
                ep = ExecutePreprocessor(timeout=600, kernel_name='python3')

                # Set the working directory to the root of the project
                root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
                os.chdir(root_dir)

                ep.preprocess(nb, {'metadata': {'path': './notebooks/'}})
        

if __name__ == '__main__':
    unittest.main()