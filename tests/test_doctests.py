import doctest
import unittest
import diPLSlib.models
import diPLSlib.functions
import diPLSlib.utils.misc
from sklearn.utils.estimator_checks import check_estimator
from diPLSlib.models import DIPLS, GCTPLS, EDPLS
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import os

class TestDocstrings(unittest.TestCase):

    # Test if all docstring examples run without errors
    def test_docstrings(self):

        # Run doctests across all modules in your_package
        doctest.testmod(diPLSlib.models)
        doctest.testmod(diPLSlib.functions)
        doctest.testmod(diPLSlib.utils.misc)

    # Test if all notebooks run without errors
    def test_notebooks(self):
        # List of notebooks to test
        notebooks = [
            './notebooks/demo_diPLS.ipynb',
            './notebooks/demo_mdiPLS.ipynb',
            './notebooks/demo_gctPLS.ipynb',
            './notebooks/demo_edPLS.ipynb'
        ]
        
        for notebook in notebooks:
            with open(notebook) as f:
                nb = nbformat.read(f, as_version=4)
                ep = ExecutePreprocessor(timeout=600, kernel_name='python3')

                # Set the working directory to the root of the project
                root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
                os.chdir(root_dir)

                ep.preprocess(nb, {'metadata': {'path': './notebooks/'}})

        # Test if diPLSlib.model classes pass check_estimator
        def test_check_estimator(self):

            for model in [DIPLS, GCTPLS, EDPLS]:
                check_estimator(model)
        

if __name__ == '__main__':
    unittest.main()