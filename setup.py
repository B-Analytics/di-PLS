import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="diPLSlib",
    version="1.0.0",
    author="Ramin Nikzad-Langerodi",
    author_email="ramin.nikzad-langerodi@scch.at",
    description="Python package for domain adaptation in multivariate regression",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/B-Analytics/di-PLS",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        'numpy',
        'matplotlib',
        'scipy',
    ]
)