# Changelog

All notable changes to this project will be documented in this file.

## [unrelease] - 2024-11-05

### Added
- 'check_estimator()' added to the doctests to validate model classes.

### Changed
- Input/Output validation added to diPLSlib.models and diPLSlib.functions.
- Changed public to private attributes added in the 'fit()' method.
- Notebooks adapted.
- 'demo_ModelSelection_SciKitLearn.ipynb' added.
- Tests excecuted properly.

### Fixed
N/A

### Removed
N/A

### Breaking Changes
- Changed type of parameter 'l' from Union[int, List[int]] to Union[float, tuple] in  'DIPLS' and 'GCTPLS' classes.

## [2.3.0] - 2024-11-06

### Added
- New feature for model selection using cross-validation.
- Additional unit tests for new features.
- Documentation for the new model selection feature.

### Changed
- Refactored code for better readability and maintainability.
- Updated dependencies to the latest versions.

### Fixed
- Fixed a bug in the `predict()` method of the `DIPLS` class.

### Removed
- Deprecated methods removed from `diPLSlib.utils`.

### Breaking Changes
- Refactored `fit()` method signature in `DIPLS` and `GCTPLS` classes.

[2.3.0]: https://github.com/B-Analytics/di-PLS/releases/tag/v2.3.0
[2.2.1]: https://github.com/B-Analytics/di-PLS/releases/tag/v2.2.1

## [2.2.1] - 2024-11-04

### Added
N/A

### Changed
N/A

### Fixed
- Bug in the extraction of the number of samples nt in the fit method corrected.
- Tested correct behavior in the notebooks.

### Removed
N/A

[2.2.1]: https://github.com/B-Analytics/di-PLS/releases/tag/v2.2.1

## [2.2.0] - 2024-11-02

### Added
- Unittests for models, functions and utils

### Changed
- DIPLS and GCTPLS classes now compatible with sklearn.
- Documentation updated.

### Fixed
- N/A

### Removed
- N/A

[2.2.0]: https://github.com/B-Analytics/di-PLS/releases/tag/v2.2.0

## [2.1.0] - 2024-11-02
### Added
- utils submodule added to outsource utility functions.
- Documentation added

### Changed
- N/A

### Fixed
- N/A

### Removed
- N/A

[2.1.0]: https://github.com/B-Analytics/di-PLS/releases/tag/v2.1.0

## [2.0.0] - 2024-10-30
### Added
- Major overhaul of the project architecture.
- New 'GCTPLS' class for Calibration Transfer.
- Demo notebook for GCT-PLS.
- Data Repository for the demo notebook.
- Changelog added.

### Changed
- Changed class names from 'model' to 'DIPLS' and 'GCTPLS'.

### Fixed
- Minor bug fixes related to predict function.

### Removed
- N/A

[2.0.0]: https://github.com/B-Analytics/di-PLS/releases/tag/v2.0.0

## [1.0.2] - 2024-10-30
### Added
- N/A

### Changed
- N/A

### Fixed
- Installation and Usage sections in documentation.

### Removed
- N/A

[1.0.2]: https://github.com/B-Analytics/di-PLS/releases/tag/v1.0.2

## [1.0.0] - 2024-10-30
### Added
- Initial release of the project.
- 'Model' class with 'fit' and 'predict' methods.
- Support for domain adaptation scenarios with multiple domains.

### Changed
- N/A

### Fixed
- N/A

### Removed
- N/A

[1.0.0]: https://github.com/B-Analytics/di-PLS/releases/tag/v1.0.0




















































